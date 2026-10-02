import os
import sys
import csv
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from piq import ssim as compute_ssim, psnr as compute_psnr
from torch.utils.data import Subset

from src.data.dataloader import MimicDataset
from src.network.client import load_model

def get_decoder_params(model, victim_model, device):
    """Return only the parameters that are used by model.decode().

    Works for every architecture: we run a dummy encode → decode, call
    .backward(), and collect the parameters whose .grad is not None.
    Those are the ones that sit on the decode path.
    """
    # Make sure everything is on the same device and in train mode
    model.train()
    dummy = torch.randn(1, 1, 256, 256, device=device)

    # Zero all grads first
    model.zero_grad()

    with torch.no_grad():
        latents = victim_model.encode(dummy)

    def _detach_recursive(obj):
        """Detach tensors in (possibly nested) tuples, re-enabling grad."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().requires_grad_(True)
        elif isinstance(obj, (tuple, list)):
            return type(obj)(_detach_recursive(x) for x in obj)
        return obj

    if isinstance(latents, tuple):
        # Detach and re-enable grad so backward flows only through decode
        latents = _detach_recursive(latents)
        out = model.decode(latents[0], latents[1])
    else:
        latents = latents.detach().requires_grad_(True)
        out = model.decode(latents)

    out.sum().backward()

    decoder_params = [p for p in model.parameters() if p.grad is not None]
    # Freeze everything that is NOT a decoder parameter
    decoder_param_set = set(id(p) for p in decoder_params)
    frozen_count = 0
    for p in model.parameters():
        if id(p) not in decoder_param_set:
            p.requires_grad = False
            frozen_count += 1

    model.zero_grad()
    print(f"  Decoder params: {len(decoder_params)}, Frozen encoder params: {frozen_count}")
    return decoder_params

def get_subset_loader(dataset, subset_size, batch_size):
    # Ensure subset_size doesn't exceed dataset size
    actual_size = min(subset_size, len(dataset))
    # Fixed seed for reproducible and nested subsets
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:actual_size].tolist()
    subset = Subset(dataset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0), actual_size

def run_adversarial_attack(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Starting Compute-Cost Adversarial Attack on {device}...")

    # 1. Load the Victim's Encoder (Pre-trained, Frozen)
    victim_model = load_model(args.model, args.base_channels, args.checkpoint, device)
    victim_model.eval()
    for param in victim_model.parameters():
        param.requires_grad = False

    # Datasets
    full_train_dataset = MimicDataset(args.train_csv, h=256, w=256)
    test_dataset = MimicDataset(args.test_csv, h=256, w=256)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # SPEEDUP: The victim encoder is frozen and never updated anywhere in this
    # script, so its output for a given test image is identical on epoch 1
    # and epoch 15, and identical across every dataset_size bucket too. The
    # original code called victim_model.encode(images) on the full test set
    # inside the per-epoch eval loop, recomputing the same forward pass over
    # and over for no reason. Encode the test set once, up front, and reuse
    # the cached latents everywhere below instead. This removes the victim
    # encoder's forward pass entirely from eval time.
    print("Precomputing victim latents for test set (one-time cost)...")
    test_latents_cache = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            latents = victim_model.encode(images)
            test_latents_cache.append((images, latents))
    print(f"Cached victim latents for {len(test_dataset)} test images.")

    # Prepare CSV Output
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, f"adversarial_compute_cost_{args.model}.csv")

    # Create or overwrite CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset_size", "epoch", "cumulative_images", "cumulative_train_time_sec", "cumulative_eval_time_sec", "test_psnr", "test_ssim"])

    # NOTE: matches the four diagnostic-usefulness thresholds used in Table 4
    # of the paper (Section 4.4). Previously this was [20.0, 25.0, 30.0],
    # which did not match what the paper actually reports.
    thresholds = [25.0, 30.0, 35.0, 40.0]

    dataset_sizes = args.dataset_sizes if args.dataset_sizes else [500, 2000, 10000, len(full_train_dataset)]

    for d_size in dataset_sizes:
        train_loader, actual_size = get_subset_loader(full_train_dataset, d_size, args.batch_size)
        print(f"\n=========================================================")
        print(f"EVALUATING DATASET SIZE: {actual_size}")
        print(f"=========================================================")

        # Fresh adversary for each dataset size run
        adversary_model = load_model(args.model, args.base_channels, checkpoint_path=None, device=device)
        adversary_decoder = adversary_model

        # FIX: Only optimize decoder parameters. The encoder half of the
        # adversary is never called (only decode() is used), so freeze it
        # to avoid wasted Adam state and ensure a clean decoder-only setup.
        decoder_params = get_decoder_params(adversary_decoder, victim_model, device)
        adversary_decoder.train()

        optimizer = Adam(decoder_params, lr=args.learning_rate)

        cumulative_images = 0
        cumulative_train_time = 0.0
        cumulative_eval_time = 0.0

        # Track threshold crossing
        crossed_thresholds = set()

        run_start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            train_time_start = time.time()

            # --- TRAINING PHASE ---
            adversary_decoder.train()
            train_loss = 0.0

            for batch_idx, images in enumerate(train_loader):
                images = images.to(device)

                with torch.no_grad():
                    latents = victim_model.encode(images)

                optimizer.zero_grad()
                if isinstance(latents, tuple):
                    reconstructed = adversary_decoder.decode(latents[0], latents[1])
                else:
                    reconstructed = adversary_decoder.decode(latents)

                loss = F.mse_loss(reconstructed, images)
                loss.backward()
                optimizer.step()

                batch_sz = images.size(0)
                train_loss += loss.item() * batch_sz
                cumulative_images += batch_sz

            train_loss /= actual_size

            # Calculate elapsed time for this epoch and update cumulative
            epoch_train_time = time.time() - train_time_start
            cumulative_train_time += epoch_train_time

            # --- TEST EVALUATION PHASE ---
            eval_time_start = time.time()
            adversary_decoder.eval()
            test_psnr = 0.0
            test_ssim = 0.0

            with torch.no_grad():
                for images, latents in test_latents_cache:
                    # NOTE: victim_model.encode() is no longer called here.
                    # `latents` comes straight from the precomputed cache built
                    # once before the dataset_sizes loop above.
                    if isinstance(latents, tuple):
                        reconstructed = adversary_decoder.decode(latents[0], latents[1])
                    else:
                        reconstructed = adversary_decoder.decode(latents)

                    # FIX: images come from MimicDataset via T.ToTensor(), which
                    # produces [0, 1] tensors (no Normalize((0.5,), (0.5,)) is
                    # applied anywhere in the pipeline). Every decoder also ends
                    # in a sigmoid, so `reconstructed` is [0, 1] too. The old
                    # code did `(x + 1.0) / 2.0` on both tensors as if they were
                    # already in [-1, 1], which instead squashed both into
                    # [0.5, 1.0], quartering the effective MSE and inflating
                    # every PSNR value here by a flat 10*log10(4) ~= 6.02 dB
                    # relative to how Table 1's legitimate-receiver PSNR is
                    # computed in evaluate_pipeline.py (plain MSE on the raw
                    # [0,1] tensors, no shift). That made the surrogate look
                    # like it crossed diagnostic thresholds far earlier than
                    # it actually did. Just use the tensors as-is.
                    orig_01 = images.clamp(0.0, 1.0)
                    recon_01 = reconstructed.clamp(0.0, 1.0)

                    psnr_val = compute_psnr(recon_01, orig_01, data_range=1.0)
                    if isinstance(psnr_val, (list, tuple)): psnr_val = psnr_val[0]
                    test_psnr += psnr_val.item() * images.size(0)

                    ssim_val = compute_ssim(recon_01, orig_01, data_range=1.0)
                    if isinstance(ssim_val, (list, tuple)): ssim_val = ssim_val[0]
                    test_ssim += ssim_val.item() * images.size(0)

            # Use type: ignore because PyTorch's base Dataset type doesn't guarantee __len__
            test_psnr /= len(test_loader.dataset)  # type: ignore
            test_ssim /= len(test_loader.dataset)  # type: ignore

            epoch_eval_time = time.time() - eval_time_start
            cumulative_eval_time += epoch_eval_time

            print(f"Epoch {epoch:2d} | Images Seen: {cumulative_images:6d} | Train Time: {cumulative_train_time:6.1f}s | Eval Time: {cumulative_eval_time:6.1f}s | Train Loss: {train_loss:.6f} | Test PSNR: {test_psnr:5.2f} | Test SSIM: {test_ssim:.4f}")

            # Log to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([actual_size, epoch, cumulative_images, cumulative_train_time, cumulative_eval_time, test_psnr, test_ssim])

            # Threshold Check
            for thresh in thresholds:
                if test_psnr >= thresh and thresh not in crossed_thresholds:
                    print(f"  >>> THRESHOLD CROSSED: {thresh} dB at Epoch {epoch} (Images: {cumulative_images}, Train Time: {cumulative_train_time:.1f}s)")
                    crossed_thresholds.add(thresh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training set CSV")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to testing set CSV")
    parser.add_argument("--model", type=str, default="unet_one")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/unet_one/checkpoints/best_model.pth")
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs per dataset size")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--dataset_sizes", type=int, nargs="+", help="Space-separated list of dataset sizes")

    args = parser.parse_args()
    run_adversarial_attack(args)