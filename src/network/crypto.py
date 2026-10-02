"""
crypto.py — AES-256-GCM encryption for latent tensors + TCP wire helpers.

Install dependency:
    pip install cryptography

Wire message layout (produced by pack(), consumed by unpack()):
    [ 4 B : header_len uint32 big-endian ]
    [ header_len B : UTF-8 JSON          ]  ← AAD: nonce, shape, model info (tamper-proof, NOT encrypted)
    [ remaining B : AES-GCM ciphertext   ]  ← encrypted latent bytes + 16-byte auth tag

TCP outer frame (added by tcp_send / tcp_recv):
    [ 8 B : total_msg_len uint64 big-endian ]
    [ total_msg_len B : wire message        ]
"""

import os
import json
import struct
import base64
import zlib

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

DEFAULT_KEY_FILE = "aes_key.bin"
NONCE_BYTES = 12   # 96-bit nonce — GCM standard
KEY_BYTES   = 32   # 256-bit key  — AES-256


# ──────────────────────────────────────────────────────────────────────────────
# LatentCrypto
# ──────────────────────────────────────────────────────────────────────────────

class LatentCrypto:
    """
    Serialize, encrypt, decrypt, and deserialize latent tensors.

    Supports two latent formats:
      • torch.Tensor                              — bottleneck-only models
                                                    (ResAE, CBAM-XL, SE-XL, InceptionAE, UNet-noskip)
      • (torch.Tensor, tuple[torch.Tensor, ...])  — compressed-skip models
                                                    (unet_one, unet_res_xl)
    """

    # ── Key management ────────────────────────────────────────────────────

    @staticmethod
    def generate_key(path: str = DEFAULT_KEY_FILE) -> bytes:
        """Generate a fresh 256-bit key and write it to disk. Run once."""
        key = os.urandom(KEY_BYTES)
        with open(path, "wb") as f:
            f.write(key)
        print(f"[crypto] Generated AES-256 key → {path}")
        return key

    @staticmethod
    def load_key(path: str = DEFAULT_KEY_FILE) -> bytes:
        """Load key from file. Both sender and receiver must use the same file."""
        with open(path, "rb") as f:
            key = f.read()
        if len(key) != KEY_BYTES:
            raise ValueError(f"Expected {KEY_BYTES}-byte key, got {len(key)}")
        return key

    # ── Serialization ─────────────────────────────────────────────────────

    @staticmethod
    def _t2mb(t: torch.Tensor, quantize_fp16: bool = True) -> tuple[dict, bytes]:
        """Tensor → (meta_dict, raw_bytes). Moves to CPU, optionally casts to float16."""
        t = t.cpu().detach()
        if quantize_fp16:
            t = t.half()
        else:
            t = t.float()
        arr = t.numpy()
        return {"shape": list(arr.shape), "dtype": str(arr.dtype)}, arr.tobytes()

    @staticmethod
    def _b2t(raw: bytes, meta: dict) -> torch.Tensor:
        """raw_bytes + meta_dict → Tensor."""
        arr = np.frombuffer(raw, dtype=np.dtype(meta["dtype"])).reshape(meta["shape"])
        return torch.from_numpy(arr.copy()).float()

    @staticmethod
    def serialize(latent, quantize_fp16: bool = True) -> tuple[dict, bytes]:
        """
        Converts latent (tensor or tensor+skips) into (meta_dict, raw_bytes).

        meta_dict describes the layout of raw_bytes so deserialize() can
        reconstruct the correct tensor(s).
        """
        if isinstance(latent, torch.Tensor):
            m, b = LatentCrypto._t2mb(latent, quantize_fp16)
            meta = {
                "has_skips": False,
                "bottleneck": {**m, "nbytes": len(b)},
            }
            return meta, b

        # Compressed-skip model: (bottleneck_tensor, (skip1, skip2, skip3, skip4))
        bottleneck, skips = latent
        bm, bb = LatentCrypto._t2mb(bottleneck, quantize_fp16)
        all_bytes  = bb
        skip_metas = []

        for s in skips:
            sm, sb = LatentCrypto._t2mb(s, quantize_fp16)
            skip_metas.append({**sm, "nbytes": len(sb)})
            all_bytes += sb

        meta = {
            "has_skips":  True,
            "bottleneck": {**bm, "nbytes": len(bb)},
            "skips":      skip_metas,
        }
        return meta, all_bytes

    @staticmethod
    def deserialize(meta: dict, raw: bytes):
        """Inverse of serialize(). Returns tensor or (tensor, tuple[tensor,...])."""
        offset = 0
        nb = meta["bottleneck"]["nbytes"]
        bottleneck = LatentCrypto._b2t(raw[offset : offset + nb], meta["bottleneck"])
        offset += nb

        if not meta["has_skips"]:
            return bottleneck

        skips = []
        for sm in meta["skips"]:
            s = LatentCrypto._b2t(raw[offset : offset + sm["nbytes"]], sm)
            skips.append(s)
            offset += sm["nbytes"]

        return bottleneck, tuple(skips)

    # ── Low-level AES-GCM ─────────────────────────────────────────────────

    @staticmethod
    def encrypt(key: bytes, plaintext: bytes, aad: bytes) -> tuple[bytes, bytes]:
        """
        Returns (nonce, ciphertext_with_16B_auth_tag).
        A fresh 96-bit nonce is generated for every call.
        """
        nonce = os.urandom(NONCE_BYTES)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
        return nonce, ciphertext

    @staticmethod
    def decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        """
        Returns plaintext.
        Raises cryptography.exceptions.InvalidTag if authentication fails
        (tampered ciphertext or wrong key).
        """
        return AESGCM(key).decrypt(nonce, ciphertext, aad)

    # ── High-level wire packing ───────────────────────────────────────────

    @staticmethod
    def pack(latent, key: bytes, model_name: str = "unknown", quantize_fp16: bool = True, apply_zlib: bool = True) -> bytes:
        """
        Full pipeline: latent → serialize → AES-256-GCM encrypt → wire bytes.

        The JSON header (model name, tensor shapes, nonce) is sent in plaintext
        but is cryptographically bound to the ciphertext via GCM AAD — any
        tampering with the header will cause decryption to fail.

        Returns bytes ready for tcp_send().
        """
        meta, raw_bytes = LatentCrypto.serialize(latent, quantize_fp16)
        
        meta["quantize_fp16"] = quantize_fp16
        meta["apply_zlib"] = apply_zlib
        
        if apply_zlib:
            raw_bytes = zlib.compress(raw_bytes, level=9)

        # Generate nonce before building the header so it can be included in AAD
        nonce = os.urandom(NONCE_BYTES)

        header = {
            "model_name":    model_name,
            "nonce_b64":     base64.b64encode(nonce).decode(),
            "latent_meta":   meta,
            "payload_bytes": len(raw_bytes),
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

        # Encrypt latent bytes with header_bytes as authenticated additional data
        ciphertext = AESGCM(key).encrypt(nonce, raw_bytes, header_bytes)

        # [ 4B: header_len ][ header_bytes ][ ciphertext ]
        return struct.pack(">I", len(header_bytes)) + header_bytes + ciphertext

    @staticmethod
    def unpack(msg: bytes, key: bytes):
        """
        Full pipeline: wire bytes → AES-256-GCM decrypt → deserialize → latent.

        Returns (latent, header_dict).
        Raises cryptography.exceptions.InvalidTag on any integrity failure.
        """
        header_len   = struct.unpack(">I", msg[:4])[0]
        header_bytes = msg[4 : 4 + header_len]
        ciphertext   = msg[4 + header_len :]

        header = json.loads(header_bytes.decode("utf-8"))
        nonce  = base64.b64decode(header["nonce_b64"])

        # Decrypt — header_bytes is the AAD, must match exactly
        raw_bytes = AESGCM(key).decrypt(nonce, ciphertext, header_bytes)

        if header["latent_meta"].get("apply_zlib", False):
            raw_bytes = zlib.decompress(raw_bytes)

        latent = LatentCrypto.deserialize(header["latent_meta"], raw_bytes)
        return latent, header


# ──────────────────────────────────────────────────────────────────────────────
# TCP send / receive helpers
# ──────────────────────────────────────────────────────────────────────────────

def tcp_send(sock, data: bytes) -> None:
    """
    Send a complete message over TCP.
    Prepends an 8-byte big-endian uint64 length prefix so the receiver
    knows exactly how many bytes to read regardless of TCP segmentation.
    """
    frame = struct.pack(">Q", len(data)) + data
    sock.sendall(frame)


def tcp_recv(sock) -> bytes:
    """
    Receive a complete length-prefixed message from a TCP socket.
    Loops on recv() until all bytes arrive (handles TCP fragmentation).
    """
    length_bytes = _recv_exactly(sock, 8)
    total = struct.unpack(">Q", length_bytes)[0]
    return _recv_exactly(sock, total)


def _recv_exactly(sock, n: int) -> bytes:
    """Read exactly n bytes from socket, raising ConnectionError on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError(
                f"Socket closed after {len(buf)}/{n} bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


# ──────────────────────────────────────────────────────────────────────────────
# CLI: key generation utility
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate AES-256 key for latent encryption")
    parser.add_argument("--key", default=DEFAULT_KEY_FILE, help="Output key file path")
    args = parser.parse_args()

    LatentCrypto.generate_key(args.key)
    print(f"Key size: {KEY_BYTES * 8} bits")
    print("Share this file securely with the receiver (server) before running the pipeline.")
