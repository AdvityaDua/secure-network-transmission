import os
import torch
import pytest
from src.network.crypto import LatentCrypto

@pytest.fixture
def temp_key_file(tmp_path):
    return str(tmp_path / "test_key.bin")

def test_key_generation(temp_key_file):
    key = LatentCrypto.generate_key(temp_key_file)
    assert len(key) == 32
    assert os.path.exists(temp_key_file)
    
    loaded_key = LatentCrypto.load_key(temp_key_file)
    assert key == loaded_key

def test_tensor_serialization():
    tensor = torch.randn(1, 1024, 16, 16)
    meta, raw_bytes = LatentCrypto.serialize(tensor, quantize_fp16=False)
    
    assert meta["has_skips"] is False
    assert meta["bottleneck"]["shape"] == [1, 1024, 16, 16]
    
    reconstructed = LatentCrypto.deserialize(meta, raw_bytes)
    assert torch.allclose(tensor, reconstructed)

def test_tensor_with_skips_serialization():
    bottleneck = torch.randn(1, 1024, 16, 16)
    skips = (
        torch.randn(1, 1, 256, 256),
        torch.randn(1, 1, 128, 128)
    )
    
    latent = (bottleneck, skips)
    meta, raw_bytes = LatentCrypto.serialize(latent, quantize_fp16=False)
    
    assert meta["has_skips"] is True
    assert len(meta["skips"]) == 2
    
    recon_bottle, recon_skips = LatentCrypto.deserialize(meta, raw_bytes)
    
    assert torch.allclose(bottleneck, recon_bottle)
    assert torch.allclose(skips[0], recon_skips[0])
    assert torch.allclose(skips[1], recon_skips[1])

def test_encryption_decryption(temp_key_file):
    key = LatentCrypto.generate_key(temp_key_file)
    plaintext = b"hello world"
    aad = b"header data"
    
    nonce, ciphertext = LatentCrypto.encrypt(key, plaintext, aad)
    assert len(nonce) == 12
    assert ciphertext != plaintext
    
    decrypted = LatentCrypto.decrypt(key, nonce, ciphertext, aad)
    assert decrypted == plaintext

def test_pack_unpack(temp_key_file):
    key = LatentCrypto.generate_key(temp_key_file)
    latent = torch.randn(1, 1024, 16, 16)
    
    wire_bytes = LatentCrypto.pack(latent, key, "test_model", quantize_fp16=False)
    assert isinstance(wire_bytes, bytes)
    
    unpacked_latent, header = LatentCrypto.unpack(wire_bytes, key)
    
    assert header["model_name"] == "test_model"
    assert torch.allclose(latent, unpacked_latent)

def test_tamper_resistance(temp_key_file):
    from cryptography.exceptions import InvalidTag
    
    key = LatentCrypto.generate_key(temp_key_file)
    latent = torch.randn(1, 1024, 16, 16)
    wire_bytes = bytearray(LatentCrypto.pack(latent, key, "test_model", quantize_fp16=False))
    
    # Tamper with the ciphertext (last byte)
    wire_bytes[-1] ^= 0xFF
    
    with pytest.raises(InvalidTag):
        LatentCrypto.unpack(bytes(wire_bytes), key)
