#!/usr/bin/env python3
"""
Simple TTS test to diagnose issues
"""
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_JIT"] = "0"

import sys

# Configure PyTorch specifics AFTER main torch import
import torch
import torch._dynamo

# REMOVE: torch._dynamo.config.suppress_errors = True

# Attempt to globally disable TorchDynamo
print("Attempting to globally disable TorchDynamo for test_simple_tts.py...")
try:
    torch._dynamo.reset()
    torch._dynamo.disable()
    print(f"  TorchDynamo is_enabled after disable(): {torch._dynamo.is_enabled()}")
    print(f"  TorchDynamo is_compiling after disable(): {torch._dynamo.is_compiling()}")
except Exception as e:
    print(f"  Warning: Could not fully disable TorchDynamo in test_simple_tts.py: {e}")

# Set torch to eager mode for inference
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def test_simple_generation():
    print("Testing simple TTS generation...")

    # Configure eSpeak paths (ensure these are correct for your system)
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = (
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )

    # Verify eSpeak paths exist
    if not os.path.exists(os.environ["PHONEMIZER_ESPEAK_PATH"]):
        print(f"eSpeak executable not found at {os.environ['PHONEMIZER_ESPEAK_PATH']}")
        return False
    if not os.path.exists(os.environ["PHONEMIZER_ESPEAK_LIBRARY"]):
        print(f"eSpeak library not found at {os.environ['PHONEMIZER_ESPEAK_LIBRARY']}")
        return False

    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        from zonos.utils import DEFAULT_DEVICE as device
        import torchaudio

        print(f"Loading model on device: {device}")
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        model.eval()  # Ensure model is in eval mode

        test_text = "Hello world"
        print(f"Testing with text: '{test_text}'")

        # No need for torch.no_grad() context if torch.set_grad_enabled(False) is global
        cond_dict = make_cond_dict(text=test_text, speaker=None, language="en-us")
        conditioning = model.prepare_conditioning(cond_dict)

        print("Generating speech...")
        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes).cpu()

        output_file = (
            "test_simple_output.wav"  # Changed name to avoid conflict with .gitignore
        )
        torchaudio.save(output_file, wavs[0], model.autoencoder.sampling_rate)

        print(f"Success! Generated: {output_file}")
        return True

    except Exception as e:
        print(f"Error during simple TTS test: {e}")
        print(f"Error type: {type(e).__name__}")
        # Add traceback for more detailed error info
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    if test_simple_generation():
        print("Simple TTS test passed!")
    else:
        print("Simple TTS test FAILED!")
        sys.exit(1)
