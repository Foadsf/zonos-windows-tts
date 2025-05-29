#!/usr/bin/env python3
"""
Simple TTS test to diagnose issues
"""

import os
import sys

# Configure PyTorch before any imports
import torch
import torch._dynamo

torch._dynamo.config.suppress_errors = True
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_JIT"] = "0"


def test_simple_generation():
    print("Testing simple TTS generation...")

    # Configure eSpeak
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = (
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )

    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        from zonos.utils import DEFAULT_DEVICE as device
        import torchaudio

        print(f"Loading model on device: {device}")
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        model.eval()

        # Test with very short text
        test_text = "Hello world"
        print(f"Testing with text: '{test_text}'")

        with torch.no_grad():
            cond_dict = make_cond_dict(text=test_text, speaker=None, language="en-us")
            conditioning = model.prepare_conditioning(cond_dict)

            print("Generating speech...")
            codes = model.generate(conditioning)
            wavs = model.autoencoder.decode(codes).cpu()

            output_file = "test_simple.wav"
            torchaudio.save(output_file, wavs[0], model.autoencoder.sampling_rate)

            print(f"Success! Generated: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

    return True


if __name__ == "__main__":
    if test_simple_generation():
        print("Simple TTS test passed!")
    else:
        print("Simple TTS test failed!")
        sys.exit(1)
