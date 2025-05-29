#!/usr/bin/env python3
"""
Quick installation test for Zonos TTS
"""

import sys
import os


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")

    try:
        import torch

        print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import torchaudio

        print(f"✓ TorchAudio {torchaudio.__version__}")
    except ImportError as e:
        print(f"✗ TorchAudio import failed: {e}")
        return False

    try:
        from phonemizer.backend import EspeakBackend

        print("✓ Phonemizer imported successfully")
    except ImportError as e:
        print(f"✗ Phonemizer import failed: {e}")
        return False

    try:
        import zonos

        print("✓ Zonos imported successfully")
    except ImportError as e:
        print(f"✗ Zonos import failed: {e}")
        return False

    return True


def test_espeak():
    """Test eSpeak availability"""
    print("\nTesting eSpeak...")

    espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    if not os.path.exists(espeak_path):
        print(f"✗ eSpeak not found at {espeak_path}")
        return False

    print(f"✓ eSpeak found at {espeak_path}")
    return True


def main():
    print("Zonos Installation Test")
    print("=" * 30)

    success = True

    if not test_imports():
        success = False

    if not test_espeak():
        success = False

    print("\n" + "=" * 30)
    if success:
        print("✓ Installation test PASSED - Ready to use TTS!")
        print("\nTo test TTS generation:")
        print('  tts.cmd "test_sample.txt"')
    else:
        print("✗ Installation test FAILED - Some components missing")
        print("\nPlease check the error messages above")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
