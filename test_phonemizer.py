#!/usr/bin/env python3
"""
Test phonemizer integration with eSpeak NG
"""

import os
import sys
import time


def main():
    print("Starting phonemizer integration test...")
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")

    # Set environment variables
    espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    espeak_lib = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

    print(f"Setting PHONEMIZER_ESPEAK_PATH to: {espeak_path}")
    print(f"Setting PHONEMIZER_ESPEAK_LIBRARY to: {espeak_lib}")

    os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_path
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib

    # Check if files exist
    print(f"eSpeak executable exists: {os.path.exists(espeak_path)}")
    print(f"eSpeak library exists: {os.path.exists(espeak_lib)}")

    if not os.path.exists(espeak_path):
        print("ERROR: eSpeak executable not found")
        sys.exit(1)

    if not os.path.exists(espeak_lib):
        print("ERROR: eSpeak library not found")
        sys.exit(1)

    try:
        print("Importing phonemizer...")
        from phonemizer.backend import EspeakBackend

        print("Successfully imported EspeakBackend")

        print("Creating EspeakBackend instance...")
        backend = EspeakBackend(language="en-us")
        print("Successfully created EspeakBackend instance")

        print("Testing phonemization...")
        test_text = ["hello world"]
        print(f"Input text: {test_text}")

        start_time = time.time()
        result = backend.phonemize(test_text, strip=True)
        end_time = time.time()

        print(f"Phonemization completed in {end_time - start_time:.2f} seconds")
        print(f"Phonemizer test successful: {result}")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Phonemizer may not be installed correctly")
        sys.exit(1)
    except Exception as e:
        print(f"Phonemizer test failed: {e}")
        print(f"Exception type: {type(e).__name__}")

        # Additional debugging
        try:
            import subprocess

            print("Testing direct eSpeak call...")
            result = subprocess.run(
                [espeak_path, "--version"], capture_output=True, text=True, timeout=10
            )
            print(f"Direct eSpeak result: {result.stdout}")
        except Exception as direct_error:
            print(f"Direct eSpeak test failed: {direct_error}")

        sys.exit(1)


if __name__ == "__main__":
    main()
