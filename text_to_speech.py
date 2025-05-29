#!/usr/bin/env python3
"""
Zonos Text-to-Speech Script
Converts text files to audio using Zonos TTS model
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torchaudio
import shutil
import subprocess

# Configure PyTorch to avoid compiler issues on Windows
import torch._dynamo

torch._dynamo.config.suppress_errors = True

# Disable torch.compile optimizations that require C++ compiler
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_JIT"] = "0"

# Set torch backend to eager mode (more compatible)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def split_text_into_chunks(text, max_chunk_size=500):
    """Split text into smaller chunks for processing"""
    # Split by sentences first
    import re

    sentences = re.split(r"[.!?]+", text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If adding this sentence would exceed the limit, start a new chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence

    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def clear_gpu_memory():
    """Clear GPU memory cache"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_and_configure_espeak():
    """Find and configure eSpeak for phonemizer"""
    espeak_ng_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    espeak_library_path = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

    if not os.path.exists(espeak_ng_path):
        print("Error: eSpeak NG executable not found")
        return False

    if not os.path.exists(espeak_library_path):
        print("Error: eSpeak NG library (libespeak-ng.dll) not found")
        return False

    print(f"Found eSpeak NG executable: {espeak_ng_path}")
    print(f"Found eSpeak NG library: {espeak_library_path}")

    # Add eSpeak directory to PATH
    espeak_dir = os.path.dirname(espeak_ng_path)
    current_path = os.environ.get("PATH", "")
    if espeak_dir not in current_path:
        os.environ["PATH"] = f"{espeak_dir};{current_path}"
        print(f"Added {espeak_dir} to PATH")

    # Set the correct environment variables for phonemizer
    os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_ng_path
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_library_path

    # Create wrapper scripts in conda environment
    create_espeak_wrapper(espeak_ng_path)

    # Test phonemizer configuration
    return test_phonemizer_config()


def create_espeak_wrapper(espeak_ng_path):
    """Create wrapper scripts for espeak commands"""
    try:
        # Get conda environment Scripts directory
        if "CONDA_PREFIX" in os.environ:
            scripts_dir = os.path.join(os.environ["CONDA_PREFIX"], "Scripts")
        else:
            scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")

        if not os.path.exists(scripts_dir):
            print(f"Warning: Scripts directory not found: {scripts_dir}")
            return False

        # Create espeak.bat wrapper
        espeak_wrapper = os.path.join(scripts_dir, "espeak.bat")
        with open(espeak_wrapper, "w") as f:
            f.write(f'@echo off\n"{espeak_ng_path}" %*\n')
        print(f"Created espeak wrapper: {espeak_wrapper}")

        # Create espeak-ng.bat wrapper
        espeak_ng_wrapper = os.path.join(scripts_dir, "espeak-ng.bat")
        with open(espeak_ng_wrapper, "w") as f:
            f.write(f'@echo off\n"{espeak_ng_path}" %*\n')
        print(f"Created espeak-ng wrapper: {espeak_ng_wrapper}")

        return True

    except Exception as e:
        print(f"Failed to create wrappers: {e}")
        return False


def test_phonemizer_config():
    """Test if phonemizer works with current configuration"""
    try:
        from phonemizer.backend import EspeakBackend

        print("Testing phonemizer configuration...")

        # Test EspeakBackend directly - this is the correct API
        try:
            print("Creating EspeakBackend...")
            backend = EspeakBackend(language="en-us")

            # Test phonemization - correct API usage
            test_text = ["hello world"]
            result = backend.phonemize(test_text, strip=True)
            print(f"Phonemizer test successful: {result}")
            return True

        except Exception as e:
            print(f"EspeakBackend test failed: {e}")

            # Try with explicit environment setup
            print("Trying with explicit environment configuration...")
            try:
                # Ensure environment variables are set
                os.environ["PHONEMIZER_ESPEAK_PATH"] = (
                    r"C:\Program Files\eSpeak NG\espeak-ng.exe"
                )
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = (
                    r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
                )

                backend = EspeakBackend(language="en-us")
                result = backend.phonemize(["test"], strip=True)
                print(f"Explicit configuration test successful: {result}")
                return True

            except Exception as e2:
                print(f"Explicit configuration test failed: {e2}")

                # Final attempt: check if we can run espeak directly
                print("Testing direct espeak execution...")
                try:
                    result = subprocess.run(
                        [
                            r"C:\Program Files\eSpeak NG\espeak-ng.exe",
                            "--phonout",
                            "--stdout",
                            "hello",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )

                    if result.returncode == 0:
                        print(f"Direct espeak test successful: {result.stdout.strip()}")
                        print("eSpeak works but phonemizer integration has issues")
                        return True
                    else:
                        print(f"Direct espeak test failed: {result.stderr}")

                except subprocess.TimeoutExpired:
                    print("Direct espeak test timed out")
                except Exception as e3:
                    print(f"Direct espeak test error: {e3}")

                return False

    except ImportError as e:
        print(f"Cannot import phonemizer: {e}")
        return False


def load_zonos_model():
    """Load and return the Zonos model"""
    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        from zonos.utils import DEFAULT_DEVICE as device

        print("Loading Zonos model...")

        # Configure PyTorch for Windows compatibility
        torch.set_grad_enabled(False)  # Disable gradients for inference

        # Load model with explicit device specification
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

        # Set model to evaluation mode
        model.eval()

        # Disable any compilation features
        if hasattr(model, "compile"):
            print("Disabling model compilation for Windows compatibility...")

        print(f"Model loaded successfully on device: {device}")
        return model

    except ImportError as e:
        print(f"Error importing Zonos: {e}")
        print("Please ensure Zonos is properly installed in your conda environment")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")

        # Try alternative loading approach
        print("Trying alternative model loading...")
        try:
            # Set environment variable to force eager execution
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            from zonos.model import Zonos
            from zonos.utils import DEFAULT_DEVICE as device

            model = Zonos.from_pretrained(
                "Zyphra/Zonos-v0.1-transformer", device=device
            )
            model.eval()
            print(f"Model loaded successfully with fallback method on device: {device}")
            return model

        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            sys.exit(1)


def read_text_file(file_path):
    """Read text from file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        sys.exit(1)


def generate_speech(model, text, output_path, speaker_file=None, language="en-us"):
    """Generate speech from text using Zonos with memory management"""
    try:
        from zonos.conditioning import make_cond_dict
        import torch
        import torchaudio
        import tempfile
        import os

        # Force eager execution mode
        with torch.no_grad():  # Disable gradients for inference
            # Clear GPU memory before starting
            clear_gpu_memory()

        # Create speaker embedding
        if speaker_file and os.path.exists(speaker_file):
            print(f"Using speaker reference: {speaker_file}")
            wav, sampling_rate = torchaudio.load(speaker_file)
            speaker = model.make_speaker_embedding(wav, sampling_rate)
        else:
            print("Using default speaker (no reference provided)")
            speaker = None

        # Check text length and decide on processing strategy
        if len(text) > 1000:
            print(
                f"Long text detected ({len(text)} characters). Processing in chunks..."
            )
            chunks = split_text_into_chunks(text, max_chunk_size=800)
            print(f"Split into {len(chunks)} chunks")

            # Process each chunk separately
            chunk_files = []

            for i, chunk in enumerate(chunks):
                print(
                    f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)..."
                )

                try:
                    # Clear memory before each chunk
                    clear_gpu_memory()

                    # Create conditioning for this chunk
                    cond_dict = make_cond_dict(
                        text=chunk, speaker=speaker, language=language
                    )
                    conditioning = model.prepare_conditioning(cond_dict)

                    # Generate audio for this chunk
                    codes = model.generate(conditioning)
                    wavs = model.autoencoder.decode(codes).cpu()

                    # Save chunk to temporary file
                    chunk_file = f"temp_chunk_{i}.wav"
                    torchaudio.save(
                        chunk_file, wavs[0], model.autoencoder.sampling_rate
                    )
                    chunk_files.append(chunk_file)

                    # Clear memory after processing chunk
                    del codes, wavs, conditioning
                    clear_gpu_memory()

                    print(f"Chunk {i+1} completed successfully")

                except Exception as e:
                    print(f"Error processing chunk {i+1}: {e}")
                    # Clean up partial files
                    for temp_file in chunk_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    raise e

            # Combine all chunks into final output
            print("Combining chunks into final audio file...")
            combine_audio_files(
                chunk_files, output_path, model.autoencoder.sampling_rate
            )

            # Clean up temporary files
            for temp_file in chunk_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        else:
            # Process normally for short text
            print("Creating conditioning...")
            cond_dict = make_cond_dict(text=text, speaker=speaker, language=language)
            conditioning = model.prepare_conditioning(cond_dict)

            # Generate audio
            print("Generating speech...")
            codes = model.generate(conditioning)
            wavs = model.autoencoder.decode(codes).cpu()

            # Save audio
            torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)

        print(f"Audio saved to: {output_path}")

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory error: {e}")
        print("\nSuggestions to resolve memory issues:")
        print("1. Try processing a shorter text (< 500 characters)")
        print("2. Close other GPU-intensive applications")
        print("3. Restart the script to clear GPU memory")
        print("4. Consider using CPU instead (slower but uses system RAM)")

        # Clear GPU memory
        clear_gpu_memory()
        sys.exit(1)

    except Exception as e:
        print(f"Error generating speech: {e}")

        # Diagnostic information
        print("\nDiagnostic information:")
        print(f"PHONEMIZER_ESPEAK_PATH: {os.environ.get('PHONEMIZER_ESPEAK_PATH')}")
        print(
            f"PHONEMIZER_ESPEAK_LIBRARY: {os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')}"
        )

        # Check if files exist
        esp_path = os.environ.get("PHONEMIZER_ESPEAK_PATH")
        esp_lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")

        if esp_path:
            print(f"eSpeak executable exists: {os.path.exists(esp_path)}")
        if esp_lib:
            print(f"eSpeak library exists: {os.path.exists(esp_lib)}")

        # GPU memory info
        if torch.cuda.is_available():
            print(
                f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            print(
                f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )

        sys.exit(1)


def combine_audio_files(file_list, output_path, sample_rate):
    """Combine multiple audio files into one"""
    import torchaudio
    import torch

    combined_audio = []

    for file_path in file_list:
        audio, sr = torchaudio.load(file_path)
        if sr != sample_rate:
            # Resample if necessary
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio = resampler(audio)
        combined_audio.append(audio)

    # Concatenate all audio chunks
    final_audio = torch.cat(combined_audio, dim=1)

    # Save combined audio
    torchaudio.save(output_path, final_audio, sample_rate)


def main():
    print("Configuring eSpeak for phonemizer...")

    if not find_and_configure_espeak():
        print("\nPhonEmizer configuration failed.")
        print("Trying to continue anyway - Zonos might have fallback mechanisms...")
        print("If this fails, please try reinstalling with: install_zonos.ps1")
    else:
        print("Phonemizer configuration successful!")

    parser = argparse.ArgumentParser(
        description="Convert text file to speech using Zonos TTS"
    )
    parser.add_argument("input_file", help="Input text file path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output audio file path (default: same as input with .wav extension)",
    )
    parser.add_argument(
        "-s", "--speaker", help="Speaker reference audio file (optional)"
    )
    parser.add_argument(
        "-l", "--language", default="en-us", help="Language code (default: en-us)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=800,
        help="Maximum chunk size for long texts (default: 800)",
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_file)
        output_path = input_path.with_suffix(".wav")

    # Ensure output directory exists
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_path}")

    # Show GPU info
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("CUDA not available - using CPU")

    # Load model
    model = load_zonos_model()

    # Read text
    text = read_text_file(args.input_file)
    print(f"Text length: {len(text)} characters")

    if len(text) > args.max_length:
        print(
            f"Long text detected. Will process in chunks of {args.max_length} characters."
        )

    # Generate speech
    generate_speech(model, text, output_path, args.speaker, args.language)

    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
