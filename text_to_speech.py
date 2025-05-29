#!/usr/bin/env python3
"""
Zonos Text-to-Speech Script
Converts text files to audio using Zonos TTS model
"""

# Ensure these are at the VERY TOP, before any other imports, especially torch
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_JIT"] = "0"  # Disables torch.jit.script and trace via env var
# os.environ["TRITON_SUPPRESS_MODULE_NOT_FOUND_ERROR"] = "1" # Optional, might reduce some noise if triton is probed

import argparse
import sys  # Keep sys import here or move to top if preferred
from pathlib import Path
import torch  # Now import torch
import torchaudio

# import shutil # Not used in the current script logic
import subprocess

# Configure PyTorch specifics AFTER main torch import
import torch._dynamo

# DO NOT suppress errors during debugging; remove or comment out:
# torch._dynamo.config.suppress_errors = True

# Attempt to globally disable TorchDynamo to prevent any JIT compilation attempts via torch.compile
# This is a stronger measure if TORCH_COMPILE_DISABLE=1 is not fully effective.
print("Attempting to globally disable TorchDynamo...")
try:
    torch._dynamo.reset()  # Reset any existing dynamo state
    torch._dynamo.disable()  # Disable dynamo frame conversion
    print(
        f"  TorchDynamo is_enabled after disable(): {torch._dynamo.is_enabled()}"
    )  # Should be False
    print(
        f"  TorchDynamo is_compiling after disable(): {torch._dynamo.is_compiling()}"
    )  # Should be False
except Exception as e:
    print(f"  Warning: Could not fully disable TorchDynamo: {e}")

# Set torch backend to eager mode (more compatible) - these are good settings for inference
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)  # Globally disable gradients for inference


def split_text_into_chunks(text, max_chunk_size=500):
    """Split text into smaller chunks for processing"""
    # Split by sentences first
    import re  # Import re here as it's only used in this function

    sentences = re.split(r"[.!?]+", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence  # Add punctuation back
            else:
                current_chunk = sentence

    if current_chunk.strip():  # Add the last chunk if it exists
        chunks.append(current_chunk.strip())

    # Handle cases where the text might be shorter than max_chunk_size or has no sentence-ending punctuation
    if not chunks and text:
        # If text is longer than max_chunk_size but wasn't split by sentences (e.g. no punctuation)
        # or if the text is simply one large block.
        # This is a fallback, primary logic relies on sentence splitting.
        # For very long single "sentences", this will still create large chunks.
        # A more robust chunker might be needed for extreme cases, but this covers many.
        if len(text) > max_chunk_size:
            # Simple character-based split if sentence splitting yields too large chunks or none
            temp_chunks = []
            for i in range(0, len(text), max_chunk_size):
                temp_chunks.append(text[i : i + max_chunk_size])
            # If sentence splitting failed but we have text, use character-based split
            if not chunks and temp_chunks:
                print(
                    f"Warning: Text splitting fell back to character-based chunking for a segment of length {len(text)}."
                )
                return temp_chunks
        elif text:  # If text is short and wasn't split
            return [text]

    return chunks if chunks else ([text] if text else [])


def clear_gpu_memory():
    """Clear GPU memory cache"""
    # import torch # torch is already imported globally
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_and_configure_espeak():
    """Find and configure eSpeak for phonemizer"""
    espeak_ng_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    espeak_library_path = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

    if not os.path.exists(espeak_ng_path):
        print(f"Error: eSpeak NG executable not found at {espeak_ng_path}")
        print(
            "Please ensure eSpeak NG is installed correctly (e.g., via 'winget install eSpeak-NG.eSpeak-NG')"
        )
        return False

    if not os.path.exists(espeak_library_path):
        print(
            f"Error: eSpeak NG library (libespeak-ng.dll) not found at {espeak_library_path}"
        )
        return False

    print(f"Found eSpeak NG executable: {espeak_ng_path}")
    print(f"Found eSpeak NG library: {espeak_library_path}")

    espeak_dir = os.path.dirname(espeak_ng_path)
    current_path = os.environ.get("PATH", "")
    if espeak_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{espeak_dir}{os.pathsep}{current_path}"
        print(f"Temporarily added {espeak_dir} to PATH for this session.")

    os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_ng_path
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_library_path

    create_espeak_wrapper(espeak_ng_path)
    return test_phonemizer_config()


def create_espeak_wrapper(espeak_ng_path):
    """Create wrapper scripts for espeak commands in conda env"""
    try:
        if "CONDA_PREFIX" in os.environ:
            scripts_dir = os.path.join(os.environ["CONDA_PREFIX"], "Scripts")
        elif sys.prefix == sys.base_prefix:  # Not in a virtual env / conda env
            print(
                "Warning: Not in a Conda environment. Skipping eSpeak wrapper creation in env Scripts."
            )
            return False
        else:  # In a virtual env (venv)
            scripts_dir = os.path.join(sys.prefix, "Scripts")

        if not os.path.exists(scripts_dir):
            print(
                f"Warning: Scripts directory not found: {scripts_dir}. Cannot create eSpeak wrappers."
            )
            return False

        # Make sure scripts_dir is writable
        if not os.access(scripts_dir, os.W_OK):
            print(
                f"Warning: Scripts directory {scripts_dir} is not writable. Cannot create eSpeak wrappers."
            )
            return False

        espeak_wrapper_path = os.path.join(scripts_dir, "espeak.bat")
        with open(espeak_wrapper_path, "w") as f:
            f.write(f'@echo off\n"{espeak_ng_path}" %*\n')
        print(f"Created espeak wrapper: {espeak_wrapper_path}")

        espeak_ng_wrapper_path = os.path.join(scripts_dir, "espeak-ng.bat")
        with open(espeak_ng_wrapper_path, "w") as f:
            f.write(f'@echo off\n"{espeak_ng_path}" %*\n')
        print(f"Created espeak-ng wrapper: {espeak_ng_wrapper_path}")
        return True

    except Exception as e:
        print(f"Failed to create eSpeak wrappers: {e}")
        return False


def test_phonemizer_config():
    """Test if phonemizer works with current configuration"""
    try:
        from phonemizer.backend import EspeakBackend

        print("Testing phonemizer configuration...")
        # Ensure environment variables are explicitly set for this test too
        os.environ["PHONEMIZER_ESPEAK_PATH"] = (
            r"C:\Program Files\eSpeak NG\espeak-ng.exe"
        )
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = (
            r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
        )

        backend = EspeakBackend(
            language="en-us", with_stress=False
        )  # Added with_stress=False for broader compatibility
        result = backend.phonemize(["hello world"], strip=True)
        print(f"Phonemizer test successful: {result}")
        return True

    except Exception as e:
        print(f"Phonemizer configuration test failed: {e}")
        print("  Details:")
        print(f"  PHONEMIZER_ESPEAK_PATH: {os.environ.get('PHONEMIZER_ESPEAK_PATH')}")
        print(
            f"  PHONEMIZER_ESPEAK_LIBRARY: {os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')}"
        )
        esp_path = os.environ.get("PHONEMIZER_ESPEAK_PATH")
        esp_lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
        if esp_path:
            print(f"  eSpeak executable exists: {os.path.exists(esp_path)}")
        if esp_lib:
            print(f"  eSpeak library exists: {os.path.exists(esp_lib)}")

        try:
            print("  Attempting direct eSpeak call for diagnostics...")
            direct_result = subprocess.run(
                [r"C:\Program Files\eSpeak NG\espeak-ng.exe", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if direct_result.returncode == 0:
                print(
                    f"  Direct eSpeak call successful: {direct_result.stdout.strip()}"
                )
            else:
                print(f"  Direct eSpeak call failed: {direct_result.stderr.strip()}")
        except Exception as e_direct:
            print(f"  Direct eSpeak call attempt failed: {e_direct}")
        return False


def load_zonos_model():
    """Load and return the Zonos model"""
    try:
        from zonos.model import Zonos
        from zonos.utils import DEFAULT_DEVICE as device  # Assuming Zonos provides this

        print("Loading Zonos model...")
        # PyTorch configurations (grad_enabled, eval mode) are handled globally or per-use

        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        model.eval()  # Ensure model is in evaluation mode

        print(f"Model loaded successfully on device: {device}")
        return model

    except ImportError as e_import:
        print(f"Error importing Zonos: {e_import}")
        print("Please ensure Zonos is properly installed in your conda environment.")
        sys.exit(1)
    except Exception as e_load:
        print(f"Error loading Zonos model: {e_load}")
        # Additional diagnostics for model loading failure
        if torch.cuda.is_available():
            print(
                f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            print(
                f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )
        sys.exit(1)


def read_text_file(file_path):
    """Read text from file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        sys.exit(1)


def generate_speech(
    model,
    text,
    output_path,
    speaker_file=None,
    language="en-us",
    max_chunk_size_arg=800,
):
    """Generate speech from text using Zonos with memory management"""
    try:
        from zonos.conditioning import make_cond_dict

        # import torch # Already imported
        # import torchaudio # Already imported
        # import tempfile # Not strictly needed if using numbered temp files
        # import os # Already imported

        # Ensure no gradients are computed during inference
        # torch.set_grad_enabled(False) is global now
        # model.eval() is done after loading

        clear_gpu_memory()

        if speaker_file and os.path.exists(speaker_file):
            print(f"Using speaker reference: {speaker_file}")
            wav, sr = torchaudio.load(speaker_file)
            speaker_embedding = model.make_speaker_embedding(wav, sr)
        else:
            print("Using default speaker (no reference provided).")
            speaker_embedding = None

        # Use the max_chunk_size from args for splitting
        chunks = split_text_into_chunks(text, max_chunk_size=max_chunk_size_arg)
        if len(text) > max_chunk_size_arg or len(chunks) > 1:
            print(
                f"Text split into {len(chunks)} chunks for processing (max_length: {max_chunk_size_arg})."
            )

        all_wavs = []
        temp_chunk_files = []

        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():  # Skip empty chunks
                continue
            print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk_text)} chars)...")
            clear_gpu_memory()

            try:
                cond_dict = make_cond_dict(
                    text=chunk_text, speaker=speaker_embedding, language=language
                )
                conditioning = model.prepare_conditioning(cond_dict)

                codes = model.generate(
                    conditioning
                )  # This is where Zonos might try to compile internally if not fully disabled
                wav_chunk = model.autoencoder.decode(codes).cpu()
                all_wavs.append(wav_chunk[0])  # Assuming batch size of 1 from Zonos

                # Optional: Save intermediate chunks if helpful for debugging very long files
                # temp_chunk_file = f"temp_chunk_{i}.wav"
                # torchaudio.save(temp_chunk_file, wav_chunk[0], model.autoencoder.sampling_rate)
                # temp_chunk_files.append(temp_chunk_file)

                del codes, wav_chunk, conditioning  # Explicitly delete to free memory
                clear_gpu_memory()
                print(f"Chunk {i+1} completed.")

            except torch.cuda.OutOfMemoryError as e_oom:
                print(f"CUDA out of memory on chunk {i+1}: {e_oom}")
                print(
                    "Try reducing --max-length further or ensure GPU has enough free VRAM."
                )
                # Clean up any temp files created so far if you were saving them
                # for f_path in temp_chunk_files: os.remove(f_path)
                sys.exit(1)
            except Exception as e_chunk:
                print(f"Error processing chunk {i+1}: {e_chunk}")
                # for f_path in temp_chunk_files: os.remove(f_path)
                raise  # Re-raise to be caught by the main try-except

        if not all_wavs:
            print("No audio generated (e.g., input text was empty or only whitespace).")
            return

        print("Combining audio chunks...")
        final_audio = torch.cat(
            all_wavs, dim=1
        )  # Concatenate along the time axis (dim=1 for [C, T])
        torchaudio.save(output_path, final_audio, model.autoencoder.sampling_rate)

        # Clean up temp chunk files if they were saved
        # for f_path in temp_chunk_files: os.remove(f_path)

        print(f"Audio saved to: {output_path}")

    except torch.cuda.OutOfMemoryError as e_oom_global:
        print(f"Global CUDA out of memory error: {e_oom_global}")
        print("\nSuggestions to resolve memory issues:")
        print("1. Reduce --max-length CLI argument (e.g., to 200-400 for 4GB VRAM).")
        print("2. Close other GPU-intensive applications.")
        print("3. Restart the script/system to clear GPU memory.")
        clear_gpu_memory()
        sys.exit(1)

    except Exception as e_global:
        print(f"Error during speech generation: {e_global}")
        # Diagnostic information
        print("\n--- Diagnostic Information ---")
        print(f"  TORCH_COMPILE_DISABLE: {os.environ.get('TORCH_COMPILE_DISABLE')}")
        print(f"  PYTORCH_JIT: {os.environ.get('PYTORCH_JIT')}")
        print(f"  TorchDynamo is_enabled: {torch._dynamo.is_enabled()}")
        print(f"  TorchDynamo is_compiling: {torch._dynamo.is_compiling()}")
        print(f"  PHONEMIZER_ESPEAK_PATH: {os.environ.get('PHONEMIZER_ESPEAK_PATH')}")
        print(
            f"  PHONEMIZER_ESPEAK_LIBRARY: {os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')}"
        )
        if torch.cuda.is_available():
            print(
                f"  GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            print(
                f"  GPU Memory Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )
        print("--- End Diagnostic Information ---")
        sys.exit(1)


# combine_audio_files is not needed if we collect tensors in memory and cat them
# def combine_audio_files(file_list, output_path, sample_rate): ...


def main():
    print("Configuring eSpeak for phonemizer...")
    if not find_and_configure_espeak():
        print("\nPhonemizer configuration failed or eSpeak not found.")
        print("Please ensure eSpeak NG is installed and configured correctly.")
        print("Attempting to continue, but phonemization errors are likely...")
    else:
        print("Phonemizer configuration appears successful.")

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
        default=800,  # Default chunk size for text splitting
        help="Maximum character length for text chunks (default: 800). Adjust based on VRAM.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    output_path_str = (
        args.output if args.output else str(Path(args.input_file).with_suffix(".wav"))
    )
    output_dir = Path(output_path_str).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_path_str}")
    print(f"Max chunk length: {args.max_length}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )
        print(f"Using GPU: {gpu_name} ({gpu_total_memory_gb:.1f} GB total VRAM)")
    else:
        print("CUDA not available - using CPU. This will be very slow.")

    model = load_zonos_model()
    text_content = read_text_file(args.input_file)
    print(f"Read text length: {len(text_content)} characters")

    if not text_content.strip():
        print(
            "Input text file is empty or contains only whitespace. Nothing to synthesize."
        )
        sys.exit(0)

    generate_speech(
        model,
        text_content,
        output_path_str,
        args.speaker,
        args.language,
        args.max_length,
    )
    print("TTS conversion process completed!")


if __name__ == "__main__":
    main()
