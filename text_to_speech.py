#!/usr/bin/env python3
"""
Zonos Text-to-Speech Script
Converts text files to audio using Zonos TTS model.
"""

# Ensure these are at the VERY TOP, before any other imports, especially torch
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_JIT"] = "0"  # Disables torch.jit.script and trace via env var

# Standard library imports
import argparse
import sys
from pathlib import Path
import subprocess
import re

# Third-party imports
import torch
import torchaudio
import torch.dynamo  # For public API status checks: torch.dynamo.is_enabled()

# Local application/submodule imports (dynamically imported in functions where needed
# to potentially speed up initial script load or avoid circular dependencies if any)
# from zonos.model import Zonos
# from zonos.conditioning import make_cond_dict
# from zonos.utils import DEFAULT_DEVICE
# from phonemizer.backend import EspeakBackend


# Configure PyTorch specifics AFTER main torch import
# Attempt to globally disable TorchDynamo to prevent any JIT compilation attempts
print("Attempting to globally disable TorchDynamo...")
try:
    torch._dynamo.reset()  # pylint: disable=protected-access
    torch._dynamo.disable()  # pylint: disable=protected-access
    # Use public API for status checks
    print(f"  TorchDynamo is_enabled after disable(): {torch.dynamo.is_enabled()}")
    print(f"  TorchDynamo is_compiling after disable(): {torch.dynamo.is_compiling()}")
except Exception as e:  # pylint: disable=broad-except
    print(f"  Warning: Could not fully disable TorchDynamo: {e}")

# Set torch backend to eager mode (more compatible) - good settings for inference
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)  # Globally disable gradients for inference


def split_text_into_chunks(text, max_chunk_size=500):
    """
    Split text into smaller chunks for processing, trying to respect sentences.
    """
    if not text:
        return []

    sentences = re.split(r"([.!?]+)", text)  # Keep delimiters
    # Re-join sentences with their delimiters
    processed_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        processed_sentences.append(
            sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
        )
    if (
        len(sentences) % 2 == 1 and sentences[-1].strip()
    ):  # Handle last part if no delimiter
        processed_sentences.append(sentences[-1])

    chunks = []
    current_chunk = ""
    for sentence in processed_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    # Fallback for very long "sentences" or text without sentence delimiters
    if not chunks and text:
        if len(text) > max_chunk_size:
            print(
                f"Warning: Text splitting fell back to character-based chunking "
                f"for a segment of length {len(text)}."
            )
            return [
                text[i : i + max_chunk_size]
                for i in range(0, len(text), max_chunk_size)
            ]
        return [text]
    if not chunks and not text:  # Should be caught by initial check but good for safety
        return []

    # Further split any chunk that is still too long (e.g., due to a single long sentence)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > max_chunk_size:
            print(
                f"Warning: A sentence-split chunk of {len(chunk)} chars still exceeds "
                f"max_chunk_size {max_chunk_size}. Further splitting."
            )
            for i in range(0, len(chunk), max_chunk_size):
                final_chunks.append(chunk[i : i + max_chunk_size])
        else:
            final_chunks.append(chunk)

    return final_chunks


def clear_gpu_memory():
    """Clear GPU memory cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def find_and_configure_espeak():
    """Find and configure eSpeak for phonemizer."""
    espeak_ng_path_str = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    espeak_library_path_str = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

    if not os.path.exists(espeak_ng_path_str):
        print(f"Error: eSpeak NG executable not found at {espeak_ng_path_str}")
        print(
            "Please ensure eSpeak NG is installed correctly "
            "(e.g., via 'winget install eSpeak-NG.eSpeak-NG')"
        )
        return False

    if not os.path.exists(espeak_library_path_str):
        print(
            f"Error: eSpeak NG library (libespeak-ng.dll) not found at "
            f"{espeak_library_path_str}"
        )
        return False

    print(f"Found eSpeak NG executable: {espeak_ng_path_str}")
    print(f"Found eSpeak NG library: {espeak_library_path_str}")

    espeak_dir = os.path.dirname(espeak_ng_path_str)
    current_path = os.environ.get("PATH", "")
    if espeak_dir not in current_path.split(os.pathsep):
        os.environ["PATH"] = f"{espeak_dir}{os.pathsep}{current_path}"
        print(f"Temporarily added {espeak_dir} to PATH for this session.")

    os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_ng_path_str
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_library_path_str

    create_espeak_wrapper(espeak_ng_path_str)
    return test_phonemizer_config()


def create_espeak_wrapper(espeak_ng_path):
    """Create wrapper scripts for espeak commands in conda env's Scripts directory."""
    try:
        if "CONDA_PREFIX" in os.environ:
            scripts_dir_path = Path(os.environ["CONDA_PREFIX"]) / "Scripts"
        elif sys.prefix != sys.base_prefix:  # In a virtual env (venv)
            scripts_dir_path = Path(sys.prefix) / "Scripts"
        else:  # Not in a virtual env / conda env
            print(
                "Warning: Not in a Conda/virtual environment. "
                "Skipping eSpeak wrapper creation in env Scripts."
            )
            return False

        scripts_dir_path.mkdir(parents=True, exist_ok=True)

        if not os.access(scripts_dir_path, os.W_OK):
            print(
                f"Warning: Scripts directory {scripts_dir_path} is not writable. "
                "Cannot create eSpeak wrappers."
            )
            return False

        espeak_wrapper_content = f'@echo off\n"{espeak_ng_path}" %*\n'
        (scripts_dir_path / "espeak.bat").write_text(
            espeak_wrapper_content, encoding="utf-8"
        )
        print(f"Created espeak wrapper: {scripts_dir_path / 'espeak.bat'}")

        (scripts_dir_path / "espeak-ng.bat").write_text(
            espeak_wrapper_content, encoding="utf-8"
        )
        print(f"Created espeak-ng wrapper: {scripts_dir_path / 'espeak-ng.bat'}")
        return True

    except (OSError, IOError) as e:  # More specific exception
        print(f"Failed to create eSpeak wrappers: {e}")
        return False


def test_phonemizer_config():
    """Test if phonemizer works with current configuration."""
    from phonemizer.backend import EspeakBackend  # Import here

    try:
        print("Testing phonemizer configuration...")
        # Ensure environment variables are explicitly set for this test too
        os.environ["PHONEMIZER_ESPEAK_PATH"] = (
            r"C:\Program Files\eSpeak NG\espeak-ng.exe"
        )
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = (
            r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
        )

        backend = EspeakBackend(language="en-us", with_stress=False)
        result = backend.phonemize(["hello world"], strip=True)
        print(f"Phonemizer test successful: {result}")
        return True

    except (ImportError, RuntimeError) as e:  # More specific exceptions
        print(f"Phonemizer configuration test failed: {e}")
        print("  Details:")
        print(f"  PHONEMIZER_ESPEAK_PATH: {os.environ.get('PHONEMIZER_ESPEAK_PATH')}")
        print(
            "  PHONEMIZER_ESPEAK_LIBRARY: "
            f"{os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')}"
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
                check=False,  # Explicit check=False
            )
            if direct_result.returncode == 0:
                print(
                    f"  Direct eSpeak call successful: {direct_result.stdout.strip()}"
                )
            else:
                print(
                    f"  Direct eSpeak call failed (stderr): {direct_result.stderr.strip()}"
                )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e_direct:
            print(f"  Direct eSpeak call attempt failed: {e_direct}")
        return False
    return True  # Should not be reached if exception occurs


def load_zonos_model():
    """Load and return the Zonos model."""
    from zonos.model import Zonos
    from zonos.utils import DEFAULT_DEVICE as device

    try:
        print("Loading Zonos model...")
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        model.eval()  # Ensure model is in evaluation mode
        print(f"Model loaded successfully on device: {device}")
        return model

    except ImportError as e_import:
        print(f"Error importing Zonos: {e_import}")
        print("Please ensure Zonos is properly installed in your conda environment.")
        sys.exit(1)
    except Exception as e_load:  # pylint: disable=broad-except
        # Broad except here as various HuggingFace/model loading errors can occur
        print(f"Error loading Zonos model: {e_load}")
        if torch.cuda.is_available():
            print(
                f"  GPU memory allocated: "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            print(
                f"  GPU memory reserved:  "
                f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )
        sys.exit(1)


def read_text_file(file_path):
    """Read text content from a file."""
    try:
        return Path(file_path).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, IOError) as e:
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
    """Generate speech from text using Zonos, handling chunking and memory."""
    from zonos.conditioning import make_cond_dict  # Import here

    try:
        clear_gpu_memory()

        speaker_embedding = None
        if speaker_file and os.path.exists(speaker_file):
            print(f"Using speaker reference: {speaker_file}")
            try:
                wav, sr = torchaudio.load(speaker_file)
                speaker_embedding = model.make_speaker_embedding(wav, sr)
            except Exception as e_speaker:  # pylint: disable=broad-except
                print(f"Warning: Could not load or process speaker file: {e_speaker}")
                print("Continuing with default speaker.")
        else:
            print("Using default speaker (no reference provided or file not found).")

        chunks = split_text_into_chunks(text, max_chunk_size=max_chunk_size_arg)
        if len(chunks) > 1 or (chunks and len(chunks[0]) != len(text)):
            print(
                f"Text split into {len(chunks)} chunks for processing "
                f"(max_length: {max_chunk_size_arg})."
            )

        all_wav_tensors = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():  # Skip empty chunks
                continue
            print(
                f"Processing chunk {i + 1}/{len(chunks)} "
                f"({len(chunk_text)} chars)..."
            )
            clear_gpu_memory()

            try:
                cond_dict = make_cond_dict(
                    text=chunk_text, speaker=speaker_embedding, language=language
                )
                conditioning = model.prepare_conditioning(cond_dict)

                codes = model.generate(conditioning)
                wav_chunk = model.autoencoder.decode(codes).cpu()
                # Assuming Zonos generate/decode returns a batch, take the first item
                all_wav_tensors.append(wav_chunk[0])

                del codes, wav_chunk, conditioning  # Explicitly delete
                clear_gpu_memory()
                print(f"Chunk {i + 1} completed.")

            except torch.cuda.OutOfMemoryError as e_oom_chunk:
                print(f"CUDA out of memory on chunk {i + 1}: {e_oom_chunk}")
                print("Try reducing --max-length or ensure GPU has enough free VRAM.")
                sys.exit(1)
            except Exception as e_chunk:  # pylint: disable=broad-except
                # Catching broadly here as Zonos internal errors might be varied
                print(f"Error processing chunk {i + 1}: {e_chunk}")
                raise  # Re-raise to be caught by the main try-except

        if not all_wav_tensors:
            print("No audio generated (e.g., input text was empty).")
            return

        print("Combining audio chunks...")
        # Concatenate along the time axis (dim=1 for [C, T] or dim=0 if [T])
        # Zonos output seems to be [1, num_samples], so dim=1 is correct.
        final_audio = torch.cat(all_wav_tensors, dim=1)
        torchaudio.save(output_path, final_audio, model.autoencoder.sampling_rate)
        print(f"Audio saved to: {output_path}")

    except torch.cuda.OutOfMemoryError as e_oom_global:
        print(f"Global CUDA out of memory error: {e_oom_global}")
        print("\nSuggestions to resolve memory issues:")
        print("1. Reduce --max-length CLI argument (e.g., to 200-400 for 4GB VRAM).")
        print("2. Close other GPU-intensive applications.")
        print("3. Restart the script/system to clear GPU memory.")
        clear_gpu_memory()
        sys.exit(1)

    except Exception as e_global:  # pylint: disable=broad-except
        print(f"Error during speech generation: {e_global}")
        print("\n--- Diagnostic Information ---")
        print(f"  TORCH_COMPILE_DISABLE: {os.environ.get('TORCH_COMPILE_DISABLE')}")
        print(f"  PYTORCH_JIT: {os.environ.get('PYTORCH_JIT')}")
        print(f"  TorchDynamo is_enabled: {torch.dynamo.is_enabled()}")  # Public API
        print(
            f"  TorchDynamo is_compiling: {torch.dynamo.is_compiling()}"
        )  # Public API
        print(f"  PHONEMIZER_ESPEAK_PATH: {os.environ.get('PHONEMIZER_ESPEAK_PATH')}")
        print(
            "  PHONEMIZER_ESPEAK_LIBRARY: "
            f"{os.environ.get('PHONEMIZER_ESPEAK_LIBRARY')}"
        )
        if torch.cuda.is_available():
            print(
                f"  GPU Memory Allocated: "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            print(
                f"  GPU Memory Reserved:  "
                f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )
        print("--- End Diagnostic Information ---")
        sys.exit(1)


def main():
    """
    Main execution function.
    Parses arguments, configures eSpeak, loads model, and generates speech.
    """
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
        help="Maximum character length for text chunks (default: 800). "
        "Adjust based on VRAM.",
    )
    args = parser.parse_args()

    input_file_path = Path(args.input_file)
    if not input_file_path.exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    output_path_str = (
        args.output if args.output else str(input_file_path.with_suffix(".wav"))
    )
    output_file_path = Path(output_path_str)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Load Zonos model (imports Zonos internally)
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
