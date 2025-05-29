# Zonos Windows TTS

> Windows-optimized Zonos TTS setup with automated installation, phonemizer integration, and GPU memory management for seamless text-to-speech generation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A comprehensive Windows setup for [Zyphra's Zonos TTS model](https://github.com/Zyphra/Zonos) with automated installation, dependency management, and production-ready command-line interface.

## ✨ Features

- **🚀 One-click installation** - Automated conda environment setup with all dependencies
- **🎯 Windows-first design** - Handles Windows-specific phonemizer and eSpeak integration, and robust PyTorch eager mode execution.
- **🧠 Smart memory management** - Automatic text chunking for long documents and GPU memory optimization
- **⚡ GPU acceleration** - CUDA support with fallback to CPU processing
- **🎵 Multiple audio formats** - Support for various input text formats and audio output options
- **🔧 Command-line friendly** - Simple batch script wrapper for easy integration
- **📋 VS Code integration** - Complete development environment configuration

## 🚀 Quick Start

### Prerequisites

- Windows 10/11
- NVIDIA GPU with 4GB+ VRAM (recommended)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Git

### Installation

1.  **Clone the repository:**

    ```cmd
    git clone https://github.com/Foadsf/zonos-windows-tts.git
    cd zonos-windows-tts
    ```

2.  **Run the automated installer:**

    ```powershell
    PowerShell -ExecutionPolicy Bypass -File install_zonos.ps1
    ```

    This script sets up a conda environment named `zonos_env` with Python 3.10, PyTorch (CUDA enabled), Zonos, and all other necessary dependencies.

3.  **Test the installation:**
    Create a file named `test_sample.txt` with some text like "Hello world! This is a test."
    Then run:
    ```cmd
    tts.cmd "test_sample.txt"
    ```
    This should generate a `test_sample.wav` file in the same directory.

### Usage

**Basic usage:**

```cmd
tts.cmd "your_text_file.txt"
```

**Advanced options:**

```cmd
# Specify output file
tts.cmd "input.txt" -o "custom_output.wav"

# Use speaker reference for voice cloning
tts.cmd "input.txt" -s "speaker_reference.wav" -l "en-us"

# Control chunk size for long texts (characters per chunk)
tts.cmd "long_document.md" --max-length 400
```

**Python interface:**
Ensure your `zonos_env` conda environment is activated.

```cmd
conda activate zonos_env
python text_to_speech.py "your_file.txt" -o "output.wav"
```

## 📁 Project Structure

```
zonos-windows-tts/
├── install_zonos.ps1          # Automated installation script
├── tts.cmd                    # Command-line wrapper
├── text_to_speech.py          # Main TTS processing script
├── test_installation.py       # Installation verification
├── test_phonemizer.py         # Phonemizer integration test
├── test_simple_tts.py         # Simple TTS functionality test (generates test_simple_output.wav)
├── .vscode/                   # VS Code configuration
│   ├── settings.json
│   ├── launch.json
│   ├── tasks.json
│   └── extensions.json
└── README.md
```

## 🔧 Configuration

### Environment Variables

The system and scripts automatically handle or rely on these critical settings for Windows compatibility:

- **`TORCH_COMPILE_DISABLE=1`**: Set internally by the Python scripts to instruct PyTorch to avoid `torch.compile`.
- **`PYTORCH_JIT=0`**: Set internally by the Python scripts to disable JIT scripting and tracing.
- **`torch._dynamo.disable()`**: Called internally within Python scripts for a more robust disabling of TorchDynamo, forcing eager execution.
- **`PHONEMIZER_ESPEAK_PATH="C:\Program Files\eSpeak NG\espeak-ng.exe"`**: Set internally by Python scripts if eSpeak is found.
- **`PHONEMIZER_ESPEAK_LIBRARY="C:\Program Files\eSpeak NG\libespeak-ng.dll"`**: Set internally by Python scripts if eSpeak is found.
- **`HF_HUB_DISABLE_SYMLINKS_WARNING=1`**: Set by `tts.cmd` to suppress Hugging Face Hub warnings common on Windows.

These settings ensure PyTorch runs in eager mode, which is crucial for avoiding compilation-related errors on Windows systems that may lack specific C++ build tools (like `cl.exe`) or a compatible Triton installation.

### GPU Memory Management

For systems with limited VRAM:

```python
# Automatic chunking for long texts is handled by text_to_speech.py
# You can control the chunk size via the --max-length argument:
# python text_to_speech.py "long_file.txt" --max-length 400

# Monitor GPU usage (run in zonos_env):
python -c "import torch; print(f'GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.1f}GB, Reserved: {torch.cuda.memory_reserved()/1024**3:.1f}GB') if torch.cuda.is_available() else print('CUDA not available')"
```

## 🐛 Known Issues & Solutions

### Installation Issues

| Issue                  | Solution                                                                                                          |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `conda not recognized` | Run `conda init cmd.exe` (or `conda init powershell`) as Administrator, then restart terminal/PowerShell.         |
| Package corruption     | Delete environment: `conda env remove -n zonos_env -y` and reinstall using `install_zonos.ps1`.                   |
| eSpeak not found       | Install via `winget install --id eSpeak-NG.eSpeak-NG --accept-package-agreements`. The scripts also attempt this. |

### Runtime Issues

| Issue                                                     | Symptom                                                                                                              | Solution                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CUDA OOM**                                              | `CUDA error: out of memory`                                                                                          | Reduce `--max-length` (e.g., to 200-400 characters for 4GB VRAM). Close other GPU-intensive applications.                                                                                                                                                                                                                                                                                                                                                                           |
| **PyTorch Compiler Errors**                               | Errors like `RuntimeError: Compiler: cl is not found.` or `RuntimeError: Cannot find a working triton installation.` | The `text_to_speech.py` and `test_simple_tts.py` scripts are designed to prevent this by setting `TORCH_COMPILE_DISABLE=1`, `PYTORCH_JIT=0`, and calling `torch._dynamo.disable()` internally. This forces PyTorch into eager execution mode, which is more compatible with standard Windows environments. If these errors still occur, ensure your PyTorch installation (via `install_zonos.ps1`) completed correctly and that no other environment settings are overriding these. |
| **Phonemizer Hanging (during installation or first run)** | Process appears to stop or timeout during phonemizer/eSpeak tests or initialization.                                 | This can sometimes happen on Windows during the first interaction with eSpeak via phonemizer. The installation script includes timeouts. Subsequent runs are usually faster. If persistent, ensure eSpeak NG is correctly installed and its path is accessible.                                                                                                                                                                                                                     |
| **Long Generation Time**                                  | Slow processing, especially on the first run or with CPU.                                                            | Normal for 4GB GPUs or CPU processing. The first run might involve model downloads/caching. Consider shorter text segments if time is critical on lower-end hardware.                                                                                                                                                                                                                                                                                                               |

## 📊 Performance Optimization

**For 4GB GPU (e.g., NVIDIA RTX A2000 Laptop, GTX 1650):**

- Use `--max-length 300-500` for long texts (adjust based on testing).
- Ensure no other GPU-heavy applications are running.
- Process very long documents in smaller, manageable input files if necessary.

**For 8GB+ GPU:**

- Default settings (e.g., `--max-length 800`) should work well.
- Can often handle `--max-length` up to 800-1000 characters, but always monitor VRAM.

## 🧠 Technical Details

### Text Processing Pipeline

1.  **Input Validation**: File existence and basic checks.
2.  **Text Chunking**: Smart sentence-aware splitting for long documents (controlled by `--max-length`).
3.  **Phonemization**: eSpeak NG integration via Phonemizer for multilingual support. Scripts ensure eSpeak paths are configured.
4.  **Model Inference**: Zonos transformer model with CUDA acceleration (if available). PyTorch runs in eager execution mode.
5.  **Audio Synthesis**: DAC autoencoder for high-quality 44kHz audio output.
6.  **Post-processing**: Audio chunk combination and saving to WAV format.

### Memory Management Strategy

```python
# Key aspects implemented in text_to_speech.py:
import torch
# torch.set_grad_enabled(False) # Globally for inference
# model.eval()                  # Set after model loading
# clear_gpu_memory()            # Called before processing chunks
# del variables                 # Explicit deletion of large tensors after use
# Chunking text                 # Reduces peak memory for individual generation steps
```

### Windows-Specific Adaptations

- **eSpeak Integration**: Automatic detection and path configuration for eSpeak NG. Wrapper scripts (`espeak.bat`, `espeak-ng.bat`) are created in the conda environment's `Scripts` directory during `text_to_speech.py` execution to aid phonemizer.
- **Path Handling**: Proper Windows path quoting and escaping in `tts.cmd` and Python scripts.
- **Process Management**: Timeout handling in `install_zonos.ps1` for potentially slow operations like phonemizer tests.
- **PyTorch Eager Execution**: `TORCH_COMPILE_DISABLE=1`, `PYTORCH_JIT=0` environment variables are set, and `torch._dynamo.disable()` is called in Python scripts to robustly enforce eager execution mode, avoiding common Windows compilation issues (e.g., missing `cl.exe` or Triton).

## 🔍 Troubleshooting

### Debug Mode / Verbose Output

PyTorch and TorchDynamo can produce extensive logs if needed:
Set these environment variables _before_ running the Python script:

```cmd
set TORCH_LOGS="+dynamo,+inductor"
set TORCHDYNAMO_VERBOSE=1
set TORCHINDUCTOR_TRACE=1
python text_to_speech.py "your_file.txt"
```

(Note: `TORCH_COMPILE_DISABLE=1` should prevent most `inductor` activity, but these are general PyTorch debugging flags.)

### Manual Testing

Activate the conda environment (`conda activate zonos_env`) then run:

```cmd
# Test full installation and basic eSpeak/PyTorch imports
python test_installation.py

# Test phonemizer and eSpeak integration specifically
python test_phonemizer.py

# Test a very simple Zonos TTS generation (outputs test_simple_output.wav)
python test_simple_tts.py
```

### Common Solutions

**Clear GPU Memory (if scripts hang or OOM errors persist after reducing chunk size):**
Sometimes, a full restart of the Python kernel or even the system can help clear stubborn GPU memory allocations.
You can also try this in a Python console within `zonos_env`:

```python
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("GPU cache cleared.")
else:
    print("CUDA not available.")
```

**Reinstall Corrupted Environment:**
If packages seem corrupted or multiple issues arise:

```cmd
conda env remove -n zonos_env -y
```

Then re-run the installer:

```powershell
PowerShell -ExecutionPolicy Bypass -File install_zonos.ps1
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1.  Clone the repository.
2.  Run `install_zonos.ps1` to set up the `zonos_env` environment.
3.  Activate the environment: `conda activate zonos_env`.
4.  Open the project in VS Code (configured settings for Python interpreter and tasks are included).
5.  Make your changes and test them using the provided test scripts (e.g., `test_simple_tts.py`, `test_installation.py`).

## 📄 License

This project is licensed under the MIT License - see the (LICENSE file if it existed, or state "MIT License"). MIT License is declared at the top badge.

## 🙏 Acknowledgments

- [Zyphra](https://github.com/Zyphra) for the amazing Zonos TTS model.
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) project for phonemization capabilities.
- [Phonemizer](https://github.com/bootphon/phonemizer) library for easy text-to-phoneme conversion.

## 📊 System Requirements

### Minimum Requirements

- Windows 10 (64-bit)
- 8GB RAM
- ~5-10GB available disk space (for environment, models, and Zonos clone)
- Python 3.10 (managed by Conda environment)
- CPU with AVX support (most modern CPUs)

### Recommended Requirements

- Windows 10/11 (64-bit)
- 16GB+ RAM
- NVIDIA GPU with 4GB+ VRAM (CUDA 12.1 compatible drivers)
- SSD for faster model loading
- ~10-15GB available disk space

## 🔗 Related Projects

- [Zonos (Original)](https://github.com/Zyphra/Zonos) - The core Zonos TTS implementation.
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) - Open-source speech synthesizer.
- [Phonemizer](https://github.com/bootphon/phonemizer) - Python library for phonemizing text.

---

**Made with ❤️ for Windows TTS enthusiasts**
