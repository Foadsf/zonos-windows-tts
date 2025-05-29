# Zonos Windows TTS

> Windows-optimized Zonos TTS setup with automated installation, phonemizer integration, and GPU memory management for seamless text-to-speech generation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A comprehensive Windows setup for [Zyphra's Zonos TTS model](https://github.com/Zyphra/Zonos) with automated installation, dependency management, and production-ready command-line interface.

## ✨ Features

- **🚀 One-click installation** - Automated conda environment setup with all dependencies
- **🎯 Windows-first design** - Handles Windows-specific phonemizer and eSpeak integration
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

1. **Clone the repository:**
   ```cmd
   git clone https://github.com/Foadsf/zonos-windows-tts.git
   cd zonos-windows-tts
   ```

````

2. **Run the automated installer:**

   ```powershell
   PowerShell -ExecutionPolicy Bypass -File install_zonos.ps1
   ```

3. **Test the installation:**
   ```cmd
   tts.cmd "test_sample.txt"
   ```

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

# Control chunk size for long texts
tts.cmd "long_document.md" --max-length 400
```

**Python interface:**

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
├── test_simple_tts.py         # Simple TTS functionality test
├── .vscode/                   # VS Code configuration
│   ├── settings.json
│   ├── launch.json
│   ├── tasks.json
│   └── extensions.json
└── README.md
```

## 🔧 Configuration

### Environment Variables

The system automatically configures these environment variables:

```bash
PHONEMIZER_ESPEAK_PATH="C:\Program Files\eSpeak NG\espeak-ng.exe"
PHONEMIZER_ESPEAK_LIBRARY="C:\Program Files\eSpeak NG\libespeak-ng.dll"
HF_HUB_DISABLE_SYMLINKS_WARNING=1
TORCH_COMPILE_DISABLE=1
```

### GPU Memory Management

For systems with limited VRAM:

```python
# Automatic chunking for long texts
python text_to_speech.py "long_file.txt" --max-length 400

# Monitor GPU usage
python -c "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB')"
```

## 🐛 Known Issues & Solutions

### Installation Issues

| Issue                  | Solution                                                             |
| ---------------------- | -------------------------------------------------------------------- |
| `conda not recognized` | Run `conda init cmd.exe` as Administrator, then restart terminal     |
| Package corruption     | Delete environment: `conda env remove -n zonos_env -y` and reinstall |
| eSpeak not found       | Install via `winget install --id eSpeak-NG.eSpeak-NG`                |

### Runtime Issues

| Issue                    | Symptom                          | Solution                                                   |
| ------------------------ | -------------------------------- | ---------------------------------------------------------- |
| **CUDA OOM**             | `CUDA error: out of memory`      | Reduce `--max-length` to 200-400 characters                |
| **Compiler Error**       | `Compiler: cl is not found`      | Script automatically handles this with eager execution     |
| **Phonemizer Hanging**   | Process stops at phonemizer test | Expected on Windows; script includes timeout handling      |
| **Long Generation Time** | Slow processing                  | Normal for 4GB GPU; consider shorter texts or CPU fallback |

### Performance Optimization

**For 4GB GPU (RTX A2000, GTX 1650, etc.):**

- Use `--max-length 400` for long texts
- Close other GPU applications
- Process shorter segments individually

**For 8GB+ GPU:**

- Default settings work well
- Can handle `--max-length 800-1000`

## 🧠 Technical Details

### Text Processing Pipeline

1. **Input Validation** - File existence and encoding checks
2. **Text Chunking** - Smart sentence-aware splitting for long documents
3. **Phonemization** - eSpeak NG integration for multilingual support
4. **Model Inference** - Zonos transformer model with CUDA acceleration
5. **Audio Synthesis** - DAC autoencoder for high-quality 44kHz output
6. **Post-processing** - Chunk combination and audio normalization

### Memory Management Strategy

```python
# Automatic memory management
clear_gpu_memory()  # Before processing
torch.no_grad()     # Disable gradients
model.eval()        # Inference mode
del variables       # Explicit cleanup
```

### Windows-Specific Adaptations

- **eSpeak Integration**: Automatic wrapper creation in conda environment
- **Path Handling**: Proper Windows path quoting and escaping
- **Process Management**: Timeout handling for hanging processes
- **Compiler Compatibility**: Torch compile disabled for Windows

## 🔍 Troubleshooting

### Debug Mode

Enable verbose logging:

```cmd
set TORCH_LOGS="+dynamo"
set TORCHDYNAMO_VERBOSE=1
python text_to_speech.py "test.txt"
```

### Manual Testing

Test individual components:

```cmd
# Test installation
python test_installation.py

# Test phonemizer
python test_phonemizer.py

# Test simple TTS
python test_simple_tts.py
```

### Common Solutions

**Reset GPU memory:**

```python
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

**Reinstall corrupted environment:**

```cmd
conda env remove -n zonos_env -y
PowerShell -ExecutionPolicy Bypass -File install_zonos.ps1
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Clone the repository
2. Run the installer
3. Open in VS Code (configured settings included)
4. Test changes with `test_installation.py`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Zyphra](https://github.com/Zyphra) for the amazing Zonos TTS model
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) for phonemization
- [Phonemizer](https://github.com/bootphon/phonemizer) for text preprocessing

## 📊 System Requirements

### Minimum Requirements

- Windows 10 (64-bit)
- 8GB RAM
- 2GB available disk space
- Python 3.10+

### Recommended Requirements

- Windows 11 (64-bit)
- 16GB+ RAM
- NVIDIA GPU with 4GB+ VRAM
- 10GB available disk space
- CUDA 12.1+

## 🔗 Related Projects

- [Zonos Original](https://github.com/Zyphra/Zonos) - Original Zonos TTS implementation
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) - Text-to-speech synthesizer
- [Phonemizer](https://github.com/bootphon/phonemizer) - Text phonemization library

---

**Made with ❤️ for Windows TTS enthusiasts**
````
