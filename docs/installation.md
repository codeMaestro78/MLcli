# Installation Guide

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, Linux
- **Memory**: 4GB RAM minimum (8GB+ recommended for deep learning)

## Installation Methods

### 1. From PyPI (Recommended)

```bash
pip install mlcli-toolkit
```

### 2. With Optional Dependencies

```bash
# Full installation with all features
pip install mlcli-toolkit[all]

# Deep learning support
pip install mlcli-toolkit[tensorflow]

# Explainability support
pip install mlcli-toolkit[explain]

# Development dependencies
pip install mlcli-toolkit[dev]
```

### 3. From Source

```bash
git clone https://github.com/codeMaestro78/mlcli.git
cd mlcli
pip install -e .
```

### 4. Using Conda

```bash
conda create -n mlcli python=3.10
conda activate mlcli
pip install mlcli-toolkit
```

## Verify Installation

```bash
# Check version
mlcli --version

# List available commands
mlcli --help

# List available models
mlcli list-models

# Run a quick test
mlcli ui
```

## Troubleshooting

### Common Issues

#### 1. TensorFlow Not Found

```bash
pip install tensorflow>=2.10
```

#### 2. XGBoost Installation Failed

```bash
# On Windows
pip install xgboost

# On macOS with M1/M2
pip install xgboost --no-binary :all:
```

#### 3. ONNX Export Issues

```bash
pip install skl2onnx onnxruntime
```

### Platform-Specific Notes

#### Windows

- Use PowerShell or Command Prompt
- May need Visual C++ Build Tools for some dependencies

#### macOS

- For M1/M2 Macs, use `tensorflow-macos` instead of `tensorflow`
- Install Xcode Command Line Tools: `xcode-select --install`

#### Linux

- Ensure `python3-dev` is installed
- For GPU support: Install CUDA and cuDNN

## Updating

```bash
pip install --upgrade mlcli-toolkit
```

## Uninstalling

```bash
pip uninstall mlcli-toolkit
```
