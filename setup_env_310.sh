#!/bin/bash
# Setup Python 3.10 Virtual Environment and Install Requirements

if ! command -v python3.10 &>/dev/null; then
    echo "Error: python3.10 not found. Please install Python 3.10 first."
    exit 1
fi

echo "Creating Python 3.10 Virtual Environment..."
python3.10 -m venv venv

echo "Activating venv..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

# 1. Install Web App & UI dependencies
echo "Installing Web App Requirements..."
pip install -r requirements/requirements_web.txt

# 2. Install DiariZen dependencies
echo "Installing DiariZen Requirements..."
pip install -r requirements/requirements_diarizen.txt

# 3. Install Pyannote dependencies
echo "Installing Pyannote Requirements..."
pip install -r requirements/requirements_pyannote.txt

# 4. Install PhoWhisper dependencies
echo "Installing PhoWhisper Requirements..."
pip install -r requirements/requirements_phowhisper.txt

# 5. Install Qwen dependencies
echo "Installing Qwen Requirements..."
pip install -r requirements/requirements_qwen.txt

echo "================================================="
echo "WARNING: requirements_diarizen.txt (and others) may"
echo "override some web dependencies."
echo "e.g., bitsandbytes from requirements_qwen.txt may"
echo "conflict with pyannote-audio version constraints"
echo "in requirements_diarizen.txt."
echo "If you encounter errors, create a separate venv"
echo "for isolated tasks."
echo "================================================="

echo "Setup Complete! To activate the environment, run:"
echo "source venv/bin/activate"
