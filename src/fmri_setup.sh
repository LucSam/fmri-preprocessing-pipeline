#!/bin/bash

# fmri_setup.sh
# Setup script for fMRI preprocessing pipeline environment
# This script creates a conda environment with all required dependencies
# Author: Lucius Fekonja
# Date: April 2025

# Set up color formatting for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}╔═════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║ $1${NC}"
    echo -e "${BLUE}╚═════════════════════════════════════════════════════════════════╝${NC}"
}

# Function to print completion message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning message
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Environment name
ENV_NAME="fmri_env"

# Check if conda is installed
print_section "Checking for Conda"
if ! command_exists conda; then
    print_error "Conda not found. Please install Miniconda or Anaconda first."
fi
print_success "Conda found."

# Check conda version
CONDA_VERSION=$(conda --version | awk '{print $2}')
echo "Conda version: $CONDA_VERSION"

# Check if environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    print_warning "Environment $ENV_NAME already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_section "Removing existing environment"
        conda env remove -n $ENV_NAME
        print_success "Environment removed."
    else
        print_warning "Setup cancelled. Using existing environment."
        exit 0
    fi
fi

# Create conda environment with Python 3.10 (more stable for scientific packages)
print_section "Creating Conda Environment"
echo "Creating environment $ENV_NAME with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y
print_success "Environment created."

# Activate environment
print_section "Activating Environment"
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
if [ $? -ne 0 ]; then
    print_error "Failed to activate environment. Try running: conda init bash"
fi
print_success "Environment activated."

# Install required packages using pip (more reliable for these packages)
print_section "Installing Python Packages"
echo "Installing required packages..."

# Install core packages with pip
pip install --upgrade pip
pip install numpy==1.26.0
pip install scipy==1.11.4
pip install matplotlib==3.8.2
pip install pandas==2.1.4
pip install scikit-learn==1.4.0
pip install nibabel==5.2.0
pip install nilearn==0.10.2
pip install seaborn==0.13.1
pip install future==0.18.3

# Install ANTs packages
echo "Installing ANTsPy and ANTsPyNet (this may take some time)..."
pip install antspyx antspynet

print_success "All Python packages installed."

# Check for required software
print_section "Checking Required Software"

software_missing=0

# Check for FSL
if ! command_exists flirt || ! command_exists fsl; then
    print_warning "FSL not found in path. Please make sure FSL is installed."
    echo "  On Ubuntu/Debian: sudo apt-get install fsl"
    echo "  On MacOS: Use FSL installer from https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation"
    software_missing=1
else
    print_success "FSL found."
fi

# Check for MRtrix3
if ! command_exists mrconvert; then
    print_warning "MRtrix3 not found in path. Please make sure MRtrix3 is installed."
    echo "  On Ubuntu/Debian: sudo apt-get install mrtrix3"
    echo "  On MacOS: brew install mrtrix3"
    software_missing=1
else
    print_success "MRtrix3 found."
fi

# Check for ANTs
if ! command_exists antsRegistrationSyNQuick.sh; then
    print_warning "ANTs not found in path. Please make sure ANTs is installed."
    echo "  On Ubuntu/Debian: sudo apt-get install ants"
    echo "  On MacOS: brew install ants"
    software_missing=1
else
    print_success "ANTs found."
fi

# Check for FreeSurfer (optional)
if ! command_exists recon-all; then
    print_warning "FreeSurfer not found in path. Only needed if using FreeSurfer atlas or SynthSeg."
    echo "  Download from: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall"
else
    print_success "FreeSurfer found."
    # Check if FREESURFER_HOME is set
    if [ -z "$FREESURFER_HOME" ]; then
        print_warning "FREESURFER_HOME environment variable not set. Set it before running the pipeline."
    fi
fi

# Verify that packages were installed correctly
print_section "Verifying Python Packages"
python -c "
import sys
packages = ['nibabel', 'numpy', 'nilearn', 'sklearn', 'scipy', 'matplotlib', 'seaborn', 'pandas']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg} is installed and importable')
    except ImportError:
        missing.append(pkg)
        print(f'✗ {pkg} could not be imported')
if missing:
    print(f'\\nWARNING: {len(missing)} packages could not be imported.')
    print('Try manually installing them with:')
    for pkg in missing:
        print(f'  pip install {pkg}')
    sys.exit(1)
"

# Create activation script
print_section "Creating Activation Script"
ACTIVATE_SCRIPT="activate_fmri_env.sh"

cat > $ACTIVATE_SCRIPT << EOF
#!/bin/bash
# Activation script for fMRI preprocessing environment
# Created: $(date)

# Activate conda environment
eval "\$(conda shell.bash hook)"
conda activate $ENV_NAME

# Make sure Python paths are set correctly
export PYTHONPATH="\$CONDA_PREFIX/lib/python\$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')/site-packages:\$PYTHONPATH"

# Make sure FSL is set up properly
if [ -f /etc/fsl/fsl.sh ]; then
    source /etc/fsl/fsl.sh
elif [ -f /usr/local/fsl/etc/fslconf/fsl.sh ]; then
    source /usr/local/fsl/etc/fslconf/fsl.sh
fi

# Set up FreeSurfer if it exists
if [ -d "/usr/local/freesurfer" ]; then
    export FREESURFER_HOME=/usr/local/freesurfer
    source \$FREESURFER_HOME/SetUpFreeSurfer.sh
fi

# Set up environment flag
export FMRI_ENV_ACTIVE=1

echo "fMRI preprocessing environment activated."
echo "Run your pipeline with:"
echo "  ./fmri-preprocessing-pipeline-1_0.sh [options]"
EOF

chmod +x $ACTIVATE_SCRIPT
print_success "Created activation script: $ACTIVATE_SCRIPT"

if [ $software_missing -eq 1 ]; then
    print_warning "Some required software is missing. Please install the missing components."
    echo "Once installed, run the pipeline with:"
    echo "  source ./$ACTIVATE_SCRIPT"
    echo "  ./fmri-preprocessing-pipeline-1_0.sh [options]"
else
    print_success "All required software found."
    echo "Run the pipeline with:"
    echo "  source ./$ACTIVATE_SCRIPT"
    echo "  ./fmri-preprocessing-pipeline-1_0.sh [options]"
fi

print_section "Setup Complete"
echo "To use this environment in the future, run:"
echo "  source ./$ACTIVATE_SCRIPT"