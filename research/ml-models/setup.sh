#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -gt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ]); then
            print_status "Python version $PYTHON_VERSION is compatible"
            return 0
        else
            print_error "Python version $PYTHON_VERSION is not compatible. Please use Python 3.8 or higher."
            return 1
        fi
    else
        print_error "Python 3 is not installed"
        return 1
    fi
}

# Function to check if pip is installed
check_pip() {
    if command_exists pip3; then
        print_status "pip is installed"
        return 0
    else
        print_error "pip is not installed"
        return 1
    fi
}

# Function to create and activate virtual environment
setup_venv() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    else
        print_warning "Virtual environment already exists"
    fi

    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate

    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install core dependencies
    pip install -r requirements.txt

    # Install optional dependencies if needed
    if [ "$1" == "--with-gpu" ]; then
        print_status "Installing GPU support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
}

# Function to check CUDA availability
check_cuda() {
    if command_exists nvidia-smi; then
        print_status "CUDA is available"
        nvidia-smi
        return 0
    else
        print_warning "CUDA is not available. Running in CPU mode."
        return 1
    fi
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check if all required packages are installed
    python3 -c "
import sys
import importlib.util
import pkg_resources

def check_package(package_name, import_name=None):
    try:
        # Try importing the package
        importlib.import_module(import_name or package_name)
        # Check if version matches requirements
        pkg_resources.require(package_name)
        return True
    except (ImportError, pkg_resources.VersionConflict, pkg_resources.DistributionNotFound) as e:
        print(f'Error with {package_name}: {str(e)}')
        return False

# List of (package_name, import_name) tuples
required_packages = [
    ('numpy', None),
    ('pandas', None),
    ('scikit-learn', 'sklearn'),  # scikit-learn is imported as sklearn
    ('matplotlib', None),
    ('seaborn', None),
    ('joblib', None),
    ('tqdm', None),
    ('wandb', None),
    ('xgboost', None),
    ('lightgbm', None),
    ('catboost', None),
    ('scipy', None),
    ('spectral', None),
    ('torch', None),
    ('torchvision', None)
]

missing_packages = []
for package_name, import_name in required_packages:
    if not check_package(package_name, import_name):
        missing_packages.append(package_name)

if missing_packages:
    print('Missing or incompatible packages:', missing_packages)
    sys.exit(1)
else:
    print('All required packages are installed and compatible')
    " || {
        print_error "Package verification failed"
        return 1
    }

    # Additional verification steps
    print_status "Running additional verifications..."
    
    # Check if we can import and use key packages
    python3 -c "
import numpy as np
import pandas as pd
from sklearn import datasets  # Note: using sklearn here
import torch

# Test numpy
arr = np.array([1, 2, 3])
print('NumPy test passed')

# Test pandas
df = pd.DataFrame({'test': [1, 2, 3]})
print('Pandas test passed')

# Test scikit-learn
iris = datasets.load_iris()
print('Scikit-learn test passed')

# Test PyTorch
x = torch.randn(2, 3)
print('PyTorch test passed')
    " || {
        print_error "Package functionality test failed"
        return 1
    }

    print_status "All verifications passed successfully"
    return 0
}

# Function to check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Check available memory
    if command_exists free; then
        MEMORY=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$MEMORY" -lt 4 ]; then
            print_warning "Less than 4GB of RAM available. Some models might not work optimally."
        fi
    fi
    
    # Check available disk space
    DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
    print_status "Available disk space: $DISK_SPACE"
    
    # Check CPU cores
    if command_exists nproc; then
        CPU_CORES=$(nproc)
        print_status "Available CPU cores: $CPU_CORES"
    fi
}

# Main setup process
main() {
    print_status "Starting setup process..."

    # Check system requirements
    check_python_version || exit 1
    check_pip || exit 1
    check_system_resources

    # Setup virtual environment
    setup_venv

    # Install dependencies
    if [ "$1" == "--with-gpu" ]; then
        check_cuda
        install_dependencies --with-gpu
    else
        install_dependencies
    fi

    # Verify installation
    verify_installation || exit 1

    print_status "Setup completed successfully!"
    print_status "To activate the virtual environment, run: source venv/bin/activate"
    print_status "To run a model, use: python run_trainer.py --config configs/<model_name>.yaml"
}

# Run main function with all arguments
main "$@"
