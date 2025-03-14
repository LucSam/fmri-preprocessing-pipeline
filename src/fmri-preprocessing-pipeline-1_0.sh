#!/bin/bash

# fmri_preprocessing_with_compcor.sh
# A comprehensive script for fMRI preprocessing with distortion correction, 
# motion correction, ICA-AROMA denoising, CompCor denoising, and connectivity analysis.
# 
# Author: Lucius Fekonja
# Version: 15.5
# Date: March 2025

# Exit immediately if a command exits with a non-zero status
set -e

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
}

usage() {
    cat <<EOF
$(print_section "fMRI Preprocessing Pipeline")

Usage: $(basename $0) [options]

Options:
  -s SUBJECT_DIR        Path to the subject directory (default: current directory)
  -o OUTPUT_DIR         Path to the output directory (default: SUBJECT_DIR/preprocessed)
  -d DICOM_DIRS         DICOM directories (comma-separated, required)
  -A ATLAS_TYPE         Type of atlas to use: 'aal', 'schaefer', or 'freesurfer' (default: 'aal')
  -a ATLAS_FILE         Path to the atlas file (required for 'aal' and 'schaefer' atlas types)
  -t T1_IMAGE           Path to the T1 image (required)
  -l LESION_MASK        Path to the lesion mask (optional)
  -f FREESURFER_SUBJECT_DIR   Path to the FreeSurfer subject directory (required for 'freesurfer' atlas type)
  -I ICA_AROMA_DIR      Path to ICA-AROMA.py (required)
  -S SEGMENTATION_TYPE  Segmentation method to use: 'atropos' or 'synthseg' (default: 'atropos')
  -C ENABLE_COMPCOR     Enable CompCor denoising: 0=disabled, 1=enabled (default: 0)
  -N COMPCOR_COMPONENTS Number of CompCor components to extract (default: 5)
  -B BANDPASS_FILTER    Apply bandpass filter with CompCor: 0=disabled, 1=enabled (default: 1)
  -L LOWPASS_HZ         Lowpass filter cutoff in Hz (default: 0.08)
  -H HIGHPASS_HZ        Highpass filter cutoff in Hz (default: 0.01)
  -h                    Display this help message

Examples:
With AAL atlas, SynthSeg segmentation, and CompCor:
./$(basename $0) -s /path/to/subject \\
  -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \\
  -A aal -a /path/to/AAL.nii \\
  -t /path/to/t1.nii.gz \\
  -S synthseg \\
  -I /path/to/ICA-AROMA.py \\
  -C 1 -N 5 -B 1 -L 0.08 -H 0.01

With Schaefer atlas and Deep Atropos segmentation:
./$(basename $0) -s /path/to/subject \\
  -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \\
  -A schaefer -a /path/to/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz \\
  -t /path/to/t1.nii.gz \\
  -S atropos \\
  -I /path/to/ICA-AROMA.py

With FreeSurfer:
./$(basename $0) -s /path/to/subject \\
  -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \\
  -A freesurfer \\
  -t /path/to/t1.nii.gz \\
  -f /path/to/freesurfer_subj_dir \\
  -I /path/to/ICA-AROMA.py

Dependencies:
  Software:
    - FSL
    - MRtrix3
    - ANTs
    - FreeSurfer (if using FreeSurfer atlas or SynthSeg segmentation)
    - Python 3
    - ICA-AROMA

  Python Packages:
    - ants (if using Deep Atropos segmentation)
    - antspynet (if using Deep Atropos segmentation)
    - nibabel
    - numpy
    - nilearn
    - scikit-learn
    - scipy
    - matplotlib
    - seaborn
    - pandas

EOF
}

# Default values
SUBJECT_DIR=$(pwd)
OUTPUT_DIR=""
DICOM_DIRS=""
ATLAS_FILE=""
ATLAS_TYPE="aal"  # Default atlas type
T1_IMAGE=""
LESION_MASK=""
SEGMENTATION_TYPE="atropos"  # Default segmentation type
ENABLE_COMPCOR=0  # Default: CompCor disabled
COMPCOR_COMPONENTS=5  # Default: 5 components
BANDPASS_FILTER=1  # Default: Apply bandpass filter
LOWPASS_HZ=0.08  # Default: 0.08 Hz lowpass
HIGHPASS_HZ=0.01  # Default: 0.01 Hz highpass

# Parse command-line options
while getopts "s:o:d:A:a:t:l:f:I:S:C:N:B:L:H:h" opt; do
  case $opt in
    s) SUBJECT_DIR="$OPTARG";;
    o) OUTPUT_DIR="$OPTARG";;
    d) DICOM_DIRS="$OPTARG";;
    A) ATLAS_TYPE="$OPTARG";;
    a) ATLAS_FILE="$OPTARG";;
    t) T1_IMAGE="$OPTARG";;
    l) LESION_MASK="$OPTARG";;
    f) FREESURFER_SUBJECT_DIR="$OPTARG";;
    I) ICA_AROMA_DIR="$OPTARG";;
    S) SEGMENTATION_TYPE="$OPTARG";;
    C) ENABLE_COMPCOR="$OPTARG";;
    N) COMPCOR_COMPONENTS="$OPTARG";;
    B) BANDPASS_FILTER="$OPTARG";;
    L) LOWPASS_HZ="$OPTARG";;
    H) HIGHPASS_HZ="$OPTARG";;
    h) usage; exit 0;;
    \?) print_error "Invalid option -$OPTARG"; usage; exit 1;;
  esac
done

# Check required arguments
if [ -z "$DICOM_DIRS" ] || [ -z "$T1_IMAGE" ] || [ -z "$ICA_AROMA_DIR" ]; then
    print_error "Missing required arguments."
    usage
    exit 1
fi

# For 'aal' and 'schaefer' atlas types, ATLAS_FILE is required
if [[ "$ATLAS_TYPE" != "freesurfer" && -z "$ATLAS_FILE" ]]; then
    print_error "Atlas file (-a) is required for atlas type '$ATLAS_TYPE'."
    usage
    exit 1
fi

# Check if FREESURFER_SUBJECT_DIR is provided when using the freesurfer atlas
if [[ "$ATLAS_TYPE" == "freesurfer" && -z "$FREESURFER_SUBJECT_DIR" ]]; then
    print_error "FreeSurfer subject directory (-f) is required when using 'freesurfer' atlas type."
    usage
    exit 1
fi

# Validate segmentation type
if [[ "$SEGMENTATION_TYPE" != "atropos" && "$SEGMENTATION_TYPE" != "synthseg" ]]; then
    print_error "Invalid segmentation type '$SEGMENTATION_TYPE'. Please use 'atropos' or 'synthseg'."
    usage
    exit 1
fi

# Set OUTPUT_DIR if not provided
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${SUBJECT_DIR}/preprocessed"
fi

# Create output directories
mkdir -p ${OUTPUT_DIR}/{nifti,motion_correction,QC,connectivity,registration,synthseg}

# Create CompCor directory if enabled
if [ "$ENABLE_COMPCOR" -eq 1 ]; then
    mkdir -p ${OUTPUT_DIR}/compcor
fi

# Set up log file
LOG_FILE="${OUTPUT_DIR}/preprocessing.log"
LOG_TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE_BACKUP="${OUTPUT_DIR}/preprocessing_${LOG_TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Function to log messages with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "${LOG_FILE}"
}

# Calculate total steps based on enabled features
TOTAL_STEPS=17  # Base steps
if [ "$ENABLE_COMPCOR" -eq 1 ]; then
    TOTAL_STEPS=$((TOTAL_STEPS + 1))  # Add CompCor step
fi

# Function to update progress bar
update_progress() {
    step=$1
    step_name=$2
    percentage=$((step * 100 / TOTAL_STEPS))

    # Create the progress bar
    num_hashes=$((percentage / 2))
    bar=$(printf "%0.s#" $(seq 1 $num_hashes))
    spaces=$((50 - num_hashes))

    # Print the progress bar to stderr
    printf "\rProgress: [%-50s] %3d%% | Step %2d/%-2d | %s" \
           "$bar$(printf "%${spaces}s")" $percentage $step $TOTAL_STEPS "${step_name}" >&2
}

# Clear the progress bar at the end
clear_progress_bar() {
    printf "\n" >&2
}

# Check for required software
# Function to check for dependencies with consistent output style
check_dependencies() {
    print_section "Checking Dependencies"
    
    # Define list of dependencies to check
    local dependencies=(
        "fsl:FSL"
        "mrconvert:MRtrix3"
        "antsRegistrationSyNQuick.sh:ANTs"
        "python3:Python3"
    )
    
    # Add SynthSeg dependency if needed
    if [ "$SEGMENTATION_TYPE" == "synthseg" ]; then
        dependencies+=("mri_synthseg:FreeSurfer SynthSeg")
    fi
    
    # Check for required software
    for dep in "${dependencies[@]}"; do
        IFS=':' read -r cmd name <<< "$dep"
        if command -v "$cmd" &> /dev/null; then
            print_success "$name is installed."
        else
            print_error "$name is not installed or not in PATH. Please install $name."
            exit 1
        fi
    done
    
    # Check for ICA-AROMA
    if [ ! -f "${ICA_AROMA_DIR}" ]; then
        print_error "ICA-AROMA.py not found at specified path: ${ICA_AROMA_DIR}"
        exit 1
    else
        print_success "ICA-AROMA is available."
    fi
    
    # Check FreeSurfer if required
    if [ "$ATLAS_TYPE" == "freesurfer" ] || [ "$SEGMENTATION_TYPE" == "synthseg" ]; then
        if [ -z "$FREESURFER_HOME" ] || ! command -v mri_label2vol &> /dev/null; then
            print_error "FreeSurfer is not installed or FREESURFER_HOME is not set. Please set up FreeSurfer."
            exit 1
        else
            print_success "FreeSurfer is installed."
        fi
    fi
    
    # Check Python packages depending on segmentation type
    log_message "Checking Python packages..."
    if [ "$SEGMENTATION_TYPE" == "atropos" ]; then
        # Include ANTsPy for Atropos segmentation
        python3 -c "
import sys

required_packages = [
    ('nibabel', 'NiBabel'),
    ('numpy', 'NumPy'),
    ('nilearn', 'NiLearn'),
    ('sklearn', 'scikit-learn'),
    ('scipy', 'SciPy'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('pandas', 'Pandas')
]

optional_packages = [
    ('ants', 'ANTsPy'),
    ('antspynet', 'ANTsPyNet')
]

# Check required packages
for package, display_name in required_packages:
    try:
        __import__(package)
        print(f'✓ {display_name} is installed.')
    except ImportError:
        print(f'✗ {display_name} is not installed. Please install it.', file=sys.stderr)
        sys.exit(1)

# For Atropos segmentation, ANTsPy and ANTsPyNet are required
for package, display_name in optional_packages:
    try:
        __import__(package)
        print(f'✓ {display_name} is installed.')
    except ImportError:
        print(f'✗ {display_name} is required for Deep Atropos segmentation but not installed.', file=sys.stderr)
        print(f'  Please install with: pip install antspyx antspynet', file=sys.stderr)
        sys.exit(1)
"
    else
        # For SynthSeg, we don't need ANTsPy
        python3 -c "
import sys

required_packages = [
    ('nibabel', 'NiBabel'),
    ('numpy', 'NumPy'),
    ('nilearn', 'NiLearn'),
    ('sklearn', 'scikit-learn'),
    ('scipy', 'SciPy'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('pandas', 'Pandas')
]

# Check required packages
for package, display_name in required_packages:
    try:
        __import__(package)
        print(f'✓ {display_name} is installed.')
    except ImportError:
        print(f'✗ {display_name} is not installed. Please install it.', file=sys.stderr)
        sys.exit(1)
"
    fi
    
    # Check return code from Python
    if [ $? -ne 0 ]; then
        print_error "Missing required Python packages. Please install them before continuing."
        exit 1
    fi
}

# Function to validate input files
validate_inputs() {
    print_section "Validating Input Files"
    
    # Check T1 image
    if [ ! -f "${T1_IMAGE}" ]; then
        print_error "T1 image not found: ${T1_IMAGE}"
        exit 1
    else
        print_success "T1 image found: ${T1_IMAGE}"
    fi
    
    # Check atlas file for non-FreeSurfer atlases
    if [ "$ATLAS_TYPE" != "freesurfer" ] && [ ! -f "${ATLAS_FILE}" ]; then
        print_error "Atlas file not found: ${ATLAS_FILE}"
        exit 1
    elif [ "$ATLAS_TYPE" != "freesurfer" ]; then
        print_success "Atlas file found: ${ATLAS_FILE}"
    fi
    
    # Check lesion mask if provided
    if [ -n "${LESION_MASK}" ]; then
        if [ ! -f "${LESION_MASK}" ]; then
            print_error "Lesion mask not found: ${LESION_MASK}"
            exit 1
        else
            print_success "Lesion mask found: ${LESION_MASK}"
        fi
    fi
    
    # Check FreeSurfer subject directory if using FreeSurfer
    if [ "$ATLAS_TYPE" == "freesurfer" ]; then
        if [ ! -d "${FREESURFER_SUBJECT_DIR}" ] || [ ! -f "${FREESURFER_SUBJECT_DIR}/mri/aparc+aseg.mgz" ]; then
            print_error "FreeSurfer subject directory invalid or aparc+aseg.mgz not found: ${FREESURFER_SUBJECT_DIR}"
            exit 1
        else
            print_success "FreeSurfer subject directory found: ${FREESURFER_SUBJECT_DIR}"
        fi
    fi
    
    # Check DICOM directories
    # Parse DICOM directories
    IFS=',' read -r -a DICOM_DIR_ARRAY <<< "$DICOM_DIRS"

    # Check if the required DICOM directories are provided
    if [ ${#DICOM_DIR_ARRAY[@]} -ne 4 ]; then
        print_error "Exactly four DICOM directories must be provided."
        exit 1
    fi

    # DICOM directories
    DICOM_DIR_010="${SUBJECT_DIR}/${DICOM_DIR_ARRAY[0]}"
    DICOM_DIR_011="${SUBJECT_DIR}/${DICOM_DIR_ARRAY[1]}"
    DICOM_DIR_012="${SUBJECT_DIR}/${DICOM_DIR_ARRAY[2]}"
    DICOM_DIR_013="${SUBJECT_DIR}/${DICOM_DIR_ARRAY[3]}"
    
    # Check if directories exist
    for dir in "${DICOM_DIR_010}" "${DICOM_DIR_011}" "${DICOM_DIR_012}" "${DICOM_DIR_013}"; do
        if [ ! -d "$dir" ]; then
            print_error "DICOM directory not found: $dir"
            exit 1
        else
            print_success "DICOM directory found: $dir"
        fi
    done
}

# Main execution
check_dependencies
validate_inputs

# Log start of preprocessing
print_section "Starting Preprocessing Pipeline"
log_message "Starting preprocessing"
update_progress 0 "Initializing"

# DICOM directories
DICOM_DIR_010="${SUBJECT_DIR}/${DICOM_DIR_ARRAY[0]}"
DICOM_DIR_011="${SUBJECT_DIR}/${DICOM_DIR_ARRAY[1]}"
DICOM_DIR_012="${SUBJECT_DIR}/${DICOM_DIR_ARRAY[2]}"
DICOM_DIR_013="${SUBJECT_DIR}/${DICOM_DIR_ARRAY[3]}"

# Convert DICOM to NIfTI
convert_if_needed() {
    local dicom_dir=$1
    local output_nii=$2
    local output_json=$3
    if [ ! -f "${output_nii}" ] || [ ! -f "${output_json}" ]; then
        log_message "Converting ${dicom_dir} to NIfTI"
        mrconvert "${dicom_dir}" "${output_nii}" -json_export "${output_json}" >> "${LOG_FILE}" 2>&1
        log_message "Conversion complete: ${output_nii}"
    else
        log_message "Skipping conversion, files already exist: ${output_nii}"
    fi
}

print_section "DICOM to NIfTI Conversion"
# Convert DICOM to NIfTI
convert_if_needed ${DICOM_DIR_010} ${OUTPUT_DIR}/nifti/010_SpinEchoFieldMap_AP.nii.gz ${OUTPUT_DIR}/nifti/010_SpinEchoFieldMap_AP.json
convert_if_needed ${DICOM_DIR_011} ${OUTPUT_DIR}/nifti/011_SpinEchoFieldMap_PA.nii.gz ${OUTPUT_DIR}/nifti/011_SpinEchoFieldMap_PA.json
convert_if_needed ${DICOM_DIR_012} ${OUTPUT_DIR}/nifti/012_rfMRI_REST_AP.nii.gz ${OUTPUT_DIR}/nifti/012_rfMRI_REST_AP.json
convert_if_needed ${DICOM_DIR_013} ${OUTPUT_DIR}/nifti/013_rfMRI_REST_AP.nii.gz ${OUTPUT_DIR}/nifti/013_rfMRI_REST_AP.json
update_progress 1 "DICOM to NIfTI"

# Define files
FMAP_AP="${OUTPUT_DIR}/nifti/010_SpinEchoFieldMap_AP.nii.gz"
FMAP_PA="${OUTPUT_DIR}/nifti/011_SpinEchoFieldMap_PA.nii.gz"
FMRI_MAIN="${OUTPUT_DIR}/nifti/013_rfMRI_REST_AP.nii.gz"
LESION_MASK="${LESION_MASK}"  # May be empty

print_section "TOPUP Distortion Correction"
# Generate TOPUP encoding file
if [ ! -f "${OUTPUT_DIR}/topup_encoding_info.txt" ]; then
    log_message "Generating TOPUP encoding file"
    PE_AP=$(jq -r '.PhaseEncodingDirection' ${OUTPUT_DIR}/nifti/010_SpinEchoFieldMap_AP.json)
    PE_PA=$(jq -r '.PhaseEncodingDirection' ${OUTPUT_DIR}/nifti/011_SpinEchoFieldMap_PA.json)
    RO_TIME_AP=$(jq -r '.TotalReadoutTime' ${OUTPUT_DIR}/nifti/010_SpinEchoFieldMap_AP.json)
    RO_TIME_PA=$(jq -r '.TotalReadoutTime' ${OUTPUT_DIR}/nifti/011_SpinEchoFieldMap_PA.json)
    echo "0 -1 0 ${RO_TIME_AP}" > ${OUTPUT_DIR}/topup_encoding_info.txt
    echo "0 1 0 ${RO_TIME_PA}" >> ${OUTPUT_DIR}/topup_encoding_info.txt
    log_message "TOPUP encoding file generated"
else
    log_message "TOPUP encoding file already exists"
fi
update_progress 2 "TOPUP encoding"

# Run TOPUP
if [ ! -f "${OUTPUT_DIR}/registration/topup_results_fieldcoef.nii.gz" ]; then
    log_message "Running TOPUP"
    fslmerge -t ${OUTPUT_DIR}/registration/fmap_all.nii.gz $FMAP_AP $FMAP_PA >> "${LOG_FILE}" 2>&1
    topup --imain=${OUTPUT_DIR}/registration/fmap_all.nii.gz --datain=${OUTPUT_DIR}/topup_encoding_info.txt --config=b02b0.cnf --out=${OUTPUT_DIR}/registration/topup_results --iout=${OUTPUT_DIR}/registration/topup_corrected >> "${LOG_FILE}" 2>&1
    log_message "TOPUP completed"
else
    log_message "Skipping TOPUP, results already exist"
fi
update_progress 3 "TOPUP"

# Apply topup to correct fMRI data
if [ ! -f "${OUTPUT_DIR}/registration/topup_corrected_fmri.nii.gz" ]; then
    log_message "Applying TOPUP correction to fMRI data"
    applytopup --imain=$FMRI_MAIN --inindex=1 --datain=${OUTPUT_DIR}/topup_encoding_info.txt --topup=${OUTPUT_DIR}/registration/topup_results --out=${OUTPUT_DIR}/registration/topup_corrected_fmri --method=jac >> "${LOG_FILE}" 2>&1
    log_message "TOPUP correction applied"
else
    log_message "Skipping TOPUP correction, file already exists"
fi
update_progress 4 "Apply TOPUP"

print_section "Motion Correction"
# Apply MCFLIRT for motion correction
if [ ! -f "${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.nii.gz" ]; then
    log_message "Applying MCFLIRT for motion correction"
    mcflirt -in ${OUTPUT_DIR}/registration/topup_corrected_fmri.nii.gz \
            -out ${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri \
            -mats -plots -rmsrel -rmsabs \
            -spline_final >> "${LOG_FILE}" 2>&1

    # Generate additional motion metrics
    fsl_motion_outliers -i ${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.nii.gz \
                        -o ${OUTPUT_DIR}/motion_correction/fd_confounds.txt \
                        -s ${OUTPUT_DIR}/motion_correction/fd_metrics.txt \
                        --fd --thresh=0.2

# Replace the existing FSL plotting commands with seaborn-based Python plotting
if [ ! -f "${OUTPUT_DIR}/QC/rot_motion.png" ] || [ ! -f "${OUTPUT_DIR}/QC/trans_motion.png" ]; then
    log_message "Generating motion parameter plots with seaborn..."
    python3 - <<EOF
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set seaborn style for consistent aesthetics
sns.set(style="whitegrid", font_scale=1.2)

try:
    # Load motion parameters
    motion_params = np.loadtxt('${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.par')
    
    # Labels for motion parameters
    labels = ['RotX', 'RotY', 'RotZ', 'TransX', 'TransY', 'TransZ']
    
    # Create DataFrame for better seaborn integration
    motion_df = pd.DataFrame(motion_params, columns=labels)
    motion_df['Volume'] = np.arange(len(motion_df))
    
    # Rotation parameters plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=motion_df.iloc[:, 0:3], dashes=False)
    plt.title('MCFLIRT estimated rotations (radians)', fontsize=14)
    plt.xlabel('Volume', fontsize=12)
    plt.ylabel('Rotation (radians)', fontsize=12)
    plt.legend(labels[:3], loc='upper right')
    plt.tight_layout()
    plt.savefig('${OUTPUT_DIR}/QC/rot_motion.png', dpi=300)
    plt.close()
    
    # Translation parameters plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=motion_df.iloc[:, 3:6], dashes=False)
    plt.title('MCFLIRT estimated translations (mm)', fontsize=14)
    plt.xlabel('Volume', fontsize=12)
    plt.ylabel('Translation (mm)', fontsize=12)
    plt.legend(labels[3:], loc='upper right')
    plt.tight_layout()
    plt.savefig('${OUTPUT_DIR}/QC/trans_motion.png', dpi=300)
    plt.close()
    
    # Combined motion plot with both rotation and translation
    plt.figure(figsize=(14, 8))
    
    # Plot on two y-axes for better visualization
    ax1 = plt.subplot(211)
    rot_plot = sns.lineplot(data=motion_df.iloc[:, 0:3], ax=ax1, dashes=False)
    ax1.set_title('Motion Parameters', fontsize=14)
    ax1.set_ylabel('Rotation (radians)', fontsize=12)
    ax1.legend(labels[:3], loc='upper right')
    ax1.set_xlabel('')
    
    ax2 = plt.subplot(212, sharex=ax1)
    trans_plot = sns.lineplot(data=motion_df.iloc[:, 3:6], ax=ax2, dashes=False)
    ax2.set_ylabel('Translation (mm)', fontsize=12)
    ax2.set_xlabel('Volume', fontsize=12)
    ax2.legend(labels[3:], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('${OUTPUT_DIR}/QC/combined_motion.png', dpi=300)
    plt.close()
    
    print("Motion parameter plots generated with seaborn.")
except Exception as e:
    print(f"Error generating motion parameter plots: {e}")
EOF
else
    log_message "Skipping motion parameter plots, files already exist."
fi

    log_message "MCFLIRT motion correction and additional metrics completed"
else
    log_message "Skipping MCFLIRT, file already exists"
fi
update_progress 5 "MCFLIRT"

# Generate motion confounds file
if [ ! -f "${OUTPUT_DIR}/motion_correction/motion_confounds.txt" ]; then
    log_message "Generating motion confounds file"
    paste ${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.par ${OUTPUT_DIR}/motion_correction/fd_metrics.txt > ${OUTPUT_DIR}/motion_correction/motion_confounds.txt
    log_message "Motion confounds file generated"
else
    log_message "Skipping motion confounds generation, file already exists"
fi
update_progress 6 "Motion Confounds"

print_section "Brain Extraction"
# Extract a 3D volume from the 4D image
log_message "Extracting 3D volume from 4D image"
if [ ! -f "${OUTPUT_DIR}/registration/fmri_3d.nii.gz" ]; then
    mrconvert ${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.nii.gz -coord 3 0 ${OUTPUT_DIR}/registration/fmri_3d.nii.gz >> "${LOG_FILE}" 2>&1
    log_message "3D volume extraction completed"
else
    log_message "Skipping 3D volume extraction, file already exists"
fi
update_progress 7 "3D Extraction"

# Brain extraction using ANTsPyNet on the 3D volume
log_message "Performing brain extraction using ANTsPyNet"
if [ ! -f "${OUTPUT_DIR}/registration/fmri_brain_mask.nii.gz" ]; then
    python3 - <<EOF
import ants
import antspynet

fmri_image = ants.image_read('${OUTPUT_DIR}/registration/fmri_3d.nii.gz')
fmri_brain_mask = antspynet.brain_extraction(fmri_image, modality='bold')
ants.image_write(fmri_brain_mask, '${OUTPUT_DIR}/registration/fmri_brain_mask.nii.gz')
fmri_brain = fmri_image * fmri_brain_mask
ants.image_write(fmri_brain, '${OUTPUT_DIR}/registration/fmri_brain.nii.gz')
print("Brain extraction completed.")
EOF
    log_message "Brain extraction completed"
else
    log_message "Skipping brain extraction, files already exist"
fi
update_progress 8 "Brain Extraction"

# Apply the brain mask to the 4D fMRI data
if [ ! -f "${OUTPUT_DIR}/registration/mcflirt_corrected_fmri_brain.nii.gz" ]; then
    log_message "Applying brain mask to 4D fMRI data using ANTsPy"
    python3 - <<EOF
import ants
import numpy as np

fmri_4d = ants.image_read('${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.nii.gz')
brain_mask = ants.image_read('${OUTPUT_DIR}/registration/fmri_brain_mask.nii.gz')

brain_mask_array = (brain_mask.numpy() > 0.5).astype(np.float32)
fmri_4d_array = fmri_4d.numpy()
fmri_4d_brain = fmri_4d_array * brain_mask_array[..., np.newaxis]

fmri_4d_brain_img = ants.from_numpy(fmri_4d_brain, origin=fmri_4d.origin, spacing=fmri_4d.spacing, direction=fmri_4d.direction)
ants.image_write(fmri_4d_brain_img, '${OUTPUT_DIR}/registration/mcflirt_corrected_fmri_brain.nii.gz')

print("Brain mask applied to 4D fMRI data.")
EOF
    log_message "Brain mask applied to 4D fMRI data"
else
    log_message "Skipping brain mask application, file already exists"
fi
update_progress 9 "Apply Brain Mask"

print_section "T1 to fMRI Registration"
# Adjust strides of T1 image
log_message "Adjusting strides of T1 image..."
if [ ! -f "${OUTPUT_DIR}/registration/t1_strides.nii.gz" ]; then
    log_message "Adjusting strides of T1 image..."
    mrconvert ${T1_IMAGE} -strides -1,-2,3 ${OUTPUT_DIR}/registration/t1_strides.nii.gz
else
    log_message "Skipping T1 strides adjustment, file already exists."
fi

# Register T1 to fMRI
log_message "Registering T1 to fMRI..."
if [ ! -f "${OUTPUT_DIR}/registration/registered_0GenericAffine.mat" ]; then
    antsRegistrationSyNQuick.sh -d 3 \
        -f ${OUTPUT_DIR}/registration/fmri_3d.nii.gz \
        -m ${OUTPUT_DIR}/registration/t1_strides.nii.gz \
        -o ${OUTPUT_DIR}/registration/registered_ \
        -n 8 \
        -t r >> "${LOG_FILE}" 2>&1

    if [ $? -ne 0 ]; then
        log_message "Error: T1 to fMRI registration failed."
        exit 1
    fi
else
    log_message "Skipping T1 to fMRI registration, transformation already exists."
fi

# Apply transformation to T1
antsApplyTransforms -d 3 \
    -i ${OUTPUT_DIR}/registration/t1_strides.nii.gz \
    -r ${OUTPUT_DIR}/registration/fmri_3d.nii.gz \
    -o ${OUTPUT_DIR}/registration/t1_in_fmri_space.nii.gz \
    -t ${OUTPUT_DIR}/registration/registered_0GenericAffine.mat \
    -n Linear >> "${LOG_FILE}" 2>&1

# Transform lesion mask to fMRI space (if it exists)
if [ -n "${LESION_MASK}" ] && [ -f "${LESION_MASK}" ] && [ ! -f "${OUTPUT_DIR}/registration/lesion_mask_in_fmri_space.nii.gz" ]; then
    log_message "Adjusting strides of lesion mask and transforming to fMRI space"
    mrconvert ${LESION_MASK} -strides -1,-2,3 ${OUTPUT_DIR}/registration/lesion_mask_strides.nii.gz
    antsApplyTransforms -d 3 \
        -i ${OUTPUT_DIR}/registration/lesion_mask_strides.nii.gz \
        -r ${OUTPUT_DIR}/registration/fmri_3d.nii.gz \
        -o ${OUTPUT_DIR}/registration/lesion_mask_in_fmri_space.nii.gz \
        -t ${OUTPUT_DIR}/registration/registered_0GenericAffine.mat \
        -n NearestNeighbor >> "${LOG_FILE}" 2>&1
    log_message "Lesion mask transformation to fMRI space completed"
else
    log_message "Skipping lesion mask transformation, file already exists or no lesion mask found"
fi
update_progress 10 "T1 and Lesion Mask to fMRI Transformation"

print_section "Tissue Segmentation"
# Perform tissue segmentation using the selected method (Deep Atropos or SynthSeg)

if [ "$SEGMENTATION_TYPE" == "atropos" ]; then
    # Use Deep Atropos segmentation
    log_message "Performing Deep Atropos segmentation on the original T1 image..."
    if [ ! -f "${OUTPUT_DIR}/registration/t1_brain_segmentation.nii.gz" ]; then
        python3 <<EOF
import ants
import antspynet

t1_image = ants.image_read('${T1_IMAGE}')

atropos = antspynet.deep_atropos(
    t1_image,
    do_preprocessing=True,
    use_spatial_priors=True
)
segmentation_image = atropos['segmentation_image']
probability_images = atropos['probability_images']

ants.image_write(segmentation_image, '${OUTPUT_DIR}/registration/t1_brain_segmentation.nii.gz')
ants.image_write(probability_images[1], '${OUTPUT_DIR}/registration/csf_prob.nii.gz')
ants.image_write(probability_images[2] + probability_images[4], '${OUTPUT_DIR}/registration/gm_prob.nii.gz')
ants.image_write(probability_images[3], '${OUTPUT_DIR}/registration/wm_prob.nii.gz')
ants.image_write(probability_images[4], '${OUTPUT_DIR}/registration/scgm_prob.nii.gz')

print("Deep Atropos segmentation completed on original T1 image.")
EOF
        log_message "Deep Atropos segmentation and probability maps generation completed"
    else
        log_message "Skipping Deep Atropos segmentation, files already exist"
    fi
    
    # Transform segmentation results to fMRI space
    log_message "Transforming segmentation results to fMRI space..."
    for image in t1_brain_segmentation csf_prob gm_prob wm_prob scgm_prob; do
        antsApplyTransforms -d 3 \
            -i ${OUTPUT_DIR}/registration/${image}.nii.gz \
            -r ${OUTPUT_DIR}/registration/fmri_3d.nii.gz \
            -o ${OUTPUT_DIR}/registration/${image}_in_fmri_space.nii.gz \
            -t ${OUTPUT_DIR}/registration/registered_0GenericAffine.mat \
            -n Linear >> "${LOG_FILE}" 2>&1
    done
    log_message "Segmentation results transformed to fMRI space"
    
elif [ "$SEGMENTATION_TYPE" == "synthseg" ]; then
    # Use SynthSeg segmentation
    # Create SynthSeg directory if it doesn't exist
    log_message "Creating SynthSeg directory..."
    mkdir -p "${OUTPUT_DIR}/synthseg"
    
    # Get tissue segmentation from MRISynthSeg
    log_message "Getting tissue segmentation from MRISynthSeg..."
    SYNTHSEG_SEG="${OUTPUT_DIR}/synthseg/segmentation.nii.gz"
    if [ ! -f "${SYNTHSEG_SEG}" ]; then
        # Run SynthSeg on T1 image
        log_message "Running SynthSeg on T1 image..."
        mri_synthseg --i "${T1_IMAGE}" --o "${SYNTHSEG_SEG}" --robust >> "${LOG_FILE}" 2>&1
        
        if [ $? -ne 0 ]; then
            log_message "Error: SynthSeg segmentation failed."
            exit 1
        fi
    fi
    
    # Convert segmentation to probability maps
    if [ ! -f "${OUTPUT_DIR}/registration/t1_brain_segmentation.nii.gz" ]; then
        log_message "Converting SynthSeg segmentation to probability maps..."
        python3 <<EOF
import nibabel as nib
import numpy as np

# Load SynthSeg segmentation
seg_img = nib.load('${SYNTHSEG_SEG}')
seg_data = seg_img.get_fdata()

# Create probability maps
# CSF labels (ventricles, CSF spaces)
csf_mask = np.zeros_like(seg_data)
for label in [24, 14, 15, 4, 5, 43, 44]:
    csf_mask += (seg_data == label)
csf_mask = (csf_mask > 0).astype(float)

# White matter labels
wm_mask = np.zeros_like(seg_data)
for label in [2, 7, 41, 46]:
    wm_mask += (seg_data == label)
wm_mask = (wm_mask > 0).astype(float)

# Gray matter labels (includes cortical and subcortical)
gm_mask = np.zeros_like(seg_data)
for label in [3, 8, 42, 47]:  # Cerebral/Cerebellar cortex
    gm_mask += (seg_data == label)
gm_mask = (gm_mask > 0).astype(float)

# Subcortical gray matter
scgm_mask = np.zeros_like(seg_data)
for label in [9, 10, 11, 12, 13, 17, 18, 48, 49, 50, 51, 52, 53, 54]:  # Subcortical structures
    scgm_mask += (seg_data == label)
scgm_mask = (scgm_mask > 0).astype(float)

# Save all probability maps using original affine
nib.save(nib.Nifti1Image(seg_data, seg_img.affine), '${OUTPUT_DIR}/registration/t1_brain_segmentation.nii.gz')
nib.save(nib.Nifti1Image(csf_mask, seg_img.affine), '${OUTPUT_DIR}/registration/csf_prob.nii.gz')
nib.save(nib.Nifti1Image(gm_mask + scgm_mask, seg_img.affine), '${OUTPUT_DIR}/registration/gm_prob.nii.gz')
nib.save(nib.Nifti1Image(wm_mask, seg_img.affine), '${OUTPUT_DIR}/registration/wm_prob.nii.gz')
nib.save(nib.Nifti1Image(scgm_mask, seg_img.affine), '${OUTPUT_DIR}/registration/scgm_prob.nii.gz')

print("SynthSeg probability maps generated successfully.")
EOF
        log_message "SynthSeg segmentation converted to probability maps"
    else
        log_message "Skipping segmentation conversion, files already exist"
    fi
    
    # Transform segmentation results to fMRI space
    log_message "Transforming SynthSeg segmentation results to fMRI space..."
    for image in t1_brain_segmentation csf_prob gm_prob wm_prob scgm_prob; do
        antsApplyTransforms -d 3 \
            -i ${OUTPUT_DIR}/registration/${image}.nii.gz \
            -r ${OUTPUT_DIR}/registration/fmri_3d.nii.gz \
            -o ${OUTPUT_DIR}/registration/${image}_in_fmri_space.nii.gz \
            -t ${OUTPUT_DIR}/registration/registered_0GenericAffine.mat \
            -n Linear >> "${LOG_FILE}" 2>&1
    done
    log_message "SynthSeg segmentation results transformed to fMRI space"
fi

update_progress 11 "Tissue Segmentation"

print_section "Atlas Registration"
# Register Atlas to fMRI data
log_message "Registering atlas to fMRI data..."

if [ "$ATLAS_TYPE" == "aal" ]; then
    ATLAS_FILE="${ATLAS_FILE}"
    ATLAS_OUTPUT="${OUTPUT_DIR}/registration/aal_registered_to_fmri.nii.gz"
    ATLAS_PREFIX="${OUTPUT_DIR}/registration/aal_to_fmri_"
    ATLAS_LABELS="${SUBJECT_DIR}/AAL_labels.txt"  # Adjust path to your AAL labels
    ATLAS_NAME="AAL"
elif [ "$ATLAS_TYPE" == "schaefer" ]; then
    ATLAS_FILE="${ATLAS_FILE}"
    ATLAS_OUTPUT="${OUTPUT_DIR}/registration/schaefer_registered_to_fmri.nii.gz"
    ATLAS_PREFIX="${OUTPUT_DIR}/registration/schaefer_to_fmri_"
    ATLAS_LABELS="${SUBJECT_DIR}/Schaefer2018_200Parcels_7Networks_order.txt"  # Adjust path to your Schaefer labels
    ATLAS_NAME="Schaefer"
elif [ "$ATLAS_TYPE" == "freesurfer" ]; then
    # Convert FreeSurfer parcellation to NIfTI
    if [ ! -f "${OUTPUT_DIR}/registration/aparc+aseg.nii.gz" ]; then
        log_message "Converting FreeSurfer parcellation to NIfTI..."
        mri_label2vol --seg ${FREESURFER_SUBJECT_DIR}/mri/aparc+aseg.mgz \
                      --temp ${FREESURFER_SUBJECT_DIR}/mri/rawavg.mgz \
                      --o ${OUTPUT_DIR}/registration/aparc+aseg.nii.gz \
                      --regheader ${FREESURFER_SUBJECT_DIR}/mri/aparc+aseg.mgz >> "${LOG_FILE}" 2>&1
        log_message "FreeSurfer parcellation converted to NIfTI."
    else
        log_message "FreeSurfer parcellation NIfTI file already exists."
    fi
    ATLAS_FILE="${OUTPUT_DIR}/registration/aparc+aseg.nii.gz"
    ATLAS_OUTPUT="${OUTPUT_DIR}/registration/freesurfer_parcellation_registered_to_fmri.nii.gz"
    ATLAS_PREFIX="${OUTPUT_DIR}/registration/freesurfer_to_fmri_"
    ATLAS_LABELS="${FREESURFER_HOME}/FreeSurferColorLUT.txt"  # FreeSurfer LUT
    ATLAS_NAME="FreeSurfer"
else
    print_error "Unknown atlas type '$ATLAS_TYPE'. Please choose from 'aal', 'schaefer', or 'freesurfer'."
    exit 1
fi

if [ ! -f "${ATLAS_OUTPUT}" ]; then
    antsRegistrationSyNQuick.sh -d 3 \
        -f ${OUTPUT_DIR}/registration/fmri_brain.nii.gz \
        -m ${ATLAS_FILE} \
        -o ${ATLAS_PREFIX} \
        -t s \
        -n 4 >> "${LOG_FILE}" 2>&1

    antsApplyTransforms -d 3 \
        -i ${ATLAS_FILE} \
        -r ${OUTPUT_DIR}/registration/fmri_brain.nii.gz \
        -o ${ATLAS_OUTPUT} \
        -t ${ATLAS_PREFIX}1Warp.nii.gz \
        -t ${ATLAS_PREFIX}0GenericAffine.mat \
        -n NearestNeighbor >> "${LOG_FILE}" 2>&1

    if [ $? -ne 0 ]; then
        log_message "Error: Atlas to fMRI registration failed."
        exit 1
    fi
    log_message "Atlas registered to fMRI space."
else
    log_message "Skipping atlas registration, file already exists."
fi
update_progress 12 "Atlas Registration"

print_section "ICA Analysis and Denoising"
# Run MELODIC

# Extract TR from JSON file
TR=$(jq -r '.RepetitionTime' ${OUTPUT_DIR}/nifti/013_rfMRI_REST_AP.json)
log_message "Extracted TR: ${TR} seconds"

if [ ! -d "${OUTPUT_DIR}/melodic.ica" ]; then
    log_message "Running MELODIC ICA..."
    melodic -i ${OUTPUT_DIR}/registration/mcflirt_corrected_fmri_brain.nii.gz \
            -o ${OUTPUT_DIR}/melodic.ica \
            --nobet \
            --mask=${OUTPUT_DIR}/registration/fmri_brain_mask.nii.gz \
            --tr=${TR} \
            --bgthreshold=3 \
            --dim=30 \
            --report \
            --Oall >> "${LOG_FILE}" 2>&1
    if [ $? -ne 0 ]; then
        log_message "Error: MELODIC ICA failed."
        exit 1
    fi
    log_message "MELODIC ICA completed"
else
    log_message "Skipping MELODIC ICA, output directory already exists"
fi
update_progress 13 "MELODIC ICA"

# Run ICA-AROMA
log_message "Running ICA-AROMA..."

# Set needed variables and convert to absolute paths
PREPROCESSED_DIR=$(realpath "${OUTPUT_DIR}")
REGISTRATION_DIR="${PREPROCESSED_DIR}/registration"
MOTION_CORRECTION_DIR="${PREPROCESSED_DIR}/motion_correction"
AROMA_OUT_DIR="${PREPROCESSED_DIR}/ICA_AROMA"
MELODIC_DIR="${PREPROCESSED_DIR}/melodic.ica"

# Input files
INPUT_FILE="${REGISTRATION_DIR}/mcflirt_corrected_fmri_brain.nii.gz"
MC_FILE="${MOTION_CORRECTION_DIR}/mcflirt_corrected_fmri.par"
MASK_FILE="${REGISTRATION_DIR}/fmri_brain_mask.nii.gz"
MNI_TEMPLATE="${FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz"

# Debug: Print all paths and check files
log_message "Checking paths:"
echo "INPUT_FILE: ${INPUT_FILE}"
echo "MC_FILE: ${MC_FILE}"
echo "MASK_FILE: ${MASK_FILE}"
echo "AROMA_OUT_DIR: ${AROMA_OUT_DIR}"

# Check if files exist
for file in "${INPUT_FILE}" "${MC_FILE}" "${MASK_FILE}" "${MNI_TEMPLATE}"; do
    if [ ! -f "${file}" ]; then
        log_message "Error: File not found: ${file}"
        exit 1
    else
        log_message "Found: ${file}"
    fi
done

if [ ! -f "${AROMA_OUT_DIR}/denoised_func_data_nonaggr.nii.gz" ]; then
    mkdir -p "${AROMA_OUT_DIR}"
    
    log_message "Creating registration to MNI space..."
    
    # Generate mean functional if it doesn't exist
    MEAN_FUNC="${REGISTRATION_DIR}/mean_func.nii.gz"
    if [ ! -f "${MEAN_FUNC}" ]; then
        fslmaths "${INPUT_FILE}" -Tmean "${MEAN_FUNC}"
    fi
    
    # Register to MNI
    FUNC_TO_MNI="${REGISTRATION_DIR}/func2mni.mat"
    flirt -in "${MEAN_FUNC}" \
          -ref "${MNI_TEMPLATE}" \
          -out "${REGISTRATION_DIR}/func2mni" \
          -omat "${FUNC_TO_MNI}" \
          -dof 12
    
    # Check if registration was successful
    if [ ! -f "${FUNC_TO_MNI}" ]; then
        log_message "Error: Registration failed, matrix file not created"
        exit 1
    fi
    
    log_message "Running ICA-AROMA with following inputs:"
    log_message "Input: ${INPUT_FILE}"
    log_message "Output: ${AROMA_OUT_DIR}"
    log_message "Motion params: ${MC_FILE}"
    log_message "Mask: ${MASK_FILE}"
    log_message "Registration matrix: ${FUNC_TO_MNI}"
    
    # Run ICA-AROMA
    python3 "${ICA_AROMA_DIR}" \
        -in "${INPUT_FILE}" \
        -out "${AROMA_OUT_DIR}" \
        -mc "${MC_FILE}" \
        -m "${MASK_FILE}" \
        -tr "${TR}" \
        -md "${MELODIC_DIR}" \
        -den both \
        -affmat "${FUNC_TO_MNI}" \
        -overwrite

    if [ $? -ne 0 ]; then
        log_message "Error: ICA-AROMA failed."
        exit 1
    fi
    log_message "ICA-AROMA completed successfully"
else
    log_message "Skipping ICA-AROMA, output already exists"
fi
update_progress 14 "ICA-AROMA"

# This is the updated CompCor implementation section to replace in the script

# Run CompCor denoising if enabled
if [ "$ENABLE_COMPCOR" -eq 1 ]; then
    print_section "CompCor Denoising"
    log_message "Running CompCor denoising..."
    
    # Create directories for CompCor output
    mkdir -p "${OUTPUT_DIR}/compcor/QC"
    
    # Run anatomical CompCor denoising
    if [ ! -f "${OUTPUT_DIR}/compcor/compcor_cleaned.nii.gz" ]; then
        log_message "Running anatomical CompCor with ${COMPCOR_COMPONENTS} components..."
        
        # Python script for CompCor implementation
        python3 - <<EOF
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from scipy import signal, stats
from nilearn.connectome import ConnectivityMeasure
from nilearn.plotting import plot_carpet

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Create more informative logging
def log_step(message):
    print(f"\n==== {message} ====")

# Set seaborn style
sns.set(style="whitegrid", context="talk", font_scale=0.8)
plt.rcParams["figure.dpi"] = 150

# Function to extract noise components using anatomical CompCor
def anatomical_compcor(data, wm_mask, csf_mask, n_components=5):
    """Extract noise components using anatomical CompCor"""
    log_step(f"Extracting aCompCor components (n={n_components})")
    
    # Ensure masks match data dimensions
    if wm_mask.shape != data.shape[:-1] or csf_mask.shape != data.shape[:-1]:
        raise ValueError(f"Mask dimensions {wm_mask.shape} do not match data dimensions {data.shape[:-1]}")
    
    # Get voxels from white matter and CSF
    data_2d = data.reshape(-1, data.shape[-1])
    
    # Extract time series from masks, handling empty masks
    wm_voxels = wm_mask.ravel()
    csf_voxels = csf_mask.ravel()
    
    # Check if masks contain enough voxels
    wm_count = np.sum(wm_voxels)
    csf_count = np.sum(csf_voxels)
    
    print(f"White matter mask contains {wm_count} voxels")
    print(f"CSF mask contains {csf_count} voxels")
    
    if wm_count < 10:
        print("WARNING: White matter mask contains very few voxels")
    if csf_count < 10:
        print("WARNING: CSF mask contains very few voxels")
    
    # Get time series
    wm_ts = data_2d[wm_voxels]
    csf_ts = data_2d[csf_voxels]
    
    # Adjust number of components if masks have too few voxels
    n_components_wm = min(n_components, wm_ts.shape[0] // 2)
    n_components_csf = min(n_components, csf_ts.shape[0] // 2)
    
    print(f"Extracting {n_components_wm} WM components and {n_components_csf} CSF components")
    
    # Handle case with too few voxels
    if n_components_wm == 0 and n_components_csf == 0:
        print("WARNING: Not enough voxels in WM and CSF masks to extract components")
        print("Returning empty components array")
        return np.zeros((data.shape[-1], 0))
    
    # Extract components
    components = []
    
    if n_components_wm > 0:
        try:
            # Transpose to get voxels x timepoints
            wm_ts_t = wm_ts.T
            
            # Remove NaN or Inf values
            wm_ts_t = np.nan_to_num(wm_ts_t, nan=0, posinf=0, neginf=0)
            
            # Run PCA
            wm_pca = PCA(n_components=n_components_wm)
            wm_components = wm_pca.fit_transform(wm_ts_t)
            
            components.append(wm_components)
            print(f"Successfully extracted {n_components_wm} WM components")
            print(f"WM variance explained: {np.sum(wm_pca.explained_variance_ratio_)*100:.2f}%")
        except Exception as e:
            print(f"Error extracting WM components: {e}")
    
    if n_components_csf > 0:
        try:
            # Transpose to get voxels x timepoints
            csf_ts_t = csf_ts.T
            
            # Remove NaN or Inf values
            csf_ts_t = np.nan_to_num(csf_ts_t, nan=0, posinf=0, neginf=0)
            
            # Run PCA
            csf_pca = PCA(n_components=n_components_csf)
            csf_components = csf_pca.fit_transform(csf_ts_t)
            
            components.append(csf_components)
            print(f"Successfully extracted {n_components_csf} CSF components")
            print(f"CSF variance explained: {np.sum(csf_pca.explained_variance_ratio_)*100:.2f}%")
        except Exception as e:
            print(f"Error extracting CSF components: {e}")
    
    if len(components) == 0:
        print("WARNING: Failed to extract any components")
        return np.zeros((data.shape[-1], 0))
    
    # Combine components
    all_components = np.hstack(components)
    print(f"Total components extracted: {all_components.shape[1]}")
    
    return all_components

# Function to apply bandpass filter
def bandpass_filter(data, tr, lowpass, highpass):
    """Apply a bandpass filter to fMRI data."""
    log_step(f"Applying bandpass filter ({highpass}-{lowpass} Hz)")
    
    fs = 1 / tr
    nyquist = fs / 2
    b, a = signal.butter(3, [highpass / nyquist, lowpass / nyquist], btype='band')

    orig_shape = data.shape
    data_2d = data.reshape(-1, orig_shape[-1])
    
    # Initialize filtered data array
    filtered_data = np.zeros_like(data_2d)
    
    # Apply filter to each voxel time series
    for i in range(data_2d.shape[0]):
        # Get time series
        ts = data_2d[i]
        
        # Skip if all zeros or contains NaN
        if np.all(ts == 0) or np.any(np.isnan(ts)):
            filtered_data[i] = ts
            continue
        
        # Apply filter
        try:
            filtered_data[i] = signal.filtfilt(b, a, ts)
        except Exception as e:
            print(f"Error filtering voxel {i}: {e}")
            filtered_data[i] = ts

    return filtered_data.reshape(orig_shape)

# Function to regress out confounds
def regress_confounds(fmri_data, confounds):
    """Regress out confounds from fMRI data."""
    log_step("Regressing out confounds")
    
    # Get dimensions
    orig_shape = fmri_data.shape
    n_voxels = np.prod(orig_shape[:-1])
    n_timepoints = orig_shape[-1]

    # Ensure confounds match timepoints
    if confounds.shape[0] != n_timepoints:
        raise ValueError(f"Confounds (T={confounds.shape[0]}) do not match fMRI data timepoints (T={n_timepoints})")

    # Reshape to 2D for regression
    data_2d = fmri_data.reshape(-1, n_timepoints).T  # Shape: (T, voxels)
    
    # Handle NaN values
    data_2d = np.nan_to_num(data_2d, nan=0, posinf=0, neginf=0)
    
    # Add intercept to confounds
    X = np.column_stack((np.ones(confounds.shape[0]), confounds))
    
    # Check for NaN in design matrix
    if np.any(np.isnan(X)):
        print("WARNING: NaN values in design matrix. Replacing with zeros.")
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Regress out confounds voxel-wise
    print(f"Performing regression on {n_voxels} voxels with {X.shape[1]} regressors")
    
    # Initialize residuals array
    residuals = np.zeros_like(data_2d)
    
    # Loop through chunks of voxels to avoid memory issues
    chunk_size = 10000
    n_chunks = int(np.ceil(data_2d.shape[1] / chunk_size))
    
    for chunk in range(n_chunks):
        start_idx = chunk * chunk_size
        end_idx = min((chunk + 1) * chunk_size, data_2d.shape[1])
        
        # Extract chunk
        chunk_data = data_2d[:, start_idx:end_idx]
        
        try:
            # Compute beta weights
            beta = np.linalg.lstsq(X, chunk_data, rcond=None)[0]
            
            # Compute residuals
            chunk_residuals = chunk_data - X @ beta
            
            # Store residuals
            residuals[:, start_idx:end_idx] = chunk_residuals
        except Exception as e:
            print(f"Error in regression for chunk {chunk}: {e}")
            # Keep original data if regression fails
            residuals[:, start_idx:end_idx] = chunk_data
    
    # Handle any remaining NaN values
    residuals = np.nan_to_num(residuals, nan=0, posinf=0, neginf=0)
    
    # Reshape back to original shape
    return residuals.T.reshape(orig_shape)

# Function to combine confounds
def combine_confounds(acompcor, motion, fd, include_fd=True):
    """Combine confounds into a single matrix with standardization."""
    log_step("Combining confounds")
    
    # Check if any components were extracted
    if acompcor.shape[1] == 0:
        print("WARNING: No aCompCor components available")
        if include_fd:
            print("Using motion parameters and FD only")
            combined = np.hstack([motion, fd.reshape(-1, 1)])
        else:
            print("Using motion parameters only")
            combined = motion
    else:
        # Combine confounds
        confounds = [acompcor]
        confounds.append(motion)
        
        if include_fd:
            confounds.append(fd.reshape(-1, 1))
        
        combined = np.hstack(confounds)
    
    print(f"Combined confounds shape: {combined.shape}")
    
    # Standardize confounds (zero mean, unit variance)
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0)
    std[std < 1e-6] = 1.0  # Avoid division by very small numbers
    
    standardized = (combined - mean) / std
    
    # Final safety check for NaN/Inf values
    standardized = np.nan_to_num(standardized, nan=0, posinf=0, neginf=0)
    
    return standardized

# Function to create tissue masks from segmentation
def create_tissue_masks(segmentation_path, fmri_data_shape, lesion_path=None):
    """Create WM and CSF masks from segmentation, ensuring proper size and values."""
    log_step("Creating tissue masks")
    
    # Load segmentation
    seg_img = nib.load(segmentation_path)
    seg_data = seg_img.get_fdata()
    
    print(f"Segmentation shape: {seg_data.shape}")
    print(f"fMRI data shape: {fmri_data_shape[:-1]}")
    
    # Check if shapes match
    if seg_data.shape != fmri_data_shape[:-1]:
        print(f"WARNING: Segmentation shape {seg_data.shape} doesn't match fMRI shape {fmri_data_shape[:-1]}")
        print("Attempting to proceed anyway...")
    
    # Get unique values in segmentation
    unique_vals = np.unique(seg_data)
    print(f"Unique segmentation values: {unique_vals}")
    
    # Define tissue labels (CSF=1, GM=2, WM=3, background=0)
    # Create conservative masks (dilated inward from boundaries)
    
    # First create basic masks
    csf_mask = (seg_data == 1)
    wm_mask = (seg_data == 3)
    
    # Get counts before erosion
    csf_count_before = np.sum(csf_mask)
    wm_count_before = np.sum(wm_mask)
    
    print(f"Initial CSF mask: {csf_count_before} voxels")
    print(f"Initial WM mask: {wm_count_before} voxels")
    
    # Erode masks to be more conservative (avoid partial volume effects)
    from scipy import ndimage
    
    # Only erode if we have enough voxels
    if wm_count_before > 100:
        wm_mask = ndimage.binary_erosion(wm_mask, iterations=1)
    
    if csf_count_before > 100:
        csf_mask = ndimage.binary_erosion(csf_mask, iterations=1)
    
    # Get counts after erosion
    csf_count_after = np.sum(csf_mask)
    wm_count_after = np.sum(wm_mask)
    
    print(f"After erosion - CSF mask: {csf_count_after} voxels")
    print(f"After erosion - WM mask: {wm_count_after} voxels")
    
    # If masks are too small after erosion, revert to original
    if csf_count_after < 10 and csf_count_before >= 10:
        print("WARNING: CSF mask too small after erosion, reverting to original")
        csf_mask = (seg_data == 1)
    
    if wm_count_after < 10 and wm_count_before >= 10:
        print("WARNING: WM mask too small after erosion, reverting to original")
        wm_mask = (seg_data == 3)
    
    # If we still don't have enough voxels, try alternate labels
    if np.sum(csf_mask) < 10:
        print("WARNING: Very few CSF voxels found. Trying ventricle labels (14, 15, 4, 43)")
        ventricle_labels = [14, 15, 4, 5, 43, 44]  # Ventricle labels in FreeSurfer segmentation
        csf_mask = np.isin(seg_data, ventricle_labels)
        print(f"Ventricle-based CSF mask: {np.sum(csf_mask)} voxels")
    
    if np.sum(wm_mask) < 10:
        print("WARNING: Very few WM voxels found. Trying alternate WM labels (2, 41)")
        wm_labels = [2, 41]  # WM labels in FreeSurfer segmentation
        wm_mask = np.isin(seg_data, wm_labels)
        print(f"Alternate WM mask: {np.sum(wm_mask)} voxels")
    
    # Exclude lesions from masks, if lesion file exists and has voxels
    if lesion_path and os.path.exists(lesion_path):
        lesion_img = nib.load(lesion_path)
        lesion_data = lesion_img.get_fdata() > 0.5
        lesion_count = np.sum(lesion_data)
        
        if lesion_count > 0:
            print(f"Excluding {lesion_count} lesion voxels from masks")
            csf_mask = csf_mask & ~lesion_data
            wm_mask = wm_mask & ~lesion_data
            
            print(f"After lesion exclusion - CSF mask: {np.sum(csf_mask)} voxels")
            print(f"After lesion exclusion - WM mask: {np.sum(wm_mask)} voxels")
    
    return wm_mask, csf_mask

# Function to generate QC plots for denoising
def plot_denoising_preview(fmri_data, cleaned_data, atlas_data, output_dir, affine, mask_path, n_sample_voxels=500):
    """Generate QC report with connectivity distributions, variance explained, and carpet plots."""
    log_step("Generating QC visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle NaN values
    fmri_data = np.nan_to_num(fmri_data, nan=0, posinf=0, neginf=0)
    cleaned_data = np.nan_to_num(cleaned_data, nan=0, posinf=0, neginf=0)
    
    # Verification of input data
    print(f"fMRI data shape: {fmri_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Atlas data shape: {atlas_data.shape}")
    
    # Check data ranges to ensure sensible values
    print(f"fMRI data range: [{np.min(fmri_data)}, {np.max(fmri_data)}]")
    print(f"Cleaned data range: [{np.min(cleaned_data)}, {np.max(cleaned_data)}]")
    
    # Load the brain mask for carpet plots
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Could not find brain mask at: {mask_path}")
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata() > 0.5
    
    # Verify mask covers brain
    print(f"Mask covers {np.sum(mask_data)} voxels")
    
    # Convert to Nifti images for nilearn
    fmri_img_nifti = nib.Nifti1Image(fmri_data, affine=affine)
    cleaned_img_nifti = nib.Nifti1Image(cleaned_data, affine=affine)
    mask_img_nifti = nib.Nifti1Image(mask_data.astype(np.int16), affine=affine)

    # Connectivity Distribution
    def sample_correlations(data, n_samples):
        # Reshape to voxels × time
        voxels = data.reshape(-1, data.shape[-1])
        
        # Find voxels with variance > 0
        var = np.var(voxels, axis=1)
        valid_voxels = np.where(var > 0)[0]
        
        if len(valid_voxels) < n_samples:
            print(f"WARNING: Only {len(valid_voxels)} valid voxels available for correlation sampling")
            n_samples = len(valid_voxels)
        
        if n_samples == 0:
            print("ERROR: No valid voxels for correlation sampling")
            return np.array([[1]])
        
        # Sample from valid voxels
        sample_idx = np.random.choice(valid_voxels, n_samples, replace=False)
        
        # Calculate correlations
        try:
            corr = np.corrcoef(voxels[sample_idx])
            # Handle NaN values
            corr = np.nan_to_num(corr, nan=0, posinf=0, neginf=0)
            return corr
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            return np.ones((n_samples, n_samples))
    
    try:
        # Sample correlations
        orig_corr = sample_correlations(fmri_data, n_sample_voxels)
        clean_corr = sample_correlations(cleaned_data, n_sample_voxels)
        
        plt.figure(figsize=(10, 5))
        
        # Extract upper triangle values (excluding diagonal)
        orig_triu = orig_corr[np.triu_indices_from(orig_corr, k=1)]
        clean_triu = clean_corr[np.triu_indices_from(clean_corr, k=1)]
        
        # Plot histograms
        sns.histplot(orig_triu, color='dodgerblue', label='Original', kde=True, bins=50, stat='density')
        sns.histplot(clean_triu, color='goldenrod', label='Denoised', kde=True, bins=50, stat='density')
        
        plt.title("Voxel Connectivity Distributions")
        plt.xlabel("Pearson's r")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(f"{output_dir}/connectivity_distributions.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error generating connectivity distributions plot: {e}")
    
    # Variance Explained Map
    try:
        total_var = np.var(fmri_data, axis=-1)
        residual_var = np.var(cleaned_data, axis=-1)
        
        # Avoid division by zero
        denom = np.where(total_var > 1e-10, total_var, 1.0)
        var_explained = (total_var - residual_var) / denom * 100
        
        # Handle edge cases
        var_explained = np.clip(var_explained, 0, 100)
        var_explained[~mask_data] = 0  # Set outside brain to zero
        
        # Select a central slice
        slice_idx = fmri_data.shape[2] // 2
        
        plt.figure(figsize=(8, 6))
        plt.imshow(var_explained[:, :, slice_idx], cmap="hot", vmax=50)
        plt.colorbar(label="% Variance Explained")
        plt.title("CompCor Regressed Variance")
        plt.savefig(f"{output_dir}/variance_explained.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error generating variance explained map: {e}")
    
    # Carpet Plot - we'll do this only once for the CompCor output
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_carpet(cleaned_img_nifti, figure=fig, axes=ax, mask_img=mask_img_nifti)
        plt.title("Carpet Plot - After CompCor Denoising")
        plt.savefig(f"{output_dir}/carpet_denoised.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error generating carpet plot: {e}")
    
    print(f"QC report saved to: {output_dir}")

# Main CompCor execution
log_step("Starting CompCor denoising")

# Load data
fmri_path = '${INPUT_FILE}'
segmentation_path = '${OUTPUT_DIR}/registration/t1_brain_segmentation_in_fmri_space.nii.gz'
lesion_path = '${OUTPUT_DIR}/registration/lesion_mask_in_fmri_space.nii.gz' if '${LESION_MASK}' else None
atlas_path = '${ATLAS_OUTPUT}'
motion_params_path = '${MC_FILE}'
fd_metrics_path = '${OUTPUT_DIR}/motion_correction/fd_metrics.txt'
mask_path = '${MASK_FILE}'
tr = ${TR}
lowpass = ${LOWPASS_HZ}
highpass = ${HIGHPASS_HZ}
n_components = ${COMPCOR_COMPONENTS}
apply_bandpass = ${BANDPASS_FILTER}

# Log parameters
print(f"CompCor Parameters:")
print(f"  TR: {tr} seconds")
print(f"  Lowpass: {lowpass} Hz")
print(f"  Highpass: {highpass} Hz")
print(f"  Components: {n_components}")
print(f"  Bandpass Filter: {'Enabled' if apply_bandpass else 'Disabled'}")

try:
    # Load fMRI data
    fmri_img = nib.load(fmri_path)
    fmri_data = fmri_img.get_fdata()
    print(f"fMRI data shape: {fmri_data.shape}")
    
    # Load motion parameters and FD metrics
    motion_params = np.loadtxt(motion_params_path)
    fd_metrics = np.loadtxt(fd_metrics_path)
    print(f"Motion parameters shape: {motion_params.shape}")
    print(f"FD metrics shape: {fd_metrics.shape}")
    
    # Load atlas data
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()
    print(f"Atlas data shape: {atlas_data.shape}")
    
    # Create tissue masks
    wm_mask, csf_mask = create_tissue_masks(segmentation_path, fmri_data.shape, lesion_path)
    print(f"WM mask: {np.sum(wm_mask)} voxels")
    print(f"CSF mask: {np.sum(csf_mask)} voxels")
    
    # Extract CompCor components
    acompcor_components = anatomical_compcor(fmri_data, wm_mask, csf_mask, n_components=n_components)
    print(f"aCompCor components shape: {acompcor_components.shape}")
    
    # Save CompCor components for reference
    np.savetxt('${OUTPUT_DIR}/compcor/acompcor_components.txt', acompcor_components)
    
    # Combine confounds
    all_confounds = combine_confounds(acompcor_components, motion_params, fd_metrics, include_fd=True)
    print(f"Combined confounds shape: {all_confounds.shape}")
    
    # Regress out confounds
    print("Regressing out confounds...")
    cleaned_data = regress_confounds(fmri_data, all_confounds)
    
    # Apply bandpass filter if requested
    if apply_bandpass:
        print(f"Applying bandpass filter: {highpass}-{lowpass} Hz")
        cleaned_data = bandpass_filter(cleaned_data, tr, lowpass, highpass)
    else:
        print("Skipping bandpass filtering")
    
    # Save cleaned data
    print("Saving CompCor cleaned data...")
    nifti_img = nib.Nifti1Image(cleaned_data, fmri_img.affine)
    output_path = '${OUTPUT_DIR}/compcor/compcor_cleaned.nii.gz'
    nib.save(nifti_img, output_path)
    print(f"Saved cleaned data to: {output_path}")
    
    # Generate QC plots
    print("Generating QC plots...")
    plot_denoising_preview(
        fmri_data, 
        cleaned_data, 
        atlas_data, 
        '${OUTPUT_DIR}/compcor/QC', 
        fmri_img.affine,
        mask_path,
        n_sample_voxels=1000
    )
    
    # Generate connectivity matrices using CompCor-cleaned data
    print("Computing connectivity matrices from CompCor-cleaned data...")
    
    # Extract time series from atlas regions
    time_series_before = []
    time_series_after = []
    valid_regions = []
    
    unique_regions = np.unique(atlas_data)
    unique_regions = unique_regions[unique_regions > 0]
    
    for region in unique_regions:
        region_mask = (atlas_data == region)
        n_voxels = np.sum(region_mask)
        
        if n_voxels > 0:
            try:
                ts_before = np.mean(fmri_data[region_mask], axis=0)
                ts_after = np.mean(cleaned_data[region_mask], axis=0)
                
                # Check for valid time series
                if not np.any(np.isnan(ts_before)) and not np.any(np.isnan(ts_after)):
                    time_series_before.append(ts_before)
                    time_series_after.append(ts_after)
                    valid_regions.append(region)
                    print(f"Region {region}: {n_voxels} voxels, valid time series extracted")
            except Exception as e:
                print(f"Error processing region {region}: {e}")
    
    time_series_before = np.array(time_series_before)
    time_series_after = np.array(time_series_after)
    
    print(f"Extracted time series from {len(valid_regions)} valid regions")
    
    if len(valid_regions) > 1:
        # Calculate connectivity measures
        try:
            pearson_before = np.corrcoef(time_series_before)
            print("Calculated Pearson correlation (before)")
        except Exception as e:
            print(f"Error calculating Pearson correlation (before): {e}")
            pearson_before = np.eye(len(valid_regions))
        
        try:
            pearson_after = np.corrcoef(time_series_after)
            print("Calculated Pearson correlation (after)")
        except Exception as e:
            print(f"Error calculating Pearson correlation (after): {e}")
            pearson_after = np.eye(len(valid_regions))
        
        try:
            spearman_before = stats.spearmanr(time_series_before.T, nan_policy='omit')[0]
            print("Calculated Spearman correlation (before)")
        except Exception as e:
            print(f"Error calculating Spearman correlation (before): {e}")
            spearman_before = pearson_before
        
        try:
            spearman_after = stats.spearmanr(time_series_after.T, nan_policy='omit')[0]
            print("Calculated Spearman correlation (after)")
        except Exception as e:
            print(f"Error calculating Spearman correlation (after): {e}")
            spearman_after = pearson_after
        
        try:
            # Handle NaN values
            time_series_before_nn = np.nan_to_num(time_series_before, nan=0, posinf=0, neginf=0)
            time_series_after_nn = np.nan_to_num(time_series_after, nan=0, posinf=0, neginf=0)
            
            partial_measure = ConnectivityMeasure(kind='partial correlation')
            partial_before = partial_measure.fit_transform([time_series_before_nn.T])[0]
            print("Calculated partial correlation (before)")
        except Exception as e:
            print(f"Error calculating partial correlation (before): {e}")
            partial_before = pearson_before
        
        try:
            partial_after = partial_measure.fit_transform([time_series_after_nn.T])[0]
            print("Calculated partial correlation (after)")
        except Exception as e:
            print(f"Error calculating partial correlation (after): {e}")
            partial_after = pearson_after
        
        # Handle NaN values in connectivity matrices
        pearson_before = np.nan_to_num(pearson_before, nan=0, posinf=0, neginf=0)
        pearson_after = np.nan_to_num(pearson_after, nan=0, posinf=0, neginf=0)
        spearman_before = np.nan_to_num(spearman_before, nan=0, posinf=0, neginf=0)
        spearman_after = np.nan_to_num(spearman_after, nan=0, posinf=0, neginf=0)
        partial_before = np.nan_to_num(partial_before, nan=0, posinf=0, neginf=0)
        partial_after = np.nan_to_num(partial_after, nan=0, posinf=0, neginf=0)
        
        # Plot and save connectivity matrices
        matrices_before = [pearson_before, spearman_before, partial_before]
        matrices_after = [pearson_after, spearman_after, partial_after]
        titles = ['Pearson', 'Spearman', 'Partial']
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Connectivity Matrix Comparison: Before and After CompCor', fontsize=16)
            
            # Create tick labels
            n_regions = len(valid_regions)
            step = max(1, n_regions // 10)  # Show at most 10 tick labels
            tick_positions = np.arange(0, n_regions, step)
            tick_labels = [str(int(valid_regions[i])) for i in tick_positions if i < len(valid_regions)]
            
            # Find global vmax for consistent colorbar
            all_corrs = np.concatenate([m.flatten() for m in matrices_before + matrices_after])
            vmax = np.percentile(np.abs(all_corrs), 99)  # Use 99th percentile to avoid outliers
            
            for idx, (before, after, title) in enumerate(zip(matrices_before, matrices_after, titles)):
                # Plot before matrix
                ax = axes[0, idx]
                im = sns.heatmap(before, cmap='RdBu_r', vmin=-vmax, vmax=vmax, center=0,
                               square=True, cbar_kws={"shrink": .8}, ax=ax)
                ax.set_title(f'Before CompCor ({title})', fontsize=12, pad=10)
                if len(tick_positions) > 0:
                    ax.set_xticks(tick_positions)
                    ax.set_yticks(tick_positions)
                    if len(tick_labels) > 0:
                        ax.set_xticklabels(tick_labels, rotation=90)
                        ax.set_yticklabels(tick_labels, rotation=0)
                
                # Plot after matrix
                ax = axes[1, idx]
                im = sns.heatmap(after, cmap='RdBu_r', vmin=-vmax, vmax=vmax, center=0,
                               square=True, cbar_kws={"shrink": .8}, ax=ax)
                ax.set_title(f'After CompCor ({title})', fontsize=12, pad=10)
                if len(tick_positions) > 0:
                    ax.set_xticks(tick_positions)
                    ax.set_yticks(tick_positions)
                    if len(tick_labels) > 0:
                        ax.set_xticklabels(tick_labels, rotation=90)
                        ax.set_yticklabels(tick_labels, rotation=0)
            
            plt.tight_layout()
            plt.savefig('${OUTPUT_DIR}/compcor/QC/connectivity_matrices_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Created connectivity matrices comparison plot")
        except Exception as e:
            print(f"Error creating connectivity matrices plot: {e}")
        
        # Save connectivity matrices
        try:
            for name, matrix in [
                ('compcor_pearson', pearson_after),
                ('compcor_spearman', spearman_after),
                ('compcor_partial', partial_after)
            ]:
                output_file = f'${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_{name}.txt'
                np.savetxt(output_file, matrix)
                print(f"Saved {name} matrix to {output_file}")
                
            # Save valid regions info
            valid_regions_df = pd.DataFrame({
                'Index': range(len(valid_regions)),
                'Region_Label': valid_regions
            })
            valid_regions_df.to_csv('${OUTPUT_DIR}/compcor/valid_regions.csv', index=False)
            print("Saved valid regions info")
        except Exception as e:
            print(f"Error saving connectivity matrices: {e}")
    else:
        print("WARNING: Not enough valid regions to calculate connectivity matrices")
    
    print("CompCor denoising and connectivity analysis completed successfully.")
except Exception as e:
    print(f"ERROR in CompCor processing: {e}")
    import traceback
    traceback.print_exc()
EOF
        
        log_message "CompCor denoising completed"
    else
        log_message "Skipping CompCor denoising, output already exists"
    fi
    
    # Update the progress counter
    update_progress 15 "CompCor"
fi

# Generate carpet plot if CompCor is not enabled
# Skip if CompCor is enabled - we already made one
if [ "$ENABLE_COMPCOR" -eq 0 ]; then
    print_section "Generate Carpet Plot"
    # Generate carpet plot for QC
    log_message "Generating carpet plot for QC..."
    python3 - <<EOF
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn style
sns.set(style="white", font_scale=1.2)

try:
    # Load the denoised data
    denoised_data_file = '${AROMA_OUT_DIR}/denoised_func_data_nonaggr.nii.gz'
    brain_mask_file = '${REGISTRATION_DIR}/fmri_brain_mask.nii.gz'
    
    # Load the data
    img = nib.load(denoised_data_file)
    data = img.get_fdata()
    mask = nib.load(brain_mask_file).get_fdata() > 0.5
    
    # Reshape data to voxels × time
    vox_data = data[mask].reshape(-1, data.shape[-1])
    
    # Calculate global signal 
    global_signal = np.mean(vox_data, axis=0)
    
    # Z-score each voxel's time series
    vox_data_zscore = (vox_data - np.mean(vox_data, axis=1, keepdims=True)) / (np.std(vox_data, axis=1, keepdims=True) + 1e-10)
    
    # Calculate correlation with global signal
    correlations = np.array([np.corrcoef(vox_data[i], global_signal)[0, 1] for i in range(vox_data.shape[0])])
    
    # Sort by correlation
    sorted_idx = np.argsort(correlations)
    vox_data_sorted = vox_data_zscore[sorted_idx]
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Plot the carpet
    plt.imshow(vox_data_sorted, aspect='auto', cmap='gray', vmin=-2, vmax=2, interpolation='none')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Z-score')
    
    # Set labels and title
    plt.xlabel('Time (TRs)')
    plt.ylabel('Voxels (sorted by correlation with global signal)')
    plt.title('Carpet Plot - ICA-AROMA Denoised fMRI Data')
    
    # Add a time scale at the bottom
    tr = ${TR}
    total_volumes = data.shape[-1]
    total_time = total_volumes * tr
    
    # Create tick positions and labels
    n_ticks = 6  # Number of time ticks
    tick_positions = np.linspace(0, total_volumes - 1, n_ticks)
    tick_labels = [f"{tick * tr:.2f}" for tick in tick_positions]
    
    plt.xticks(tick_positions, tick_labels)
    plt.xlabel('time (s)')
    
    # Save the plot with high resolution
    plt.tight_layout()
    plt.savefig('${OUTPUT_DIR}/QC/carpet_plot_simple.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Carpet plot created successfully.")
    
except Exception as e:
    print(f"Error generating carpet plot: {e}")
EOF
    update_progress 16 "Carpet Plot"
else
    # Skip the carpet plot step since it's already done in CompCor
    log_message "Skipping separate carpet plot, using the one from CompCor QC"
    update_progress 16 "Carpet Plot (skipped)"
fi

print_section "Connectivity Analysis"
# Perform connectivity analysis using ICA-AROMA denoised data
log_message "Performing connectivity analysis using ICA-AROMA denoised data..."

python3 <<EOF
import nibabel as nib
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def robust_standardize(X):
    """Apply robust standardization to time series"""
    from sklearn.covariance import MinCovDet
    try:
        # Check for NaN values
        if np.isnan(X).any():
            print("Warning: NaN values detected in time series. Using simple standardization...")
            # Get mean ignoring NaNs
            mean = np.nanmean(X, axis=1)
            # Get std ignoring NaNs
            std = np.nanstd(X, axis=1)
            # Replace zeros in std to avoid division by zero
            std[std == 0] = 1.0
            # Standardize
            X_std = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_std[i] = (X[i] - mean[i]) / std[i]
            return X_std
        else:
            # Use robust method if no NaNs
            robust_cov = MinCovDet(random_state=42).fit(X.T)
            robust_mean = robust_cov.location_
            robust_std = np.sqrt(np.diag(robust_cov.covariance_))
            # Replace zeros in std to avoid division by zero
            robust_std[robust_std < 1e-10] = 1.0
            return (X - robust_mean[:, np.newaxis]) / robust_std[:, np.newaxis]
    except Exception as e:
        print(f"Warning: Robust standardization failed: {e}")
        # Fallback to simple standardization
        try:
            mean = np.nanmean(X, axis=1)
            std = np.nanstd(X, axis=1)
            std[std < 1e-10] = 1.0
            X_std = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_std[i] = (X[i] - mean[i]) / std[i]
            return X_std
        except Exception as e2:
            print(f"Warning: Simple standardization failed: {e2}")
            # Return original data
            return X

# Load ICA-AROMA denoised data
fmri_img = nib.load('${AROMA_OUT_DIR}/denoised_func_data_nonaggr.nii.gz')
fmri_data = fmri_img.get_fdata()
print(f"fMRI data shape: {fmri_data.shape}")

# Load atlas and lesion mask
atlas_img = nib.load('${ATLAS_OUTPUT}')
atlas_data = atlas_img.get_fdata()
unique_regions = np.unique(atlas_data)[1:]  # Exclude zero
print(f"Found {len(unique_regions)} unique regions in atlas")

if os.path.exists('${OUTPUT_DIR}/registration/lesion_mask_in_fmri_space.nii.gz'):
    lesion_mask = nib.load('${OUTPUT_DIR}/registration/lesion_mask_in_fmri_space.nii.gz').get_fdata() > 0.5
    print("Using lesion mask")
else:
    lesion_mask = np.zeros(atlas_data.shape, dtype=bool)
    print("No lesion mask found")

# Extract time series from atlas regions
print("Extracting time series from atlas regions...")
time_series = []
valid_regions = []  # Keep track of regions with valid data
region_counts = []  # Keep track of voxel count per region

for i, region in enumerate(unique_regions):
    region_mask = (atlas_data == region)
    if lesion_mask is not None:
        region_mask = region_mask & (~lesion_mask)
    voxel_count = np.sum(region_mask)
    region_counts.append(voxel_count)
    
    if voxel_count > 0:
        # Extract time series for this region
        region_time_series = np.mean(fmri_data[region_mask], axis=0)
        if not np.all(np.isnan(region_time_series)):
            time_series.append(region_time_series)
            valid_regions.append(int(region))
        else:
            print(f"Warning: Region {region} has all NaN values. Replacing with zeros.")
            time_series.append(np.zeros(fmri_data.shape[-1]))
            valid_regions.append(int(region))
    else:
        print(f"Warning: Region {region} has no voxels after masking. Filling with zeros.")
        time_series.append(np.zeros(fmri_data.shape[-1]))
        valid_regions.append(int(region))

print(f"Extracted time series from {len(valid_regions)} regions")

# Convert to numpy array
time_series = np.array(time_series)
print(f"Time series shape: {time_series.shape}")

# Check for NaN values
nan_count = np.isnan(time_series).sum()
if nan_count > 0:
    print(f"Warning: Found {nan_count} NaN values in time series, out of {time_series.size} total values.")
    print("Handling NaN values...")
    
    # Handle NaNs in time series by interpolation
    for i in range(time_series.shape[0]):
        mask = np.isnan(time_series[i])
        if np.any(mask):
            indices = np.arange(len(time_series[i]))
            valid_indices = indices[~mask]
            if len(valid_indices) > 0:  # Only interpolate if we have some valid points
                valid_values = time_series[i, ~mask]
                # Linear interpolation
                time_series[i, mask] = np.interp(
                    indices[mask], valid_indices, valid_values,
                    left=np.nanmean(valid_values), right=np.nanmean(valid_values)
                )
            else:
                # If all values are NaN, replace with zeros
                print(f"Warning: Region {valid_regions[i]} has all NaN values. Replacing with zeros.")
                time_series[i] = np.zeros_like(time_series[i])

# Standardize time series
print("Standardizing time series...")
time_series_std = robust_standardize(time_series)

def fisher_z_transform(r):
    """Apply Fisher's Z-transform with proper clipping"""
    r_clipped = np.clip(r, -0.99999, 0.99999)
    return 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))

def inverse_fisher_z_transform(z):
    """Convert back from Fisher's Z-scores"""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def z_normalize_correlations(corr_matrix):
    """Z-normalize correlation matrix"""
    z_matrix = fisher_z_transform(corr_matrix)
    z_normalized = (z_matrix - np.nanmean(z_matrix)) / np.nanstd(z_matrix)
    return inverse_fisher_z_transform(z_normalized)

# Calculate connectivity matrices
print("Calculating connectivity matrices...")

# Pearson correlation (handles NaNs)
try:
    pearson_corr = np.corrcoef(time_series_std)
except Exception as e:
    print(f"Warning: Pearson correlation failed: {e}")
    pearson_corr = np.ones((time_series_std.shape[0], time_series_std.shape[0]))
    for i in range(time_series_std.shape[0]):
        for j in range(time_series_std.shape[0]):
            if i == j:
                pearson_corr[i, j] = 1.0
            else:
                # Calculate correlation manually, handling NaNs
                x = time_series_std[i]
                y = time_series_std[j]
                valid = ~np.isnan(x) & ~np.isnan(y)
                if np.sum(valid) > 1:
                    pearson_corr[i, j] = np.corrcoef(x[valid], y[valid])[0, 1]
                else:
                    pearson_corr[i, j] = 0.0

# Spearman correlation
try:
    spearman_corr = stats.spearmanr(time_series_std.T, nan_policy='omit')[0]
except Exception as e:
    print(f"Warning: Could not compute Spearman correlation: {e}")
    print("Using Pearson correlation as fallback")
    spearman_corr = pearson_corr

# Partial correlation with handling for NaNs
try:
    # First replace NaNs with zeros for partial correlation
    time_series_no_nan = np.nan_to_num(time_series_std, nan=0.0)
    partial_measure = ConnectivityMeasure(kind='partial correlation', standardize=False)
    partial_corr = partial_measure.fit_transform([time_series_no_nan.T])[0]
except Exception as e:
    print(f"Warning: Could not compute partial correlation: {e}")
    print("Using Pearson correlation as fallback for partial correlation")
    partial_corr = pearson_corr

# Calculate z-normalized versions (handle potential NaNs)
try:
    pearson_corr_normalized = z_normalize_correlations(pearson_corr)
    spearman_corr_normalized = z_normalize_correlations(spearman_corr)
    partial_corr_normalized = z_normalize_correlations(partial_corr)
except Exception as e:
    print(f"Warning: Could not compute z-normalized correlations: {e}")
    print("Using original correlations")
    pearson_corr_normalized = pearson_corr
    spearman_corr_normalized = spearman_corr
    partial_corr_normalized = partial_corr

# Save valid regions info
valid_regions_df = pd.DataFrame({
    'Index': range(len(valid_regions)),
    'Region_Label': valid_regions,
    'Voxel_Count': region_counts
})
valid_regions_df.to_csv('${OUTPUT_DIR}/connectivity/valid_regions.csv', index=False)

# Save all matrices
for name, matrix in [
    ('pearson', pearson_corr),
    ('spearman', spearman_corr),
    ('partial', partial_corr),
    ('pearson_znorm', pearson_corr_normalized),
    ('spearman_znorm', spearman_corr_normalized),
    ('partial_znorm', partial_corr_normalized)
]:
    np.savetxt(f'${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_{name}.txt', matrix)
    print(f"Saved {name} connectivity matrix")

def plot_correlation_matrices(pearson_corr, spearman_corr, partial_corr, 
                            pearson_corr_normalized, spearman_corr_normalized, partial_corr_normalized,
                            output_file):
    try:
        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Functional Connectivity Matrices - {os.path.basename("${ATLAS_NAME}")} Atlas', fontsize=16)

        # Create more readable tick labels with fewer ticks
        n_regions = len(pearson_corr)
        step = max(n_regions // 20, 1)  # Show ~20 ticks
        tick_positions = np.arange(0, n_regions, step)
        tick_labels = [str(valid_regions[i]) for i in tick_positions if i < len(valid_regions)]
        
        matrices = [
            (pearson_corr, 'Pearson Correlation'),
            (spearman_corr, 'Spearman Correlation'),
            (partial_corr, 'Partial Correlation'),
            (pearson_corr_normalized, 'Z-normalized Pearson Correlation'),
            (spearman_corr_normalized, 'Z-normalized Spearman Correlation'),
            (partial_corr_normalized, 'Z-normalized Partial Correlation')
        ]

        for idx, (matrix, title) in enumerate(matrices):
            row, col = divmod(idx, 3)
            ax = axes[row, col]
            
            # Plot heatmap with robust vmin/vmax
            vmin = np.nanpercentile(matrix, 1)
            vmax = np.nanpercentile(matrix, 99)
            
            mask = np.isnan(matrix)
            
            sns.heatmap(matrix, 
                    cmap='RdBu_r', 
                    vmin=max(-1, vmin), 
                    vmax=min(1, vmax), 
                    center=0,
                    square=True, 
                    cbar_kws={"shrink": .8}, 
                    ax=ax,
                    mask=mask)
            
            # Set specific tick locations and labels
            tick_pos = [p for p in tick_positions if p < n_regions]
            ax.set_xticks(tick_pos)
            ax.set_yticks(tick_pos)
            
            if len(tick_labels) > 0 and len(tick_pos) > 0:
                ax.set_xticklabels([tick_labels[i] for i in range(len(tick_pos)) if i < len(tick_labels)], rotation=90)
                ax.set_yticklabels([tick_labels[i] for i in range(len(tick_pos)) if i < len(tick_labels)], rotation=0)
            
            # Adjust label parameters
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel('ROI', labelpad=10)
            ax.set_ylabel('ROI', labelpad=10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved connectivity matrix visualization to {output_file}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot correlation matrices: {e}")

# Plot and save correlation matrices
try:
    plot_correlation_matrices(
        pearson_corr, spearman_corr, partial_corr,
        pearson_corr_normalized, spearman_corr_normalized, partial_corr_normalized,
        f'${OUTPUT_DIR}/QC/functional_connectivity_matrices.png'
    )
except Exception as e:
    print(f"Warning: Could not generate connectivity matrix plots: {e}")

# Generate additional QC visualizations
def plot_connectivity_distributions():
    try:
        plt.figure(figsize=(15, 5))
        
        # Get upper triangular indices
        triu_idx = np.triu_indices_from(pearson_corr, k=1)
        
        # Extract values
        pearson_triu = pearson_corr[triu_idx]
        pearson_triu = pearson_triu[~np.isnan(pearson_triu)]
        
        spearman_triu = spearman_corr[triu_idx]
        spearman_triu = spearman_triu[~np.isnan(spearman_triu)]
        
        partial_triu = partial_corr[triu_idx]
        partial_triu = partial_triu[~np.isnan(partial_triu)]
        
        # Plot distributions for original correlations
        plt.subplot(1, 2, 1)
        
        # Plot KDE
        try:
            sns.kdeplot(pearson_triu, label='Pearson', fill=True, alpha=0.3)
            sns.kdeplot(spearman_triu, label='Spearman', fill=True, alpha=0.3)
            sns.kdeplot(partial_triu, label='Partial', fill=True, alpha=0.3)
        except Exception as kde_err:
            print(f"KDE plot error: {kde_err}")
            # Fallback to histograms
            plt.hist(pearson_triu, alpha=0.3, label='Pearson', bins=30)
            plt.hist(spearman_triu, alpha=0.3, label='Spearman', bins=30)
            plt.hist(partial_triu, alpha=0.3, label='Partial', bins=30)
            
        plt.title('Original Correlation Distributions')
        plt.xlabel('Correlation Value')
        plt.ylabel('Density')
        plt.legend()
        
        # Extract values
        pearson_norm_triu = pearson_corr_normalized[triu_idx]
        pearson_norm_triu = pearson_norm_triu[~np.isnan(pearson_norm_triu)]
        
        spearman_norm_triu = spearman_corr_normalized[triu_idx]
        spearman_norm_triu = spearman_norm_triu[~np.isnan(spearman_norm_triu)]
        
        partial_norm_triu = partial_corr_normalized[triu_idx]
        partial_norm_triu = partial_norm_triu[~np.isnan(partial_norm_triu)]
        
        # Plot distributions for z-normalized correlations
        plt.subplot(1, 2, 2)
        
        # Plot KDE
        try:
            sns.kdeplot(pearson_norm_triu, label='Pearson', fill=True, alpha=0.3)
            sns.kdeplot(spearman_norm_triu, label='Spearman', fill=True, alpha=0.3)
            sns.kdeplot(partial_norm_triu, label='Partial', fill=True, alpha=0.3)
        except Exception as kde_err:
            print(f"KDE plot error: {kde_err}")
            # Fallback to histograms
            plt.hist(pearson_norm_triu, alpha=0.3, label='Pearson', bins=30)
            plt.hist(spearman_norm_triu, alpha=0.3, label='Spearman', bins=30)
            plt.hist(partial_norm_triu, alpha=0.3, label='Partial', bins=30)
            
        plt.title('Z-normalized Correlation Distributions')
        plt.xlabel('Correlation Value')
        plt.ylabel('Density')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('${OUTPUT_DIR}/QC/connectivity_distributions.png', dpi=300, bbox_inches='tight')
        print(f"Saved connectivity distributions plot")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate connectivity distribution plots: {e}")

# Generate additional QC plots
try:
    plot_connectivity_distributions()
except Exception as e:
    print(f"Warning: Could not generate connectivity distributions: {e}")

print("Connectivity analysis and visualization completed successfully.")
EOF
update_progress 17 "Connectivity Analysis" 

print_section "Generating Summary Report"
# Generate summary report
log_message "Generating summary report..."

# Set up HTML conditional sections based on CompCor being enabled
if [ "$ENABLE_COMPCOR" -eq 1 ]; then
    ENABLE_COMPCOR_HTML_START=""
    ENABLE_COMPCOR_HTML_END=""
    ENABLE_COMPCOR_LIST_ITEM="    <li>CompCor Denoising (${COMPCOR_COMPONENTS} components)</li>"
    ENABLE_COMPCOR_TXT="Enabled (${COMPCOR_COMPONENTS} components, Bandpass: ${HIGHPASS_HZ}-${LOWPASS_HZ} Hz)"
    COMPCOR_SECTION="CompCor Files (compcor/):
- CompCor Denoised fMRI: ${OUTPUT_DIR}/compcor/compcor_cleaned.nii.gz
- aCompCor Components: ${OUTPUT_DIR}/compcor/acompcor_components.txt
- CompCor QC: ${OUTPUT_DIR}/compcor/QC/ (variance maps, connectivity plots)"
else
    ENABLE_COMPCOR_HTML_START="<!-- CompCor disabled -->"
    ENABLE_COMPCOR_HTML_END="<!-- End CompCor section -->"
    ENABLE_COMPCOR_LIST_ITEM=""
    ENABLE_COMPCOR_TXT="Disabled"
    COMPCOR_SECTION=""
fi

# Set up segmentation method info for the report
if [ "$SEGMENTATION_TYPE" == "atropos" ]; then
    SEGMENTATION_METHOD_HTML="Deep Atropos (ANTsPyNet)"
    SEGMENTATION_METHOD_TXT="Deep Atropos (ANTsPyNet)"
else
    SEGMENTATION_METHOD_HTML="SynthSeg (FreeSurfer)"
    SEGMENTATION_METHOD_TXT="SynthSeg (FreeSurfer)"
fi

# Create HTML summary report
cat > "${OUTPUT_DIR}/summary_report.html" <<EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>fMRI Preprocessing Summary Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            border-bottom: 1px solid #3498db;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        .info-box {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
        }
        .parameter {
            font-family: monospace;
            background-color: #f5f5f5;
            padding: 2px 5px;
            border-radius: 3px;
        }
        .qc-image {
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            border: 1px solid #ddd;
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
        }
        .image-item {
            flex: 1;
            min-width: 300px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .footer {
            margin-top: 50px;
            border-top: 1px solid #ddd;
            padding-top: 10px;
            font-size: 0.9em;
            color: #777;
        }
        .accordion {
            background-color: #f8f9fa;
            color: #444;
            cursor: pointer;
            padding: 18px;
            width: 100%;
            text-align: left;
            border: none;
            outline: none;
            transition: 0.4s;
            border-left: 4px solid #3498db;
            margin: 5px 0;
            font-weight: bold;
        }
        .active, .accordion:hover {
            background-color: #e9ecef;
        }
        .panel {
            padding: 0 18px;
            background-color: white;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.2s ease-out;
        }
    </style>
</head>
<body>
    <h1>fMRI Preprocessing Pipeline Summary Report</h1>
    
    <div class="info-box">
        <p><strong>Date:</strong> $(date '+%Y-%m-%d %H:%M:%S')</p>
        <p><strong>Subject Directory:</strong> ${SUBJECT_DIR}</p>
        <p><strong>Output Directory:</strong> ${OUTPUT_DIR}</p>
    </div>
    
    <h2>Input Parameters</h2>
    <table>
        <tr>
            <th>Parameter</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Atlas Type</td>
            <td>${ATLAS_TYPE}</td>
        </tr>
        <tr>
            <td>T1 Image</td>
            <td>${T1_IMAGE}</td>
        </tr>
        <tr>
            <td>Segmentation Method</td>
            <td>${SEGMENTATION_METHOD_HTML}</td>
        </tr>
        <tr>
            <td>Lesion Mask</td>
            <td>${LESION_MASK:-"Not provided"}</td>
        </tr>
        <tr>
            <td>TR (seconds)</td>
            <td>${TR}</td>
        </tr>
        <tr>
            <td>CompCor</td>
            <td>${ENABLE_COMPCOR_TXT}</td>
        </tr>
    </table>
    
    <h2>Processing Steps Completed</h2>
    <ol>
        <li>DICOM to NIfTI Conversion</li>
        <li>TOPUP Distortion Correction</li>
        <li>MCFLIRT Motion Correction</li>
        <li>Brain Extraction (ANTsPyNet)</li>
        <li>T1 Registration to fMRI Space</li>
        <li>Tissue Segmentation (${SEGMENTATION_METHOD_HTML})</li>
        <li>Atlas Registration (${ATLAS_TYPE})</li>
        <li>MELODIC ICA (Components: 30)</li>
        <li>ICA-AROMA Denoising</li>
        ${ENABLE_COMPCOR_LIST_ITEM}
        <li>Functional Connectivity Analysis</li>
    </ol>
    
    <h2>Quality Control Visualizations</h2>
    
    <h3>Motion Parameters</h3>
    <div class="image-container">
        <div class="image-item">
            <img class="qc-image" src="QC/rot_motion.png" alt="Rotation Motion Parameters">
            <p>Rotation motion parameters (radians) in X, Y, and Z directions.</p>
        </div>
        <div class="image-item">
            <img class="qc-image" src="QC/trans_motion.png" alt="Translation Motion Parameters">
            <p>Translation motion parameters (mm) in X, Y, and Z directions.</p>
        </div>
    </div>
    
    <h3>ICA-AROMA Components</h3>
    <div class="image-container">
        <div class="image-item">
            <img class="qc-image" src="QC/ica_aroma_classification.png" alt="ICA-AROMA Classification">
            <p>Classification of ICA components as motion-related or non-motion-related.</p>
        </div>
    </div>
    
    <h3>Carpet Plot</h3>
    <div class="image-container">
        <div class="image-item">
            <img class="qc-image" src="${ENABLE_COMPCOR_HTML_START}compcor/QC/carpet_denoised.png${ENABLE_COMPCOR_HTML_END}${ENABLE_COMPCOR_HTML_START}${ENABLE_COMPCOR_HTML_END}QC/carpet_plot_simple.png" alt="Carpet Plot">
            <p>Carpet plot showing voxel-wise time series data after denoising.</p>
        </div>
    </div>
    
    <!-- CompCor QC section (only shown if CompCor was enabled) -->
    ${ENABLE_COMPCOR_HTML_START}
    <h3>CompCor Denoising</h3>
    <div class="image-container">
        <div class="image-item">
            <img class="qc-image" src="compcor/QC/variance_explained.png" alt="CompCor Variance Explained">
            <p>Spatial map of variance explained by CompCor regression.</p>
        </div>
        <div class="image-item">
            <img class="qc-image" src="compcor/QC/connectivity_distributions.png" alt="CompCor Connectivity Distributions">
            <p>Distribution of connectivity values before and after CompCor denoising.</p>
        </div>
    </div>
    <div class="image-container">
        <div class="image-item">
            <img class="qc-image" src="compcor/QC/connectivity_matrices_comparison.png" alt="CompCor Connectivity Matrices">
            <p>Comparison of connectivity matrices before and after CompCor denoising.</p>
        </div>
    </div>
    ${ENABLE_COMPCOR_HTML_END}
    
    <h3>Functional Connectivity</h3>
    <div class="image-container">
        <div class="image-item">
            <img class="qc-image" src="QC/functional_connectivity_matrices.png" alt="Functional Connectivity Matrices">
            <p>Functional connectivity matrices using different correlation methods.</p>
        </div>
        <div class="image-item">
            <img class="qc-image" src="QC/connectivity_distributions.png" alt="Connectivity Distributions">
            <p>Distribution of connectivity values across different correlation methods.</p>
        </div>
    </div>
    
    <h2>Output Files</h2>
    <p>Below is a comprehensive list of key output files organized by directory:</p>
    
    <button class="accordion">NIfTI Files (nifti/)</button>
    <div class="panel">
        <table>
            <tr>
                <th>Description</th>
                <th>Path</th>
            </tr>
            <tr>
                <td>Field Map (AP)</td>
                <td>${OUTPUT_DIR}/nifti/010_SpinEchoFieldMap_AP.nii.gz</td>
            </tr>
            <tr>
                <td>Field Map (PA)</td>
                <td>${OUTPUT_DIR}/nifti/011_SpinEchoFieldMap_PA.nii.gz</td>
            </tr>
            <tr>
                <td>fMRI Data</td>
                <td>${OUTPUT_DIR}/nifti/013_rfMRI_REST_AP.nii.gz</td>
            </tr>
            <tr>
                <td>Field Map Metadata (AP)</td>
                <td>${OUTPUT_DIR}/nifti/010_SpinEchoFieldMap_AP.json</td>
            </tr>
            <tr>
                <td>Field Map Metadata (PA)</td>
                <td>${OUTPUT_DIR}/nifti/011_SpinEchoFieldMap_PA.json</td>
            </tr>
            <tr>
                <td>fMRI Metadata</td>
                <td>${OUTPUT_DIR}/nifti/013_rfMRI_REST_AP.json</td>
            </tr>
        </table>
    </div>

    <button class="accordion">Registration Files (registration/)</button>
    <div class="panel">
        <table>
            <tr>
                <th>Description</th>
                <th>Path</th>
            </tr>
            <tr>
                <td>TOPUP Corrected fMRI</td>
                <td>${OUTPUT_DIR}/registration/topup_corrected_fmri.nii.gz</td>
            </tr>
            <tr>
                <td>T1 in fMRI Space</td>
                <td>${OUTPUT_DIR}/registration/t1_in_fmri_space.nii.gz</td>
            </tr>
            <tr>
                <td>fMRI Brain Mask</td>
                <td>${OUTPUT_DIR}/registration/fmri_brain_mask.nii.gz</td>
            </tr>
            <tr>
                <td>fMRI Brain</td>
                <td>${OUTPUT_DIR}/registration/fmri_brain.nii.gz</td>
            </tr>
            <tr>
                <td>fMRI 3D Volume</td>
                <td>${OUTPUT_DIR}/registration/fmri_3d.nii.gz</td>
            </tr>
            <tr>
                <td>Brain-Extracted fMRI (4D)</td>
                <td>${OUTPUT_DIR}/registration/mcflirt_corrected_fmri_brain.nii.gz</td>
            </tr>
            <tr>
                <td>Tissue Segmentation</td>
                <td>${OUTPUT_DIR}/registration/t1_brain_segmentation.nii.gz</td>
            </tr>
            <tr>
                <td>CSF Probability Map</td>
                <td>${OUTPUT_DIR}/registration/csf_prob.nii.gz</td>
            </tr>
            <tr>
                <td>GM Probability Map</td>
                <td>${OUTPUT_DIR}/registration/gm_prob.nii.gz</td>
            </tr>
            <tr>
                <td>WM Probability Map</td>
                <td>${OUTPUT_DIR}/registration/wm_prob.nii.gz</td>
            </tr>
            <tr>
                <td>Atlas in fMRI Space</td>
                <td>${ATLAS_OUTPUT}</td>
            </tr>
            <tr>
                <td>Lesion Mask in fMRI Space</td>
                <td>${OUTPUT_DIR}/registration/lesion_mask_in_fmri_space.nii.gz</td>
            </tr>
            <tr>
                <td>TOPUP Results</td>
                <td>${OUTPUT_DIR}/registration/topup_results_fieldcoef.nii.gz</td>
            </tr>
        </table>
    </div>

    <button class="accordion">Motion Correction Files (motion_correction/)</button>
    <div class="panel">
        <table>
            <tr>
                <th>Description</th>
                <th>Path</th>
            </tr>
            <tr>
                <td>Motion-Corrected fMRI</td>
                <td>${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.nii.gz</td>
            </tr>
            <tr>
                <td>Motion Parameters</td>
                <td>${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.par</td>
            </tr>
            <tr>
                <td>Framewise Displacement Metrics</td>
                <td>${OUTPUT_DIR}/motion_correction/fd_metrics.txt</td>
            </tr>
            <tr>
                <td>Motion Confounds</td>
                <td>${OUTPUT_DIR}/motion_correction/motion_confounds.txt</td>
            </tr>
        </table>
    </div>

    <button class="accordion">ICA-AROMA Files (ICA_AROMA/)</button>
    <div class="panel">
        <table>
            <tr>
                <th>Description</th>
                <th>Path</th>
            </tr>
            <tr>
                <td>Denoised fMRI (Non-aggressive)</td>
                <td>${AROMA_OUT_DIR}/denoised_func_data_nonaggr.nii.gz</td>
            </tr>
            <tr>
                <td>Denoised fMRI (Aggressive)</td>
                <td>${AROMA_OUT_DIR}/denoised_func_data_aggr.nii.gz</td>
            </tr>
            <tr>
                <td>Classified Motion ICs</td>
                <td>${AROMA_OUT_DIR}/classified_motion_ICs.txt</td>
            </tr>
            <tr>
                <td>Feature Scores</td>
                <td>${AROMA_OUT_DIR}/feature_scores.txt</td>
            </tr>
            <tr>
                <td>Classification Overview</td>
                <td>${AROMA_OUT_DIR}/classification_overview.txt</td>
            </tr>
        </table>
    </div>
    
    ${ENABLE_COMPCOR_HTML_START}
    <button class="accordion">CompCor Files (compcor/)</button>
    <div class="panel">
        <table>
            <tr>
                <th>Description</th>
                <th>Path</th>
            </tr>
            <tr>
                <td>CompCor Denoised fMRI</td>
                <td>${OUTPUT_DIR}/compcor/compcor_cleaned.nii.gz</td>
            </tr>
            <tr>
                <td>aCompCor Components</td>
                <td>${OUTPUT_DIR}/compcor/acompcor_components.txt</td>
            </tr>
            <tr>
                <td>Connectivity Matrices Comparison</td>
                <td>${OUTPUT_DIR}/compcor/QC/connectivity_matrices_comparison.png</td>
            </tr>
            <tr>
                <td>Connectivity Distributions</td>
                <td>${OUTPUT_DIR}/compcor/QC/connectivity_distributions.png</td>
            </tr>
            <tr>
                <td>Variance Explained Map</td>
                <td>${OUTPUT_DIR}/compcor/QC/variance_explained.png</td>
            </tr>
            <tr>
                <td>Carpet Plot</td>
                <td>${OUTPUT_DIR}/compcor/QC/carpet_denoised.png</td>
            </tr>
        </table>
    </div>
    ${ENABLE_COMPCOR_HTML_END}

    <button class="accordion">Connectivity Analysis Files (connectivity/)</button>
    <div class="panel">
        <table>
            <tr>
                <th>Description</th>
                <th>Path</th>
            </tr>
            <tr>
                <td>Pearson Correlation Matrix (ICA-AROMA)</td>
                <td>${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_pearson.txt</td>
            </tr>
            <tr>
                <td>Spearman Correlation Matrix (ICA-AROMA)</td>
                <td>${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_spearman.txt</td>
            </tr>
            <tr>
                <td>Partial Correlation Matrix (ICA-AROMA)</td>
                <td>${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_partial.txt</td>
            </tr>
            <tr>
                <td>Z-normalized Pearson Matrix (ICA-AROMA)</td>
                <td>${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_pearson_znorm.txt</td>
            </tr>
            <tr>
                <td>Z-normalized Spearman Matrix (ICA-AROMA)</td>
                <td>${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_spearman_znorm.txt</td>
            </tr>
            <tr>
                <td>Z-normalized Partial Matrix (ICA-AROMA)</td>
                <td>${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_partial_znorm.txt</td>
            </tr>
            ${ENABLE_COMPCOR_HTML_START}
            <tr>
                <td>Pearson Correlation Matrix (CompCor)</td>
                <td>${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_compcor_pearson.txt</td>
            </tr>
            <tr>
                <td>Spearman Correlation Matrix (CompCor)</td>
                <td>${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_compcor_spearman.txt</td>
            </tr>
            <tr>
                <td>Partial Correlation Matrix (CompCor)</td>
                <td>${OUTPUT_DIR}/connectivity/functional_connectivity_matrix_compcor_partial.txt</td>
            </tr>
            ${ENABLE_COMPCOR_HTML_END}
            <tr>
                <td>Valid Regions Information</td>
                <td>${OUTPUT_DIR}/connectivity/valid_regions.csv</td>
            </tr>
        </table>
    </div>

    <button class="accordion">Quality Control Files (QC/)</button>
    <div class="panel">
        <table>
            <tr>
                <th>Description</th>
                <th>Path</th>
            </tr>
            <tr>
                <td>Rotation Motion Plot</td>
                <td>${OUTPUT_DIR}/QC/rot_motion.png</td>
            </tr>
            <tr>
                <td>Translation Motion Plot</td>
                <td>${OUTPUT_DIR}/QC/trans_motion.png</td>
            </tr>
            <tr>
                <td>Combined Motion Plot</td>
                <td>${OUTPUT_DIR}/QC/combined_motion.png</td>
            </tr>
            <tr>
                <td>ICA-AROMA Classification</td>
                <td>${OUTPUT_DIR}/QC/ica_aroma_classification.png</td>
            </tr>
            <tr>
                <td>Carpet Plot</td>
                <td>${OUTPUT_DIR}/QC/carpet_plot_simple.png</td>
            </tr>
            <tr>
                <td>Functional Connectivity Matrices</td>
                <td>${OUTPUT_DIR}/QC/functional_connectivity_matrices.png</td>
            </tr>
            <tr>
                <td>Connectivity Distributions</td>
                <td>${OUTPUT_DIR}/QC/connectivity_distributions.png</td>
            </tr>
        </table>
    </div>

    <button class="accordion">MELODIC ICA Files (melodic.ica/)</button>
    <div class="panel">
        <p>The MELODIC ICA directory contains numerous files related to the independent component analysis.</p>
        <p>Key files include:</p>
        <ul>
            <li>melodic_IC.nii.gz - The spatial maps of the independent components</li>
            <li>melodic_mix - The timecourses of the components</li>
            <li>melodic_FTmix - The Fourier spectra of the component timecourses</li>
            <li>report/ - Directory containing HTML report with component visualizations</li>
        </ul>
    </div>
    
    <button class="accordion">SynthSeg Files (synthseg/)</button>
    <div class="panel">
        <p>The SynthSeg directory contains output files from the FreeSurfer SynthSeg segmentation.</p>
        <p>Key files include:</p>
        <ul>
            <li>segmentation.nii.gz - The FreeSurfer-style segmentation of the T1 image</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>For detailed processing logs, refer to: ${LOG_FILE}</p>
        <p>Generated by fMRI Preprocessing Pipeline v15.5</p>
        <p>© $(date +%Y) Lucius Fekonja</p>
    </div>

    <script>
    var acc = document.getElementsByClassName("accordion");
    var i;

    for (i = 0; i < acc.length; i++) {
        acc[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var panel = this.nextElementSibling;
            if (panel.style.maxHeight) {
                panel.style.maxHeight = null;
            } else {
                panel.style.maxHeight = panel.scrollHeight + "px";
            } 
        });
    }
    </script>
</body>
</html>
EOF

# Also update the plain text summary report for backward compatibility
cat > "${OUTPUT_DIR}/summary_report.txt" <<EOF
fMRI Preprocessing Pipeline Summary
=================================
Date: $(date '+%Y-%m-%d %H:%M:%S')
Subject Directory: ${SUBJECT_DIR}
Output Directory: ${OUTPUT_DIR}

Input Parameters
---------------
Atlas Type: ${ATLAS_TYPE}
T1 Image: ${T1_IMAGE}
Segmentation Method: ${SEGMENTATION_METHOD_TXT}
Lesion Mask: ${LESION_MASK:-"Not provided"}
TR (seconds): ${TR}
CompCor: ${ENABLE_COMPCOR_TXT}

Processing Steps Completed
------------------------
1. DICOM to NIfTI Conversion
2. TOPUP Distortion Correction
3. MCFLIRT Motion Correction
4. Brain Extraction (ANTsPyNet)
5. T1 Registration to fMRI Space
6. Tissue Segmentation (${SEGMENTATION_METHOD_TXT})
7. Atlas Registration
8. MELODIC ICA (Components: 30)
9. ICA-AROMA Denoising
10. Functional Connectivity Analysis

Quality Metrics
-------------
- Motion Statistics: ${OUTPUT_DIR}/motion_correction/fd_metrics.txt
- QC Visualizations: ${OUTPUT_DIR}/QC/
- Connectivity Matrices: ${OUTPUT_DIR}/connectivity/

COMPREHENSIVE OUTPUT FILE LISTING
================================

NIfTI Files (nifti/):
- Field Maps: ${OUTPUT_DIR}/nifti/010_SpinEchoFieldMap_AP.nii.gz, ${OUTPUT_DIR}/nifti/011_SpinEchoFieldMap_PA.nii.gz
- fMRI Data: ${OUTPUT_DIR}/nifti/013_rfMRI_REST_AP.nii.gz
- JSON Metadata: Corresponding .json files for all NIfTI files

Registration Files (registration/):
- TOPUP Corrected fMRI: ${OUTPUT_DIR}/registration/topup_corrected_fmri.nii.gz
- T1 in fMRI Space: ${OUTPUT_DIR}/registration/t1_in_fmri_space.nii.gz
- Brain Masks: ${OUTPUT_DIR}/registration/fmri_brain_mask.nii.gz, ${OUTPUT_DIR}/registration/fmri_brain.nii.gz
- Tissue Segmentation: ${OUTPUT_DIR}/registration/t1_brain_segmentation.nii.gz
- Probability Maps: CSF, GM, WM probability maps
- Atlas in fMRI Space: ${ATLAS_OUTPUT}
- Lesion Mask in fMRI Space: ${OUTPUT_DIR}/registration/lesion_mask_in_fmri_space.nii.gz

Motion Correction Files (motion_correction/):
- Motion-Corrected fMRI: ${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.nii.gz
- Motion Parameters: ${OUTPUT_DIR}/motion_correction/mcflirt_corrected_fmri.par
- Framewise Displacement: ${OUTPUT_DIR}/motion_correction/fd_metrics.txt
- Motion Confounds: ${OUTPUT_DIR}/motion_correction/motion_confounds.txt

ICA-AROMA Files (ICA_AROMA/):
- Denoised fMRI (Non-aggressive): ${AROMA_OUT_DIR}/denoised_func_data_nonaggr.nii.gz
- Denoised fMRI (Aggressive): ${AROMA_OUT_DIR}/denoised_func_data_aggr.nii.gz
- Classified Motion ICs: ${AROMA_OUT_DIR}/classified_motion_ICs.txt
- Feature Scores & Classification: Feature scores and classification overview files

${COMPCOR_SECTION}

Connectivity Analysis Files (connectivity/):
- Correlation Matrices: Pearson, Spearman, and Partial correlation matrices
- Z-normalized Matrices: Z-normalized versions of all correlation matrices
- Region Information: ${OUTPUT_DIR}/connectivity/valid_regions.csv

Quality Control Files (QC/):
- Motion Plots: Rotation, translation, and combined motion plots
- ICA-AROMA Visualization: Classification plot
- Carpet Plot: Data quality visualization
- Connectivity Visualizations: Matrix and distribution plots

MELODIC ICA Files (melodic.ica/):
- Spatial Maps: melodic_IC.nii.gz
- Timecourses: melodic_mix
- Fourier Spectra: melodic_FTmix
- HTML Report: report/00index.html with component visualizations

SynthSeg Files (synthseg/):
- FreeSurfer SynthSeg Segmentation: ${OUTPUT_DIR}/synthseg/segmentation.nii.gz

For detailed processing logs, please refer to: ${LOG_FILE}
EOF

log_message "Summary report generated"
update_progress 18 "Summary Report"

# Create a backup of the log file
cp "${LOG_FILE}" "${LOG_FILE_BACKUP}"

# Clear the progress bar
clear_progress_bar

# Finalize the report with the appropriate denoising method and segmentation method used
if [ "$ENABLE_COMPCOR" -eq 1 ]; then
    DENOISING_METHOD="ICA-AROMA and CompCor"
else
    DENOISING_METHOD="ICA-AROMA"
fi

print_section "Processing Complete"
log_message "Preprocessing pipeline completed successfully using ${DENOISING_METHOD} denoising and ${SEGMENTATION_TYPE} segmentation."

echo -e "${GREEN}Preprocessing pipeline completed successfully. Results are in ${OUTPUT_DIR}${NC}"
echo -e "${BLUE}Denoising method used: ${DENOISING_METHOD}${NC}"
echo -e "${BLUE}Segmentation method used: ${SEGMENTATION_TYPE}${NC}"
echo -e "${BLUE}Please check the summary report at ${OUTPUT_DIR}/summary_report.html for an overview of the preprocessing steps and results.${NC}"