# fMRI Preprocessing Pipeline

**Version:** 1.0  
**Authors:** Lucius Fekonja, Onurhan Karatay  
**Last Updated:** March 2025

## Overview

This pipeline provides a comprehensive set of tools for preprocessing functional MRI (fMRI) data. It includes distortion correction, motion correction, advanced denoising techniques (ICA-AROMA and CompCor), and functional connectivity matrix creation, making it suitable for both clinical and research applications.

## Features

- **DICOM to NIfTI conversion** using MRtrix3
- **Distortion correction** using FSL's TOPUP
- **Motion correction** using FSL's MCFLIRT
- **Brain extraction** using ANTsPyNet
- **T1 to fMRI registration** using ANTs
- **Tissue segmentation** using choice of ANTs Deep Atropos or FreeSurfer SynthSeg
- **Atlas registration** with support for AAL, Schaefer, and FreeSurfer parcellations
- **ICA-based denoising** with MELODIC and ICA-AROMA
- **CompCor denoising** with optional bandpass filtering
- **Functional connectivity matrix creation**
- **Extensive quality control visualizations**
- **Detailed HTML and text summary reports**

## Software Requirements

### Required Software

- **FSL** (FMRIB Software Library)
- **MRtrix3**
- **ANTs** (Advanced Normalization Tools)
- **Python 3** with the following packages:
  - nibabel
  - numpy
  - nilearn
  - scikit-learn
  - scipy
  - matplotlib
  - seaborn
  - pandas
- **ICA-AROMA**
- **FreeSurfer** (required for FreeSurfer atlas or SynthSeg segmentation)

### Optional Software

- **ANTsPy** and **ANTsPyNet** for Deep Atropos segmentation and improved brain extraction

## Input Data Requirements

The script expects the following input data:

1. **DICOM directories** for:
   - Spin Echo Field Map (Anterior-Posterior)
   - Spin Echo Field Map (Posterior-Anterior)
   - Two fMRI REST scans (Anterior-Posterior)
2. **T1-weighted structural image** in NIfTI format
3. **Atlas file** (depending on the selected atlas type):
   - AAL atlas
   - Schaefer atlas
   - FreeSurfer output directory (if using FreeSurfer)
4. **Lesion mask** (optional) in NIfTI format

## Usage

### Basic Command

```bash
./fmri-preprocessing-pipeline-1_0.sh -s /path/to/subject \
  -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \
  -A aal -a /path/to/AAL.nii \
  -t /path/to/t1.nii.gz \
  -S atropos \
  -I /path/to/ICA-AROMA.py \
  -C 1 -N 5 -B 1 -L 0.08 -H 0.01
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `-s` | Subject directory path | No (defaults to current directory) |
| `-o` | Output directory path | No (defaults to `SUBJECT_DIR/preprocessed`) |
| `-d` | DICOM directories (comma-separated) | Yes |
| `-A` | Atlas type: 'aal', 'schaefer', or 'freesurfer' | No (defaults to 'aal') |
| `-a` | Path to atlas file (required for 'aal' and 'schaefer') | Conditional |
| `-t` | Path to T1 image | Yes |
| `-l` | Path to lesion mask | No |
| `-f` | Path to FreeSurfer subject directory (for 'freesurfer' atlas) | Conditional |
| `-I` | Path to ICA-AROMA.py | Yes |
| `-S` | Segmentation method: 'atropos' or 'synthseg' | No (defaults to 'atropos') |
| `-C` | Enable CompCor denoising: 0=disabled, 1=enabled | No (defaults to 0) |
| `-N` | Number of CompCor components to extract | No (defaults to 5) |
| `-B` | Apply bandpass filter with CompCor: 0=disabled, 1=enabled | No (defaults to 1) |
| `-L` | Lowpass filter cutoff in Hz | No (defaults to 0.08) |
| `-H` | Highpass filter cutoff in Hz | No (defaults to 0.01) |
| `-h` | Display help message | No |

### Example Commands

#### Using AAL Atlas with Deep Atropos segmentation and CompCor

```bash
./fmri-preprocessing-pipeline-1_0.sh -s /path/to/subject \
  -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \
  -A aal -a /path/to/AAL.nii \
  -t /path/to/t1.nii.gz \
  -S atropos \
  -I /path/to/ICA-AROMA.py \
  -C 1 -N 5 -B 1 -L 0.08 -H 0.01
```

#### Using Schaefer Atlas with SynthSeg segmentation (without CompCor)

```bash
./fmri-preprocessing-pipeline-1_0.sh -s /path/to/subject \
  -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \
  -A schaefer -a /path/to/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz \
  -t /path/to/t1.nii.gz \
  -S synthseg \
  -I /path/to/ICA-AROMA.py
```

#### Using FreeSurfer Atlas

```bash
./fmri-preprocessing-pipeline-1_0.sh -s /path/to/subject \
  -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \
  -A freesurfer \
  -t /path/to/t1.nii.gz \
  -f /path/to/freesurfer_subj_dir \
  -I /path/to/ICA-AROMA.py
```

#### Including Lesion Mask with CompCor Denoising

```bash
./fmri-preprocessing-pipeline-1_0.sh -s /path/to/subject \
  -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \
  -A aal -a /path/to/AAL.nii \
  -t /path/to/t1.nii.gz \
  -l /path/to/lesion.nii.gz \
  -I /path/to/ICA-AROMA.py \
  -C 1 -N 8 -B 1 -L 0.08 -H 0.01
```

## Processing Steps

1. **Dependency and input validation**
   - Checks for required software and validates input files

2. **DICOM to NIfTI conversion**
   - Converts DICOM images to NIfTI format using MRtrix3
   - Extracts metadata (e.g., TR, phase encoding directions)

3. **TOPUP distortion correction**
   - Generates TOPUP encoding file
   - Runs TOPUP to estimate susceptibility-induced distortions
   - Applies correction to fMRI data

4. **Motion correction**
   - Applies MCFLIRT to correct for head motion
   - Generates motion parameters, FD metrics, and QC plots
   - Creates motion confounds file

5. **Brain extraction**
   - Extracts a 3D volume from the 4D fMRI
   - Applies ANTsPyNet brain extraction
   - Creates brain mask and applies it to 4D fMRI

6. **T1 to fMRI registration**
   - Adjusts strides of T1 image
   - Registers T1 to fMRI space
   - Transforms lesion mask (if provided)

7. **Tissue segmentation**
   - Uses selected segmentation method (Deep Atropos or SynthSeg)
   - Generates probability maps for CSF, GM, and WM
   - Transforms segmentation results to fMRI space

8. **Atlas registration**
   - Registers selected atlas to fMRI space
   - Transforms FreeSurfer parcellation (if selected)

9. **ICA analysis and denoising**
   - Runs MELODIC ICA
   - Performs ICA-AROMA to identify and remove motion components
   - Generates QC visualizations for component classification

10. **CompCor denoising (optional)**
    - Extracts noise components from WM and CSF regions
    - Regresses out noise components and motion parameters
    - Applies bandpass filtering if enabled
    - Generates CompCor QC visualizations

11. **Carpet plot generation**
    - Creates carpet plot for quality assessment
    - Shows voxel-wise time series data by tissue type

12. **Connectivity analysis**
    - Extracts time series from atlas regions
    - Calculates Pearson, Spearman, and partial correlations
    - Generates connectivity matrices and visualizations
    - Creates Z-normalized connectivity matrices

13. **Report generation**
    - Creates detailed HTML summary report
    - Provides text-based summary report
    - Includes comprehensive file listings

## Output Structure

After running the pipeline, the following directory structure is created:

```
preprocessed/
├── nifti/                       # Converted NIfTI files and JSON metadata
│   ├── 010_SpinEchoFieldMap_AP.nii.gz
│   ├── 011_SpinEchoFieldMap_PA.nii.gz
│   ├── 012_rfMRI_REST_AP.nii.gz
│   └── 013_rfMRI_REST_AP.nii.gz
│
├── registration/                # Registration-related files
│   ├── fmri_3d.nii.gz
│   ├── fmri_brain.nii.gz
│   ├── fmri_brain_mask.nii.gz
│   ├── t1_in_fmri_space.nii.gz
│   ├── t1_brain_segmentation*.nii.gz
│   ├── csf_prob.nii.gz
│   ├── gm_prob.nii.gz
│   ├── wm_prob.nii.gz
│   ├── topup_*
│   └── atlas_registered_to_fmri.nii.gz
│
├── motion_correction/           # Motion correction results
│   ├── mcflirt_corrected_fmri.nii.gz
│   ├── mcflirt_corrected_fmri.par
│   ├── fd_metrics.txt
│   └── motion_confounds.txt
│
├── melodic.ica/                 # MELODIC ICA results
│   ├── melodic_IC.nii.gz
│   ├── melodic_mix
│   └── report/
│
├── ICA_AROMA/                   # ICA-AROMA denoising results
│   ├── denoised_func_data_nonaggr.nii.gz
│   ├── denoised_func_data_aggr.nii.gz
│   └── classified_motion_ICs.txt
│
├── compcor/                     # CompCor denoising results (if enabled)
│   ├── compcor_cleaned.nii.gz
│   ├── acompcor_components.txt
│   └── QC/
│
├── connectivity/                # Connectivity analysis results
│   ├── functional_connectivity_matrix_*.txt
│   └── valid_regions.csv
│
├── synthseg/                    # SynthSeg results (if used)
│   └── segmentation.nii.gz
│
├── QC/                          # Quality control visualizations
│   ├── rot_motion.png
│   ├── trans_motion.png
│   ├── combined_motion.png
│   ├── ica_aroma_classification.png
│   ├── carpet_plot_simple.png
│   ├── functional_connectivity_matrices.png
│   └── connectivity_distributions.png
│
├── preprocessing.log            # Processing log
├── preprocessing_timestamp.log  # Backup log file
├── summary_report.html          # HTML summary report
└── summary_report.txt           # Text summary report
```

## Key Output Files

### Preprocessed Data

- **ICA-AROMA Denoised fMRI (Non-aggressive):** `ICA_AROMA/denoised_func_data_nonaggr.nii.gz`
- **ICA-AROMA Denoised fMRI (Aggressive):** `ICA_AROMA/denoised_func_data_aggr.nii.gz`
- **CompCor Denoised fMRI:** `compcor/compcor_cleaned.nii.gz` (if CompCor enabled)
- **Brain-Extracted fMRI:** `registration/mcflirt_corrected_fmri_brain.nii.gz`
- **T1 in fMRI Space:** `registration/t1_in_fmri_space.nii.gz`
- **Atlas in fMRI Space:** `registration/atlas_registered_to_fmri.nii.gz`

### Connectivity Results

- **Pearson Correlation Matrix:** `connectivity/functional_connectivity_matrix_pearson.txt`
- **Spearman Correlation Matrix:** `connectivity/functional_connectivity_matrix_spearman.txt`
- **Partial Correlation Matrix:** `connectivity/functional_connectivity_matrix_partial.txt`
- **Z-normalized Matrices:** `connectivity/functional_connectivity_matrix_*_znorm.txt`
- **CompCor Correlation Matrices:** `connectivity/functional_connectivity_matrix_compcor_*.txt` (if CompCor enabled)

### Quality Control

- **Motion Parameters:** `motion_correction/mcflirt_corrected_fmri.par`
- **Framewise Displacement:** `motion_correction/fd_metrics.txt`
- **Motion Plots:** `QC/rot_motion.png`, `QC/trans_motion.png`
- **ICA-AROMA Classification:** `QC/ica_aroma_classification.png`
- **Carpet Plot:** `QC/carpet_plot_simple.png` or `compcor/QC/carpet_denoised.png`
- **Connectivity Visualizations:** `QC/functional_connectivity_matrices.png`
- **CompCor QC:** `compcor/QC/variance_explained.png`, `compcor/QC/connectivity_matrices_comparison.png` (if CompCor enabled)

### Reports

- **HTML Summary Report:** `summary_report.html`
- **Text Summary Report:** `summary_report.txt`
- **Processing Log:** `preprocessing.log`



## Customization

### Advanced Options

The script offers several advanced customization options:

- **Segmentation method selection:** Choose between `atropos` (Deep Atropos with ANTsPyNet) and `synthseg` (FreeSurfer SynthSeg)
- **CompCor parameters:**
  - Enable/disable CompCor denoising with the `-C` option
  - Adjust number of components with `-N` option
  - Enable/disable bandpass filtering with `-B` option
  - Set custom filter frequencies with `-L` and `-H` options
- **MELODIC component count:** Modify the `--dim=30` parameter in the MELODIC call
- **ICA-AROMA denoising approach:** Use non-aggressive (`nonaggr`) denoising for connectivity analysis
- **Atlas registration approach:** Adjust the `-t s` parameter in the ANTs registration call

### Processing Multiple Subjects

For batch processing of multiple subjects, create a wrapper script:

```bash
#!/bin/bash

# Set paths
ATLAS_FILE="/path/to/atlas.nii"
T1_DIR="/path/to/t1_images"
SUBJECT_DIR="/path/to/subjects"
ICA_AROMA_DIR="/path/to/ICA-AROMA.py"

# Set processing options
SEGMENTATION_TYPE="atropos"  # or "synthseg"
ENABLE_COMPCOR=1  # 1=enabled, 0=disabled
COMPCOR_COMPONENTS=5
BANDPASS_FILTER=1  # 1=enabled, 0=disabled
LOWPASS_HZ=0.08
HIGHPASS_HZ=0.01

# Process each subject
for subject in $(ls $SUBJECT_DIR); do
  echo "Processing subject: $subject"
  
  ./fmri-preprocessing-pipeline-1_0.sh \
    -s "$SUBJECT_DIR/$subject" \
    -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \
    -A aal -a "$ATLAS_FILE" \
    -t "$T1_DIR/${subject}_t1.nii.gz" \
    -I "$ICA_AROMA_DIR" \
    -S "$SEGMENTATION_TYPE" \
    -C "$ENABLE_COMPCOR" \
    -N "$COMPCOR_COMPONENTS" \
    -B "$BANDPASS_FILTER" \
    -L "$LOWPASS_HZ" \
    -H "$HIGHPASS_HZ"
done
```

## Citation

If you use this pipeline in your research, please cite:

```
Fekonja, L., & Karatay, O. (2025). fMRI Preprocessing Pipeline (Version 1.0) [Computer software].
```

## Acknowledgments

This pipeline integrates and builds upon several neuroimaging tools:

- FSL (FMRIB Software Library)
- ANTs (Advanced Normalization Tools)
- MRtrix3
- ICA-AROMA
- CompCor denoising algorithm
- FreeSurfer (including SynthSeg)
- Python libraries: nibabel, nilearn, ANTsPy, ANTsPyNet, and more

## Contact

For questions, feedback, or support, please contact:

**Authors:** Lucius Fekonja, Onurhan Karatay
