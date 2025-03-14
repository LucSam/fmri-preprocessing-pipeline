# fMRI Preprocessing Pipeline

A comprehensive fMRI preprocessing pipeline with distortion correction, motion correction, advanced denoising techniques (ICA-AROMA and CompCor), and functional connectivity matrix construction, making it suitable for both clinical and research applications.

## Features

- DICOM to NIfTI conversion using MRtrix3
- Distortion correction using FSL's TOPUP
- Motion correction using FSL's MCFLIRT
- Brain extraction using ANTsPyNet
- T1 to fMRI registration using ANTs
- Tissue segmentation using choice of ANTs Deep Atropos or FreeSurfer SynthSeg
- Atlas registration with support for AAL, Schaefer, and FreeSurfer parcellations
- ICA-based denoising with MELODIC and ICA-AROMA
- CompCor denoising with optional bandpass filtering
- Functional connectivity matrix creation
- Extensive quality control visualizations
- Detailed HTML and text summary reports

## Documentation

For detailed documentation, please see [documentation](docs/README.md).

## Usage

```bash
./fmri_preprocessing15_5.sh -s /path/to/subject \
  -d "010_SpinEchoFieldMap_AP,011_SpinEchoFieldMap_PA,012_rfMRI_REST_AP,013_rfMRI_REST_AP" \
  -A aal -a /path/to/AAL.nii \
  -t /path/to/t1.nii.gz \
  -S atropos \
  -I /path/to/ICA-AROMA.py \
  -C 1 -N 5 -B 1 -L 0.08 -H 0.01