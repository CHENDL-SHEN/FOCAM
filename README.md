# Fractional-Order–Guided SAM for Breast Ultrasound Lesion Segmentation 

This repository is the official implementation of "Fractional-Order–Guided SAM for Breast Ultrasound Lesion Segmentation". 

## Prerequisite
- Python 3.6, PyTorch 1.8.0, and more in requirements.txt
- CUDA 11.1
- 1 x  RTX 3090 GPUs

## Usage

### 1. Install python dependencies
```bash
python3 -m pip install -r requirements.txt
```
### 2. Generate Class Activation Maps (CAMs) for SAM prompt
- step1. Train classification network with FO loss.
    ```python
    python FF_busi_s1_train_cam_fotv.py
- step2. Infer CAMs by using the trained CLs.Net.
    ```python 
    python FF_busi_s1_infer_cam.py  
### 3. CAMs-guided adaptive SAM segmentation
- Obtain SAM segmentation results under box prompt from FOCAM.
    ```python
    python FF_busi_s2_refine_sam.py
