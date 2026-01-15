# 3D-CT-Scan-Classification
A 3D Convolutional Neural Network (CNN) pipeline to classify volumetric CT scans (NIfTI format) for COVID-19 detection.

# Project Overview
This project implements a **"3D Convolutional Neural Network CNN)"** to classify volumetric CT scans. Unlike standard 2D image classification, this model processes the entire 3D volume of the lungs to detect the presence of viral pneumonia (COVID-19).

The pipeline handles **NIfTI (.nii)** medical imaging files, performing 3D preprocessing, normalization, and volume resizing before feeding data into a custom 3D CNN architecture built with **TensorFlow/Keras**.

# Dataset
The model was trained on a subset of the **MosMedData** dataset (~140 training volumes).

# Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Processing:** Nibabel (for NIfTI files), NumPy, Scipy
* **Visualization:** Matplotlib
* **Environment:** Google Colab (T4 GPU)

## Neural Network Architecture
The model uses a custom 3D architecture designed to capture spatial dependencies in volumetric data:
* **Input:** 64x64x32 3D volumes.
* **Layers:** 3D Convolutional Layers -> MaxPooling3D -> Batch Normalization.
* **Regularization:** L2 Regularization and Dropout (0.4) to mitigate overfitting.
* **Output:** Sigmoid activation for binary classification (Normal vs. Abnormal).

# Performance & Challenges
The "Small Data" Challenge
* **Training Accuracy:** ~100% (Model successfully learned features).
* **Validation Accuracy:** ~52-60% (Limited generalization).

**Analysis:**
The divergence between training and validation performance highlights a classic **data bottleneck**. While the 3D pipeline functions correctly, the dataset size (140 samples) is insufficient for a deep 3D CNN to generalize robustly against unseen anatomical variations. To achieve clinical viability, this architecture would require a dataset order of magnitude larger (1k-10k scans).

# Visualizations
<img width="389" height="411" alt="Slice" src="https://github.com/user-attachments/assets/ec5083b0-6902-4867-a32a-0745059bd7c3" />
<img width="366" height="103" alt="Performence" src="https://github.com/user-attachments/assets/0e724ca3-834d-4132-aba1-c4f2c979e6db" />
<img width="1233" height="470" alt="Graph" src="https://github.com/user-attachments/assets/a1c2c4d1-bf12-427d-8243-b7fa6c5c4b7b" />

