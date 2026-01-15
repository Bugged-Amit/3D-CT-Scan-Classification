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
