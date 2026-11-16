# Signal Reconstruction Using Autoencoders and Deep Learning

A comparative study of traditional signal-processing techniques, Machine Learning, and Deep Learning approaches for denoising and reconstructing ECG signals.

## Overview

Electrocardiogram (ECG) signals are often contaminated by noise, which can affect the quality of signal analysis. This project explores different techniques to remove noise and reconstruct clean ECG signals.

The project compares traditional filtering methods with Machine Learning and Deep Learning models using the **MIT-BIH Arrhythmia Database**.

## Objectives

- Load and preprocess real-world ECG signals.
- Introduce Gaussian noise into clean ECG signals.
- Apply traditional signal-processing techniques.
- Implement Machine Learning and Deep Learning models for signal reconstruction.
- Compare reconstruction performance using quantitative evaluation metrics.

## Dataset

- **Dataset:** MIT-BIH Arrhythmia Database
- **Source:** [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- **Record used:** Record 100
- **Sampling frequency:** 360 Hz
- **Signal segmentation:** 512 samples per segment with 50% overlap.
- **Noise:** Gaussian noise with standard deviation 0.3.

## Methodology

The project implements and compares the following approaches:

### 1. Traditional Signal Processing

- **Butterworth Filter:** A fourth-order bandpass filter with cutoff frequencies of 0.5–40 Hz.
- **Wavelet Denoising:** Daubechies-4 (db4) wavelet decomposition with soft thresholding.

### 2. Machine Learning

- **Random Forest Regressor:** Used to learn the relationship between noisy and clean ECG signal segments.

### 3. Deep Learning

- **1D Convolutional Autoencoder:** A neural network architecture designed to learn compressed signal representations and reconstruct denoised ECG signals.

## Evaluation Metrics

The reconstruction performance is evaluated using:

- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **SNR:** Signal-to-Noise Ratio
- **Correlation:** Correlation between reconstructed and reference signals

## Results

The following results were obtained during the experiments:

| Method | RMSE | MAE | SNR | Correlation |
|---|---:|---:|---:|---:|
| Noisy ECG | 0.3001 | 0.2395 | 11.0625 | 0.9630 |
| Butterworth Filter | 0.7269 | 0.4375 | 3.3771 | 0.7975 |
| Wavelet Denoising | 0.2169 | 0.1317 | 13.8825 | 0.9811 |
| Random Forest | 0.8224 | 0.4103 | 2.3056 | 0.7765 |
| **Autoencoder** | **0.1137** | **0.0861** | **19.4880** | **0.9945** |

### Key Findings

- The 1D Convolutional Autoencoder achieved the best reconstruction performance among the implemented methods.
- Wavelet denoising performed best among the traditional signal-processing techniques.
- Butterworth filtering and Random Forest did not outperform the noisy baseline under the current experimental settings.

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- PyWavelets
- Scikit-learn
- PyTorch
- Streamlit
- WFDB

## Project Structure

```text
Signal-Reconstruction/
│
├── app.py
├── check_setup.py
├── inspect_dataset.py
├── requirements.txt
│
├── dataset/
│   ├── record_100.npz
│   ├── processed_ecg.npz
│   ├
