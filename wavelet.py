import numpy as np
import pywt
import matplotlib.pyplot as plt
from pathlib import Path

print("Starting Wavelet Denoising...")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"

# Load noisy ECG
data = np.load(DATA_DIR / "noisy_ecg.npz")

noisy_segments = data["noisy_segments"]
clean_segments = data["clean_segments"]
fs = int(data["fs"])

# Wavelet denoising function
def wavelet_denoise(signal):

    # Wavelet decomposition
    coeffs = pywt.wavedec(signal, "db4", level=4)

    # Estimate noise level
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Calculate threshold
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply soft thresholding to detail coefficients
    denoised_coeffs = [coeffs[0]]

    for detail in coeffs[1:]:
        denoised = pywt.threshold(
            detail, threshold, mode="soft"
        )
        denoised_coeffs.append(denoised)

    # Reconstruct signal
    reconstructed = pywt.waverec(
        denoised_coeffs, "db4"
    )

    return reconstructed[:len(signal)]


# Apply denoising to all segments
wavelet_segments = []

for segment in noisy_segments:
    result = wavelet_denoise(segment)
    wavelet_segments.append(result)

wavelet_segments = np.array(
    wavelet_segments, dtype=np.float32
)

# Save results
np.savez(
    DATA_DIR / "wavelet_ecg.npz",
    wavelet_segments=wavelet_segments,
    clean_segments=clean_segments,
    fs=fs
)

print("Wavelet denoising completed!")
print("Number of processed segments:", len(wavelet_segments))

# Plot comparison
time = np.arange(512) / fs

plt.figure(figsize=(12, 5))

plt.plot(time, clean_segments[0], label="Clean ECG")
plt.plot(time, noisy_segments[0], label="Noisy ECG", alpha=0.4)
plt.plot(time, wavelet_segments[0], label="Wavelet Denoised")

plt.title("ECG Reconstruction Using Wavelet Denoising")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()