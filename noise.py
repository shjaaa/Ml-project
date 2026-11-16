import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("Adding noise to ECG signal...")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"

# Load preprocessed ECG
data = np.load(DATA_DIR / "processed_ecg.npz")

clean_signal = data["signal"]
fs = int(data["fs"])

# Generate reproducible Gaussian noise
np.random.seed(42)

noise_level = 0.3
noise = np.random.normal(
    0, noise_level, len(clean_signal)
)

# Add noise to clean ECG
noisy_signal = clean_signal + noise

# Divide noisy signal into segments
window_size = 512
stride = 256

noisy_segments = []
clean_segments = []

for i in range(0, len(clean_signal) - window_size + 1, stride):

    noisy_segments.append(noisy_signal[i:i + window_size])
    clean_segments.append(clean_signal[i:i + window_size])

noisy_segments = np.array(noisy_segments, dtype=np.float32)
clean_segments = np.array(clean_segments, dtype=np.float32)

# Save paired data
np.savez(
    DATA_DIR / "noisy_ecg.npz",
    noisy_segments=noisy_segments,
    clean_segments=clean_segments,
    fs=fs
)

print("Noise addition completed!")
print("Number of segments:", len(noisy_segments))

# Compare clean and noisy ECG
plt.figure(figsize=(12, 5))

time = np.arange(512) / fs

plt.plot(time, clean_segments[0], label="Clean ECG")
plt.plot(time, noisy_segments[0], label="Noisy ECG", alpha=0.7)

plt.title("Clean vs Noisy ECG Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()