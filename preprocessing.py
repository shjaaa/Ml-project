import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("Starting ECG preprocessing...")

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load original ECG signal
data = np.load(DATA_DIR / "record_100.npz")
signal = data["signal"]
fs = int(data["fs"])

print("Original signal loaded:", len(signal), "samples")

# Step 1: Normalize using Z-score normalization
mean = np.mean(signal)
std = np.std(signal)

normalized_signal = (signal - mean) / std

print("Signal normalization completed!")

# Step 2: Divide signal into segments
window_size = 512
stride = 256

segments = []

for i in range(0, len(normalized_signal) - window_size + 1, stride):
    segment = normalized_signal[i:i + window_size]
    segments.append(segment)

segments = np.array(segments, dtype=np.float32)

print("Segmentation completed!")
print("Number of segments:", len(segments))
print("Samples per segment:", window_size)

# Step 3: Save processed data
np.savez(
    DATA_DIR / "processed_ecg.npz",
    signal=normalized_signal,
    segments=segments,
    fs=fs
)

print("Processed data saved successfully!")

# Step 4: Plot one segment
plt.figure(figsize=(12, 4))
plt.plot(np.arange(window_size) / fs, segments[0])
plt.title("Normalized ECG - One Segment")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")
plt.grid()
plt.tight_layout()
plt.show()