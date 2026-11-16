import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

print("Starting Butterworth filtering...")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"

# Load noisy ECG
data = np.load(DATA_DIR / "noisy_ecg.npz")

noisy_segments = data["noisy_segments"]
clean_segments = data["clean_segments"]
fs = int(data["fs"])

# Design Butterworth bandpass filter
low_cutoff = 0.5
high_cutoff = 40

order = 4

b, a = butter(
    order,
    [low_cutoff, high_cutoff],
    btype="bandpass",
    fs=fs
)

# Filter each noisy segment
filtered_segments = []

for segment in noisy_segments:
    filtered = filtfilt(b, a, segment)
    filtered_segments.append(filtered)

filtered_segments = np.array(
    filtered_segments,
    dtype=np.float32
)

# Save filtered data
np.savez(
    DATA_DIR / "butterworth_ecg.npz",
    filtered_segments=filtered_segments,
    clean_segments=clean_segments,
    fs=fs
)

print("Butterworth filtering completed!")
print("Number of filtered segments:", len(filtered_segments))

# Plot comparison
time = np.arange(512) / fs

plt.figure(figsize=(12, 5))

plt.plot(time, clean_segments[0], label="Clean ECG")
plt.plot(time, noisy_segments[0], label="Noisy ECG", alpha=0.5)
plt.plot(time, filtered_segments[0], label="Butterworth Filtered")

plt.title("ECG Reconstruction Using Butterworth Filter")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()