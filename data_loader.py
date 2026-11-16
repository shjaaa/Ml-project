import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("Loading MIT-BIH ECG dataset...")

# Project directories
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
DATA_DIR.mkdir(exist_ok=True)

# Load record 100
record = wfdb.rdrecord("100", pn_dir="mitdb")

# Extract ECG signal (first channel)
signal = record.p_signal[:, 0].astype(np.float32)
fs = record.fs

print("Dataset loaded successfully!")
print("Signal length:", len(signal))
print("Sampling frequency:", fs, "Hz")
print("Duration:", round(len(signal) / fs, 2), "seconds")

# Save the signal
np.savez(DATA_DIR / "record_100.npz", signal=signal, fs=fs)

print("Dataset saved in:", DATA_DIR)

# Plot first 5 seconds
plt.figure(figsize=(12, 4))
plt.plot(np.arange(5 * fs) / fs, signal[:5 * fs])
plt.title("MIT-BIH ECG Signal - Record 100")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid()
plt.tight_layout()
plt.show()