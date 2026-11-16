import numpy as np
from pathlib import Path

print("Starting performance evaluation...")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"

# Load clean and noisy signals
noisy_data = np.load(DATA_DIR / "noisy_ecg.npz")
butter_data = np.load(DATA_DIR / "butterworth_ecg.npz")
wavelet_data = np.load(DATA_DIR / "wavelet_ecg.npz")

clean = noisy_data["clean_segments"].flatten()
noisy = noisy_data["noisy_segments"].flatten()

butter = butter_data["filtered_segments"].flatten()
wavelet = wavelet_data["wavelet_segments"].flatten()


# Evaluation function
def evaluate(reference, reconstructed):

    error = reference - reconstructed

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    snr = 10 * np.log10(
        np.sum(reference ** 2) /
        (np.sum(error ** 2) + 1e-12)
    )

    correlation = np.corrcoef(
        reference, reconstructed
    )[0, 1]

    return rmse, mae, snr, correlation


# Evaluate all methods
methods = {
    "Noisy ECG": noisy,
    "Butterworth": butter,
    "Wavelet": wavelet
}

# Calculate noisy ECG baseline SNR
_, _, noisy_snr, _ = evaluate(clean, noisy)

print("\nPERFORMANCE COMPARISON")
print("-" * 75)
print(f"{'Method':<15} {'RMSE':<12} {'MAE':<12} {'SNR':<12} {'Correlation':<12}")
print("-" * 75)

for name, signal in methods.items():

    rmse, mae, snr, corr = evaluate(clean, signal)

    improvement = snr - noisy_snr

    print(
        f"{name:<15} {rmse:<12.4f} {mae:<12.4f} "
        f"{snr:<12.4f} {corr:<12.4f}"
    )

    print(f"  SNR improvement: {improvement:.4f} dB")

print("-" * 75)
print("Evaluation completed!")