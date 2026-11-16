import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path

print("Starting unified model comparison...")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"

# Load datasets
noisy_data = np.load(DATA_DIR / "noisy_ecg.npz")
butter_data = np.load(DATA_DIR / "butterworth_ecg.npz")
wavelet_data = np.load(DATA_DIR / "wavelet_ecg.npz")

X = noisy_data["noisy_segments"]
y = noisy_data["clean_segments"]

# Use the same test segments for every method
split = int(0.8 * len(X))

X_test = X[split:]
y_test = y[split:]

butter = butter_data["filtered_segments"][split:]
wavelet = wavelet_data["wavelet_segments"][split:]

# Load Random Forest
rf_model = joblib.load(DATA_DIR / "random_forest.pkl")
rf_predictions = rf_model.predict(X_test)


# Define Autoencoder
class ECGAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 2, stride=2)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Load Autoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = ECGAutoencoder().to(device)

autoencoder.load_state_dict(
    torch.load(
        DATA_DIR / "ecg_autoencoder.pth",
        map_location=device
    )
)

autoencoder.eval()

with torch.no_grad():
    input_tensor = torch.tensor(
        X_test, dtype=torch.float32
    ).unsqueeze(1).to(device)

    ae_predictions = autoencoder(input_tensor)
    ae_predictions = ae_predictions.cpu().numpy().squeeze(1)


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
        reference.flatten(),
        reconstructed.flatten()
    )[0, 1]

    return rmse, mae, snr, correlation


# Compare methods
methods = {
    "Noisy ECG": X_test,
    "Butterworth": butter,
    "Wavelet": wavelet,
    "Random Forest": rf_predictions,
    "Autoencoder": ae_predictions
}

print("\nFINAL PERFORMANCE COMPARISON")
print("-" * 75)
print(f"{'Method':<16} {'RMSE':<12} {'MAE':<12} {'SNR':<12} {'Correlation':<12}")
print("-" * 75)

for name, prediction in methods.items():

    rmse, mae, snr, corr = evaluate(y_test, prediction)

    print(
        f"{name:<16} {rmse:<12.4f} {mae:<12.4f} "
        f"{snr:<12.4f} {corr:<12.4f}"
    )

print("-" * 75)
print("Comparison completed!")