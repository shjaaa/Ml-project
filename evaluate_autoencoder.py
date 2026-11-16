import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from autoencoder import ECGAutoencoder

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = np.load(DATA_DIR / "noisy_ecg.npz")

X = data["noisy_segments"]
y = data["clean_segments"]

# Same test split as training
split = int(0.8 * len(X))

X_test = X[split:]
y_test = y[split:]

# Load trained model
model = ECGAutoencoder().to(device)

model.load_state_dict(
    torch.load(
        DATA_DIR / "ecg_autoencoder.pth",
        map_location=device
    )
)

model.eval()

# Predict
with torch.no_grad():

    test_tensor = torch.tensor(
        X_test, dtype=torch.float32
    ).unsqueeze(1).to(device)

    predictions = model(test_tensor)

    predictions = predictions.cpu().numpy().squeeze(1)

# Calculate metrics
error = y_test - predictions

rmse = np.sqrt(np.mean(error ** 2))
mae = np.mean(np.abs(error))

snr = 10 * np.log10(
    np.sum(y_test ** 2) /
    (np.sum(error ** 2) + 1e-12)
)

correlation = np.corrcoef(
    y_test.flatten(),
    predictions.flatten()
)[0, 1]

print("\nAUTOENCODER RESULTS")
print("RMSE:", round(rmse, 4))
print("MAE:", round(mae, 4))
print("SNR:", round(snr, 4))
print("Correlation:", round(correlation, 4))

# Plot reconstruction
time = np.arange(512) / 360

plt.figure(figsize=(12, 5))

plt.plot(time, y_test[0], label="Clean ECG")
plt.plot(time, X_test[0], label="Noisy ECG", alpha=0.4)
plt.plot(time, predictions[0], label="Autoencoder Output")

plt.title("ECG Reconstruction Using 1D Autoencoder")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()