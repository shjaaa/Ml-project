import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

print("Starting Autoencoder training...")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
data = np.load(DATA_DIR / "noisy_ecg.npz")

X = data["noisy_segments"]
y = data["clean_segments"]

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Train-test split
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Define Autoencoder
class ECGAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                32, 16, kernel_size=2, stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                16, 1, kernel_size=2, stride=2
            )
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = ECGAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001
)

# Training
epochs = 20

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for noisy, clean in train_loader:

        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()

        output = model(noisy)
        loss = criterion(output, clean)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {average_loss:.6f}"
    )

# Save model
torch.save(
    model.state_dict(),
    DATA_DIR / "ecg_autoencoder.pth"
)

print("Autoencoder training completed!")
print("Model saved successfully!")