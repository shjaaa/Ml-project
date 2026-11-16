import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Starting Random Forest training...")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"

# Load data
data = np.load(DATA_DIR / "noisy_ecg.npz")

X = data["noisy_segments"]
y = data["clean_segments"]

print("Input shape:", X.shape)
print("Target shape:", y.shape)

# Split training and testing data
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

print("Training completed!")

# Predict
predictions = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(
    y_test.flatten(), predictions.flatten()
))

mae = mean_absolute_error(
    y_test.flatten(), predictions.flatten()
)

correlation = np.corrcoef(
    y_test.flatten(), predictions.flatten()
)[0, 1]

print("\nRANDOM FOREST RESULTS")
print("RMSE:", round(rmse, 4))
print("MAE:", round(mae, 4))
print("Correlation:", round(correlation, 4))

# Save model
import joblib

joblib.dump(model, DATA_DIR / "random_forest.pkl")

print("Random Forest model saved!")