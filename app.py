
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib

from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ECG Signal Reconstruction",
    page_icon="❤️",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.title("ECG Signal Reconstruction Using Deep Learning")
st.markdown(
    "Compare traditional signal processing, machine learning "
    "and deep learning methods for ECG denoising."
)

st.markdown("""
### Project Overview

**Objective:** To reconstruct clean ECG signals from noisy
observations using traditional signal processing,
machine learning and deep learning techniques.

**Dataset:** MIT-BIH Arrhythmia Database — Record 100

**Sampling Frequency:** 360 Hz

**Segment Length:** 512 samples

**Reconstruction Techniques:**
- Butterworth Bandpass Filtering
- Wavelet Denoising
- Random Forest Regression
- 1D Convolutional Autoencoder

**Evaluation Metrics:** RMSE, MAE, SNR and Correlation
""")

st.divider()


# --------------------------------------------------
# AUTOENCODER ARCHITECTURE
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------

@st.cache_data
def load_data():

    noisy = np.load(DATA_DIR / "noisy_ecg.npz")
    butter = np.load(DATA_DIR / "butterworth_ecg.npz")
    wavelet = np.load(DATA_DIR / "wavelet_ecg.npz")

    return {
        "clean": noisy["clean_segments"],
        "noisy": noisy["noisy_segments"],
        "butterworth": butter["filtered_segments"],
        "wavelet": wavelet["wavelet_segments"]
    }


@st.cache_resource
def load_models():

    rf_model = joblib.load(
        DATA_DIR / "random_forest.pkl"
    )

    ae_model = ECGAutoencoder().to(device)

    weights = torch.load(
        DATA_DIR / "ecg_autoencoder.pth",
        map_location=device
    )

    ae_model.load_state_dict(weights)
    ae_model.eval()

    return rf_model, ae_model


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(clean, reconstructed):

    rmse = np.sqrt(
        mean_squared_error(clean, reconstructed)
    )

    mae = mean_absolute_error(clean, reconstructed)

    noise_power = np.sum(
        (clean - reconstructed) ** 2
    )

    signal_power = np.sum(clean ** 2)

    snr = 10 * np.log10(
        signal_power / (noise_power + 1e-12)
    )

    correlation = np.corrcoef(
        clean, reconstructed
    )[0, 1]

    return rmse, mae, snr, correlation


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

try:

    data = load_data()
    rf_model, ae_model = load_models()

    st.success("Datasets and trained models loaded successfully!")

    st.caption(f"Device: {device}")

    segment_no = st.slider(
        "Select ECG segment",
        min_value=0,
        max_value=len(data["clean"]) - 1,
        value=0
    )

    clean = data["clean"][segment_no]
    noisy = data["noisy"][segment_no]

    # Random Forest prediction
    rf_prediction = rf_model.predict(
        noisy.reshape(1, -1)
    )[0]

    # Autoencoder prediction
    input_tensor = torch.tensor(
        noisy,
        dtype=torch.float32
    ).reshape(1, 1, -1).to(device)

    with torch.no_grad():
        ae_prediction = ae_model(input_tensor)

    ae_prediction = (
        ae_prediction.cpu().numpy().reshape(-1)
    )

    # All reconstruction methods
    results = {
        "Clean ECG (Reference)": clean,
        "Noisy ECG": noisy,
        "Butterworth": data["butterworth"][segment_no],
        "Wavelet": data["wavelet"][segment_no],
        "Random Forest": rf_prediction,
        "Autoencoder": ae_prediction
    }

    st.subheader("ECG Signal Comparison")

    selected_methods = st.multiselect(
        "Choose signals to display",
        options=list(results.keys()),
        default=list(results.keys())
    )

    fig, ax = plt.subplots(figsize=(13, 5))

    for method in selected_methods:
        ax.plot(
            results[method],
            label=method,
            linewidth=1
        )

    ax.set_title(
        f"ECG Reconstruction - Segment {segment_no}"
    )
    ax.set_xlabel("Sample")
    ax.set_ylabel("Normalized Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------
    # PERFORMANCE COMPARISON
    # --------------------------------------------------

    st.subheader("Performance Metrics")

    metric_methods = {
        "Noisy ECG": noisy,
        "Butterworth": results["Butterworth"],
        "Wavelet": results["Wavelet"],
        "Random Forest": rf_prediction,
        "Autoencoder": ae_prediction
    }

    metric_rows = []

    for name, prediction in metric_methods.items():

        rmse, mae, snr, corr = calculate_metrics(
            clean, prediction
        )

        metric_rows.append({
            "Method": name,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "SNR (dB)": round(snr, 4),
            "Correlation": round(corr, 4)
        })

    st.dataframe(
        metric_rows,
        use_container_width=True,
        hide_index=True
    )

    
    # --------------------------------------------------
    # DOWNLOAD RECONSTRUCTED ECG
    # --------------------------------------------------

    st.subheader("Download Reconstructed Signal")

    download_method = st.selectbox(
        "Choose reconstruction method",
        [
            "Noisy ECG",
            "Butterworth",
            "Wavelet",
            "Random Forest",
            "Autoencoder"
        ]
    )

    selected_signal = metric_methods[download_method]

    csv_data = __import__("pandas").DataFrame({
        "Sample": np.arange(len(selected_signal)),
        "Reconstructed_ECG": selected_signal
    }).to_csv(index=False)

    st.download_button(
        label="Download ECG as CSV",
        data=csv_data,
        file_name=f"{download_method.lower().replace(' ', '_')}_segment_{segment_no}.csv",
        mime="text/csv"
    )

    
    # --------------------------------------------------
    # PERFORMANCE CHARTS
    # --------------------------------------------------

    st.subheader("Visual Performance Comparison")

    metrics_df = __import__("pandas").DataFrame(metric_rows)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Error Comparison (lower is better)**")

        fig1, ax1 = plt.subplots(figsize=(7, 4))

        ax1.bar(metrics_df["Method"], metrics_df["RMSE"],
                label="RMSE")
        ax1.bar(metrics_df["Method"], metrics_df["MAE"],
                label="MAE", alpha=0.7)

        ax1.set_ylabel("Error")
        ax1.tick_params(axis="x", rotation=30)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        st.markdown("**Signal Quality Comparison**")

        fig2, ax2 = plt.subplots(figsize=(7, 4))

        ax2.bar(metrics_df["Method"], metrics_df["Correlation"])

        ax2.set_ylabel("Correlation")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="x", rotation=30)
        ax2.grid(axis="y", alpha=0.3)

        st.pyplot(fig2)
        plt.close(fig2)

    
    st.markdown("**Signal-to-Noise Ratio (higher is better)**")

    fig3, ax3 = plt.subplots(figsize=(10, 4))

    ax3.bar(metrics_df["Method"], metrics_df["SNR (dB)"])

    ax3.set_ylabel("SNR (dB)")
    ax3.set_title("SNR Comparison")
    ax3.tick_params(axis="x", rotation=25)
    ax3.grid(axis="y", alpha=0.3)

    st.pyplot(fig3)
    plt.close(fig3)


    # --------------------------------------------------
    # PROJECT METHODOLOGY
    # --------------------------------------------------

    st.subheader("Project Methodology")

    with st.expander("View the complete workflow"):

        st.markdown("""
        1. **Data Acquisition:** ECG signals are obtained
           from the MIT-BIH Arrhythmia Database.

        2. **Preprocessing:** Signals are normalized and
           divided into overlapping segments.

        3. **Noise Addition:** Gaussian noise is introduced
           to simulate noisy ECG signals.

        4. **Traditional Denoising:** Butterworth filtering
           and wavelet denoising are applied.

        5. **Machine Learning:** A Random Forest regressor
           learns the mapping from noisy to clean ECG.

        6. **Deep Learning:** A 1D convolutional
           autoencoder reconstructs clean ECG signals.

        7. **Evaluation:** Methods are compared using
           RMSE, MAE, SNR and correlation.
        """)


    st.info(
        "Lower RMSE and MAE indicate less reconstruction error. "
        "Higher SNR and correlation indicate better similarity "
        "to the clean reference signal."
    )

except Exception as e:

    st.error(f"Application error: {e}")
    st.exception(e)
