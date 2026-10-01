import os
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Voice Classifier App", page_icon="🎙️", layout="wide"
)

st.title("🎙️ Voice & Acoustic Feature Classifier")
st.write(
    "Upload real audio files or tabular acoustic data to run predictions."
)

# 1. Load the Model
MODEL_PATH = "svm_model.joblib"  # Falls back to model.pkl if not found
if not os.path.exists(MODEL_PATH) and os.path.exists("model.pkl"):
    MODEL_PATH = "model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    st.sidebar.success(f"Loaded model: `{MODEL_PATH}`")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# List of 20 features expected by the model
FEATURE_NAMES = [
    "meanfreq",
    "sd",
    "median",
    "Q25",
    "Q75",
    "IQR",
    "skew",
    "kurt",
    "sp.ent",
    "sfm",
    "mode",
    "centroid",
    "meanfun",
    "minfun",
    "maxfun",
    "meandom",
    "mindom",
    "maxdom",
    "dfrange",
    "modindx",
]


# Feature Extraction Helper for Raw Audio
def extract_audio_features(audio_path):
    import librosa

    y, sr = librosa.load(audio_path, sr=None)

    # Spectral centroid stats
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    meanfreq = np.mean(spec_cent) / 1000
    sd = np.std(spec_cent) / 1000
    median = np.median(spec_cent) / 1000
    Q25 = np.percentile(spec_cent, 25) / 1000
    Q75 = np.percentile(spec_cent, 75) / 1000
    IQR = Q75 - Q25
    centroid = meanfreq

    # Signal distribution
    skew = float(stats.skew(y))
    kurt = float(stats.kurtosis(y))

    # Flatness & Mode
    spec_flat = librosa.feature.spectral_flatness(y=y)
    sp_ent = np.mean(spec_flat)
    sfm = np.mean(spec_flat)
    mode = float(stats.mode(spec_cent, axis=None).mode[0]) / 1000

    # Fundamental frequency stats
    f0, _, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    f0_clean = f0[~np.isnan(f0)] / 1000 if f0 is not None else [0]

    meanfun = np.mean(f0_clean) if len(f0_clean) > 0 else 0
    minfun = np.min(f0_clean) if len(f0_clean) > 0 else 0
    maxfun = np.max(f0_clean) if len(f0_clean) > 0 else 0

    # Rolloff & dominant frequency stats
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    meandom = np.mean(rolloff) / 1000
    mindom = np.min(rolloff) / 1000
    maxdom = np.max(rolloff) / 1000
    dfrange = maxdom - mindom
    modindx = np.mean(np.abs(np.diff(y)))

    return np.array(
        [
            [
                meanfreq,
                sd,
                median,
                Q25,
                Q75,
                IQR,
                skew,
                kurt,
                sp_ent,
                sfm,
                mode,
                centroid,
                meanfun,
                minfun,
                maxfun,
                meandom,
                mindom,
                maxdom,
                dfrange,
                modindx,
            ]
        ]
    )


# App Tabs Interface
tab1, tab2, tab3 = st.tabs(
    ["📁 Batch CSV Input", "🎙️ Raw Audio (.wav)", "🎛️ Manual Parameters"]
)

# TAB 1: CSV Prediction
with tab1:
    st.subheader("Upload CSV File with Acoustic Features")
    uploaded_csv = st.file_uploader("Choose a CSV file", type=["csv"], key="csv")

    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.write("### Data Preview", df.head())

        if st.button("Run Batch Prediction", type="primary"):
            missing = [col for col in FEATURE_NAMES if col not in df.columns]
            if missing:
                st.error(f"Missing expected columns: {', '.join(missing)}")
            else:
                X = df[FEATURE_NAMES]
                df["Prediction"] = model.predict(X)
                st.success("Batch prediction completed!")
                st.dataframe(df[["Prediction"] + FEATURE_NAMES])

                # Download button
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Output CSV",
                    csv_out,
                    "predictions.csv",
                    "text/csv",
                )

# TAB 2: Audio File Prediction
with tab2:
    st.subheader("Upload Audio File")
    audio_file = st.file_uploader(
        "Upload a .wav or .mp3 recording", type=["wav", "mp3"], key="audio"
    )

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Extract Features & Predict", type="primary"):
            with st.spinner("Extracting features using librosa..."):
                # Save temp file
                temp_filename = "temp_audio.wav"
                with open(temp_filename, "wb") as f:
                    f.write(audio_file.getbuffer())

                try:
                    features = extract_audio_features(temp_filename)
                    pred = model.predict(features)[0]
                    st.success(f"**Predicted Result:** {pred}")

                    # Display extracted features table
                    feat_df = pd.DataFrame(features, columns=FEATURE_NAMES)
                    st.write("### Extracted Acoustic Features:", feat_df)

                except Exception as err:
                    st.error(f"Failed to process audio file: {err}")
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)

# TAB 3: Manual Slider / Input Field Prediction
with tab3:
    st.subheader("Manual Feature Input")
    cols = st.columns(3)
    manual_inputs = {}

    for i, feature in enumerate(FEATURE_NAMES):
        with cols[i % 3]:
            manual_inputs[feature] = st.number_input(
                label=feature, value=0.00000, format="%.5f"
            )

    if st.button("Predict Single Sample", type="primary"):
        input_array = np.array([list(manual_inputs.values())])
        pred = model.predict(input_array)[0]
        st.success(f"**Prediction:** {pred}")
