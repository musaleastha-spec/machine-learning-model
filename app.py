import os
import tempfile
import joblib
import numpy as np
import scipy.stats as stats
import librosa
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Voice Intelligence AI Studio", 
    page_icon="🎙️", 
    layout="centered"
)

# 2. Inject Custom CSS & Styled HTML Header Component
st.markdown("""
    <style>
        /* Base page tweaks */
        .stApp {
            background-color: #0f172a;
        }
        /* Custom Header Card */
        .custom-card {
            background-color: #1e293b;
            padding: 25px 30px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            border: 1px solid #334155;
            margin-bottom: 25px;
        }
        .custom-card h1 {
            color: #6366f1;
            font-size: 28px;
            margin: 0 0 8px 0;
            font-weight: 700;
        }
        .custom-card p {
            color: #94a3b8;
            font-size: 14px;
            margin: 0;
        }
        /* Status Badge */
        .badge {
            display: inline-block;
            background: #0284c7;
            color: white;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>

    <div class="custom-card">
        <h1>🎙️ Voice Intelligence Studio</h1>
        <p>Classify audio files or raw acoustic feature vectors in real time using SVM Machine Learning.</p>
        <div class="badge">Model Status: Active</div>
    </div>
""", unsafe_allow_html=True)

# 3. Model Loader
@st.cache_resource
def load_model():
    try:
        return joblib.load("svm_model.joblib")
    except Exception:
        return joblib.load("model.pkl")

model = load_model()

# 4. Feature Extraction Logic (20 Features)
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    meanfreq = np.mean(spec_cent) / 1000
    sd = np.std(spec_cent) / 1000
    median = np.median(spec_cent) / 1000
    Q25 = np.percentile(spec_cent, 25) / 1000
    Q75 = np.percentile(spec_cent, 75) / 1000
    IQR = Q75 - Q25
    centroid = meanfreq

    skew = float(stats.skew(y))
    kurt = float(stats.kurtosis(y))

    spec_flat = librosa.feature.spectral_flatness(y=y)
    sp_ent = float(np.mean(spec_flat))
    sfm = sp_ent
    mode = float(stats.mode(spec_cent, axis=None).mode[0]) / 1000

    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_clean = f0[~np.isnan(f0)] / 1000 if f0 is not None else [0]

    meanfun = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
    minfun = float(np.min(f0_clean)) if len(f0_clean) > 0 else 0.0
    maxfun = float(np.max(f0_clean)) if len(f0_clean) > 0 else 0.0

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    meandom = float(np.mean(rolloff)) / 1000
    mindom = float(np.min(rolloff)) / 1000
    maxdom = float(np.max(rolloff)) / 1000
    dfrange = maxdom - mindom
    modindx = float(np.mean(np.abs(np.diff(y))))

    return np.array([[
        meanfreq, sd, median, Q25, Q75, IQR, skew, kurt,
        sp_ent, sfm, mode, centroid, meanfun, minfun, maxfun,
        meandom, mindom, maxdom, dfrange, modindx
    ]])

# 5. Interface Tabs
tab1, tab2 = st.tabs(["🎵 Audio Upload (.wav/.mp3)", "🎛️ Manual Parameters (20 Features)"])

# TAB 1: File Upload
with tab1:
    st.markdown("### Upload Audio Recording")
    uploaded_file = st.file_uploader("Select or drop a file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("Analyze Audio File", type="primary"):
            with st.spinner("Processing audio with librosa..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    features = extract_audio_features(tmp_path)
                    prediction = model.predict(features)[0]
                    
                    # Styled HTML Result Card
                    st.markdown(f"""
                        <div style="background-color: #0f172a; padding: 15px 20px; border-radius: 10px; border: 1px solid #38bdf8; margin-top: 15px;">
                            <h4 style="color: #94a3b8; margin: 0 0 5px 0;">Classification Output:</h4>
                            <h2 style="color: #38bdf8; margin: 0;">{prediction}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

# TAB 2: Feature Parameters Input Grid
with tab2:
    st.markdown("### Acoustic Feature Vector Input")
    
    if st.button("⚡ Fill Demo Values"):
        st.session_state["demo_filled"] = True

    features_list = [
        "meanfreq", "sd", "median", "Q25", "Q75", "IQR", "skew", "kurt",
        "sp.ent", "sfm", "mode", "centroid", "meanfun", "minfun", "maxfun",
        "meandom", "mindom", "maxdom", "dfrange", "modindx"
    ]
    
    demo_defaults = [
        0.18102, 0.05580, 0.18801, 0.14120, 0.22482, 0.08362, 1.41150, 4.92130,
        0.92310, 0.44210, 0.18320, 0.18102, 0.11720, 0.01576, 0.27586, 0.73291,
        0.00781, 5.27344, 5.26563, 0.13411
    ]

    use_demo = st.session_state.get("demo_filled", False)
    
    cols = st.columns(4)
    user_inputs = []
    
    for idx, feature in enumerate(features_list):
        col = cols[idx % 4]
        default_val = demo_defaults[idx] if use_demo else 0.0
        val = col.number_input(feature, value=default_val, format="%.5f")
        user_inputs.append(val)

    if st.button("Predict From Parameters", type="primary"):
        input_data = np.array(user_inputs).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        
        st.markdown(f"""
            <div style="background-color: #0f172a; padding: 15px 20px; border-radius: 10px; border: 1px solid #38bdf8; margin-top: 15px;">
                <h4 style="color: #94a3b8; margin: 0 0 5px 0;">Classification Output:</h4>
                <h2 style="color: #38bdf8; margin: 0;">{prediction}</h2>
            </div>
        """, unsafe_allow_html=True)
