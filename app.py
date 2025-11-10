import streamlit as st
import pandas as pd
import joblib, pickle
from pathlib import Path

st.set_page_config(page_title="Credit Risk Prediction", page_icon="🏦", layout="centered")

# ============== LOAD ARTIFACTS ==============
@st.cache_resource
def load_artifacts():
    roots = [Path("models"), Path("../models")]
    model_path = pre_path = meta_path = None
    for r in roots:
        if (r / "xgb_credit_risk_model.pkl").exists():
            model_path = r / "xgb_credit_risk_model.pkl"
        if (r / "preprocessor.pkl").exists():
            pre_path = r / "preprocessor.pkl"
        if (r / "model_metadata.pkl").exists():
            meta_path = r / "model_metadata.pkl"

    if not model_path:
        raise FileNotFoundError("File model 'xgb_credit_risk_model.pkl' tidak ditemukan.")
    model = joblib.load(model_path)

    preprocessor = None
    if pre_path and pre_path.exists():
        preprocessor = joblib.load(pre_path)

    metadata = {}
    if meta_path and meta_path.exists():
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

    # deteksi apakah model adalah pipeline (punya 'steps' atau 'named_steps')
    is_pipeline = hasattr(model, "steps") or hasattr(model, "named_steps")
    return model, preprocessor, metadata, is_pipeline

model, preprocessor, metadata, IS_PIPELINE = load_artifacts()

def align_columns(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    feats = metadata.get("feature_names_original")
    if isinstance(feats, list) and len(feats) > 0:
        for c in feats:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[feats]
    return df

# ============== UI ==============
st.title("🏦 Credit Risk Prediction App")
st.write("Prediksi kemungkinan default pinjaman berdasarkan data peminjam.")

st.sidebar.header("Input Data Peminjam")
def user_input():
    return pd.DataFrame([{
        "person_age": st.sidebar.slider("Usia", 18, 100, 30),
        "person_income": st.sidebar.number_input("Pendapatan Tahunan ($)", 1000, 1_000_000, 50_000),
        "person_home_ownership": st.sidebar.selectbox("Kepemilikan Rumah", ["RENT","OWN","MORTGAGE","OTHER"]),
        "person_emp_length": st.sidebar.slider("Lama Bekerja (tahun)", 0, 60, 5),
        "loan_intent": st.sidebar.selectbox("Tujuan Pinjaman",
                      ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"]),
        "loan_grade": st.sidebar.selectbox("Grade Pinjaman", ["A","B","C","D","E","F","G"]),
        "loan_amnt": st.sidebar.number_input("Jumlah Pinjaman ($)", 500, 100_000, 10_000),
        "loan_int_rate": st.sidebar.slider("Suku Bunga Pinjaman (%)", 1.0, 40.0, 12.5, 0.1),
        "loan_percent_income": st.sidebar.slider("Persentase Pinjaman dari Pendapatan", 0.0, 1.0, 0.2, 0.01),
        "cb_person_default_on_file": st.sidebar.selectbox("Riwayat Default", ["Y","N"]),
        "cb_person_cred_hist_length": st.sidebar.slider("Lama Riwayat Kredit (tahun)", 0, 50, 5),
    }])

input_df = user_input()
st.subheader("Data Input Peminjam")
st.write(input_df)

# ============== PREDIKSI ==============
if st.button("Prediksi Risiko Kredit"):
    try:
        aligned = align_columns(input_df.copy(), metadata)

        if IS_PIPELINE:
            # model sudah termasuk preprocessor → JANGAN transform manual
            proba = model.predict_proba(aligned)[0]
        else:
            if preprocessor is None:
                raise RuntimeError("Preprocessor tidak ditemukan, tetapi model bukan pipeline.")
            X_proc = preprocessor.transform(aligned)  # transform SEKALI
            proba = model.predict_proba(X_proc)[0]

        pred = int(proba[1] >= 0.5)

        st.subheader("Hasil Prediksi")
        if pred == 1:
            st.error("🚨 RISIKO TINGGI: Peminjam berpotensi DEFAULT")
        else:
            st.success("✅ RISIKO RENDAH: Peminjam berpotensi BAYAR")

        st.write(f"Probabilitas Non-Default: {proba[0]:.2%}")
        st.write(f"Probabilitas Default: {proba[1]:.2%}")

        st.subheader("Distribusi Probabilitas")
        chart_df = pd.DataFrame(
            {"Kategori": ["Non-Default","Default"], "Prob": [proba[0], proba[1]]}
        ).set_index("Kategori")
        st.bar_chart(chart_df)

        # debug kecil (opsional): info tipe model
        with st.expander("Debug info"):
            st.write("Model is pipeline:", IS_PIPELINE)

    except Exception as e:
        st.error(f"Error dalam prediksi: {e}")
        with st.expander("Trace hint"):
            st.write("Jika pesan error menyebut ColumnTransformer expecting 11 features, "
                     "itu berarti input mentah harus dikasih ke pipeline (hapus transform manual).")

# ============== INFO ==============
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("App ini otomatis mendeteksi apakah file model adalah Pipeline atau estimator murni, "
                "agar tidak terjadi double-preprocess.")
