import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Heartlytic Pro | Heart Attack Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS untuk tampilan Premium
st.markdown("""
    <style>
    /* Mengatur judul utama agar adaptif */
    .sidebar-card {
        background: rgba(255, 75, 75, 0.07);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 75, 75, 0.15);
        margin-bottom: 15px;
    }
    .sidebar-card p, .sidebar-card b, .sidebar-card span {
        color: var(--text-color) !important;
    }
    .description-line {
        font-family: 'Outfit', sans-serif;
        font-size: 1rem;
        font-weight: 400;
        
        /* Menggunakan warna teks sistem dengan sedikit transparansi (opsional) */
        color: var(--text-color);
        opacity: 0.8; 
        
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 10px;
        text-align: justify;
        width: 100%;
        border-top: 1px solid rgba(255, 75, 75, 0.2);
        padding-top: 10px;
    }
    .card-title {
        color: #FF4B4B !important;
        font-weight: bold;
        margin-bottom: 8px;
        display: block;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-color);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD ASSETS & MODEL ---
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

@st.cache_resource
def load_resources():
    # Model Logistic Regression v1.6.1
    model = joblib.load('model_logreg.pkl')
    features = [
        'Age', 'Cholesterol', 'Heart Rate', 'Diabetes', 'Family History', 
        'Smoking', 'Obesity', 'Alcohol Consumption', 'Exercise Hours Per Week', 
        'Previous Heart Problems', 'Medication Use', 'Stress Level', 
        'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides', 
        'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Systolic_BP', 
        'Diastolic_BP', 'Sex_Male', 'Diet_Healthy', 'Diet_Unhealthy', 
        'Continent_Asia', 'Continent_Australia', 'Continent_Europe', 
        'Continent_North America', 'Continent_South America', 'Hemisphere_Southern Hemisphere'
    ]
    return model, features

model, all_features = load_resources()
lottie_heart = load_lottie("https://lottie.host/80e72251-872c-4614-a957-30e713589b21/HqQ6h72WzZ.json")

# --- SIDEBAR ---
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    if lottie_heart:
        st_lottie(lottie_heart, height=150, key="heart_sidebar")
    
    st.markdown(
        f"""<h2 style='text-align: center; letter-spacing: 2px; color: #FF4B4B; margin-top: -80px; margin-bottom: 0px;'> HEARTLYTIC <span style='color: var(--text-color); font-weight: bold;'>PRO</span></h2>
        """, unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 0.8rem; color: #888; margin-top: -60px;'>"
        "AI Cardiac Analytics</p>", 
        unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
        <div class="sidebar-card">
            <span class="card-title">🧠 Intelligence Engine</span>
            <p style='font-size: 0.85rem; line-height: 1.4;'>
                Menggunakan algoritma <b>Logistic Regression</b> yang telah divalidasi untuk mendeteksi anomali risiko jantung.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📖 Panduan Penggunaan"):
        st.markdown("""
            <div style='font-size: 0.85rem;'>
            1. Isi <b>Indikator Klinis</b> sesuai hasil laboratorium.<br><br>
            2. Lengkapi <b>Profil Gaya Hidup</b>.<br><br>
            3. Sistem akan menghitung skor probabilitas secara real-time.
            </div>
        """, unsafe_allow_html=True)
    
    st.caption("© 2026 Heartlytic | Portofolio Data Science")

# --- HEADER UTAMA ---
st.markdown("<h1 style='text-align: left; color: var(--text-color);'>Heart Attack Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.1rem; color: #6b7280; border-left: 4px solid #FF4B4B; padding-left: 15px;'>Advanced AI-Driven Heart Attack Risk Prediction</p>", unsafe_allow_html=True)
st.divider()

# --- FORM INPUT ---
with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(["👤 Demografi", "🩺 Data Klinis", "🏃 Gaya Hidup"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Umur (Tahun)", 1, 120, 35)
        with c2:
            gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            
    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            sys_bp = st.number_input("Tekanan Darah Sistolik (mmHg)", 80, 250, 120)
            dia_bp = st.number_input("Tekanan Darah Diastolik (mmHg)", 40, 150, 80)
            chol = st.number_input("Total Kolesterol (mg/dL)", 100, 500, 190)
        with c4:
            bmi_input = st.number_input("Indeks Massa Tubuh (BMI)", 10.0, 60.0, 24.0, step=0.1)
            # --- LOGIKA FEEDBACK BMI ---
            if bmi_input < 18.5:
                st.info(f"🦴 Status: **Berat Badan Kurang**")
                obesity_status = 0
            elif 18.5 <= bmi_input <= 25.0:
                st.success(f"✅ Status: **Berat Badan Ideal**")
                obesity_status = 0
            elif 25.1 <= bmi_input <= 29.9:
                st.warning(f"⚠️ Status: **Kelebihan Berat Badan**")
                obesity_status = 0
            else:
                st.error(f"🚨 Status: **Obesitas**")
                obesity_status = 1
            
            diabetes = st.segmented_control("Riwayat Diabetes", options=["Tidak", "Ya"], default="Tidak")
            
            
    with tab3:
        c5, c6 = st.columns(2)
        with c5:
            ex_hours = st.number_input("Olahraga per Minggu (Jam)", 0.0, 168.0, 3.0, step=0.5)
            sleep = st.select_slider("Durasi Tidur per Hari (Jam)", options=list(range(1, 13)), value=7)
        with c6:
            smoking = st.segmented_control("Kebiasaan Merokok", options=["Tidak", "Ya"], default="Tidak")
            alcohol = st.segmented_control("Konsumsi Alkohol", options=["Tidak", "Ya"], default="Tidak")

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("ANALISIS RISIKO SEKARANG")

# --- LOGIKA VALIDASI & OUTPUT ---
if submit:
    #_Validasi Input Kritis_
    errors = []
    if sys_bp > 250 or sys_bp < 80:
        errors.append("Tekanan Darah Sistolik tidak valid (Rentang: 80-250 mmHg)")
    if dia_bp > 150 or dia_bp < 40:
        errors.append("Tekanan Darah Diastolik tidak valid (Rentang: 40-150 mmHg)")
    if chol > 500 or chol < 100:
        errors.append("Kadar Kolesterol tidak valid (Rentang: 100-500 mg/dL)")
    #_Cek Apakah Ada Error
    if errors:
        for err in errors:
            st.error(f"❌ {err}")
        st.stop()
    #_Jika Valid, Jalankan Model
    if model:
        # Default value untuk fitur yang dihapus/tidak diinput user
        input_data = {feat: 0 for feat in all_features}
        
        # Mapping input ke fitur model
        input_data.update({
            'Age': age, 
            'Sex_Male': 1 if gender == "Laki-laki" else 0,
            'Systolic_BP': sys_bp, 
            'Diastolic_BP': dia_bp, 
            'Cholesterol': chol,
            'BMI': bmi_input,
            'Diabetes': 1 if diabetes == "Ya" else 0,
            'Obesity': obesity_status,
            'Exercise Hours Per Week': ex_hours, 
            'Sleep Hours Per Day': sleep,
            'Smoking': 1 if smoking == "Ya" else 0,
            'Alcohol Consumption': 1 if alcohol == "Ya" else 0,
            # Placeholder data konstan untuk fitur model lainnya
            'Heart Rate': 72, 
            'Triglycerides': 150,
            'Stress Level': 5,
            'Income': 100000
        })

        df_input = pd.DataFrame([input_data])[all_features]
        probability = model.predict_proba(df_input)[0][1]
        prediction = 1 if probability > 0.5 else 0

        st.subheader("📊 Hasil Analisis")
        r1, r2, r3 = st.columns(3)
        with r1: st.metric("Status Risiko", "TINGGI" if prediction == 1 else "RENDAH")
        with r2: st.metric("Skor Probabilitas", f"{probability:.1%}")
        with r3: 
            urg = "Aman" if probability < 0.3 else "Waspada" if probability < 0.7 else "Kritis"
            st.metric("Level Urgensi", urg)
        
        st.progress(probability)
        if prediction == 1:
            st.error("🚨 Segera konsultasikan hasil ini dengan dokter spesialis jantung.")
        else:
            st.success("✅ Parameter Anda saat ini menunjukkan profil risiko rendah.")
