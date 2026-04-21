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
    if lottie_heart:
        st_lottie(lottie_heart, height=150, key="heart_sidebar")
    
    st.markdown("<h2 style='text-align: center; letter-spacing: 2px;'>HEARTLYTIC <span style='var(--text-color);font-weight: bold;'>PRO</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #888;'>Professional Edition</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
        <div class="sidebar-card">
            <span class="card-title">🧠 Intelligence Engine</span>
            <p style='font-size: 0.85rem; line-height: 1.4;'>
                Menggunakan algoritma <b>Logistic Regression</b> yang telah divalidasi untuk mendeteksi anomali risiko jantung.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📖 Panduan Analisis"):
        st.markdown("""
            <div style='font-size: 0.85rem;'>
            1. Isi <b>Indikator Klinis</b> sesuai hasil laboratorium.<br><br>
            2. Lengkapi <b>Profil Gaya Hidup</b>.<br><br>
            3. Sistem akan menghitung skor probabilitas secara real-time.
            </div>
        """, unsafe_allow_html=True)
    
    st.caption("© 2026 Heartlytic | Portofolio Data Science")

# --- 4. HEADER UTAMA ---
col_head1, col_head2 = st.columns([5, 1])
with col_head1:
    st.markdown("<h1 class='main-title'>Heart Attack Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: #555;'>Analisis Risiko Kardiovaskular Berbasis Kecerdasan Buatan</p>", unsafe_allow_html=True)
with col_head2:
    st.write("")
    st.markdown("""<div style="background-color: #d4edda; color: #155724; padding: 10px; border-radius: 10px; text-align: center; font-weight: bold; border: 1px solid #c3e6cb;">REAL TIME</div>""", unsafe_allow_html=True)

st.divider()

# --- 5. FORM INPUT (BAHASA INDONESIA) ---
with st.form("prediction_form"):
    tab1, tab2 = st.tabs(["🩺 Data Klinis (Medis)", "🏃 Gaya Hidup (Harian)"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Umur (Tahun)", 1, 120, 35)
            gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            sys_bp = st.number_input("Tekanan Darah Sistolik (mmHg)", 80, 250, 120)
            heart_rate = st.number_input("Detak Jantung (BPM)", 40, 200, 72)
        with c2:
            chol = st.number_input("Total Kolesterol (mg/dL)", 100, 500, 190)
            trig = st.number_input("Trigliserida (mg/dL)", 50, 600, 150)
            dia_bp = st.number_input("Tekanan Darah Diastolik (mmHg)", 40, 150, 80)
            
    with tab2:
        # Layout Utama Gaya Hidup
        col_main1, col_main2 = st.columns(2, gap="large")

        with col_main1:
            # Grup 1: Durasi & Istirahat
            st.markdown("<p style='font-weight: 700; color: #FF4B4B; margin-bottom: 15px;'> MANAJEMEN WAKTU</p>", unsafe_allow_html=True)
            
            # durasi olahraga
            ex_hours = st.number_input(
                "Olahraga per Minggu (Jam)",
                min_value=0.0, max_value=168.0, value=3.0, step=0.5,
                help="Masukkan total durasi aktivitas fisik dalam satu minggu."
            )
            
            # Select Slider untuk tidur (Lebih interaktif dari number_input)
            sleep = st.select_slider(
                "Durasi Tidur per Hari (Jam)",
                options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                value=7,
                help="Durasi tidur harian yang disarankan adalah 7-9 jam."
            )
            
            # Feedback Real-time berdasarkan input tidur
            if sleep < 6:
                st.caption("⚠️ Durasi tidur kurang dari 6 jam dapat meningkatkan tingkat stres jantung.")
            elif sleep >= 7:
                st.caption("✅ Durasi tidur Anda sudah dalam rentang ideal.")

        with col_main2:
            # Grup 2: Kondisi Kesehatan
            st.markdown("<p style='font-weight: 700; color: #FF4B4B; margin-bottom: 15px;'> PROFIL KESEHATAN</p>", unsafe_allow_html=True)
            
            # Menggunakan Segmented Control (Tampilan lebih modern)
            diabetes = st.segmented_control(
                "Riwayat Diabetes",
                options=["Tidak", "Ya"],
                default="Tidak",
                help="Pilih 'Ya' jika Anda memiliki diagnosis diabetes dari dokter."
            )
            
            st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
            
            # Obesitas
            bmi_input = st.number_input(
                "Indeks Massa Tubuh (BMI)",
                min_value=10.0, max_value=60.0, value=24.0, step=0.1,
                help="Masukkan nilai BMI Anda (Berat Badan / Tinggi Badan^2)."
            )

            obesity_status = 1 if bmi_input > 30 else 0

            
            # Feedback visual singkat mengenai BMI
            if bmi_input > 30:
                st.warning(f"Status: Obesitas. BMI {bmi_input} berada di atas ambang batas normal.")
            elif bmi_input < 18.5:
                st.info(f"Status: Berat badan kurang. BMI: {bmi_input}")
            else:
                st.success(f"Status: Berat badan ideal. BMI: {bmi_input}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

    submit = st.form_submit_button("ANALISIS RISIKO SEKARANG")

# --- 6. LOGIKA VALIDASI & OUTPUT ---
if submit:
    # Pengecekan Error
    errors = []
    if sys_bp > 250: errors.append(f"Tekanan Sistolik ({sys_bp}) terlalu tinggi.")
    if chol > 500: errors.append(f"Kolesterol ({chol}) terlalu tinggi.")
    if dia_bp > 150: errors.append(f"Tekanan Diastolik ({dia_bp}) terlalu tinggi.")
    
    if errors:
        st.error("❌ ANALISIS DIBATALKAN: Mohon perbaiki data.")
        for e in errors: st.warning(e)
        st.stop() 

    else:
        if model:
            # Mapping data ke fitur model
            input_data = {feat: 0 for feat in all_features}
            input_data.update({
                'Age': age, 'Sex_Male': 1 if gender == "Laki-laki" else 0,
                'Systolic_BP': sys_bp, 'Diastolic_BP': dia_bp, 'Cholesterol': chol,
                'Triglycerides': trig, 'Heart Rate': heart_rate, 'Sleep Hours Per Day': sleep,
                'Exercise Hours Per Week': ex_hours, 'Diabetes': 1 if diabetes == "Ya" else 0,
                'Obesity': obesity_status, 'Income': 150000, 'BMI': bmi_input, 'Stress Level': 5
            })

            df_input = pd.DataFrame([input_data])[all_features]
            prediction = model.predict(df_input)[0]
            probability = model.predict_proba(df_input)[0][1]

            st.divider()
            st.subheader("📊 Hasil Analisis Prediktif")
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