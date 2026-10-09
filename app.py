import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import re
import pandas as pd
import plotly.express as px
from datetime import datetime

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ---------------------------------------------------------
st.set_page_config(
    page_title="Global AI E-Waste Detector Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# CSS Kustom (Diperbaiki: unsafe_allow_html)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. MANAJEMEN API KEY & SIDEBAR
# ---------------------------------------------------------
api_key = ""
if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
    api_key = st.secrets["GEMINI_API_KEY"]

with st.sidebar:
    st.title("⚡ AI Core Settings")
    st.caption("Universal E-Waste Detection Engine")
    
    if api_key:
        st.success("🟢 API Key Terhubung (Secrets)")
    else:
        api_key = st.text_input("Masukkan Gemini API Key:", type="password", help="Dapatkan dari Google AI Studio")
    
    st.markdown("---")
    st.markdown("### 📋 Standar Klasifikasi PBB")
    st.info("Mengadopsi **UN Global E-Waste Monitor** (UNU/UNITAR).")
    st.markdown("---")
    st.caption("v5.1 Fixed Engine")

# ---------------------------------------------------------
# 3. CORE AI ENGINE
# ---------------------------------------------------------
def discover_active_models(key):
    active_models = []
    try:
        genai.configure(api_key=key)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                active_models.append(m.name)
    except Exception:
        pass

    priority_models = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]
    
    candidates = []
    for model in priority_models:
        if model not in candidates:
            candidates.append(model)
    for model in active_models:
        if model not in candidates:
            candidates.append(model)
            
    return candidates

def parse_json_robust(text_content):
    if not text_content:
        raise ValueError("Respon AI kosong.")
        
    cleaned = text_content.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    
    start_idx = cleaned.find('{')
    if start_idx != -1:
        json_candidate = cleaned[start_idx:]
        try:
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(json_candidate)
            return parsed
        except Exception:
            pass

    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    return json.loads(cleaned)

def analyze_ewaste_master(image, key):
    genai.configure(api_key=key)
    
    prompt = """
    Bertindaklah sebagai Pakar Pengolahan Sampah Elektronik (E-Waste Specialist) berstandar Internasional PBB.
    Analisis gambar ini dengan teliti untuk mengidentifikasi segala bentuk sampah elektronik, komponen, perangkat rumah tangga, modul IT, PCB, atau aksesoris elektronik.
    
    Hasilkan output STRICTLY dalam format JSON MURNI tanpa teks atau pendahuluan di luar JSON.

    Struktur JSON yang wajib dihasilkan:
    {
        "nama_objek": "Nama spesifik perangkat/komponen yang teridentifikasi",
        "kategori_un": "Salah satu dari 6 Kategori UN E-Waste (1. Temperature Exchange Equipment / 2. Screens & Monitors / 3. Lamps / 4. Large Equipment / 5. Small Equipment / 6. Small IT & Telecommunication)",
        "deskripsi": "Penjelasan detail mengenai kondisi, tipe, dan kegunaan asli objek",
        "tingkat_bahaya": "Sangat Tinggi / Tinggi / Sedang / Rendah",
        "skor_bahaya": 8,
        "bahan_berbahaya": ["Contoh: Timbal (Solder)", "Raksa", "Kadmium", "BFR", "CFC/Freon"],
        "potensi_logam_mulia": {
            "Emas (Au)": "Tinggi / Sedang / Rendah / Tidak Ada",
            "Perak (Ag)": "Tinggi / Sedang / Rendah / Tidak Ada",
            "Tembaga (Cu)": "Tinggi / Sedang / Rendah / Tidak Ada"
        },
        "instruksi_penanganan": [
            "Langkah 1: Pengamanan awal dan pembongkaran aman",
            "Langkah 2: Pemisahan komponen berbahaya/baterai",
            "Langkah 3: Opsi daur ulang / penyetoran ke fasilitas resmi"
        ],
        "dapat_didaur_ulang_persen": 80
    }
    """
    
    candidate_models = discover_active_models(key)
    generation_config = {
        "temperature": 0.1,
        "response_mime_type": "application/json"
    }
    
    last_error = ""

    for model_name in candidate_models:
        try:
            try:
                model = genai.GenerativeModel(model_name, generation_config=generation_config)
                res = model.generate_content([prompt, image])
            except Exception:
                model = genai.GenerativeModel(model_name)
                res = model.generate_content([prompt, image])

            if res and res.text:
                parsed_data = parse_json_robust(res.text)
                return parsed_data, model_name, None
        except Exception as e:
            last_error = str(e)
            continue

    return None, None, f"Gagal menganalisis gambar: {last_error}"

# ---------------------------------------------------------
# 4. TAMPILAN UTAMA APLIKASI
# ---------------------------------------------------------
st.markdown('<div class="main-header">⚡ Global AI E-Waste Detector Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistem Pengenal & Analisis Bahaya Sampah Elektronik Berbasis Multi-Vision AI</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Analisis E-Waste", "📊 Dashboard & Riwayat", "📖 Standar UN E-Waste"])

# --- TAB 1: ANALISIS ---
with tab1:
    col_input, col_output = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.subheader("1. Masukkan Gambar Perangkat")
        source = st.radio("Pilih Sumber Gambar:", ["Kamera Langsung 📷", "Unggah Berkas 📁"], horizontal=True)
        
        input_image = None
        if "Kamera" in source:
            cam_file = st.camera_input("Ambil Foto E-Waste")
            if cam_file:
                input_image = Image.open(cam_file)
        else:
            uploaded_file = st.file_uploader("Unggah berkas gambar (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file:
                input_image = Image.open(uploaded_file)
                
        if input_image:
            st.image(input_image, caption="Gambar Siap Dianalisis", use_container_width=True)
            analyze_btn = st.button("🚀 Jalankan Analisis AI Universal", type="primary", use_container_width=True)

    with col_output:
        st.subheader("2. Hasil Identifikasi & Analisis Bahaya")
        
        if 'analyze_btn' in locals() and analyze_btn:
            if not api_key:
                st.error("⚠️ **API Key Belum Diisi!** Masukkan Gemini API Key pada sidebar atau simpan di Streamlit Secrets.")
            elif input_image is None:
                st.warning("⚠️ **Gambar Belum Ada!** Ambil foto atau unggah berkas gambar e-waste terlebih dahulu.")
            else:
                with st.spinner("🧠 Menghubungkan ke Gemini AI & menganalisis e-waste..."):
                    data, model_used, err = analyze_ewaste_master(input_image, api_key)
                    
                    if err:
                        st.error(f"❌ {err}")
                    else:
                        st.session_state.detection_history.append({
                            "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "nama": data.get("nama_objek", "Tidak diketahui"),
                            "kategori": data.get("kategori_un", "Umum"),
                            "bahaya": data.get("tingkat_bahaya", "Sedang"),
                            "skor_bahaya": data.get("skor_bahaya", 5),
                            "daur_ulang": data.get("dapat_didaur_ulang_persen", 0),
                            "model": model_used
                        })
                        
                        st.success(f"✅ **Analisis Berhasil** (Engine: `{model_used}`)")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Perangkat Terdeteksi", data.get("nama_objek", "-"))
                        m2.metric("Tingkat Bahaya", data.get("tingkat_bahaya", "-"), delta=f"Skor {data.get('skor_bahaya', 0)}/10", delta_color="inverse")
                        m3.metric("Potensi Daur Ulang", f"{data.get('dapat_didaur_ulang_persen', 0)}%")
                        
                        st.markdown("---")
                        st.markdown(f"**📂 Kategori UN E-Waste:** `{data.get('kategori_un', '-')}`")
                        st.markdown(f"**📝 Deskripsi Objek:** {data.get('deskripsi', '-')}")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("🚨 **Bahan & Zat Berbahaya:**")
                            for bahan in data.get("bahan_berbahaya", []):
                                st.write(f"- {bahan}")
                                
                        with col_b:
                            st.markdown("💎 **Potensi Kandungan Logam Mulia:**")
                            lm = data.get("potensi_logam_mulia", {})
                            for k, v in lm.items():
                                st.write(f"- **{k}:** {v}")
                                
                        st.markdown("---")
                        st.markdown("🛠️ **Instruksi Penanganan & Daur Ulang Aman:**")
                        for idx, step in enumerate(data.get("instruksi_penanganan", []), 1):
                            st.write(f"**{idx}.** {step}")

# --- TAB 2: DASHBOARD ---
with tab2:
    st.subheader("📊 Rekapitulasi & Statistik Analisis Sesi Ini")
    
    if len(st.session_state.detection_history) > 0:
        df = pd.DataFrame(st.session_state.detection_history)
        st.dataframe(df, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_pie = px.pie(df, names="kategori", title="Distribusi Kategori UN E-Waste", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            fig_bar = px.bar(df, x="nama", y="daur_ulang", color="bahaya", title="Tingkat Kemungkinan Daur Ulang (%)")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Laporan Rekapitulasi (CSV)",
            data=csv_data,
            file_name=f"rekap_ewaste_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Belum ada riwayat analisis pada sesi ini. Lakukan identifikasi di Tab 1 untuk mengisi rekapitulasi.")

# --- TAB 3: STANDAR PBB ---
with tab3:
    st.subheader("📖 6 Kategori Standar UN Global E-Waste Monitor")
    st.markdown("""
    Berdasarkan standar **United Nations University (UNU)** dan **UNITAR**, e-waste dikelompokkan ke dalam 6 kategori utama:
    
    1. **Temperature Exchange Equipment**: Pendingin udara (AC), kulkas, freezer, heat pump.
    2. **Screens & Monitors**: TV, monitor komputer, laptop, tablet, layar LCD/OLED/CRT.
    3. **Lamps**: Lampu fluoresen, lampu neon, lampu LED, lampu HID.
    4. **Large Equipment**: Mesin cuci, pengering pakaian, kompor listrik, mesin fotokopi besar, panel surya.
    5. **Small Equipment**: Vacuum cleaner, microwave, kipas angin, toaster, kamera, alat pemotong rambut, pemutar musik.
    6. **Small IT & Telecommunication Equipment**: Handphone, Smartphone, router Wi-Fi, PC Desktop, printer kecil, kalkulator.
    """)
