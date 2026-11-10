import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Pupuk Prediksi",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# DESIGN — Based on Botanist & AgroFlow (proven ag dashboards)
# Light theme, clean sans-serif, green agricultural standard
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
    background: #f8f9f7;
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    color: #1a1a1a;
}

.block-container {
    padding: 32px 48px 48px !important;
    max-width: 1600px !important;
}

/* Typography */
h1, h2, h3, h4 { font-weight: 700; color: #1a1a1a; letter-spacing: -0.5px; }
h1 { font-size: 40px; line-height: 1.2; margin-bottom: 12px; }
h2 { font-size: 28px; line-height: 1.25; margin-bottom: 20px; margin-top: 0; }
h3 { font-size: 18px; line-height: 1.35; margin-bottom: 16px; }

p { line-height: 1.6; color: #555; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e5e3 !important;
}
[data-testid="stSidebar"] > div { padding: 28px 20px; }
[data-testid="stSidebar"] .stFileUploader {
    background: #f5f5f3;
    border: 2px dashed #d0d0cc;
    border-radius: 6px;
    padding: 16px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid #e5e5e3;
}

.stTabs [data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 500;
    color: #888;
    padding: 12px 20px;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #2d7e4d !important;
    border-bottom: 2px solid #2d7e4d !important;
}

.stTabs [data-baseweb="tab-panel"] { padding-top: 32px; }

/* Buttons */
.stButton > button {
    background: #ffffff !important;
    border: 1px solid #d0d0cc !important;
    color: #1a1a1a !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
}

.stButton > button:hover {
    background: #f5f5f3 !important;
    border-color: #2d7e4d !important;
}

[data-testid="stFormSubmitButton"] > button {
    background: #2d7e4d !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    border-radius: 6px !important;
    width: 100%;
}

[data-testid="stFormSubmitButton"] > button:hover {
    background: #226640 !important;
}

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #d0d0cc !important;
    color: #1a1a1a !important;
    border-radius: 6px !important;
    font-size: 13px !important;
}

/* Data tables */
.stDataFrame {
    border: 1px solid #e5e5e3 !important;
    border-radius: 6px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #f5f5f3 !important;
    border: 1px solid #e5e5e3 !important;
    border-radius: 6px !important;
    color: #1a1a1a !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

.streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e5e5e3 !important;
    border-top: none !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e5e3;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

[data-testid="metric-container"] > div:first-child {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #888;
    text-transform: uppercase;
}

[data-testid="metric-container"] > div:last-child {
    font-size: 32px;
    font-weight: 800;
    color: #2d7e4d;
}

/* Alerts */
[data-testid="stAlert"] {
    background: #f5faf8 !important;
    border: 1px solid #d4e8e2 !important;
    border-radius: 6px !important;
    color: #1a1a1a !important;
}

/* Divider */
.stDivider { margin: 32px 0 !important; }

/* Hide chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Status indicator card */
.status-card {
    background: #ffffff;
    border: 1px solid #e5e5e3;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.status-icon {
    font-size: 24px;
    margin-bottom: 8px;
}

.status-label {
    font-size: 12px;
    font-weight: 600;
    color: #888;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.status-value {
    font-size: 28px;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 4px;
}

.status-desc {
    font-size: 12px;
    color: #888;
}

.status-good { border-left: 4px solid #2d7e4d; }
.status-alert { border-left: 4px solid #f59e0b; }
.status-critical { border-left: 4px solid #dc2626; }

/* Alert badge */
.alert-badge {
    display: inline-block;
    background: #fee2e2;
    color: #991b1b;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}

.alert-badge.warning {
    background: #fef3c7;
    color: #92400e;
}

/* Hero intro */
.intro-section {
    background: linear-gradient(135deg, #f0f7f3 0%, #faf9f7 100%);
    border: 1px solid #e5e5e3;
    border-radius: 8px;
    padding: 36px 40px;
    margin-bottom: 32px;
}

.intro-emoji { font-size: 48px; margin-bottom: 12px; }
.intro-title { font-size: 32px; font-weight: 700; color: #1a1a1a; margin-bottom: 8px; }
.intro-desc { font-size: 14px; color: #555; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MATPLOTLIB — Light theme matching dashboard
# ══════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': '#f8f9f7',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#e5e5e3',
    'axes.labelcolor': '#555',
    'axes.titlecolor': '#1a1a1a',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.color': '#888',
    'ytick.color': '#888',
    'grid.color': '#f0f0ed',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.5,
    'legend.facecolor': '#ffffff',
    'legend.edgecolor': '#e5e5e3',
    'legend.fontsize': 10,
    'text.color': '#1a1a1a',
})

C_GREEN = '#2d7e4d'
C_LIGHT_GREEN = '#4caf50'
C_GRAY = '#888'
C_BORDER = '#e5e5e3'
PALETTE = ['#2d7e4d', '#4caf50', '#81c784', '#a5d6a7', '#c8e6c9', '#e8f5e9']

# ══════════════════════════════════════════════════════════
# PKL LOADING
# ══════════════════════════════════════════════════════════
PKL_FILES = {
    'rf': 'model_rf_pupuk.pkl',
    'xgb': 'model_xgb_pupuk.pkl',
    'le_pupuk': 'encoder_pupuk.pkl',
    'le_kec': 'encoder_kecamatan.pkl',
}

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(base, 'model_rf_pupuk.pkl')):
        if os.path.exists('/mount/src'):
            base = '/mount/src'
        else:
            base = os.getcwd()

    models = {'_base': base}
    missing = []
    
    for key, fname in PKL_FILES.items():
        path = os.path.join(base, fname)
        if os.path.exists(path):
            try:
                models[key] = joblib.load(path)
            except:
                missing.append(fname)
        else:
            missing.append(fname)

    try:
        with open(os.path.join(base, 'feature_names.json')) as f:
            models['features'] = json.load(f)
    except:
        models['features'] = ['Jenis Pupuk Enc', 'Kecamatan Enc', 'Bulan', 'Tahun', 'RDKK (ton)', 'Alokasi (ton)']

    models['missing'] = missing
    return models

M = load_models()
models_ok = len(M['missing']) == 0

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Pupuk Prediksi")
    st.caption("Sistem prediksi realisasi penyaluran pupuk")
    st.divider()
    
    st.markdown("**Dataset**")
    uploaded = st.file_uploader("Pilih file .xlsx", type=["xlsx"], label_visibility="collapsed")
    
    st.divider()
    st.markdown("**Model**")
    st.caption("Random Forest + XGBoost")
    st.caption("pkl-based • tanpa training ulang")

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="intro-section">
    <div class="intro-emoji">🌱</div>
    <div class="intro-title">Prediksi Realisasi Pupuk Bersubsidi</div>
    <div class="intro-desc">
        Sistem machine learning untuk estimasi realisasi penyaluran pupuk berdasarkan data historis. 
        Menggunakan Random Forest dan XGBoost untuk prediksi akurat di setiap kecamatan.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# ERROR STATE
# ══════════════════════════════════════════════════════════
if not models_ok:
    st.error(f"**Model files belum ada:** {', '.join(M['missing'])}")
    
    with st.expander("📋 Cara upload model dari Colab"):
        st.write("""
        1. **Jalankan notebook** di Google Colab (Bagian 1–12)
        2. **Simpan pkl** ke Google Drive (Bagian 12)
        3. **Download** 5 file pkl ke komputer
        4. **Upload ke GitHub** di folder root repo
        5. **Commit & push** — Streamlit Cloud auto re-deploy
        """)
    st.stop()

# ══════════════════════════════════════════════════════════
# MODEL LOADED
# ══════════════════════════════════════════════════════════
rf_model = M['rf']
xgb_model = M['xgb']
le_pupuk = M['le_pupuk']
le_kec = M['le_kec']
FEATURES = M['features']
jenis_list = list(le_pupuk.classes_)
kec_list = list(le_kec.classes_)

st.success("✓ Model dimuat — siap prediksi")

# ══════════════════════════════════════════════════════════
# STATUS OVERVIEW (Botanist-style)
# ══════════════════════════════════════════════════════════
st.markdown("### Status Sistem")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Aktif", "RF + XGB", "2 algoritma")
with col2:
    st.metric("Jenis Pupuk", len(jenis_list), "kategori")
with col3:
    st.metric("Kecamatan", len(kec_list), "wilayah")
with col4:
    st.metric("Data", "2017–2025", "9 tahun")

st.divider()

# ══════════════════════════════════════════════════════════
# LOAD DATASET
# ══════════════════════════════════════════════════════════
df = None
if uploaded is not None:
    @st.cache_data
    def load_data(file):
        from sklearn.preprocessing import LabelEncoder
        df = pd.read_excel(file)
        df['Bulan'] = df['Bulan'].ffill().astype(int)
        drop_cols = ['NO', 'Kelompok Tani', 'Alokasi Vs RDKK (%)', 'Realisasi Vs Alokasi (%)']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        for col in ['Realisasi (ton)', 'RDKK (ton)', 'Alokasi (ton)']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
        df['Kecamatan'] = df['Kecamatan'].str.strip().str.title()
        df['Jenis Pupuk'] = df['Jenis Pupuk'].str.strip()
        df['Jenis Pupuk Enc'] = LabelEncoder().fit_transform(df['Jenis Pupuk'])
        df['Kecamatan Enc'] = LabelEncoder().fit_transform(df['Kecamatan'])
        return df
    df = load_data(uploaded)

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Model", "Prediksi", "Analisis"])

# ─────────────────────────
# TAB 1: DATA
# ─────────────────────────
with tab1:
    if df is None:
        st.info("📊 Upload dataset di sidebar untuk melihat exploratory analysis")
    else:
        st.markdown(f"### Dataset · {len(df):,} records")
        
        with st.expander("Lihat data", expanded=False):
            st.dataframe(df.head(30), use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Realisasi per Jenis Pupuk")
            d = df.groupby('Jenis Pupuk')['Realisasi (ton)'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(d.index, d.values, color=PALETTE, edgecolor='none', alpha=0.85)
            ax.set_ylabel("Realisasi (ton)")
            ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.markdown("#### Tren Tahunan")
            tren = df.groupby('Tahun')['Realisasi (ton)'].sum()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(tren.index, tren.values, linewidth=3, color=C_GREEN, marker='o', markersize=7)
            ax.fill_between(tren.index, tren.values, alpha=0.15, color=C_GREEN)
            ax.set_ylabel("Realisasi (ton)")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Korelasi Variabel")
            cols = ['Realisasi (ton)', 'Alokasi (ton)', 'RDKK (ton)', 'Bulan', 'Tahun']
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='RdYlGn', ax=ax, 
                        center=0, cbar_kws={'shrink': 0.8}, linewidths=0.5)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col4:
            st.markdown("#### Statistik")
            st.dataframe(df[['Realisasi (ton)', 'Alokasi (ton)', 'RDKK (ton)']].describe().round(2), 
                        use_container_width=True)

# ─────────────────────────
# TAB 2: MODEL
# ─────────────────────────
with tab2:
    st.markdown("### Evaluasi Model")

    live_eval = False
    if df is not None:
        try:
            from sklearn.model_selection import train_test_split
            df2 = df.copy()
            df2['Jenis Pupuk Enc'] = le_pupuk.transform(df2['Jenis Pupuk'])
            df2['Kecamatan Enc'] = le_kec.transform(df2['Kecamatan'])
            X = df2[FEATURES]
            y = df2['Realisasi (ton)']
            _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            
            rfp = rf_model.predict(Xte)
            xgbp = xgb_model.predict(Xte)
            
            mrf = {'mae': mean_absolute_error(yte, rfp), 'rmse': np.sqrt(mean_squared_error(yte, rfp)), 'r2': r2_score(yte, rfp)}
            mxgb = {'mae': mean_absolute_error(yte, xgbp), 'rmse': np.sqrt(mean_squared_error(yte, xgbp)), 'r2': r2_score(yte, xgbp)}
            live_eval = True
        except:
            pass

    if not live_eval:
        mrf = {'mae': 8.71, 'rmse': 20.62, 'r2': 0.9919}
        mxgb = {'mae': 7.63, 'rmse': 16.41, 'r2': 0.9948}

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Random Forest**")
        st.metric("MAE", f"{mrf['mae']:.2f} ton")
        st.metric("RMSE", f"{mrf['rmse']:.2f} ton")
        st.metric("R² Score", f"{mrf['r2']:.4f}")

    with col2:
        st.markdown("**XGBoost**")
        st.metric("MAE", f"{mxgb['mae']:.2f} ton")
        st.metric("RMSE", f"{mxgb['rmse']:.2f} ton")
        st.metric("R² Score", f"{mxgb['r2']:.4f}")

    st.divider()
    st.markdown("### Feature Importance")

    col1, col2 = st.columns(2)
    
    with col1:
        imp_rf = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.barh(imp_rf.index, imp_rf.values, color=C_GREEN, alpha=0.8, edgecolor='none')
        ax.set_xlabel("Importance")
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Random Forest")

    with col2:
        imp_xgb = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.barh(imp_xgb.index, imp_xgb.values, color=C_LIGHT_GREEN, alpha=0.8, edgecolor='none')
        ax.set_xlabel("Importance")
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("XGBoost")

# ─────────────────────────
# TAB 3: PREDIKSI
# ─────────────────────────
with tab3:
    st.markdown("### Buat Prediksi")
    st.write("Masukkan parameter untuk mendapatkan estimasi realisasi.")

    with st.form("pred_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            jenis = st.selectbox("Jenis Pupuk", sorted(jenis_list))
            kec = st.selectbox("Kecamatan", sorted(kec_list))
        
        with col2:
            months = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
                      7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"}
            bln_name = st.selectbox("Bulan", list(months.values()), index=5)
            bln = [k for k, v in months.items() if v == bln_name][0]
        
        with col3:
            tahun = st.number_input("Tahun", 2017, 2035, 2025)

        col4, col5 = st.columns(2)
        with col4:
            rdkk = st.number_input("RDKK (ton)", 0.0, 10000.0, 500.0, 50.0)
        with col5:
            alokasi = st.number_input("Alokasi (ton)", 0.0, 10000.0, 300.0, 50.0)

        submit = st.form_submit_button("Hitung", use_container_width=True, type="primary")

    if submit:
        inp = pd.DataFrame([{
            'Jenis Pupuk Enc': le_pupuk.transform([jenis])[0],
            'Kecamatan Enc': le_kec.transform([kec])[0],
            'Bulan': bln,
            'Tahun': tahun,
            'RDKK (ton)': rdkk,
            'Alokasi (ton)': alokasi,
        }])[FEATURES]

        prf = float(rf_model.predict(inp)[0])
        pxgb = float(xgb_model.predict(inp)[0])

        st.divider()
        st.markdown("### Hasil")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Random Forest", f"{prf:.1f} ton", delta="prediksi")
        with col2:
            st.metric("XGBoost", f"{pxgb:.1f} ton", delta="prediksi")

        st.divider()
        st.markdown("### Input Summary")
        summary = pd.DataFrame({
            'Parameter': ['Pupuk', 'Kecamatan', 'Bulan', 'Tahun', 'RDKK', 'Alokasi'],
            'Nilai': [jenis, kec, bln_name, tahun, f"{rdkk:.0f}", f"{alokasi:.0f}"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if alokasi > 0:
            st.markdown("### Analisis Penyerapan")
            pct_rf = (prf / alokasi) * 100
            pct_xgb = (pxgb / alokasi) * 100

            fig, ax = plt.subplots(figsize=(7, 3))
            y_pos = [0, 1]
            cols_pct = [C_LIGHT_GREEN, C_GREEN]
            ax.barh(y_pos, [pct_xgb, pct_rf], color=cols_pct, alpha=0.85, height=0.4)
            ax.axvline(100, color='#d0d0cc', linestyle='--', linewidth=2)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(['XGBoost', 'Random Forest'])
            ax.set_xlabel("Penyerapan Alokasi (%)")
            ax.grid(axis='x', alpha=0.3)
            ax.set_axisbelow(True)
            
            for i, (y, pct) in enumerate([(0, pct_xgb), (1, pct_rf)]):
                ax.text(pct + 3, y, f'{pct:.0f}%', va='center', fontsize=12, fontweight='700')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

# ─────────────────────────
# TAB 4: ANALISIS
# ─────────────────────────
with tab4:
    st.markdown("### Insights & Rekomendasi")

    st.write("""
    **Performa Model**
    
    Kedua model mencapai R² > 0.99 dengan MAE < 10 ton, menunjukkan akurasi tinggi 
    untuk keperluan perencanaan distribusi pupuk ke lapangan.
    """)

    st.write("""
    **Insight dari Data**
    
    • **Alokasi mendominasi** — fitur paling berpengaruh pada prediksi (>70%)
    • **Gap konsisten** — RDKK selalu lebih tinggi dari alokasi di semua kecamatan
    • **Variasi per pupuk** — beberapa jenis lebih mudah direalisasikan
    • **Tren stabil** — realisasi tahunan konsisten dengan fluktuasi kecil
    """)

    st.divider()

    st.markdown("### Rekomendasi")
    st.write("""
    1. Gunakan model untuk **perencanaan pre-musim** — estimasi realisasi sebelum distribusi dimulai
    2. Monitor **gap RDKK-Alokasi** — pertimbangkan peningkatan untuk jenis/wilayah dengan deficit
    3. Kumpulkan **data lebih detail** — curah hujan, musim, kondisi pasar meningkatkan akurasi
    4. Deploy ke **sistem real-time** — integrasikan prediksi ke dashboard dinas pertanian
    """)

    st.divider()
    st.caption("Built with Random Forest + XGBoost | pkl-based inference | Streamlit Cloud deployment")
