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
# DESIGN SYSTEM — Dark Agricultural Theme
# Typography: Lora (serif) + Inter (sans)
# Palette: Forest green + Earth tone
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
    background: #0f0f0f;
    color: #e8e8e8;
}

.block-container {
    padding: 48px 60px 72px !important;
    max-width: 1600px !important;
}

/* Typography base */
h1, h2, h3, h4 { font-family: 'Lora', serif; font-weight: 600; letter-spacing: -0.8px; color: #e8e8e8; }
h1 { font-size: 56px; line-height: 1.15; margin-bottom: 16px; }
h2 { font-size: 36px; line-height: 1.2; margin-bottom: 28px; margin-top: 0; }
h3 { font-size: 22px; line-height: 1.3; margin-bottom: 20px; }

p, span, div { font-family: 'Inter', system-ui, sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1a1a !important;
    border-right: 1px solid #252525 !important;
}
[data-testid="stSidebar"] > div { padding: 40px 24px; }
[data-testid="stSidebar"] * { color: #e8e8e8 !important; }

[data-testid="stSidebar"] .stFileUploader {
    background: #252525;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 12px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #252525;
}

.stTabs [data-baseweb="tab"] {
    font-size: 14px;
    font-weight: 500;
    color: #888;
    padding: 16px 24px;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    font-family: 'Inter', sans-serif;
}

.stTabs [aria-selected="true"] {
    color: #3a9d5d !important;
    border-bottom: 2px solid #3a9d5d !important;
}

.stTabs [data-baseweb="tab-panel"] { padding-top: 40px; }

/* Buttons */
.stButton > button {
    background: #252525 !important;
    border: 1px solid #333 !important;
    color: #e8e8e8 !important;
    border-radius: 3px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    font-family: 'Inter', sans-serif;
}

.stButton > button:hover {
    background: #333 !important;
}

[data-testid="stFormSubmitButton"] > button {
    background: #3a9d5d !important;
    border: none !important;
    color: #0f0f0f !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    border-radius: 3px !important;
    width: 100%;
    font-family: 'Inter', sans-serif;
}

[data-testid="stFormSubmitButton"] > button:hover {
    background: #4ab56a !important;
}

/* Inputs */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    color: #e8e8e8 !important;
    border-radius: 3px !important;
    font-size: 13px !important;
}

.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: #3a9d5d !important;
}

/* Data tables */
.stDataFrame { border: 1px solid #252525 !important; border-radius: 3px !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #1a1a1a !important;
    border: 1px solid #252525 !important;
    border-radius: 3px !important;
    color: #e8e8e8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif;
}

.streamlit-expanderContent {
    background: #0f0f0f !important;
    border: 1px solid #252525 !important;
    border-top: none !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #1a1a1a;
    border: 1px solid #252525;
    border-radius: 3px;
    padding: 20px 24px;
}

[data-testid="metric-container"] > div:first-child {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    color: #888;
    text-transform: none;
}

[data-testid="metric-container"] > div:last-child {
    font-size: 32px;
    font-weight: 700;
    color: #3a9d5d;
}

/* Alerts */
[data-testid="stAlert"] {
    background: #1a1a1a !important;
    border: 1px solid #252525 !important;
    border-radius: 3px !important;
    color: #e8e8e8 !important;
}

/* Divider */
.stDivider { margin: 40px 0 !important; }

/* Hide chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Custom hero section */
.hero-section {
    text-align: center;
    padding: 60px 40px;
    margin-bottom: 40px;
}

.hero-emoji { font-size: 80px; margin-bottom: 28px; }

.hero-title {
    font-family: 'Lora', serif;
    font-size: 52px;
    font-weight: 700;
    color: #e8e8e8;
    margin-bottom: 12px;
    line-height: 1.2;
    letter-spacing: -1px;
}

.hero-desc {
    font-size: 16px;
    color: #888;
    line-height: 1.7;
    max-width: 700px;
    margin: 0 auto;
}

/* Stat blocks for prediction results */
.stat-block {
    background: #1a1a1a;
    border: 1px solid #252525;
    border-radius: 3px;
    padding: 24px;
    border-left: 3px solid #3a9d5d;
}

.stat-block.alt { border-left-color: #d4a574; }

.stat-label {
    font-size: 11px;
    color: #888;
    letter-spacing: 1px;
    margin-bottom: 12px;
    font-weight: 600;
    text-transform: none;
}

.stat-value {
    font-size: 40px;
    font-weight: 700;
    color: #e8e8e8;
    line-height: 1;
}

.stat-unit {
    font-size: 12px;
    color: #666;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MATPLOTLIB — Match design system
# ══════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor': '#1a1a1a',
    'axes.edgecolor': '#252525',
    'axes.labelcolor': '#888',
    'axes.titlecolor': '#e8e8e8',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.color': '#666',
    'ytick.color': '#666',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'grid.color': '#252525',
    'grid.linewidth': 0.7,
    'grid.alpha': 0.5,
    'legend.facecolor': '#1a1a1a',
    'legend.edgecolor': '#252525',
    'legend.fontsize': 11,
    'text.color': '#e8e8e8',
})

C_GREEN = '#3a9d5d'
C_EARTH = '#d4a574'
C_SUBTLE = '#5b8c6a'
C_WARM = '#c99a6e'
PALETTE = [C_GREEN, C_EARTH, C_SUBTLE, C_WARM, '#6b5b4f', '#8a7d73']

# ══════════════════════════════════════════════════════════
# MODEL LOADING
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
            except Exception as e:
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
    st.markdown("**Pupuk Prediksi**")
    st.caption("Realisasi Penyaluran · 2017–2025")
    st.divider()
    
    st.markdown("**Dataset**")
    uploaded = st.file_uploader("Upload .xlsx", type=["xlsx"], label_visibility="collapsed")
    
    st.divider()
    st.markdown("**Model**")
    st.caption("Random Forest + XGBoost")
    st.caption("pkl-based inference — tidak ada training ulang")

# ══════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-section">
    <div class="hero-emoji">🌱</div>
    <div class="hero-title">Prediksi Realisasi Pupuk Bersubsidi</div>
    <div class="hero-desc">
        Model machine learning untuk estimasi realisasi distribusi pupuk berdasarkan data historis 
        2017–2025 dari Lombok Barat. Menggunakan Random Forest dan XGBoost untuk prediksi akurat.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════
# ERROR STATE
# ══════════════════════════════════════════════════════════
if not models_ok:
    st.error(f"Model files missing: {', '.join(M['missing'])}")
    
    with st.expander("Cara mengatasi — upload model dari Colab"):
        st.write("""
        **Langkah 1: Generate model di Google Colab**
        1. Jalankan notebook `Prediksi_Pupuk_Bersubsidi.ipynb`
        2. Selesaikan Bagian 1 sampai Bagian 11 (training)
        3. Jalankan Bagian 12 untuk menyimpan pkl ke Google Drive
        4. Jalankan cell download untuk mengunduh 5 file pkl
        
        **Langkah 2: Upload ke GitHub**
        1. Taruh semua file pkl di root repo
        2. Commit dan push ke branch main
        3. Streamlit Cloud akan otomatis re-deploy
        """)
    st.stop()

# ══════════════════════════════════════════════════════════
# MODEL LOADED — EXTRACT
# ══════════════════════════════════════════════════════════
rf_model = M['rf']
xgb_model = M['xgb']
le_pupuk = M['le_pupuk']
le_kec = M['le_kec']
FEATURES = M['features']
jenis_list = list(le_pupuk.classes_)
kec_list = list(le_kec.classes_)

st.success("✓ Model siap digunakan")

# ══════════════════════════════════════════════════════════
# KEY METRICS
# ══════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Algoritma", "RF + XGB")
with col2:
    st.metric("Jenis Pupuk", len(jenis_list))
with col3:
    st.metric("Kecamatan", len(kec_list))
with col4:
    st.metric("Rentang Data", "2017–25")

st.divider()

# ══════════════════════════════════════════════════════════
# LOAD DATASET (optional)
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
tab1, tab2, tab3, tab4 = st.tabs(["Eksplorasi Data", "Evaluasi Model", "Buat Prediksi", "Temuan"])

# ─────────────────────────
# TAB 1: EKSPLORASI DATA
# ─────────────────────────
with tab1:
    if df is None:
        st.info("📊 Upload file .xlsx di sidebar untuk melihat exploratory analysis")
    else:
        st.header("Dataset")
        st.write(f"{len(df):,} records × {len(df.columns)} fitur")
        
        with st.expander("Lihat data mentah"):
            st.dataframe(df.head(25), use_container_width=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Realisasi per Jenis Pupuk")
            d = df.groupby('Jenis Pupuk')['Realisasi (ton)'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.bar(d.index, d.values, color=PALETTE, edgecolor='none', alpha=0.85)
            ax.set_ylabel("Realisasi (ton)")
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            plt.xticks(rotation=25, ha='right')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.subheader("Tren Realisasi Tahunan")
            tren = df.groupby('Tahun')['Realisasi (ton)'].sum()
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.plot(tren.index, tren.values, linewidth=3, color=C_GREEN, marker='o', markersize=8)
            ax.fill_between(tren.index, tren.values, alpha=0.15, color=C_GREEN)
            ax.set_ylabel("Realisasi (ton)")
            ax.set_xlabel("Tahun")
            ax.grid(alpha=0.3)
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Korelasi Variabel")
            cols_c = ['Bulan', 'Tahun', 'RDKK (ton)', 'Alokasi (ton)', 'Realisasi (ton)']
            fig, ax = plt.subplots(figsize=(6, 5))
            cmap = sns.diverging_palette(150, 30, s=70, l=50, as_cmap=True)
            sns.heatmap(df[cols_c].corr(), annot=True, fmt='.2f', cmap=cmap, ax=ax, 
                        cbar_kws={'shrink': 0.8}, linewidths=0.5)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col4:
            st.subheader("Statistik Deskriptif")
            st.dataframe(df[['Realisasi (ton)', 'RDKK (ton)', 'Alokasi (ton)']].describe().round(2), use_container_width=True)

# ─────────────────────────
# TAB 2: EVALUASI MODEL
# ─────────────────────────
with tab2:
    st.header("Performa Model")

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
            
            mrf = {
                'mae': mean_absolute_error(yte, rfp),
                'rmse': np.sqrt(mean_squared_error(yte, rfp)),
                'r2': r2_score(yte, rfp)
            }
            mxgb = {
                'mae': mean_absolute_error(yte, xgbp),
                'rmse': np.sqrt(mean_squared_error(yte, xgbp)),
                'r2': r2_score(yte, xgbp)
            }
            live_eval = True
        except Exception as e:
            st.warning(f"Evaluasi live gagal: {str(e)[:50]}")

    if not live_eval:
        mrf = {'mae': 8.71, 'rmse': 20.62, 'r2': 0.9919}
        mxgb = {'mae': 7.63, 'rmse': 16.41, 'r2': 0.9948}
        st.caption("Data dari notebook Colab")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Random Forest")
        st.metric("MAE", f"{mrf['mae']:.2f} ton")
        st.metric("RMSE", f"{mrf['rmse']:.2f} ton")
        st.metric("R² Score", f"{mrf['r2']:.4f}")

    with col2:
        st.subheader("XGBoost")
        st.metric("MAE", f"{mxgb['mae']:.2f} ton")
        st.metric("RMSE", f"{mxgb['rmse']:.2f} ton")
        st.metric("R² Score", f"{mxgb['r2']:.4f}")

    st.divider()
    st.subheader("Feature Importance")

    col1, col2 = st.columns(2)
    
    with col1:
        imp_rf = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.barh(imp_rf.index, imp_rf.values, color=C_GREEN, alpha=0.7, edgecolor='none')
        ax.set_xlabel("Importance Score")
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("Random Forest — fitur dengan pengaruh terbesar")

    with col2:
        imp_xgb = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.barh(imp_xgb.index, imp_xgb.values, color=C_EARTH, alpha=0.7, edgecolor='none')
        ax.set_xlabel("Importance Score")
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.caption("XGBoost — fitur dengan pengaruh terbesar")

# ─────────────────────────
# TAB 3: BUAT PREDIKSI
# ─────────────────────────
with tab3:
    st.header("Prediksi Manual")
    st.write("Masukkan data untuk mendapatkan prediksi realisasi penyaluran pupuk.")

    with st.form("form_pred"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            jenis = st.selectbox("Jenis Pupuk", sorted(jenis_list), key='j')
            kec = st.selectbox("Kecamatan", sorted(kec_list), key='k')
        
        with col2:
            months = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
                      7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"}
            bln_name = st.selectbox("Bulan", list(months.values()), index=5)
            bln = [k for k, v in months.items() if v == bln_name][0]
        
        with col3:
            tahun = st.number_input("Tahun", 2017, 2035, 2025)

        col4, col5 = st.columns(2)
        with col4:
            rdkk = st.number_input("RDKK (ton)", 0.0, 10000.0, 500.0, step=50.0)
        with col5:
            alokasi = st.number_input("Alokasi (ton)", 0.0, 10000.0, 300.0, step=50.0)

        submit = st.form_submit_button("Hitung Prediksi", use_container_width=True, type="primary")

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
        st.subheader("Hasil Prediksi")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-block">
                <div class="stat-label">Random Forest</div>
                <div class="stat-value">{prf:.1f}</div>
                <div class="stat-unit">ton realisasi diprediksi</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stat-block alt">
                <div class="stat-label">XGBoost</div>
                <div class="stat-value" style="color: {C_EARTH};">{pxgb:.1f}</div>
                <div class="stat-unit">ton realisasi diprediksi</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Ringkasan Input")
        summary = pd.DataFrame({
            'Parameter': ['Pupuk', 'Kecamatan', 'Bulan', 'Tahun', 'RDKK', 'Alokasi'],
            'Nilai': [jenis, kec, bln_name, tahun, f"{rdkk:.0f}", f"{alokasi:.0f}"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if alokasi > 0:
            st.subheader("Analisis Penyerapan")
            pct_rf = (prf / alokasi) * 100
            pct_xgb = (pxgb / alokasi) * 100

            fig, ax = plt.subplots(figsize=(7, 3))
            y_pos = [0, 1]
            ax.barh(y_pos, [pct_xgb, pct_rf], color=[C_EARTH, C_GREEN], alpha=0.8, height=0.5)
            ax.axvline(100, color='#555', linestyle='--', linewidth=2, label='Target 100%')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(['XGBoost', 'Random Forest'])
            ax.set_xlabel("Persentase Penyerapan Alokasi (%)")
            ax.set_xlim(0, max(pct_rf, pct_xgb) * 1.2)
            ax.grid(axis='x', alpha=0.3)
            ax.set_axisbelow(True)
            
            for i, (y, pct) in enumerate([(0, pct_xgb), (1, pct_rf)]):
                ax.text(pct + 2, y, f'{pct:.1f}%', va='center', fontsize=12, fontweight='600')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

# ─────────────────────────
# TAB 4: TEMUAN
# ─────────────────────────
with tab4:
    st.header("Insights & Rekomendasi")

    st.write("""
    **Performa Model**
    
    Kedua model mencapai akurasi tinggi dengan R² > 0.99, artinya model menjelaskan lebih dari 99% 
    variasi data. Error rata-rata (MAE) kurang dari 10 ton menunjukkan prediksi cukup reliabel untuk 
    keperluan perencanaan distribusi.
    """)

    st.write("""
    **Insight dari Data**
    
    • Alokasi (ton) adalah fitur paling berpengaruh pada kedua model — menentukan lebih dari 70% hasil prediksi
    • Gap konsisten antara RDKK dan alokasi di semua kecamatan menunjukkan pola struktural
    • Variasi realisasi antar jenis pupuk signifikan — beberapa jenis lebih mudah direalisasikan
    • Tren tahunan stabil dengan sedikit fluktuasi musiman
    """)

    st.divider()

    st.subheader("Rekomendasi Operasional")
    
    st.write("""
    1. **Gunakan model untuk perencanaan alokasi**
       Estimasi realisasi sebelum musim tanam untuk mengoptimalkan distribusi pupuk ke kecamatan
    
    2. **Monitor gap RDKK–Alokasi**
       Jika alokasi terus lebih rendah, pertimbangkan peningkatan untuk jenis/wilayah tertentu
    
    3. **Data lebih granular meningkatkan akurasi**
       Tambahkan fitur curah hujan, musim pertanian, atau kondisi pasar lokal
    
    4. **Integrasikan ke dashboard real-time**
       Dashboard ini memudahkan dinas pertanian akses prediksi kapan saja tanpa komputasi manual
    """)

    st.divider()
    st.caption("Built with Random Forest + XGBoost | pkl-based inference | Streamlit Cloud deployment")
