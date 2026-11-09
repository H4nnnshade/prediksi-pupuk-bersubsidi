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

# ══════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pupuk Bersubsidi · Prediksi ML",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# CSS — DARK MINIMALIS  |  Palet: #202734 #36415A #B7D894 #E0F8C7
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #202734 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    color: #E0F8C7;
}
[data-testid="stMain"] { background-color: #202734 !important; }
.block-container { padding: 32px 44px 56px !important; max-width: 1300px !important; }
[data-testid="stSidebar"] {
    background-color: #1A2130 !important;
    border-right: 1px solid rgba(183,216,148,0.08) !important;
}
[data-testid="stSidebar"] > div { padding: 28px 20px; }
[data-testid="stSidebar"] * { color: #B7D894 !important; }
[data-testid="stSidebar"] .stFileUploader {
    background: rgba(183,216,148,0.04);
    border: 1px dashed rgba(183,216,148,0.2);
    border-radius: 8px; padding: 12px;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(183,216,148,0.08);
    border-radius: 8px; padding: 4px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px; font-size: 13px; font-weight: 500;
    color: rgba(183,216,148,0.5) !important;
    padding: 8px 20px; letter-spacing: -0.1px;
    border: none !important; background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(183,216,148,0.1) !important;
    color: #E0F8C7 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 28px; }
.stButton > button {
    background: rgba(183,216,148,0.07) !important;
    border: 1px solid rgba(183,216,148,0.18) !important;
    color: #E0F8C7 !important; border-radius: 6px !important;
    font-size: 13px !important; font-weight: 500 !important; padding: 8px 18px !important;
}
.stButton > button:hover {
    background: rgba(183,216,148,0.13) !important;
    border-color: rgba(183,216,148,0.35) !important;
}
[data-testid="stFormSubmitButton"] > button {
    background: #B7D894 !important; color: #202734 !important;
    border: none !important; font-weight: 600 !important;
    font-size: 14px !important; padding: 10px 24px !important;
    border-radius: 6px !important; width: 100%;
}
[data-testid="stFormSubmitButton"] > button:hover { background: #cae8aa !important; }
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(183,216,148,0.12) !important;
    color: #E0F8C7 !important; border-radius: 6px !important; font-size: 13px !important;
}
.stDataFrame { border: 1px solid rgba(183,216,148,0.08) !important; border-radius: 8px !important; overflow: hidden; }
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(183,216,148,0.08) !important;
    border-radius: 8px !important; color: #B7D894 !important;
    font-size: 13px !important; font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.01) !important;
    border: 1px solid rgba(183,216,148,0.06) !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.wm { display: flex; align-items: center; gap: 10px; padding-bottom: 22px; border-bottom: 1px solid rgba(183,216,148,0.08); margin-bottom: 24px; }
.wm-dot { width: 30px; height: 30px; border-radius: 8px; background: linear-gradient(135deg, #B7D894 0%, #E0F8C7 100%); flex-shrink: 0; }
.wm-label { font-size: 14px; font-weight: 600; color: #E0F8C7 !important; letter-spacing: -0.3px; }
.wm-sub   { font-size: 11px; color: rgba(183,216,148,0.4) !important; letter-spacing: 0.5px; text-transform: uppercase; }
.nav-row { display: flex; align-items: center; gap: 10px; padding: 7px 8px; border-radius: 6px; font-size: 13px; font-weight: 400; color: rgba(183,216,148,0.6) !important; margin-bottom: 2px; }
.ph { margin-bottom: 32px; }
.ph-eyebrow { font-size: 11px; font-weight: 600; color: rgba(183,216,148,0.45); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px; }
.ph-title   { font-size: 30px; font-weight: 600; color: #E0F8C7; letter-spacing: -1px; line-height: 1.1; margin-bottom: 10px; }
.ph-desc    { font-size: 14px; color: rgba(183,216,148,0.5); font-weight: 400; letter-spacing: -0.1px; line-height: 1.6; }
.kpi-wrap { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin-bottom: 32px; }
.kpi { background: rgba(255,255,255,0.025); border: 1px solid rgba(183,216,148,0.08); border-radius: 8px; padding: 16px 18px; }
.kpi-lbl  { font-size: 10px; font-weight: 600; color: rgba(183,216,148,0.4); text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 8px; }
.kpi-val  { font-size: 22px; font-weight: 600; color: #E0F8C7; letter-spacing: -0.5px; line-height: 1; }
.kpi-unit { font-size: 11px; color: rgba(183,216,148,0.3); margin-top: 4px; }
.slbl { font-size: 11px; font-weight: 600; color: rgba(183,216,148,0.4); text-transform: uppercase; letter-spacing: 1px; margin: 28px 0 14px; padding-bottom: 10px; border-bottom: 1px solid rgba(183,216,148,0.07); }
.sc { background: rgba(255,255,255,0.025); border: 1px solid rgba(183,216,148,0.08); border-radius: 8px; padding: 20px; margin-bottom: 14px; }
.sc-ttl { font-size: 11px; font-weight: 600; color: rgba(183,216,148,0.4); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 14px; }
.mb { background: rgba(255,255,255,0.025); border: 1px solid rgba(183,216,148,0.08); border-radius: 8px; padding: 20px 22px; height: 100%; }
.mb-rf  { border-top: 2px solid #B7D894; }
.mb-xgb { border-top: 2px solid #36415A; }
.mb-name { font-size: 13px; font-weight: 600; color: #E0F8C7; letter-spacing: -0.2px; margin-bottom: 18px; }
.mb-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 10px; }
.mb-item { background: rgba(255,255,255,0.03); border-radius: 6px; padding: 12px 14px; }
.mb-key  { font-size: 10px; font-weight: 600; color: rgba(183,216,148,0.35); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
.mb-val  { font-size: 19px; font-weight: 600; color: #E0F8C7; letter-spacing: -0.5px; }
.mb-val-g { color: #B7D894; }
.istrip { background: rgba(183,216,148,0.05); border: 1px solid rgba(183,216,148,0.12); border-radius: 6px; padding: 11px 16px; font-size: 13px; color: rgba(183,216,148,0.75); margin-bottom: 24px; display: flex; align-items: center; gap: 10px; }
.pr { background: rgba(255,255,255,0.025); border: 1px solid rgba(183,216,148,0.1); border-radius: 8px; padding: 32px; text-align: center; }
.pr-accent { border-color: rgba(183,216,148,0.3); }
.pr-model  { font-size: 11px; font-weight: 600; color: rgba(183,216,148,0.4); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
.pr-num    { font-size: 48px; font-weight: 700; color: #E0F8C7; letter-spacing: -2px; line-height: 1; }
.pr-num-g  { color: #B7D894; }
.pr-unit   { font-size: 13px; color: rgba(183,216,148,0.4); margin-top: 8px; }
.cl { display: flex; gap: 12px; padding: 13px 0; border-bottom: 1px solid rgba(183,216,148,0.06); }
.cl-dot { width: 5px; height: 5px; border-radius: 50%; background: #B7D894; margin-top: 7px; flex-shrink: 0; }
.cl-txt { font-size: 14px; color: rgba(183,216,148,0.65); line-height: 1.65; }
.cl-txt strong { color: #E0F8C7; }
.fc-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-top: 36px; }
.fc { background: rgba(255,255,255,0.02); border: 1px solid rgba(183,216,148,0.07); border-radius: 8px; padding: 20px; text-align: left; }
.fc-icon  { font-size: 20px; margin-bottom: 10px; }
.fc-title { font-size: 13px; font-weight: 600; color: #E0F8C7; letter-spacing: -0.2px; margin-bottom: 6px; }
.fc-desc  { font-size: 12px; color: rgba(183,216,148,0.4); line-height: 1.65; }
.upload-step { display: flex; gap: 14px; align-items: flex-start; padding: 10px 0; }
.step-num { width: 24px; height: 24px; border-radius: 50%; background: rgba(183,216,148,0.1); border: 1px solid rgba(183,216,148,0.2); font-size: 11px; font-weight: 600; color: #B7D894; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.step-txt { font-size: 13px; color: rgba(183,216,148,0.6); line-height: 1.6; }
.step-txt code { background: rgba(183,216,148,0.08); padding: 1px 6px; border-radius: 3px; font-size: 12px; color: #E0F8C7; font-family: monospace; }
.ft { margin-top: 56px; padding-top: 20px; border-top: 1px solid rgba(183,216,148,0.07); display: flex; justify-content: space-between; align-items: center; }
.ft-l { font-size: 12px; color: rgba(183,216,148,0.3); }
.ft-r { font-size: 11px; color: rgba(183,216,148,0.2); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MATPLOTLIB THEME
# ══════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': '#202734', 'axes.facecolor': '#252E3D',
    'axes.edgecolor': 'none', 'axes.labelcolor': 'rgba(183,216,148,0.5)',
    'axes.titlecolor': '#E0F8C7', 'axes.titlesize': 11, 'axes.titleweight': '600',
    'axes.labelsize': 9, 'xtick.color': 'rgba(183,216,148,0.4)',
    'ytick.color': 'rgba(183,216,148,0.4)', 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'grid.color': '#2A3344', 'grid.linewidth': 0.8,
    'legend.facecolor': '#252E3D', 'legend.edgecolor': 'rgba(183,216,148,0.1)',
    'legend.fontsize': 8, 'text.color': '#B7D894', 'font.family': 'DejaVu Sans',
})

CG = '#B7D894'; CG2 = '#E0F8C7'; CB = '#36415A'; CG3 = '#7A9B6E'
CLRS = [CG, CG2, '#6BAE4A', '#8FC47A', CG3]

# ══════════════════════════════════════════════════════════
# LOAD MODEL dari pkl (root repo GitHub)
# ══════════════════════════════════════════════════════════
PKL_FILES = {
    'rf'      : 'model_rf_pupuk.pkl',
    'xgb'     : 'model_xgb_pupuk.pkl',
    'le_pupuk': 'encoder_pupuk.pkl',
    'le_kec'  : 'encoder_kecamatan.pkl',
}

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    models = {}; missing = []
    for key, fname in PKL_FILES.items():
        path = os.path.join(base, fname)
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            missing.append(fname)
    feat_path = os.path.join(base, 'feature_names.json')
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            models['features'] = json.load(f)
    else:
        models['features'] = ['Jenis Pupuk Enc','Kecamatan Enc','Bulan','Tahun','RDKK (ton)','Alokasi (ton)']
    models['missing'] = missing
    return models

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="wm">
        <div class="wm-dot"></div>
        <div>
            <div class="wm-label">Pupuk Bersubsidi</div>
            <div class="wm-sub">Prediksi ML · 2017–2025</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="slbl" style="margin-top:0;">Dataset (Opsional)</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px;color:rgba(183,216,148,0.4);margin-bottom:10px;">Upload .xlsx untuk Tab EDA. Model load dari pkl otomatis.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Dataset .xlsx", type=["xlsx"], label_visibility="collapsed")

    st.markdown('<div class="slbl">Fitur Model</div>', unsafe_allow_html=True)
    for f in ["Jenis Pupuk","Kecamatan","Bulan","Tahun","RDKK (ton)","Alokasi (ton)"]:
        st.markdown(f'<div class="nav-row"><span style="color:rgba(183,216,148,0.25);font-size:9px;">◆</span>{f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="slbl">Target</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-row" style="color:#E0F8C7 !important;"><span style="color:#B7D894;font-size:9px;">◆</span>Realisasi (ton)</div>', unsafe_allow_html=True)

    st.markdown('<div class="slbl">Algoritma</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-row">🌲&nbsp; Random Forest (Tuned)</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-row">⚡&nbsp; XGBoost (Tuned)</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════
M = load_models()
models_ok = len(M['missing']) == 0

st.markdown("""
<div class="ph">
    <div class="ph-eyebrow">Machine Learning · Regresi</div>
    <div class="ph-title">Prediksi Realisasi Pupuk Bersubsidi</div>
    <div class="ph-desc">Random Forest &amp; XGBoost · Dataset Lombok Barat 2017–2025</div>
</div>
""", unsafe_allow_html=True)

# ── Jika pkl belum ada ──
if not models_ok:
    st.markdown(f"""
    <div class="istrip" style="border-color:rgba(220,80,80,0.3);background:rgba(220,80,80,0.05);">
        <span>⚠️</span>
        <span>File pkl tidak ditemukan: <strong style="color:#f08080;">{', '.join(M['missing'])}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sc"><div class="sc-ttl">Langkah — Google Colab</div>', unsafe_allow_html=True)
        for num, step in [
            ("1","Buka <code>Prediksi_Pupuk_Bersubsidi.ipynb</code> di Google Colab"),
            ("2","Jalankan Bagian 1 sampai Bagian 11"),
            ("3","Jalankan <strong>Bagian 12</strong> → pkl tersimpan ke <code>MyDrive/pupuk_model/</code>"),
            ("4","Jalankan cell download → 5 file diunduh ke komputer"),
        ]:
            st.markdown(f'<div class="upload-step"><div class="step-num">{num}</div><div class="step-txt">{step}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sc"><div class="sc-ttl">Langkah — GitHub & Deploy</div>', unsafe_allow_html=True)
        for num, step in [
            ("5","Taruh semua pkl + <code>feature_names.json</code> di root repo GitHub"),
            ("6","Commit & push ke GitHub"),
            ("7","Streamlit Cloud otomatis re-deploy"),
            ("8","App load model instan — tanpa training ulang"),
        ]:
            st.markdown(f'<div class="upload-step"><div class="step-num">{num}</div><div class="step-txt">{step}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sc-ttl" style="margin-top:16px;">File yang dibutuhkan di repo</div>', unsafe_allow_html=True)
        for f in ['model_rf_pupuk.pkl','model_xgb_pupuk.pkl','encoder_pupuk.pkl','encoder_kecamatan.pkl','feature_names.json']:
            st.markdown(f'<div class="nav-row"><code style="font-size:12px;color:#E0F8C7;">{f}</code></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ── Model ok ──
rf_model  = M['rf']
xgb_model = M['xgb']
le_pupuk  = M['le_pupuk']
le_kec    = M['le_kec']
FEATURES  = M['features']
jenis_list = list(le_pupuk.classes_)
kec_list   = list(le_kec.classes_)

st.markdown("""
<div class="istrip">
    <span>✅</span>
    <span>Model berhasil di-load dari pkl. Siap prediksi — tidak perlu training ulang.</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LOAD DATASET (opsional)
# ══════════════════════════════════════════════════════════
df = None
if uploaded is not None:
    @st.cache_data
    def load_data(file):
        from sklearn.preprocessing import LabelEncoder
        df = pd.read_excel(file)
        df['Bulan'] = df['Bulan'].ffill().astype(int)
        drop = ['NO','Kelompok Tani','Alokasi Vs RDKK (%)','Realisasi Vs Alokasi (%)']
        df = df.drop(columns=[c for c in drop if c in df.columns])
        for col in ['Realisasi (ton)','RDKK (ton)','Alokasi (ton)']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        df['Kecamatan']   = df['Kecamatan'].str.strip().str.title()
        df['Jenis Pupuk'] = df['Jenis Pupuk'].str.strip()
        df['Jenis Pupuk Enc'] = LabelEncoder().fit_transform(df['Jenis Pupuk'])
        df['Kecamatan Enc']   = LabelEncoder().fit_transform(df['Kecamatan'])
        return df
    df = load_data(uploaded)

# ══════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="kpi-wrap">
    <div class="kpi"><div class="kpi-lbl">Model</div><div class="kpi-val" style="font-size:16px;">RF + XGB</div><div class="kpi-unit">pkl loaded</div></div>
    <div class="kpi"><div class="kpi-lbl">RF Trees</div><div class="kpi-val">{rf_model.n_estimators}</div><div class="kpi-unit">n_estimators</div></div>
    <div class="kpi"><div class="kpi-lbl">XGB Trees</div><div class="kpi-val">{xgb_model.n_estimators}</div><div class="kpi-unit">n_estimators</div></div>
    <div class="kpi"><div class="kpi-lbl">Jenis Pupuk</div><div class="kpi-val">{len(jenis_list)}</div><div class="kpi-unit">kategori</div></div>
    <div class="kpi"><div class="kpi-lbl">Kecamatan</div><div class="kpi-val">{len(kec_list)}</div><div class="kpi-unit">wilayah</div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & EDA", "🤖 Evaluasi Model", "📈 Feature Importance", "🔮 Prediksi Manual", "📋 Kesimpulan"
])

# ─────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────
with tab1:
    if df is None:
        st.markdown("""
        <div class="istrip">
            <span>ℹ️</span>
            <span>Upload <strong>Data-pupuk-bersubsidi.xlsx</strong> di sidebar untuk EDA.
            Tab lain tetap berfungsi penuh tanpa dataset.</span>
        </div>
        <div class="fc-grid">
            <div class="fc"><div class="fc-icon">📊</div><div class="fc-title">Realisasi per Jenis Pupuk</div><div class="fc-desc">Bar chart total 2017–2025 per jenis</div></div>
            <div class="fc"><div class="fc-icon">📈</div><div class="fc-title">Tren per Tahun</div><div class="fc-desc">Line chart total realisasi tahunan</div></div>
            <div class="fc"><div class="fc-icon">🔥</div><div class="fc-title">Heatmap Korelasi</div><div class="fc-desc">Korelasi 7 variabel numerik</div></div>
            <div class="fc"><div class="fc-icon">📏</div><div class="fc-title">RDKK vs Alokasi vs Realisasi</div><div class="fc-desc">Bar grouped per 8 kecamatan</div></div>
            <div class="fc"><div class="fc-icon">🔵</div><div class="fc-title">Scatter Alokasi vs Realisasi</div><div class="fc-desc">Per jenis pupuk + garis ideal 1:1</div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="slbl">Dataset</div>', unsafe_allow_html=True)
        with st.expander("Lihat Data — 10 baris pertama"):
            st.dataframe(df.head(10), use_container_width=True)
        with st.expander("Statistik Deskriptif"):
            st.dataframe(df.describe().round(3), use_container_width=True)

        st.markdown('<div class="slbl">Visualisasi</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sc"><div class="sc-ttl">Total Realisasi per Jenis Pupuk</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5,3.5))
            d = df.groupby('Jenis Pupuk')['Realisasi (ton)'].sum().sort_values(ascending=False)
            bars = ax.bar(d.index, d.values, color=CLRS[:len(d)], edgecolor='none', width=0.5, zorder=3)
            ax.set_xlabel("Jenis Pupuk"); ax.set_ylabel("Realisasi (ton)")
            ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
            for bar, v in zip(bars, d.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+d.max()*0.012, f'{v:,.0f}', ha='center', fontsize=7, color=CG, fontweight='600')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="sc"><div class="sc-ttl">Tren Realisasi per Tahun</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5,3.5))
            tren = df.groupby('Tahun')['Realisasi (ton)'].sum()
            ax.plot(tren.index, tren.values, marker='o', color=CG, linewidth=2, markersize=7, markerfacecolor='#202734', markeredgewidth=2, zorder=3)
            ax.fill_between(tren.index, tren.values, alpha=0.07, color=CG)
            ax.set_xlabel("Tahun"); ax.set_ylabel("Realisasi (ton)"); ax.set_xticks(tren.index)
            ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="sc"><div class="sc-ttl">Heatmap Korelasi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5,4.2))
            cols_c = ['Bulan','Tahun','RDKK (ton)','Alokasi (ton)','Jenis Pupuk Enc','Kecamatan Enc','Realisasi (ton)']
            cmap = sns.diverging_palette(145, 300, s=70, l=40, as_cmap=True)
            sns.heatmap(df[cols_c].corr(), annot=True, fmt='.2f', cmap=cmap, ax=ax, linewidths=0.5, linecolor='#202734', annot_kws={'size':7.5,'weight':'600','color':CG2}, vmin=-1, vmax=1, center=0, cbar_kws={'shrink':0.75})
            ax.tick_params(axis='x', rotation=30); ax.tick_params(axis='y', rotation=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="sc"><div class="sc-ttl">RDKK vs Alokasi vs Realisasi per Kecamatan</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5,4.2))
            kec = df.groupby('Kecamatan')[['RDKK (ton)','Alokasi (ton)','Realisasi (ton)']].sum()
            x = np.arange(len(kec)); w = 0.26
            ax.bar(x-w, kec['RDKK (ton)'],     w, label='RDKK',     color=CB,  edgecolor='none')
            ax.bar(x,   kec['Alokasi (ton)'],   w, label='Alokasi',  color=CG3, edgecolor='none')
            ax.bar(x+w, kec['Realisasi (ton)'], w, label='Realisasi',color=CG,  edgecolor='none')
            ax.set_xticks(x); ax.set_xticklabels(kec.index, rotation=35, ha='right', fontsize=7)
            ax.set_ylabel("Total (ton)"); ax.legend(fontsize=8)
            ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sc"><div class="sc-ttl">Scatter: Alokasi vs Realisasi per Jenis Pupuk</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10,3.8))
        for i, p in enumerate(sorted(df['Jenis Pupuk'].unique())):
            sub = df[df['Jenis Pupuk']==p]
            ax.scatter(sub['Alokasi (ton)'], sub['Realisasi (ton)'], label=p, alpha=0.45, s=22, color=CLRS[i%len(CLRS)], zorder=3)
        mv = max(df['Alokasi (ton)'].max(), df['Realisasi (ton)'].max())
        ax.plot([0,mv],[0,mv],'--', color='rgba(183,216,148,0.25)', lw=1.5, label='Ideal')
        ax.set_xlabel("Alokasi (ton)"); ax.set_ylabel("Realisasi (ton)")
        ax.legend(bbox_to_anchor=(1.01,1), loc='upper left', fontsize=8)
        ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────
# TAB 2 — EVALUASI MODEL
# ─────────────────────────────
with tab2:
    live_eval = False
    if df is not None:
        try:
            from sklearn.model_selection import train_test_split
            df2 = df.copy()
            df2['Jenis Pupuk Enc'] = le_pupuk.transform(df2['Jenis Pupuk'])
            df2['Kecamatan Enc']   = le_kec.transform(df2['Kecamatan'])
            X = df2[FEATURES]; y = df2['Realisasi (ton)']
            _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            rfp  = rf_model.predict(Xte)
            xgbp = xgb_model.predict(Xte)
            def Mtr(yt, yp):
                return {'MAE': mean_absolute_error(yt,yp), 'MSE': mean_squared_error(yt,yp),
                        'RMSE': np.sqrt(mean_squared_error(yt,yp)), 'R2': r2_score(yt,yp)}
            mRF = Mtr(yte, rfp); mXGB = Mtr(yte, xgbp)
            live_eval = True
        except Exception as e:
            st.warning(f"Gagal evaluasi live: {e}")

    if not live_eval:
        # Hasil dari notebook run
        mRF  = {'MAE': 8.71,  'MSE': 424.98, 'RMSE': 20.62, 'R2': 0.9919}
        mXGB = {'MAE': 7.63,  'MSE': 269.29, 'RMSE': 16.41, 'R2': 0.9948}
        st.markdown("""
        <div class="istrip" style="border-color:rgba(255,200,80,0.2);background:rgba(255,200,80,0.04);">
            <span>📌</span>
            <span>Menampilkan hasil evaluasi dari notebook (Colab). Upload dataset untuk hitung ulang live.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="slbl">Kartu Metrik</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="mb mb-rf">
            <div class="mb-name">🌲 Random Forest (Tuned)</div>
            <div class="mb-grid">
                <div class="mb-item"><div class="mb-key">MAE</div><div class="mb-val mb-val-g">{mRF['MAE']:.4f} <span style="font-size:11px;opacity:.5">ton</span></div></div>
                <div class="mb-item"><div class="mb-key">RMSE</div><div class="mb-val mb-val-g">{mRF['RMSE']:.4f} <span style="font-size:11px;opacity:.5">ton</span></div></div>
                <div class="mb-item"><div class="mb-key">MSE</div><div class="mb-val">{mRF['MSE']:.2f}</div></div>
                <div class="mb-item"><div class="mb-key">R² Score</div><div class="mb-val mb-val-g">{mRF['R2']:.4f}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="mb mb-xgb">
            <div class="mb-name">⚡ XGBoost (Tuned)</div>
            <div class="mb-grid">
                <div class="mb-item"><div class="mb-key">MAE</div><div class="mb-val">{mXGB['MAE']:.4f} <span style="font-size:11px;opacity:.5">ton</span></div></div>
                <div class="mb-item"><div class="mb-key">RMSE</div><div class="mb-val">{mXGB['RMSE']:.4f} <span style="font-size:11px;opacity:.5">ton</span></div></div>
                <div class="mb-item"><div class="mb-key">MSE</div><div class="mb-val">{mXGB['MSE']:.2f}</div></div>
                <div class="mb-item"><div class="mb-key">R² Score</div><div class="mb-val">{mXGB['R2']:.4f}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="slbl">Tabel Perbandingan</div>', unsafe_allow_html=True)
    tbl = pd.DataFrame({
        'Model'     : ['🌲 Random Forest (Tuned)','⚡ XGBoost (Tuned)'],
        'MAE (ton)' : [mRF['MAE'],  mXGB['MAE']],
        'MSE'       : [mRF['MSE'],  mXGB['MSE']],
        'RMSE (ton)': [mRF['RMSE'], mXGB['RMSE']],
        'R² Score'  : [mRF['R2'],   mXGB['R2']]
    }).round(4)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown('<div class="slbl">Grafik Metrik</div>', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    for col, key, ttl in [(ca,'R2','R² Score'),(cb,'RMSE','RMSE (ton)'),(cc,'MAE','MAE (ton)')]:
        with col:
            st.markdown('<div class="sc">', unsafe_allow_html=True)
            st.markdown(f'<div class="sc-ttl">{ttl}</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(3.8,3))
            vals = [mRF[key], mXGB[key]]
            bars = ax.bar(['Random\nForest','XGBoost'], vals, color=[CG,CB], edgecolor='none', width=0.45, zorder=3)
            if key=='R2': ax.set_ylim(0.98, 1.0)
            ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+(0.0002 if key=='R2' else max(vals)*0.015),
                        f'{v:.4f}' if key=='R2' else f'{v:.2f}',
                        ha='center', fontsize=9, fontweight='600', color=CG2)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

    if live_eval:
        st.markdown('<div class="slbl">Aktual vs Prediksi</div>', unsafe_allow_html=True)
        cx, cy = st.columns(2)
        for col, pred, color, lbl in [(cx, rfp, CG,'🌲 Random Forest'),(cy, xgbp, CB,'⚡ XGBoost')]:
            with col:
                st.markdown(f'<div class="sc"><div class="sc-ttl">{lbl} — Aktual vs Prediksi</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5,4))
                ax.scatter(yte, pred, alpha=0.4, color=color, s=20, zorder=3)
                mv = max(yte.max(), pred.max())
                ax.plot([0,mv],[0,mv],'--', color='rgba(183,216,148,0.2)', lw=1.5, label='Ideal')
                ax.set_xlabel("Aktual (ton)"); ax.set_ylabel("Prediksi (ton)")
                ax.legend(fontsize=8); ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────
# TAB 3 — FEATURE IMPORTANCE
# ─────────────────────────────
with tab3:
    st.markdown('<div class="slbl">Feature Importance Score</div>', unsafe_allow_html=True)
    st.markdown('<div class="istrip"><span>ℹ</span> Semakin tinggi importance score, semakin besar pengaruh fitur terhadap prediksi realisasi.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, model_obj, color, lbl in [(c1,rf_model,CG,'🌲 Random Forest (Tuned)'),(c2,xgb_model,CB,'⚡ XGBoost (Tuned)')]:
        with col:
            imp = pd.Series(model_obj.feature_importances_, index=FEATURES).sort_values()
            st.markdown(f'<div class="sc"><div class="sc-ttl">{lbl}</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.5))
            bars = ax.barh(imp.index, imp.values, color=color, edgecolor='none', height=0.5, zorder=3)
            ax.xaxis.grid(True, zorder=0, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
            ax.set_xlabel("Importance Score")
            for bar, v in zip(bars, imp.values):
                ax.text(v+imp.max()*0.01, bar.get_y()+bar.get_height()/2, f'{v:.4f}', va='center', fontsize=8, fontweight='600', color=CG2)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            t = imp.sort_values(ascending=False).reset_index()
            t.columns = ['Fitur','Importance']; t['Importance'] = t['Importance'].round(4)
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────
# TAB 4 — PREDIKSI MANUAL
# ─────────────────────────────
with tab4:
    st.markdown('<div class="slbl">Input Parameter</div>', unsafe_allow_html=True)

    with st.form("form_pred"):
        c1, c2, c3 = st.columns(3)
        with c1:
            jenis   = st.selectbox("Jenis Pupuk", sorted(jenis_list))
            kec_sel = st.selectbox("Kecamatan",   sorted(kec_list))
        with c2:
            bmap = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
                    7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"}
            bl   = st.selectbox("Bulan", list(bmap.values()), index=5)
            bnum = [k for k,v in bmap.items() if v==bl][0]
            thn  = st.number_input("Tahun", min_value=2017, max_value=2035, value=2025)
        with c3:
            rdkk    = st.number_input("RDKK (ton)",    min_value=0.0, value=500.0, step=10.0)
            alokasi = st.number_input("Alokasi (ton)", min_value=0.0, value=300.0, step=10.0)
        sub = st.form_submit_button("Hitung Prediksi", use_container_width=True, type="primary")

    if sub:
        inp = pd.DataFrame([{
            'Jenis Pupuk Enc': le_pupuk.transform([jenis])[0],
            'Kecamatan Enc'  : le_kec.transform([kec_sel])[0],
            'Bulan': bnum, 'Tahun': thn,
            'RDKK (ton)': rdkk, 'Alokasi (ton)': alokasi
        }])[FEATURES]

        prf  = rf_model.predict(inp)[0]
        pxgb = xgb_model.predict(inp)[0]

        st.markdown('<div class="slbl">Hasil Prediksi</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="pr">
                <div class="pr-model">🌲 Random Forest (Tuned)</div>
                <div class="pr-num pr-num-g">{prf:.2f}</div>
                <div class="pr-unit">ton realisasi diprediksi</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="pr pr-accent">
                <div class="pr-model">⚡ XGBoost (Tuned)</div>
                <div class="pr-num">{pxgb:.2f}</div>
                <div class="pr-unit">ton realisasi diprediksi</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="slbl">Ringkasan Input</div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            inp_df = pd.DataFrame({'Parameter':['Jenis Pupuk','Kecamatan','Bulan','Tahun','RDKK (ton)','Alokasi (ton)'],
                                   'Nilai'    :[jenis,kec_sel,bl,thn,rdkk,alokasi]})
            st.dataframe(inp_df, use_container_width=True, hide_index=True)
        with s2:
            if alokasi > 0:
                st.markdown('<div class="sc"><div class="sc-ttl">% Penyerapan Alokasi</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(4,2.5))
                p1 = (prf/alokasi)*100; p2 = (pxgb/alokasi)*100
                bars = ax.barh(['XGBoost','Random\nForest'],[p2,p1],color=[CB,CG],edgecolor='none',height=0.4,zorder=3)
                ax.axvline(100,color='rgba(183,216,148,0.2)',linestyle='--',lw=1.2)
                ax.set_xlabel("% Penyerapan")
                ax.xaxis.grid(True, zorder=0, linestyle='--', alpha=0.5); ax.set_axisbelow(True)
                for bar, v in zip(bars,[p2,p1]):
                    ax.text(v+1, bar.get_y()+bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=9, fontweight='600', color=CG2)
                ax.set_xlim(0, max(p1,p2)*1.25)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────
# TAB 5 — KESIMPULAN
# ─────────────────────────────
with tab5:
    best_lbl = "XGBoost (Tuned) ⚡"
    # Ambil mXGB dari tab2 (sudah dihitung)
    st.markdown(f"""
    <div class="istrip">
        <span>🏆</span>
        <span>Model terbaik: <strong style="color:#E0F8C7;">{best_lbl}</strong>
        &nbsp;·&nbsp; R² = <strong style="color:#E0F8C7;">{mXGB['R2']:.4f}</strong>
        &nbsp;·&nbsp; RMSE = <strong style="color:#E0F8C7;">{mXGB['RMSE']:.4f} ton</strong></span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sc"><div class="sc-ttl">Alur Proyek</div>', unsafe_allow_html=True)
        for title, desc in [
            ("Load Data","Dataset pupuk bersubsidi 2017–2025 dari Google Drive"),
            ("EDA","Eksplorasi distribusi, tren, dan pola — 5 visualisasi"),
            ("Preprocessing","ffill Bulan, drop leakage, label encoding, imputasi"),
            ("Modeling","RF & XGBoost, 80/20 split"),
            ("Tuning","GridSearchCV 5-fold CV"),
            ("Evaluasi","MAE, MSE, RMSE, R² dibandingkan"),
            ("Feature Importance","Fitur paling berpengaruh diidentifikasi"),
            ("Simpan pkl","Model & encoder ke Google Drive"),
            ("Deploy Streamlit","app.py load pkl — prediksi tanpa training ulang"),
        ]:
            st.markdown(f'<div class="cl"><div class="cl-dot"></div><div class="cl-txt"><strong>{title}</strong> — {desc}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sc"><div class="sc-ttl">Temuan Utama</div>', unsafe_allow_html=True)
        for title, desc in [
            ("Alokasi (ton)","fitur paling berpengaruh di kedua model"),
            ("Gap RDKK vs Alokasi","terjadi di semua kecamatan"),
            ("R² > 0.99","kedua model sangat akurat"),
            ("XGBoost Tuned","performa terbaik secara keseluruhan"),
            ("pkl deployment","load instan tanpa training ulang"),
        ]:
            st.markdown(f'<div class="cl"><div class="cl-dot"></div><div class="cl-txt"><strong>{title}</strong> — {desc}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sc-ttl" style="margin-top:16px;">Saran</div>', unsafe_allow_html=True)
        for s in [
            "Gunakan model untuk estimasi realisasi sebelum musim tanam",
            "Data tanpa duplikasi akan meningkatkan akurasi",
            "Streamlit memudahkan dinas pertanian akses prediksi",
        ]:
            st.markdown(f'<div class="cl"><div class="cl-dot" style="background:#36415A;"></div><div class="cl-txt">{s}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="slbl">Hasil Evaluasi Final</div>', unsafe_allow_html=True)
    tbl_f = pd.DataFrame({
        'Model'      : ['🌲 Random Forest (Tuned)','⚡ XGBoost (Tuned)'],
        'MAE (ton)'  : [mRF['MAE'],  mXGB['MAE']],
        'MSE'        : [mRF['MSE'],  mXGB['MSE']],
        'RMSE (ton)' : [mRF['RMSE'], mXGB['RMSE']],
        'R² Score'   : [mRF['R2'],   mXGB['R2']]
    }).round(4)
    st.dataframe(tbl_f, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="ft">
    <div class="ft-l">Prediksi Realisasi Penyaluran Pupuk Bersubsidi · Pemrograman 5</div>
    <div class="ft-r">Random Forest · XGBoost · pkl deployment · Streamlit Cloud</div>
</div>
""", unsafe_allow_html=True)
