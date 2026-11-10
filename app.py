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
    page_title="Prediksi Pupuk Bersubsidi",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# CSS — LIGHT MODE  |  Stripe elegance × Notion warmth
# Font: Inter (Google Fonts)
# Palet: #ffffff · #f6f5f4 · #061b31 · #533afd · #64748d · #e5edf5
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── RESET ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── BASE ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.stApp {
    background-color: #ffffff !important;
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    color: rgba(0,0,0,0.88);
}
.block-container {
    padding: 36px 52px 72px !important;
    max-width: 1280px !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background-color: #f6f5f4 !important;
    border-right: 1px solid rgba(0,0,0,0.08) !important;
}
[data-testid="stSidebar"] > div { padding: 32px 22px; }
[data-testid="stSidebar"] * { color: rgba(0,0,0,0.88) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #061b31 !important; }
[data-testid="stSidebar"] .stFileUploader {
    background: #ffffff;
    border: 1px dashed rgba(0,0,0,0.15);
    border-radius: 8px; padding: 12px;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(0,0,0,0.1);
    padding: 0; gap: 0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    font-size: 13.5px; font-weight: 500;
    color: #64748d !important;
    padding: 10px 20px;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    letter-spacing: -0.1px;
}
.stTabs [aria-selected="true"] {
    color: #061b31 !important;
    border-bottom: 2px solid #533afd !important;
    background: transparent !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 32px; }

/* ── BUTTONS ── */
.stButton > button {
    background: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    color: rgba(0,0,0,0.88) !important;
    border-radius: 4px !important;
    font-size: 14px !important; font-weight: 500 !important;
    padding: 8px 16px !important; letter-spacing: -0.1px !important;
    box-shadow: rgba(50,50,93,0.08) 0px 2px 5px, rgba(0,0,0,0.05) 0px 1px 2px !important;
}
.stButton > button:hover {
    box-shadow: rgba(50,50,93,0.15) 0px 4px 12px, rgba(0,0,0,0.08) 0px 2px 5px !important;
    border-color: rgba(0,0,0,0.2) !important;
}
[data-testid="stFormSubmitButton"] > button {
    background: #533afd !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 10px 24px !important; border-radius: 4px !important;
    width: 100%;
    box-shadow: rgba(83,58,253,0.3) 0px 4px 12px, rgba(0,0,0,0.1) 0px 2px 4px !important;
    letter-spacing: -0.1px !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: #4434d4 !important;
    box-shadow: rgba(83,58,253,0.4) 0px 6px 20px, rgba(0,0,0,0.12) 0px 2px 6px !important;
}

/* ── INPUTS ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #e5edf5 !important;
    color: rgba(0,0,0,0.88) !important;
    border-radius: 4px !important;
    font-size: 14px !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: #533afd !important;
    box-shadow: 0 0 0 2px rgba(83,58,253,0.1) !important;
}

/* ── DATAFRAME ── */
.stDataFrame {
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 8px !important; overflow: hidden;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: #f6f5f4 !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-radius: 6px !important;
    color: rgba(0,0,0,0.88) !important;
    font-size: 14px !important; font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}

/* ── HIDE CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ══════════════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════════════ */

/* Logo + brand */
.brand {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 24px;
    border-bottom: 1px solid rgba(0,0,0,0.08);
    margin-bottom: 24px;
}
.brand-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, #533afd 0%, #7b68ee 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0;
}
.brand-name { font-size: 15px; font-weight: 700; color: #061b31 !important; letter-spacing: -0.3px; }
.brand-sub  { font-size: 11px; color: #a39e98 !important; letter-spacing: 0.3px; }

/* Nav items sidebar */
.nav-item {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 10px; border-radius: 5px;
    font-size: 13px; color: #615d59 !important;
    margin-bottom: 2px; cursor: default;
}
.nav-dot { width: 5px; height: 5px; border-radius: 50%; background: #c5bfbb; flex-shrink: 0; }

/* Section label */
.slbl {
    font-size: 10.5px; font-weight: 600; color: #a39e98;
    text-transform: uppercase; letter-spacing: 1.1px;
    margin: 24px 0 12px; padding-bottom: 8px;
    border-bottom: 1px solid rgba(0,0,0,0.07);
}

/* Page header */
.ph { margin-bottom: 36px; }
.ph-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f2f9ff; color: #097fe8;
    font-size: 11.5px; font-weight: 600; letter-spacing: 0.5px;
    padding: 4px 10px; border-radius: 9999px;
    margin-bottom: 16px;
}
.ph-title {
    font-size: 36px; font-weight: 700; color: #061b31;
    letter-spacing: -1.2px; line-height: 1.08; margin-bottom: 12px;
}
.ph-desc {
    font-size: 16px; color: #64748d; font-weight: 400;
    line-height: 1.6; letter-spacing: -0.1px; max-width: 580px;
}

/* KPI strip */
.kpi-wrap {
    display: grid; grid-template-columns: repeat(5,1fr);
    gap: 12px; margin-bottom: 36px;
}
.kpi {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08); border-radius: 8px;
    padding: 18px 20px;
    box-shadow: rgba(50,50,93,0.06) 0px 2px 8px, rgba(0,0,0,0.04) 0px 1px 3px;
}
.kpi-lbl { font-size: 10px; font-weight: 600; color: #a39e98; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.kpi-val { font-size: 24px; font-weight: 700; color: #061b31; letter-spacing: -0.8px; line-height: 1; }
.kpi-unit { font-size: 11px; color: #a39e98; margin-top: 4px; }

/* Info strip */
.istrip {
    background: #f2f9ff;
    border: 1px solid rgba(9,127,232,0.2);
    border-radius: 6px; padding: 12px 16px;
    font-size: 13.5px; color: #615d59;
    margin-bottom: 28px;
    display: flex; align-items: flex-start; gap: 10px;
}
.istrip.warn { background: #fff8f0; border-color: rgba(221,91,0,0.2); }
.istrip.err  { background: #fff2f2; border-color: rgba(220,60,60,0.2); }

/* Section card */
.sc {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08); border-radius: 8px;
    padding: 22px 24px; margin-bottom: 14px;
    box-shadow: rgba(0,0,0,0.04) 0px 4px 18px,
                rgba(0,0,0,0.027) 0px 2px 7.85px,
                rgba(0,0,0,0.02) 0px 0.8px 2.93px;
}
.sc-ttl {
    font-size: 10.5px; font-weight: 600; color: #a39e98;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px;
}

/* Model cards */
.mb {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08); border-radius: 8px;
    padding: 22px 24px; height: 100%;
    box-shadow: rgba(50,50,93,0.08) 0px 8px 24px, rgba(0,0,0,0.05) 0px 2px 6px;
}
.mb-rf  { border-top: 3px solid #533afd; }
.mb-xgb { border-top: 3px solid #15be53; }
.mb-name { font-size: 14px; font-weight: 700; color: #061b31; letter-spacing: -0.2px; margin-bottom: 20px; }
.mb-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 10px; }
.mb-item { background: #f6f5f4; border-radius: 6px; padding: 14px 16px; }
.mb-key  { font-size: 10px; font-weight: 600; color: #a39e98; text-transform: uppercase; letter-spacing: 0.9px; margin-bottom: 5px; }
.mb-val  { font-size: 20px; font-weight: 700; color: #061b31; letter-spacing: -0.5px; }
.mb-val-b { color: #533afd; }
.mb-val-g { color: #108c3d; }

/* Upload steps */
.upload-step { display: flex; gap: 14px; align-items: flex-start; padding: 10px 0; }
.step-num {
    width: 26px; height: 26px; border-radius: 50%;
    background: #533afd; color: #ffffff;
    font-size: 11px; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.step-txt { font-size: 13.5px; color: #615d59; line-height: 1.6; padding-top: 2px; }
.step-txt code {
    background: #f6f5f4; padding: 1px 6px;
    border-radius: 3px; font-size: 12px; color: #061b31;
    font-family: monospace; border: 1px solid rgba(0,0,0,0.08);
}

/* Prediksi result */
.pr {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.1); border-radius: 10px;
    padding: 32px; text-align: center;
    box-shadow: rgba(50,50,93,0.1) 0px 20px 40px -15px, rgba(0,0,0,0.07) 0px 10px 20px -10px;
}
.pr-accent { border-top: 3px solid #533afd; }
.pr-green  { border-top: 3px solid #15be53; }
.pr-model  { font-size: 11px; font-weight: 600; color: #a39e98; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px; }
.pr-num    { font-size: 52px; font-weight: 700; color: #061b31; letter-spacing: -2.5px; line-height: 1; }
.pr-num-b  { color: #533afd; }
.pr-num-g  { color: #108c3d; }
.pr-unit   { font-size: 13px; color: #a39e98; margin-top: 10px; font-weight: 500; }

/* Conclusion list */
.cl { display: flex; gap: 12px; padding: 13px 0; border-bottom: 1px solid rgba(0,0,0,0.06); }
.cl-dot { width: 6px; height: 6px; border-radius: 50%; background: #533afd; margin-top: 7px; flex-shrink: 0; }
.cl-txt { font-size: 14px; color: #615d59; line-height: 1.65; }
.cl-txt strong { color: #061b31; font-weight: 600; }

/* Feature cards grid */
.fc-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-top: 20px; }
.fc {
    background: #f6f5f4;
    border: 1px solid rgba(0,0,0,0.07); border-radius: 10px;
    padding: 22px; text-align: left;
}
.fc-icon  { font-size: 22px; margin-bottom: 10px; }
.fc-title { font-size: 14px; font-weight: 600; color: #061b31; letter-spacing: -0.2px; margin-bottom: 6px; }
.fc-desc  { font-size: 13px; color: #615d59; line-height: 1.65; }

/* Pill badge */
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: #f2f9ff; color: #097fe8;
    font-size: 11.5px; font-weight: 600;
    padding: 3px 10px; border-radius: 9999px;
    letter-spacing: 0.1px;
}
.pill.green { background: rgba(21,190,83,0.1); color: #108c3d; }
.pill.purple { background: rgba(83,58,253,0.08); color: #533afd; }

/* Footer */
.ft {
    margin-top: 72px; padding-top: 24px;
    border-top: 1px solid rgba(0,0,0,0.08);
    display: flex; justify-content: space-between; align-items: center;
}
.ft-l { font-size: 12.5px; color: #a39e98; }
.ft-r { font-size: 11.5px; color: #c5bfbb; }

/* Divider */
.divider { height: 1px; background: rgba(0,0,0,0.08); margin: 28px 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MATPLOTLIB THEME — light, Stripe-inspired
# ══════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.facecolor':   '#f6f5f4',
    'axes.edgecolor':   'none',
    'axes.labelcolor':  '#64748d',
    'axes.titlecolor':  '#061b31',
    'axes.titlesize':   11,
    'axes.titleweight': '600',
    'axes.labelsize':   9,
    'xtick.color':      '#a39e98',
    'ytick.color':      '#a39e98',
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'grid.color':       '#e5edf5',
    'grid.linewidth':   0.8,
    'legend.facecolor': '#ffffff',
    'legend.edgecolor': '#e5edf5',
    'legend.fontsize':  8,
    'text.color':       '#061b31',
    'font.family':      'DejaVu Sans',
})

# Palet warna chart
CP  = '#533afd'   # purple (primary)
CG  = '#15be53'   # green
CN  = '#061b31'   # navy
CS  = '#64748d'   # slate
CO  = '#f96bee'   # accent
CLRS = [CP, CG, '#ea2261', '#fbbf24', CS]

# ══════════════════════════════════════════════════════════
# PKL config
# ══════════════════════════════════════════════════════════
PKL_FILES = {
    'rf'      : 'model_rf_pupuk.pkl',
    'xgb'     : 'model_xgb_pupuk.pkl',
    'le_pupuk': 'encoder_pupuk.pkl',
    'le_kec'  : 'encoder_kecamatan.pkl',
}

@st.cache_resource
def load_models():
    candidates = [
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd(),
        '/mount/src/' + os.path.basename(os.getcwd()),
    ]
    base = candidates[0]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, 'model_rf_pupuk.pkl')):
            base = cand
            break

    models = {'_base': base}; missing = []
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
    <div class="brand">
        <div class="brand-icon">🌱</div>
        <div>
            <div class="brand-name">PupukML</div>
            <div class="brand-sub">Prediksi Realisasi · 2017–2025</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="slbl" style="margin-top:0;">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:12.5px;color:#a39e98;margin-bottom:10px;">Upload .xlsx untuk EDA. Model load otomatis dari pkl.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Dataset .xlsx", type=["xlsx"], label_visibility="collapsed")

    st.markdown('<div class="slbl">Fitur Model</div>', unsafe_allow_html=True)
    for f, icon in [("Jenis Pupuk","🌿"),("Kecamatan","📍"),("Bulan","📅"),("Tahun","🗓️"),("RDKK (ton)","📋"),("Alokasi (ton)","📦")]:
        st.markdown(f'<div class="nav-item"><div class="nav-dot"></div>{icon} {f}</div>', unsafe_allow_html=True)

    st.markdown('<div class="slbl">Target Prediksi</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item" style="color:#533afd !important;font-weight:600;"><div class="nav-dot" style="background:#533afd;"></div>⚡ Realisasi (ton)</div>', unsafe_allow_html=True)

    st.markdown('<div class="slbl">Algoritma</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item"><span class="pill purple" style="font-size:11px;">RF</span> &nbsp;Random Forest</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item"><span class="pill green" style="font-size:11px;">XGB</span> &nbsp;XGBoost</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════
M = load_models()
models_ok = len(M['missing']) == 0

# ── PAGE HEADER ──
st.markdown("""
<div class="ph">
    <div class="ph-eyebrow">🤖 Machine Learning &nbsp;·&nbsp; Regresi</div>
    <div class="ph-title">Prediksi Realisasi<br>Penyaluran Pupuk Bersubsidi</div>
    <div class="ph-desc">
        Dua model ensemble — Random Forest &amp; XGBoost —
        memprediksi realisasi distribusi pupuk bersubsidi
        berdasarkan data historis Lombok Barat 2017–2025.
    </div>
</div>
""", unsafe_allow_html=True)

# ── PKL MISSING STATE ──
if not models_ok:
    st.markdown(f"""
    <div class="istrip err">
        <span>⚠️</span>
        <div>File pkl belum tersedia: <strong>{', '.join(M['missing'])}</strong><br>
        <span style="font-size:12px;color:#a39e98;">Ikuti langkah di bawah untuk generate dan deploy pkl ke GitHub.</span></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Debug Info"):
        st.code(f"""Base dir : {M.get('_base','?')}
CWD      : {os.getcwd()}
Files    :
{chr(10).join(sorted(os.listdir(M.get('_base','.'))[:30]))}""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sc"><div class="sc-ttl">Langkah 1 — Google Colab</div>', unsafe_allow_html=True)
        for num, step in [
            ("1","Buka <code>Prediksi_Pupuk_Bersubsidi.ipynb</code> di Google Colab"),
            ("2","Jalankan Bagian 1 sampai Bagian 11 — training selesai"),
            ("3","Jalankan <strong>Bagian 12</strong> → pkl tersimpan ke <code>MyDrive/pupuk_model/</code>"),
            ("4","Jalankan cell download → 5 file diunduh ke komputer"),
        ]:
            st.markdown(f'<div class="upload-step"><div class="step-num">{num}</div><div class="step-txt">{step}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="sc"><div class="sc-ttl">Langkah 2 — GitHub & Deploy</div>', unsafe_allow_html=True)
        for num, step in [
            ("5","Taruh semua pkl + <code>feature_names.json</code> di root repo GitHub"),
            ("6","Commit & push ke branch main"),
            ("7","Streamlit Cloud otomatis re-deploy"),
            ("8","App load model instan — tanpa training ulang"),
        ]:
            st.markdown(f'<div class="upload-step"><div class="step-num">{num}</div><div class="step-txt">{step}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div><div class="sc-ttl">File yang dibutuhkan di root repo</div>', unsafe_allow_html=True)
        for f in ['model_rf_pupuk.pkl','model_xgb_pupuk.pkl','encoder_pupuk.pkl','encoder_kecamatan.pkl','feature_names.json']:
            st.markdown(f'<div style="font-size:13px;padding:4px 0;color:#615d59;"><code style="background:#f6f5f4;padding:2px 7px;border-radius:3px;border:1px solid rgba(0,0,0,0.08);font-size:12px;color:#061b31;">{f}</code></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ── MODEL OK ──
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
    <span>Model berhasil di-load dari pkl &nbsp;·&nbsp;
    <strong style="color:#097fe8;">Siap prediksi</strong> — tidak perlu training ulang.</span>
</div>
""", unsafe_allow_html=True)

# ── KPI STRIP ──
st.markdown(f"""
<div class="kpi-wrap">
    <div class="kpi">
        <div class="kpi-lbl">Algoritma</div>
        <div class="kpi-val" style="font-size:17px;letter-spacing:-0.3px;">RF + XGB</div>
        <div class="kpi-unit">pkl loaded ✓</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">RF Trees</div>
        <div class="kpi-val">{rf_model.n_estimators}</div>
        <div class="kpi-unit">n_estimators</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">XGB Trees</div>
        <div class="kpi-val">{xgb_model.n_estimators}</div>
        <div class="kpi-unit">n_estimators</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">Jenis Pupuk</div>
        <div class="kpi-val">{len(jenis_list)}</div>
        <div class="kpi-unit">kategori</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">Kecamatan</div>
        <div class="kpi-val">{len(kec_list)}</div>
        <div class="kpi-unit">wilayah</div>
    </div>
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
# TABS
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Data & EDA",
    "🤖  Evaluasi Model",
    "📈  Feature Importance",
    "🔮  Prediksi Manual",
    "📋  Kesimpulan",
])

# ─────────────────────────────────────────────
# TAB 1 — EDA
# ─────────────────────────────────────────────
with tab1:
    if df is None:
        st.markdown("""
        <div class="istrip warn">
            <span>ℹ️</span>
            <span>Upload <strong>Data-pupuk-bersubsidi.xlsx</strong> di sidebar untuk melihat EDA.
            Tab Evaluasi, Feature Importance, dan Prediksi tetap berfungsi penuh.</span>
        </div>
        <div class="fc-grid">
            <div class="fc"><div class="fc-icon">📊</div>
                <div class="fc-title">Realisasi per Jenis Pupuk</div>
                <div class="fc-desc">Bar chart total realisasi 2017–2025 per kategori pupuk</div></div>
            <div class="fc"><div class="fc-icon">📈</div>
                <div class="fc-title">Tren Realisasi Tahunan</div>
                <div class="fc-desc">Line chart total realisasi per tahun dengan fill area</div></div>
            <div class="fc"><div class="fc-icon">🔥</div>
                <div class="fc-title">Heatmap Korelasi</div>
                <div class="fc-desc">Korelasi antar 7 variabel numerik dataset</div></div>
            <div class="fc"><div class="fc-icon">📏</div>
                <div class="fc-title">RDKK vs Alokasi vs Realisasi</div>
                <div class="fc-desc">Bar grouped per 8 kecamatan Lombok Barat</div></div>
            <div class="fc"><div class="fc-icon">🔵</div>
                <div class="fc-title">Scatter Alokasi vs Realisasi</div>
                <div class="fc-desc">Per jenis pupuk + garis ideal 1:1</div></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="slbl">Dataset</div>', unsafe_allow_html=True)
        with st.expander(f"Lihat Data — {len(df):,} baris, {len(df.columns)} kolom"):
            st.dataframe(df.head(15), use_container_width=True)
        with st.expander("Statistik Deskriptif"):
            st.dataframe(df.describe().round(3), use_container_width=True)

        st.markdown('<div class="slbl">Visualisasi</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="sc"><div class="sc-ttl">Total Realisasi per Jenis Pupuk</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            d = df.groupby('Jenis Pupuk')['Realisasi (ton)'].sum().sort_values(ascending=False)
            bars = ax.bar(d.index, d.values, color=CLRS[:len(d)], edgecolor='none', width=0.5, zorder=3)
            ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
            ax.set_ylabel("Realisasi (ton)"); ax.set_xlabel("")
            for bar, v in zip(bars, d.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+d.max()*0.012,
                        f'{v:,.0f}', ha='center', fontsize=7.5, color=CN, fontweight='600')
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="sc"><div class="sc-ttl">Tren Realisasi per Tahun</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            tren = df.groupby('Tahun')['Realisasi (ton)'].sum()
            ax.plot(tren.index, tren.values, marker='o', color=CP, linewidth=2.5,
                    markersize=8, markerfacecolor='#ffffff', markeredgewidth=2.5, zorder=3)
            ax.fill_between(tren.index, tren.values, alpha=0.08, color=CP)
            ax.set_xlabel("Tahun"); ax.set_ylabel("Realisasi (ton)")
            ax.set_xticks(tren.index); ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="sc"><div class="sc-ttl">Heatmap Korelasi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 4.4))
            cols_c = ['Bulan','Tahun','RDKK (ton)','Alokasi (ton)','Jenis Pupuk Enc','Kecamatan Enc','Realisasi (ton)']
            cmap = sns.diverging_palette(230, 150, s=70, l=55, as_cmap=True)
            sns.heatmap(df[cols_c].corr(), annot=True, fmt='.2f', cmap=cmap, ax=ax,
                        linewidths=0.5, linecolor='#ffffff',
                        annot_kws={'size':7.5,'weight':'600','color':CN},
                        vmin=-1, vmax=1, center=0, cbar_kws={'shrink':0.75})
            ax.tick_params(axis='x', rotation=30); ax.tick_params(axis='y', rotation=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="sc"><div class="sc-ttl">RDKK vs Alokasi vs Realisasi per Kecamatan</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 4.4))
            kec = df.groupby('Kecamatan')[['RDKK (ton)','Alokasi (ton)','Realisasi (ton)']].sum()
            x = np.arange(len(kec)); w = 0.26
            ax.bar(x-w, kec['RDKK (ton)'],     w, label='RDKK',     color=CS,  edgecolor='none', zorder=3)
            ax.bar(x,   kec['Alokasi (ton)'],   w, label='Alokasi',  color='#fbbf24', edgecolor='none', zorder=3)
            ax.bar(x+w, kec['Realisasi (ton)'], w, label='Realisasi',color=CP,  edgecolor='none', zorder=3)
            ax.set_xticks(x); ax.set_xticklabels(kec.index, rotation=35, ha='right', fontsize=7)
            ax.set_ylabel("Total (ton)"); ax.legend(fontsize=8)
            ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sc"><div class="sc-ttl">Scatter: Alokasi vs Realisasi per Jenis Pupuk</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, p in enumerate(sorted(df['Jenis Pupuk'].unique())):
            sub = df[df['Jenis Pupuk'] == p]
            ax.scatter(sub['Alokasi (ton)'], sub['Realisasi (ton)'],
                       label=p, alpha=0.5, s=25, color=CLRS[i % len(CLRS)], zorder=3)
        mv = max(df['Alokasi (ton)'].max(), df['Realisasi (ton)'].max())
        ax.plot([0, mv], [0, mv], '--', color='#a39e98', lw=1.5, label='Ideal 1:1')
        ax.set_xlabel("Alokasi (ton)"); ax.set_ylabel("Realisasi (ton)")
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 2 — EVALUASI MODEL
# ─────────────────────────────────────────────
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
                return {'MAE': mean_absolute_error(yt,yp),
                        'MSE': mean_squared_error(yt,yp),
                        'RMSE': np.sqrt(mean_squared_error(yt,yp)),
                        'R2': r2_score(yt,yp)}
            mRF = Mtr(yte, rfp); mXGB = Mtr(yte, xgbp)
            live_eval = True
        except Exception as e:
            st.warning(f"Gagal evaluasi live: {e}")

    if not live_eval:
        mRF  = {'MAE': 8.71,  'MSE': 424.98, 'RMSE': 20.62, 'R2': 0.9919}
        mXGB = {'MAE': 7.63,  'MSE': 269.29, 'RMSE': 16.41, 'R2': 0.9948}
        st.markdown("""
        <div class="istrip warn">
            <span>📌</span>
            <span>Menampilkan hasil dari notebook (Colab). Upload dataset di sidebar untuk evaluasi live.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="slbl">Kartu Metrik</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="mb mb-rf">
            <div class="mb-name">🌲 Random Forest (Tuned)</div>
            <div class="mb-grid">
                <div class="mb-item"><div class="mb-key">MAE</div><div class="mb-val mb-val-b">{mRF['MAE']:.4f} <span style="font-size:11px;color:#a39e98;">ton</span></div></div>
                <div class="mb-item"><div class="mb-key">RMSE</div><div class="mb-val mb-val-b">{mRF['RMSE']:.4f} <span style="font-size:11px;color:#a39e98;">ton</span></div></div>
                <div class="mb-item"><div class="mb-key">MSE</div><div class="mb-val">{mRF['MSE']:.2f}</div></div>
                <div class="mb-item"><div class="mb-key">R² Score</div><div class="mb-val mb-val-b">{mRF['R2']:.4f}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="mb mb-xgb">
            <div class="mb-name">⚡ XGBoost (Tuned)</div>
            <div class="mb-grid">
                <div class="mb-item"><div class="mb-key">MAE</div><div class="mb-val mb-val-g">{mXGB['MAE']:.4f} <span style="font-size:11px;color:#a39e98;">ton</span></div></div>
                <div class="mb-item"><div class="mb-key">RMSE</div><div class="mb-val mb-val-g">{mXGB['RMSE']:.4f} <span style="font-size:11px;color:#a39e98;">ton</span></div></div>
                <div class="mb-item"><div class="mb-key">MSE</div><div class="mb-val">{mXGB['MSE']:.2f}</div></div>
                <div class="mb-item"><div class="mb-key">R² Score</div><div class="mb-val mb-val-g">{mXGB['R2']:.4f}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="slbl">Perbandingan Visual</div>', unsafe_allow_html=True)
    ca, cb, cc = st.columns(3)
    for col, key, ttl, fmt in [
        (ca, 'R2',   'R² Score',   '{:.4f}'),
        (cb, 'RMSE', 'RMSE (ton)', '{:.2f}'),
        (cc, 'MAE',  'MAE (ton)',  '{:.2f}'),
    ]:
        with col:
            st.markdown('<div class="sc">', unsafe_allow_html=True)
            st.markdown(f'<div class="sc-ttl">{ttl}</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(4, 3.2))
            vals = [mRF[key], mXGB[key]]
            bars = ax.bar(['Random\nForest','XGBoost'], vals,
                          color=[CP, CG], edgecolor='none', width=0.45, zorder=3)
            if key == 'R2': ax.set_ylim(0.98, 1.0)
            ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height() + (0.0003 if key=='R2' else max(vals)*0.015),
                        fmt.format(v), ha='center', fontsize=9.5, fontweight='700', color=CN)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="slbl">Tabel Perbandingan</div>', unsafe_allow_html=True)
    tbl = pd.DataFrame({
        'Model'      : ['🌲 Random Forest (Tuned)', '⚡ XGBoost (Tuned)'],
        'MAE (ton)'  : [mRF['MAE'],  mXGB['MAE']],
        'MSE'        : [mRF['MSE'],  mXGB['MSE']],
        'RMSE (ton)' : [mRF['RMSE'], mXGB['RMSE']],
        'R² Score'   : [mRF['R2'],   mXGB['R2']],
    }).round(4)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    if live_eval:
        st.markdown('<div class="slbl">Aktual vs Prediksi</div>', unsafe_allow_html=True)
        cx, cy = st.columns(2)
        for col, pred, color, lbl in [
            (cx, rfp,  CP, '🌲 Random Forest'),
            (cy, xgbp, CG, '⚡ XGBoost'),
        ]:
            with col:
                st.markdown(f'<div class="sc"><div class="sc-ttl">{lbl} — Aktual vs Prediksi</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5.2, 4))
                ax.scatter(yte, pred, alpha=0.45, color=color, s=22, zorder=3)
                mv = max(float(yte.max()), float(pred.max()))
                ax.plot([0, mv], [0, mv], '--', color='#a39e98', lw=1.5, label='Ideal')
                ax.set_xlabel("Aktual (ton)"); ax.set_ylabel("Prediksi (ton)")
                ax.legend(fontsize=8)
                ax.yaxis.grid(True, zorder=0, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 3 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="slbl">Feature Importance Score</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="istrip">
        <span>ℹ️</span>
        <span>Semakin tinggi importance score, semakin besar pengaruh fitur terhadap hasil prediksi realisasi.</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, model_obj, color, lbl in [
        (c1, rf_model,  CP, '🌲 Random Forest (Tuned)'),
        (c2, xgb_model, CG, '⚡ XGBoost (Tuned)'),
    ]:
        with col:
            imp = pd.Series(model_obj.feature_importances_, index=FEATURES).sort_values()
            st.markdown(f'<div class="sc"><div class="sc-ttl">{lbl}</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            bars = ax.barh(imp.index, imp.values, color=color,
                           edgecolor='none', height=0.5, zorder=3)
            ax.xaxis.grid(True, zorder=0, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
            ax.set_xlabel("Importance Score")
            for bar, v in zip(bars, imp.values):
                ax.text(v + imp.max()*0.015, bar.get_y()+bar.get_height()/2,
                        f'{v:.4f}', va='center', fontsize=8.5, fontweight='600', color=CN)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            t = imp.sort_values(ascending=False).reset_index()
            t.columns = ['Fitur','Importance']; t['Importance'] = t['Importance'].round(4)
            st.dataframe(t, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 4 — PREDIKSI MANUAL
# ─────────────────────────────────────────────
with tab4:
    st.markdown('<div class="slbl">Input Parameter Prediksi</div>', unsafe_allow_html=True)

    with st.form("form_pred"):
        c1, c2, c3 = st.columns(3)
        with c1:
            jenis   = st.selectbox("Jenis Pupuk", sorted(jenis_list))
            kec_sel = st.selectbox("Kecamatan", sorted(kec_list))
        with c2:
            bmap = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
                    7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"}
            bl   = st.selectbox("Bulan", list(bmap.values()), index=5)
            bnum = [k for k, v in bmap.items() if v == bl][0]
            thn  = st.number_input("Tahun", min_value=2017, max_value=2035, value=2025)
        with c3:
            rdkk    = st.number_input("RDKK (ton)",    min_value=0.0, value=500.0, step=10.0)
            alokasi = st.number_input("Alokasi (ton)", min_value=0.0, value=300.0, step=10.0)
        sub = st.form_submit_button("⚡ Hitung Prediksi", use_container_width=True, type="primary")

    if sub:
        inp = pd.DataFrame([{
            'Jenis Pupuk Enc': le_pupuk.transform([jenis])[0],
            'Kecamatan Enc'  : le_kec.transform([kec_sel])[0],
            'Bulan': bnum, 'Tahun': thn,
            'RDKK (ton)': rdkk, 'Alokasi (ton)': alokasi
        }])[FEATURES]

        prf  = float(rf_model.predict(inp)[0])
        pxgb = float(xgb_model.predict(inp)[0])

        st.markdown('<div class="slbl">Hasil Prediksi</div>', unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="pr pr-accent">
                <div class="pr-model">🌲 Random Forest (Tuned)</div>
                <div class="pr-num pr-num-b">{prf:.2f}</div>
                <div class="pr-unit">ton realisasi diprediksi</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="pr pr-green">
                <div class="pr-model">⚡ XGBoost (Tuned)</div>
                <div class="pr-num pr-num-g">{pxgb:.2f}</div>
                <div class="pr-unit">ton realisasi diprediksi</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown('<div class="slbl">Ringkasan Input</div>', unsafe_allow_html=True)
            inp_tbl = pd.DataFrame({
                'Parameter': ['Jenis Pupuk','Kecamatan','Bulan','Tahun','RDKK (ton)','Alokasi (ton)'],
                'Nilai'    : [jenis, kec_sel, bl, thn, rdkk, alokasi]
            })
            st.dataframe(inp_tbl, use_container_width=True, hide_index=True)
        with s2:
            if alokasi > 0:
                st.markdown('<div class="sc"><div class="sc-ttl">% Penyerapan Alokasi</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(4.2, 2.6))
                p1 = (prf  / alokasi) * 100
                p2 = (pxgb / alokasi) * 100
                bars = ax.barh(['XGBoost','Random\nForest'], [p2, p1],
                               color=[CG, CP], edgecolor='none', height=0.4, zorder=3)
                ax.axvline(100, color='#a39e98', linestyle='--', lw=1.5)
                ax.set_xlabel("% Penyerapan")
                ax.xaxis.grid(True, zorder=0, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
                for bar, v in zip(bars, [p2, p1]):
                    ax.text(v + 1, bar.get_y()+bar.get_height()/2,
                            f'{v:.1f}%', va='center', fontsize=9.5, fontweight='600', color=CN)
                ax.set_xlim(0, max(p1, p2) * 1.3)
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 5 — KESIMPULAN
# ─────────────────────────────────────────────
with tab5:
    st.markdown(f"""
    <div class="istrip">
        <span>🏆</span>
        <span>Model terbaik: <strong style="color:#061b31;">XGBoost (Tuned)</strong>
        &nbsp;·&nbsp; R² = <strong style="color:#533afd;">{mXGB['R2']:.4f}</strong>
        &nbsp;·&nbsp; RMSE = <strong style="color:#533afd;">{mXGB['RMSE']:.4f} ton</strong>
        &nbsp;·&nbsp; <span class="pill green">Sangat Akurat</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sc"><div class="sc-ttl">Alur Proyek</div>', unsafe_allow_html=True)
        for title, desc in [
            ("📥 Load Data",         "Dataset pupuk bersubsidi 2017–2025, 1082+ baris"),
            ("🔍 EDA",               "Distribusi, tren, korelasi — 5 visualisasi interaktif"),
            ("⚙️ Preprocessing",     "ffill Bulan, drop leakage, label encoding, imputasi median"),
            ("🤖 Modeling",          "RF & XGBoost, split 80/20, random_state=42"),
            ("🔧 Tuning",            "GridSearchCV 5-fold cross-validation"),
            ("📊 Evaluasi",          "MAE, MSE, RMSE, R² — perbandingan 4 varian model"),
            ("📈 Feature Importance","Alokasi (ton) dominan di kedua model"),
            ("💾 Simpan pkl",        "Model & encoder tersimpan ke Google Drive"),
            ("🚀 Deploy Streamlit",  "app.py load pkl — prediksi instan tanpa training ulang"),
        ]:
            st.markdown(f'<div class="cl"><div class="cl-dot"></div><div class="cl-txt"><strong>{title}</strong> — {desc}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sc"><div class="sc-ttl">Temuan Utama</div>', unsafe_allow_html=True)
        for title, desc in [
            ("Alokasi (ton)",         "fitur paling berpengaruh di kedua model — langsung menentukan realisasi"),
            ("Gap RDKK vs Alokasi",   "konsisten terjadi di semua kecamatan — alokasi selalu < RDKK"),
            ("R² > 0.99",             "kedua model sangat akurat, bias sangat rendah"),
            ("XGBoost > RF",          "XGBoost unggul di MAE, RMSE, dan R² secara konsisten"),
            ("pkl deployment",        "load instan di Streamlit Cloud — tidak perlu training ulang"),
        ]:
            st.markdown(f'<div class="cl"><div class="cl-dot"></div><div class="cl-txt"><strong>{title}</strong> — {desc}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="divider" style="margin:16px 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sc-ttl">Saran</div>', unsafe_allow_html=True)
        for s in [
            "Gunakan model untuk estimasi realisasi sebelum musim tanam dimulai",
            "Data tanpa duplikasi & lebih granular akan meningkatkan akurasi",
            "Tambahkan fitur curah hujan / musim untuk konteks agronomis",
            "Dashboard ini memudahkan dinas pertanian akses prediksi real-time",
        ]:
            st.markdown(f'<div class="cl"><div class="cl-dot" style="background:#15be53;"></div><div class="cl-txt">{s}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="slbl">Evaluasi Final</div>', unsafe_allow_html=True)
    tbl_f = pd.DataFrame({
        'Model'      : ['🌲 Random Forest (Tuned)', '⚡ XGBoost (Tuned)'],
        'MAE (ton)'  : [mRF['MAE'],  mXGB['MAE']],
        'MSE'        : [mRF['MSE'],  mXGB['MSE']],
        'RMSE (ton)' : [mRF['RMSE'], mXGB['RMSE']],
        'R² Score'   : [mRF['R2'],   mXGB['R2']],
    }).round(4)
    st.dataframe(tbl_f, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="ft">
    <div class="ft-l">Prediksi Realisasi Penyaluran Pupuk Bersubsidi &nbsp;·&nbsp; Pemrograman 5</div>
    <div class="ft-r">Random Forest · XGBoost · pkl deployment · Streamlit Cloud</div>
</div>
""", unsafe_allow_html=True)
