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
# CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pupuk Prediksi",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# CSS — Minimal light dashboard (no template feel)
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

/* BASE */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
    background: #fafaf8;
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    color: #2d2d2d;
}

.block-container {
    padding: 24px 32px 48px !important;
    max-width: 1400px !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #ffffff !important;
}
[data-testid="stSidebar"] > div { padding: 20px 16px; }
[data-testid="stSidebar"] * { color: #2d2d2d !important; }

[data-testid="stSidebar"] .stFileUploader {
    background: #f5f5f3;
    border: 1px solid #e0e0dc;
    border-radius: 6px;
    padding: 10px;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #e0e0dc;
    padding: 0;
}

.stTabs [data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 400;
    color: #888;
    padding: 12px 16px;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: #2d2d2d !important;
    border-bottom: 2px solid #5b5b5b !important;
    font-weight: 500 !important;
}

.stTabs [data-baseweb="tab-panel"] { padding-top: 24px; }

/* BUTTONS */
.stButton > button {
    background: #f5f5f3 !important;
    border: 1px solid #d0d0cc !important;
    color: #2d2d2d !important;
    border-radius: 5px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}

.stButton > button:hover {
    background: #efefeb !important;
    border-color: #c0c0bb !important;
}

[data-testid="stFormSubmitButton"] > button {
    background: #2d2d2d !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    border-radius: 5px !important;
    width: 100%;
}

[data-testid="stFormSubmitButton"] > button:hover {
    background: #1a1a1a !important;
}

/* INPUTS */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #e0e0dc !important;
    color: #2d2d2d !important;
    border-radius: 5px !important;
    font-size: 13px !important;
}

/* DATA */
.stDataFrame { border: 1px solid #e0e0dc !important; border-radius: 6px !important; }

/* EXPANDER */
.streamlit-expanderHeader {
    background: #f5f5f3 !important;
    border: 1px solid #e0e0dc !important;
    border-radius: 5px !important;
    color: #2d2d2d !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

.streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e0e0dc !important;
    border-top: none !important;
}

/* HIDE CHROME */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MATPLOTLIB
# ══════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor': '#fafaf8',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#e0e0dc',
    'axes.labelcolor': '#888',
    'axes.titlecolor': '#2d2d2d',
    'axes.titlesize': 13,
    'axes.titleweight': '600',
    'axes.labelsize': 11,
    'xtick.color': '#888',
    'ytick.color': '#888',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'grid.color': '#f0f0ed',
    'grid.linewidth': 0.8,
    'legend.facecolor': '#ffffff',
    'legend.edgecolor': '#e0e0dc',
    'legend.fontsize': 10,
    'text.color': '#2d2d2d',
})

C1 = '#2d2d2d'  # dark
C2 = '#5b5b5b'  # gray
C3 = '#888'     # light gray
C4 = '#ea5455'  # accent red
C5 = '#4caf50'  # green
CLRS = [C1, C4, C5, '#ff9800', '#2196f3', '#9c27b0']

# ══════════════════════════════════════════════════════════
# MODEL LOAD
# ══════════════════════════════════════════════════════════
PKL_FILES = {
    'rf'      : 'model_rf_pupuk.pkl',
    'xgb'     : 'model_xgb_pupuk.pkl',
    'le_pupuk': 'encoder_pupuk.pkl',
    'le_kec'  : 'encoder_kecamatan.pkl',
}

@st.cache_resource
def load_models():
    # Try multiple paths — Streamlit Cloud mounts at /mount/src/REPO
    base = os.path.dirname(os.path.abspath(__file__))
    
    # If __file__ not in mounted repo, try /mount/src
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
                err_msg = str(e) if str(e) else type(e).__name__
                missing.append(f"{fname} (exists but load failed: {err_msg[:50]})")
        else:
            missing.append(f"{fname} (file not found at {path})")

    feat_path = os.path.join(base, 'feature_names.json')
    if os.path.exists(feat_path):
        try:
            with open(feat_path) as f:
                models['features'] = json.load(f)
        except:
            models['features'] = ['Jenis Pupuk Enc','Kecamatan Enc','Bulan','Tahun','RDKK (ton)','Alokasi (ton)']
    else:
        models['features'] = ['Jenis Pupuk Enc','Kecamatan Enc','Bulan','Tahun','RDKK (ton)','Alokasi (ton)']

    models['missing'] = missing
    return models

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("**Pupuk Prediksi**", help="ML realisasi 2017–2025")
    st.divider()
    
    st.markdown("##### Data")
    uploaded = st.file_uploader("Upload .xlsx untuk EDA", type=["xlsx"], label_visibility="collapsed")
    
    st.markdown("##### Model")
    st.caption("Random Forest + XGBoost")
    st.caption("Load dari pkl — tanpa training ulang")

# ══════════════════════════════════════════════════════════
# LOAD & CHECK
# ══════════════════════════════════════════════════════════
M = load_models()
models_ok = len(M['missing']) == 0

# ══════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("## 🌱", help="Prediksi realisasi pupuk")
with col2:
    st.markdown("#### Prediksi Realisasi Penyaluran Pupuk Bersubsidi")
    st.caption("Machine Learning · Random Forest + XGBoost · Dataset 2017–2025")

st.divider()

# ══════════════════════════════════════════════════════════
# ERROR STATE — PKL MISSING
# ══════════════════════════════════════════════════════════
if not models_ok:
    st.error(f"Model files missing: {', '.join(M['missing'])}")
    
    with st.expander("Setup Instructions"):
        st.write("""
        **Step 1: Google Colab**
        1. Jalankan notebook Bagian 1–11
        2. Jalankan Bagian 12 (simpan pkl ke Google Drive)
        3. Jalankan cell download
        
        **Step 2: GitHub**
        1. Upload 5 file pkl ke root repo
        2. Commit & push
        3. Streamlit Cloud auto re-deploy
        """)
    
    with st.expander("Debug Info"):
        st.code(f"Base: {M.get('_base', '?')}\nCWD: {os.getcwd()}")
    
    st.stop()

# ══════════════════════════════════════════════════════════
# SUCCESS — MODEL OK
# ══════════════════════════════════════════════════════════
rf_model  = M['rf']
xgb_model = M['xgb']
le_pupuk  = M['le_pupuk']
le_kec    = M['le_kec']
FEATURES  = M['features']
jenis_list = list(le_pupuk.classes_)
kec_list   = list(le_kec.classes_)

st.success("Model loaded ✓")

# ══════════════════════════════════════════════════════════
# DASHBOARD — METRICS
# ══════════════════════════════════════════════════════════
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("RF Trees", rf_model.n_estimators)
with col2:
    st.metric("XGB Trees", xgb_model.n_estimators)
with col3:
    st.metric("Jenis Pupuk", len(jenis_list))
with col4:
    st.metric("Kecamatan", len(kec_list))
with col5:
    st.metric("Data Span", "2017–2025")

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
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Model", "Prediksi", "Insights"])

# ─────────────────────────
# TAB: DATA
# ─────────────────────────
with tab1:
    if df is None:
        st.info("Upload .xlsx di sidebar untuk melihat data & EDA")
    else:
        st.subheader(f"Dataset · {len(df)} rows × {len(df.columns)} cols")
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(df.describe())
        with col2:
            st.write(df.groupby('Jenis Pupuk')['Realisasi (ton)'].sum().sort_values(ascending=False))
        
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            d = df.groupby('Jenis Pupuk')['Realisasi (ton)'].sum().sort_values(ascending=False)
            ax.bar(d.index, d.values, color=CLRS[:len(d)], alpha=0.7, edgecolor=C2, linewidth=0.5)
            ax.set_ylabel("Realisasi (ton)")
            ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            tren = df.groupby('Tahun')['Realisasi (ton)'].sum()
            ax.plot(tren.index, tren.values, marker='o', linewidth=2, markersize=6, color=C1)
            ax.fill_between(tren.index, tren.values, alpha=0.1, color=C1)
            ax.set_ylabel("Realisasi (ton)")
            ax.set_xlabel("Tahun")
            ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            st.pyplot(fig, use_container_width=True)

# ─────────────────────────
# TAB: MODEL
# ─────────────────────────
with tab2:
    st.subheader("Model Evaluation")
    
    live_eval = False
    if df is not None:
        try:
            from sklearn.model_selection import train_test_split
            df2 = df.copy()
            df2['Jenis Pupuk Enc'] = le_pupuk.transform(df2['Jenis Pupuk'])
            df2['Kecamatan Enc']   = le_kec.transform(df2['Kecamatan'])
            X = df2[FEATURES]
            y = df2['Realisasi (ton)']
            _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            rfp  = rf_model.predict(Xte)
            xgbp = xgb_model.predict(Xte)
            
            def metrics(yt, yp):
                return {
                    'MAE': mean_absolute_error(yt, yp),
                    'RMSE': np.sqrt(mean_squared_error(yt, yp)),
                    'R2': r2_score(yt, yp)
                }
            
            mrf = metrics(yte, rfp)
            mxgb = metrics(yte, xgbp)
            live_eval = True
        except:
            pass
    
    if not live_eval:
        mrf = {'MAE': 8.71, 'RMSE': 20.62, 'R2': 0.9919}
        mxgb = {'MAE': 7.63, 'RMSE': 16.41, 'R2': 0.9948}
        st.caption("Pre-calculated from notebook")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Random Forest")
        st.metric("MAE", f"{mrf['MAE']:.2f}")
        st.metric("RMSE", f"{mrf['RMSE']:.2f}")
        st.metric("R²", f"{mrf['R2']:.4f}")
    
    with col2:
        st.write("#### XGBoost")
        st.metric("MAE", f"{mxgb['MAE']:.2f}")
        st.metric("RMSE", f"{mxgb['RMSE']:.2f}")
        st.metric("R²", f"{mxgb['R2']:.4f}")
    
    st.divider()
    st.subheader("Feature Importance")
    col1, col2 = st.columns(2)
    
    with col1:
        imp = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(imp.index, imp.values, color=C1, alpha=0.6, edgecolor=C2, linewidth=0.5)
        ax.set_xlabel("Importance")
        st.pyplot(fig, use_container_width=True)
        st.caption("Random Forest")
    
    with col2:
        imp = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(imp.index, imp.values, color=C4, alpha=0.6, edgecolor=C2, linewidth=0.5)
        ax.set_xlabel("Importance")
        st.pyplot(fig, use_container_width=True)
        st.caption("XGBoost")

# ─────────────────────────
# TAB: PREDIKSI
# ─────────────────────────
with tab3:
    st.subheader("Prediksi Manual")
    
    with st.form("pred_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            jenis = st.selectbox("Jenis Pupuk", sorted(jenis_list))
            kec = st.selectbox("Kecamatan", sorted(kec_list))
        
        with col2:
            bulan_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                         7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            bln_name = st.selectbox("Bulan", list(bulan_map.values()), index=5)
            bln = [k for k, v in bulan_map.items() if v == bln_name][0]
            tahun = st.number_input("Tahun", 2017, 2035, 2025)
        
        with col3:
            rdkk = st.number_input("RDKK (ton)", 0.0, 10000.0, 500.0)
            alokasi = st.number_input("Alokasi (ton)", 0.0, 10000.0, 300.0)
        
        submit = st.form_submit_button("Prediksi", use_container_width=True)
    
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
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Random Forest", f"{prf:.2f} ton")
        with col2:
            st.metric("XGBoost", f"{pxgb:.2f} ton")
        
        st.divider()
        st.subheader("Input Summary")
        tbl = pd.DataFrame({
            'Parameter': ['Jenis Pupuk','Kecamatan','Bulan','Tahun','RDKK','Alokasi'],
            'Value': [jenis, kec, bln_name, tahun, f"{rdkk:.0f}", f"{alokasi:.0f}"]
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

# ─────────────────────────
# TAB: INSIGHTS
# ─────────────────────────
with tab4:
    st.subheader("Insights & Conclusions")
    
    st.markdown("""
    **Model Performance**
    - XGBoost outperforms Random Forest on all metrics
    - R² > 0.99 — very accurate predictions
    - MAE < 10 ton — low prediction error
    
    **Key Findings**
    - Alokasi (ton) is the dominant feature
    - Consistent gap between RDKK and Alokasi across kecamatan
    - Strong seasonal patterns in realization
    
    **Recommendations**
    - Use model for budget planning
    - Monitor allocation vs realization monthly
    - Investigate allocation gaps
    """)
    
    st.divider()
    st.caption("Built with Random Forest + XGBoost | Streamlit Cloud Deploy | pkl-based inference")
