import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ══════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Prediksi Pupuk Bersubsidi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# PALET WARNA — LIGHT MODE PERTANIAN
# ══════════════════════════════════════════════════════════
# Hijau utama   : #2E7D32
# Hijau muda    : #66BB6A
# Hijau terang  : #E8F5E9
# Hijau border  : #A5D6A7
# Kuning aksen  : #F9A825
# Biru data     : #1565C0
# Abu teks      : #37474F
# Putih kartu   : #FFFFFF
# Abu latar     : #F5F7F5

# ══════════════════════════════════════════════════════════
# CSS — LIGHT MODE PREMIUM
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Root & Background ── */
    .stApp {
        background-color: #F5F7F5;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 60%, #388E3C 100%);
        border-right: none;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] .stFileUploader div {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }
    [data-testid="stSidebar"] .stInfo {
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 10px;
        color: #FFFFFF !important;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%);
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 24px;
        box-shadow: 0 4px 20px rgba(46,125,50,0.25);
    }
    .hero-icon { font-size: 4rem; line-height: 1; }
    .hero-title {
        font-size: 1.9rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.80);
        margin: 0;
    }
    .hero-badge {
        margin-left: auto;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.8rem;
        color: #FFFFFF;
        white-space: nowrap;
    }

    /* ── Kartu Statistik ── */
    .stat-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 20px 22px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        text-align: center;
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #78909C;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #2E7D32;
        line-height: 1.1;
    }
    .stat-unit {
        font-size: 0.75rem;
        color: #90A4AE;
        margin-top: 2px;
    }

    /* ── Section Header ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 24px 0 16px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #E8F5E9;
    }
    .section-icon {
        background: #E8F5E9;
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 1.2rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1B5E20;
        margin: 0;
    }

    /* ── Chart Card ── */
    .chart-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 16px;
    }
    .chart-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #37474F;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Metrik Model ── */
    .model-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-top: 4px solid #2E7D32;
    }
    .model-card-xgb { border-top-color: #F9A825; }
    .model-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1B5E20;
        margin-bottom: 16px;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        flex-wrap: wrap;
    }
    .metric-item {
        background: #F5F7F5;
        border-radius: 10px;
        padding: 12px 16px;
        flex: 1;
        min-width: 100px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #90A4AE;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 800;
        color: #2E7D32;
        margin-top: 4px;
    }

    /* ── Prediksi Result ── */
    .pred-card {
        background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        border: 2px solid #A5D6A7;
        box-shadow: 0 4px 16px rgba(46,125,50,0.12);
    }
    .pred-card-xgb {
        background: linear-gradient(135deg, #FFFDE7, #FFF9C4);
        border-color: #FFD54F;
        box-shadow: 0 4px 16px rgba(249,168,37,0.12);
    }
    .pred-model-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: #546E7A;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .pred-value {
        font-size: 3rem;
        font-weight: 900;
        color: #1B5E20;
        line-height: 1;
    }
    .pred-value-xgb { color: #E65100; }
    .pred-unit {
        font-size: 1rem;
        color: #78909C;
        margin-top: 6px;
    }

    /* ── Upload Zone ── */
    .upload-hero {
        background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
        border: 2px dashed #A5D6A7;
        border-radius: 16px;
        padding: 48px 32px;
        text-align: center;
        margin: 20px 0;
    }
    .upload-icon { font-size: 3.5rem; margin-bottom: 16px; }
    .upload-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 8px;
    }
    .upload-sub { font-size: 0.9rem; color: #78909C; }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #546E7A;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #E8F5E9 !important;
        color: #1B5E20 !important;
    }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #FFFFFF;
        border-radius: 10px;
        font-weight: 600;
        color: #2E7D32;
    }

    /* ── Badge ── */
    .badge {
        display: inline-block;
        background: #E8F5E9;
        color: #2E7D32;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.78rem;
        font-weight: 700;
        margin: 2px;
        border: 1px solid #A5D6A7;
    }
    .badge-yellow {
        background: #FFFDE7;
        color: #E65100;
        border-color: #FFD54F;
    }
    .badge-blue {
        background: #E3F2FD;
        color: #1565C0;
        border-color: #90CAF9;
    }

    /* ── Info box ── */
    .info-box {
        background: #E3F2FD;
        border-left: 4px solid #1565C0;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #0D47A1;
    }
    .success-box {
        background: #E8F5E9;
        border-left: 4px solid #2E7D32;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #1B5E20;
        font-weight: 600;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 20px;
        color: #90A4AE;
        font-size: 0.8rem;
        margin-top: 32px;
        border-top: 1px solid #E0E0E0;
    }

    /* ── Hide default streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# MATPLOTLIB THEME — LIGHT & CLEAN
# ══════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.facecolor'  : '#FFFFFF',
    'axes.facecolor'    : '#FAFAFA',
    'axes.edgecolor'    : '#E0E0E0',
    'axes.labelcolor'   : '#37474F',
    'axes.titlecolor'   : '#1B5E20',
    'axes.titlesize'    : 11,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 9,
    'xtick.color'       : '#546E7A',
    'ytick.color'       : '#546E7A',
    'xtick.labelsize'   : 8,
    'ytick.labelsize'   : 8,
    'grid.color'        : '#EEEEEE',
    'grid.linewidth'    : 0.8,
    'legend.fontsize'   : 8,
    'legend.framealpha' : 0.9,
    'font.family'       : 'DejaVu Sans',
})

# Palet warna chart
C_GREEN   = '#2E7D32'
C_GREEN2  = '#66BB6A'
C_YELLOW  = '#F9A825'
C_BLUE    = '#1565C0'
C_BLUE2   = '#42A5F5'
C_ORANGE  = '#E65100'
C_GREY    = '#90A4AE'
PALETTE   = [C_GREEN, C_YELLOW, C_BLUE, C_ORANGE, C_GREEN2]

# ══════════════════════════════════════════════════════════
# FUNGSI LOAD & PROSES DATA
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_and_process(file):
    df = pd.read_excel(file)
    df['Bulan'] = df['Bulan'].ffill().astype(int)
    kolom_drop = ['NO', 'Kelompok Tani', 'Alokasi Vs RDKK (%)', 'Realisasi Vs Alokasi (%)']
    df = df.drop(columns=[c for c in kolom_drop if c in df.columns])
    for col in ['Realisasi (ton)', 'RDKK (ton)', 'Alokasi (ton)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    df['Kecamatan']   = df['Kecamatan'].str.strip().str.title()
    df['Jenis Pupuk'] = df['Jenis Pupuk'].str.strip()
    le_pupuk = LabelEncoder()
    le_kec   = LabelEncoder()
    df['Jenis Pupuk Enc'] = le_pupuk.fit_transform(df['Jenis Pupuk'])
    df['Kecamatan Enc']   = le_kec.fit_transform(df['Kecamatan'])
    return df, le_pupuk, le_kec


@st.cache_resource
def train_models(_df):
    fitur  = ['Jenis Pupuk Enc', 'Kecamatan Enc', 'Bulan', 'Tahun', 'RDKK (ton)', 'Alokasi (ton)']
    target = 'Realisasi (ton)'
    X = _df[fitur]
    y = _df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    return {
        'rf_model' : rf, 'xgb_model': xgb,
        'X_test'   : X_test, 'y_test': y_test,
        'rf_pred'  : rf_pred, 'xgb_pred': xgb_pred,
        'fitur'    : fitur,
        'metrics'  : {
            'RF': {
                'MAE' : mean_absolute_error(y_test, rf_pred),
                'MSE' : mean_squared_error(y_test, rf_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'R2'  : r2_score(y_test, rf_pred)
            },
            'XGBoost': {
                'MAE' : mean_absolute_error(y_test, xgb_pred),
                'MSE' : mean_squared_error(y_test, xgb_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
                'R2'  : r2_score(y_test, xgb_pred)
            }
        }
    }

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌾 Pupuk Bersubsidi")
    st.markdown("**Prediksi Realisasi Penyaluran**")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset Excel",
        type=["xlsx"],
        help="Upload file Data-pupuk-bersubsidi.xlsx"
    )

    st.markdown("---")
    st.markdown("### 📌 Fitur Model")
    for f in ["Jenis Pupuk", "Kecamatan", "Bulan", "Tahun", "RDKK (ton)", "Alokasi (ton)"]:
        st.markdown(f"&nbsp;&nbsp;✦ {f}")

    st.markdown("### 🎯 Target")
    st.markdown("&nbsp;&nbsp;**Realisasi (ton)**")

    st.markdown("---")
    st.markdown("### 🤖 Algoritma")
    st.markdown("&nbsp;&nbsp;🌲 Random Forest Regressor")
    st.markdown("&nbsp;&nbsp;⚡ XGBoost Regressor")

    st.markdown("---")
    st.caption("Pemrograman 5 · ML End-to-End · 2017–2025")

# ══════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-icon">🌾</div>
    <div>
        <div class="hero-title">Prediksi Realisasi Penyaluran Pupuk Bersubsidi</div>
        <div class="hero-sub">Menggunakan Algoritma Random Forest &amp; XGBoost Regressor &nbsp;|&nbsp; Dataset Lombok Barat 2017–2025</div>
    </div>
    <div class="hero-badge">🎓 Pemrograman 5</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LANDING PAGE (sebelum upload)
# ══════════════════════════════════════════════════════════
if uploaded_file is None:
    st.markdown("""
    <div class="upload-hero">
        <div class="upload-icon">📂</div>
        <div class="upload-title">Upload Dataset untuk Memulai</div>
        <div class="upload-sub">Upload file <strong>Data-pupuk-bersubsidi.xlsx</strong> di sidebar kiri<br>
        Aplikasi akan otomatis memproses data dan melatih model</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="chart-card" style="text-align:center;">
            <div style="font-size:2.5rem;">📊</div>
            <div style="font-weight:700; color:#1B5E20; margin:8px 0 4px;">Eksplorasi Data</div>
            <div style="font-size:0.85rem; color:#78909C;">Visualisasi 5 chart interaktif dari data pupuk bersubsidi</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="chart-card" style="text-align:center;">
            <div style="font-size:2.5rem;">🤖</div>
            <div style="font-weight:700; color:#1B5E20; margin:8px 0 4px;">Dua Model ML</div>
            <div style="font-size:0.85rem; color:#78909C;">Random Forest vs XGBoost dengan evaluasi MAE, RMSE, R²</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="chart-card" style="text-align:center;">
            <div style="font-size:2.5rem;">🔮</div>
            <div style="font-weight:700; color:#1B5E20; margin:8px 0 4px;">Prediksi Manual</div>
            <div style="font-size:0.85rem; color:#78909C;">Input data baru → prediksi realisasi dalam ton secara langsung</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════
# PROSES DATA
# ══════════════════════════════════════════════════════════
with st.spinner("⏳ Memproses data dan melatih model..."):
    df, le_pupuk, le_kec = load_and_process(uploaded_file)
    hasil = train_models(df)

st.markdown("""
<div class="success-box">
    ✅ Dataset berhasil dimuat dan model selesai dilatih — silakan jelajahi tab di bawah.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# STATISTIK RINGKAS — ROW ATAS
# ══════════════════════════════════════════════════════════
c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    (c1, "Total Data",     f"{df.shape[0]:,}",                          "baris"),
    (c2, "Jenis Pupuk",    str(df['Jenis Pupuk'].nunique()),             "jenis"),
    (c3, "Kecamatan",      str(df['Kecamatan'].nunique()),               "wilayah"),
    (c4, "Rentang Tahun",  f"{int(df['Tahun'].min())}–{int(df['Tahun'].max())}", ""),
    (c5, "R² Terbaik",     f"{max(hasil['metrics']['RF']['R2'], hasil['metrics']['XGBoost']['R2']):.4f}", "score"),
]
for col, label, value, unit in stats:
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB NAVIGASI
# ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Data & EDA",
    "🤖  Evaluasi Model",
    "📈  Feature Importance",
    "🔮  Prediksi Manual",
    "📋  Kesimpulan"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — DATA & EDA
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🗂️</div>
        <div class="section-title">Eksplorasi Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📄 Lihat Data (10 Baris Pertama)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📐 Statistik Deskriptif", expanded=False):
        st.dataframe(df.describe().round(3), use_container_width=True)

    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <div class="section-title">Visualisasi Data</div>
    </div>
    """, unsafe_allow_html=True)

    # Baris 1
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📦 Total Realisasi per Jenis Pupuk</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        data_pupuk = df.groupby('Jenis Pupuk')['Realisasi (ton)'].sum().sort_values(ascending=False)
        bars = ax.bar(data_pupuk.index, data_pupuk.values, color=PALETTE, edgecolor='white', linewidth=1.2, zorder=3)
        ax.set_xlabel("Jenis Pupuk")
        ax.set_ylabel("Total Realisasi (ton)")
        ax.set_title("Realisasi per Jenis Pupuk (2017–2025)")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, data_pupuk.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                    f'{v:,.0f}', ha='center', fontsize=7.5, fontweight='bold', color='#37474F')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">📈 Tren Realisasi per Tahun</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 3.8))
        tren = df.groupby('Tahun')['Realisasi (ton)'].sum()
        ax.plot(tren.index, tren.values, marker='o', color=C_GREEN, linewidth=2.5,
                markersize=8, markerfacecolor='white', markeredgewidth=2.5, zorder=3)
        ax.fill_between(tren.index, tren.values, alpha=0.12, color=C_GREEN)
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Total Realisasi (ton)")
        ax.set_title("Tren Realisasi 2017–2025")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(tren.index)
        for x, y in zip(tren.index, tren.values):
            ax.annotate(f'{y:,.0f}', (x, y), textcoords="offset points",
                        xytext=(0, 10), ha='center', fontsize=7, color='#37474F')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Baris 2
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🌡️ Heatmap Korelasi Antar Variabel</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cols_corr = ['Bulan', 'Tahun', 'RDKK (ton)', 'Alokasi (ton)',
                     'Jenis Pupuk Enc', 'Kecamatan Enc', 'Realisasi (ton)']
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        sns.heatmap(df[cols_corr].corr(), annot=True, fmt='.2f', cmap=cmap,
                    ax=ax, linewidths=0.5, linecolor='white',
                    annot_kws={'size': 8, 'weight': 'bold'},
                    vmin=-1, vmax=1, center=0,
                    cbar_kws={'shrink': 0.8})
        ax.set_title("Korelasi Antar Variabel")
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">🏘️ RDKK vs Alokasi vs Realisasi per Kecamatan</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        kec_sum = df.groupby('Kecamatan')[['RDKK (ton)', 'Alokasi (ton)', 'Realisasi (ton)']].sum()
        x = np.arange(len(kec_sum))
        w = 0.26
        ax.bar(x - w, kec_sum['RDKK (ton)'],    w, label='RDKK',     color=C_BLUE,   edgecolor='white')
        ax.bar(x,     kec_sum['Alokasi (ton)'],  w, label='Alokasi',  color=C_GREEN,  edgecolor='white')
        ax.bar(x + w, kec_sum['Realisasi (ton)'],w, label='Realisasi',color=C_YELLOW, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(kec_sum.index, rotation=35, ha='right', fontsize=7.5)
        ax.set_ylabel("Total (ton)")
        ax.set_title("Perbandingan per Kecamatan")
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Baris 3
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">🔵 Scatter Plot: Alokasi vs Realisasi per Jenis Pupuk</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, pupuk in enumerate(sorted(df['Jenis Pupuk'].unique())):
        sub = df[df['Jenis Pupuk'] == pupuk]
        ax.scatter(sub['Alokasi (ton)'], sub['Realisasi (ton)'],
                   label=pupuk, alpha=0.55, s=35, color=PALETTE[i % len(PALETTE)], zorder=3)
    max_val = max(df['Alokasi (ton)'].max(), df['Realisasi (ton)'].max())
    ax.plot([0, max_val], [0, max_val], '--', color='#E53935', lw=1.5, label='Ideal (Alokasi=Realisasi)')
    ax.set_xlabel("Alokasi (ton)")
    ax.set_ylabel("Realisasi (ton)")
    ax.set_title("Scatter: Alokasi vs Realisasi")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — EVALUASI MODEL
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🤖</div>
        <div class="section-title">Evaluasi & Perbandingan Model</div>
    </div>
    """, unsafe_allow_html=True)

    metrics = hasil['metrics']

    col1, col2 = st.columns(2)

    with col1:
        m = metrics['RF']
        st.markdown(f"""
        <div class="model-card">
            <div class="model-name">🌲 Random Forest Regressor</div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-label">MAE</div>
                    <div class="metric-val">{m['MAE']:.4f}</div>
                    <div style="font-size:0.7rem;color:#90A4AE;">ton</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">MSE</div>
                    <div class="metric-val">{m['MSE']:.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-val">{m['RMSE']:.4f}</div>
                    <div style="font-size:0.7rem;color:#90A4AE;">ton</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-val">{m['R2']:.4f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        m = metrics['XGBoost']
        st.markdown(f"""
        <div class="model-card model-card-xgb">
            <div class="model-name" style="color:#E65100;">⚡ XGBoost Regressor</div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-label">MAE</div>
                    <div class="metric-val" style="color:#E65100;">{m['MAE']:.4f}</div>
                    <div style="font-size:0.7rem;color:#90A4AE;">ton</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">MSE</div>
                    <div class="metric-val" style="color:#E65100;">{m['MSE']:.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-val" style="color:#E65100;">{m['RMSE']:.4f}</div>
                    <div style="font-size:0.7rem;color:#90A4AE;">ton</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-val" style="color:#E65100;">{m['R2']:.4f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabel
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📋</div>
        <div class="section-title">Tabel Perbandingan</div>
    </div>
    """, unsafe_allow_html=True)
    tabel = pd.DataFrame({
        'Model'   : ['🌲 Random Forest', '⚡ XGBoost'],
        'MAE (ton)' : [metrics['RF']['MAE'],  metrics['XGBoost']['MAE']],
        'MSE'       : [metrics['RF']['MSE'],  metrics['XGBoost']['MSE']],
        'RMSE (ton)': [metrics['RF']['RMSE'], metrics['XGBoost']['RMSE']],
        'R² Score'  : [metrics['RF']['R2'],   metrics['XGBoost']['R2']]
    }).round(4)
    st.dataframe(tabel, use_container_width=True, hide_index=True)

    # Grafik perbandingan
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📉</div>
        <div class="section-title">Visualisasi Perbandingan</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    model_labels = ['Random Forest', 'XGBoost']
    colors_model = [C_GREEN, C_YELLOW]

    with col_a:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">R² Score (Lebih Tinggi = Lebih Baik)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.2))
        vals = [metrics['RF']['R2'], metrics['XGBoost']['R2']]
        bars = ax.bar(model_labels, vals, color=colors_model, edgecolor='white', width=0.45, zorder=3)
        ax.set_ylim(0.98, 1.0)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                    f'{v:.4f}', ha='center', fontsize=9, fontweight='bold', color='#37474F')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">RMSE (Lebih Kecil = Lebih Baik)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.2))
        vals = [metrics['RF']['RMSE'], metrics['XGBoost']['RMSE']]
        bars = ax.bar(model_labels, vals, color=colors_model, edgecolor='white', width=0.45, zorder=3)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        ax.set_ylabel("ton")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{v:.2f}', ha='center', fontsize=9, fontweight='bold', color='#37474F')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">MAE (Lebih Kecil = Lebih Baik)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.2))
        vals = [metrics['RF']['MAE'], metrics['XGBoost']['MAE']]
        bars = ax.bar(model_labels, vals, color=colors_model, edgecolor='white', width=0.45, zorder=3)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        ax.set_ylabel("ton")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{v:.2f}', ha='center', fontsize=9, fontweight='bold', color='#37474F')
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Actual vs Predicted
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🎯</div>
        <div class="section-title">Aktual vs Prediksi</div>
    </div>
    """, unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    for col, model_name, pred, color, title in [
        (col_x, 'RF',      hasil['rf_pred'],  C_GREEN,  '🌲 Random Forest'),
        (col_y, 'XGBoost', hasil['xgb_pred'], C_YELLOW, '⚡ XGBoost'),
    ]:
        with col:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="chart-title">{title}: Aktual vs Prediksi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(hasil['y_test'], pred, alpha=0.5, color=color, s=25, zorder=3, edgecolors='white', linewidths=0.3)
            max_v = max(hasil['y_test'].max(), pred.max())
            ax.plot([0, max_v], [0, max_v], '--', color='#E53935', lw=1.5, label='Ideal')
            ax.set_xlabel("Aktual (ton)")
            ax.set_ylabel("Prediksi (ton)")
            ax.set_title(f"{title} — Aktual vs Prediksi")
            ax.legend()
            ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📈</div>
        <div class="section-title">Feature Importance</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="info-box">ℹ️ Fitur dengan importance score tertinggi adalah variabel yang paling berpengaruh terhadap prediksi realisasi pupuk.</div>', unsafe_allow_html=True)

    fitur = hasil['fitur']
    col_a, col_b = st.columns(2)

    for col, model_key, model_obj, color, label in [
        (col_a, 'RF',      hasil['rf_model'],  C_GREEN,  '🌲 Random Forest'),
        (col_b, 'XGBoost', hasil['xgb_model'], C_YELLOW, '⚡ XGBoost'),
    ]:
        with col:
            imp = pd.Series(model_obj.feature_importances_, index=fitur).sort_values()
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="chart-title">{label}</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 3.8))
            bars = ax.barh(imp.index, imp.values, color=color, edgecolor='white', height=0.6, zorder=3)
            ax.xaxis.grid(True, zorder=0); ax.set_axisbelow(True)
            ax.set_xlabel("Importance Score")
            ax.set_title(f"Feature Importance — {label}")
            for bar, v in zip(bars, imp.values):
                ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{v:.4f}', va='center', fontsize=8, fontweight='bold', color='#37474F')
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            tbl = imp.sort_values(ascending=False).reset_index()
            tbl.columns = ['Fitur', 'Importance Score']
            tbl['Importance Score'] = tbl['Importance Score'].round(4)
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — PREDIKSI MANUAL
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🔮</div>
        <div class="section-title">Prediksi Realisasi Pupuk</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Masukkan parameter di bawah ini untuk memprediksi berapa ton pupuk yang akan terealisasi.")

    with st.form("form_prediksi"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🌱 Jenis & Lokasi**")
            jenis_pupuk = st.selectbox("Jenis Pupuk", sorted(df['Jenis Pupuk'].unique()))
            kecamatan   = st.selectbox("Kecamatan",   sorted(df['Kecamatan'].unique()))

        with col2:
            st.markdown("**📅 Waktu**")
            bulan_names = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",
                           6:"Juni",7:"Juli",8:"Agustus",9:"September",
                           10:"Oktober",11:"November",12:"Desember"}
            bulan_label = st.selectbox("Bulan", list(bulan_names.values()), index=5)
            bulan       = [k for k, v in bulan_names.items() if v == bulan_label][0]
            tahun       = st.number_input("Tahun", min_value=2017, max_value=2030, value=2025)

        with col3:
            st.markdown("**📦 Kuantitas (ton)**")
            rdkk    = st.number_input("RDKK (ton)",    min_value=0.0, value=500.0, step=10.0)
            alokasi = st.number_input("Alokasi (ton)", min_value=0.0, value=300.0, step=10.0)

        submitted = st.form_submit_button(
            "🔮 Prediksi Sekarang",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        pupuk_enc = le_pupuk.transform([jenis_pupuk])[0]
        kec_enc   = le_kec.transform([kecamatan])[0]
        input_df  = pd.DataFrame([{
            'Jenis Pupuk Enc': pupuk_enc,
            'Kecamatan Enc'  : kec_enc,
            'Bulan'          : bulan,
            'Tahun'          : tahun,
            'RDKK (ton)'     : rdkk,
            'Alokasi (ton)'  : alokasi
        }])

        pred_rf  = hasil['rf_model'].predict(input_df)[0]
        pred_xgb = hasil['xgb_model'].predict(input_df)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Hasil Prediksi")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-model-name">🌲 Random Forest</div>
                <div class="pred-value">{pred_rf:.2f}</div>
                <div class="pred-unit">ton terealisasi</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown(f"""
            <div class="pred-card pred-card-xgb">
                <div class="pred-model-name">⚡ XGBoost</div>
                <div class="pred-value pred-value-xgb">{pred_xgb:.2f}</div>
                <div class="pred-unit">ton terealisasi</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Ringkasan input
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**📋 Ringkasan Input**")
            input_sum = pd.DataFrame({
                'Parameter': ['Jenis Pupuk','Kecamatan','Bulan','Tahun','RDKK (ton)','Alokasi (ton)'],
                'Nilai'    : [jenis_pupuk, kecamatan, bulan_label, tahun, rdkk, alokasi]
            })
            st.dataframe(input_sum, use_container_width=True, hide_index=True)

        with col_s2:
            if alokasi > 0:
                st.markdown("**📊 Estimasi Penyerapan Alokasi**")
                pct_rf  = (pred_rf  / alokasi) * 100
                pct_xgb = (pred_xgb / alokasi) * 100

                fig, ax = plt.subplots(figsize=(4, 2.5))
                bars = ax.barh(['XGBoost', 'Random Forest'], [pct_xgb, pct_rf],
                               color=[C_YELLOW, C_GREEN], edgecolor='white', height=0.4)
                ax.axvline(100, color='#E53935', linestyle='--', lw=1.5, label='100% Alokasi')
                ax.set_xlabel("% Penyerapan Alokasi")
                ax.set_title("Estimasi Penyerapan")
                ax.legend(fontsize=8)
                for bar, v in zip(bars, [pct_xgb, pct_rf]):
                    ax.text(v + 1, bar.get_y() + bar.get_height()/2,
                            f'{v:.1f}%', va='center', fontsize=9, fontweight='bold')
                ax.set_xlim(0, max(pct_rf, pct_xgb) * 1.2)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════
# TAB 5 — KESIMPULAN
# ══════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📋</div>
        <div class="section-title">Kesimpulan & Temuan</div>
    </div>
    """, unsafe_allow_html=True)

    m_rf  = hasil['metrics']['RF']
    m_xgb = hasil['metrics']['XGBoost']
    best  = "XGBoost ⚡" if m_xgb['R2'] > m_rf['R2'] else "Random Forest 🌲"

    st.markdown(f"""
    <div class="success-box">
        🏆 Model Terbaik: <strong>{best}</strong> &nbsp;|&nbsp;
        R² Score: <strong>{max(m_rf['R2'], m_xgb['R2']):.4f}</strong> &nbsp;|&nbsp;
        RMSE: <strong>{min(m_rf['RMSE'], m_xgb['RMSE']):.4f} ton</strong>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="chart-card">
            <div class="chart-title">📌 Alur Proyek</div>
            <ol style="font-size:0.9rem; color:#37474F; line-height:2;">
                <li><strong>Load Data</strong> — Dataset pupuk bersubsidi 2017–2025</li>
                <li><strong>EDA</strong> — Eksplorasi distribusi & pola data</li>
                <li><strong>Preprocessing</strong> — ffill, drop leakage, encoding</li>
                <li><strong>Visualisasi</strong> — 5 chart interaktif</li>
                <li><strong>Modeling</strong> — RF & XGBoost dilatih & dievaluasi</li>
                <li><strong>Evaluasi</strong> — MAE, MSE, RMSE, R² Score</li>
                <li><strong>Feature Importance</strong> — Fitur paling berpengaruh</li>
                <li><strong>Prediksi Manual</strong> — Simulasi input baru</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="chart-card">
            <div class="chart-title">🔍 Temuan Utama</div>
            <ul style="font-size:0.9rem; color:#37474F; line-height:2.2;">
                <li><strong>Alokasi (ton)</strong> adalah fitur paling berpengaruh terhadap realisasi</li>
                <li>Gap besar antara <strong>RDKK vs Alokasi</strong> di semua kecamatan</li>
                <li>Kedua model mencapai <strong>R² &gt; 0.99</strong> — performa sangat baik</li>
                <li>Model terbaik: <strong>{best}</strong></li>
            </ul>
            <div class="chart-title" style="margin-top:16px;">💡 Saran</div>
            <ul style="font-size:0.9rem; color:#37474F; line-height:2.2;">
                <li>Gunakan model untuk estimasi realisasi <strong>sebelum musim tanam</strong></li>
                <li>Data tanpa duplikasi akan meningkatkan akurasi model</li>
                <li>Deploy via Streamlit memudahkan dinas pertanian menggunakannya</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Tabel hasil final
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <div class="section-title">Hasil Evaluasi Final</div>
    </div>
    """, unsafe_allow_html=True)

    tabel_final = pd.DataFrame({
        'Model'      : ['🌲 Random Forest', '⚡ XGBoost'],
        'MAE (ton)'  : [m_rf['MAE'],  m_xgb['MAE']],
        'MSE'        : [m_rf['MSE'],  m_xgb['MSE']],
        'RMSE (ton)' : [m_rf['RMSE'], m_xgb['RMSE']],
        'R² Score'   : [m_rf['R2'],   m_xgb['R2']]
    }).round(4)
    st.dataframe(tabel_final, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    🌾 Prediksi Realisasi Penyaluran Pupuk Bersubsidi &nbsp;|&nbsp;
    Pemrograman 5 · Machine Learning End-to-End &nbsp;|&nbsp;
    Dataset Lombok Barat 2017–2025
</div>
""", unsafe_allow_html=True)
