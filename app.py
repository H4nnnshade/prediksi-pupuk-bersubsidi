import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ──────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Pupuk Bersubsidi",
    page_icon="🌾",
    layout="wide"
)

# ──────────────────────────────────────────
# CSS STYLING
# ──────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2e7d32;
        text-align: center;
        padding: 10px 0 0 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f1f8e9;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 15px 20px;
        margin: 5px 0;
    }
    .result-box {
        background: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
        text-align: center;
    }
    .result-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1b5e20;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────
st.markdown('<div class="main-title">🌾 Prediksi Realisasi Penyaluran Pupuk Bersubsidi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Menggunakan Algoritma Random Forest dan XGBoost Regressor | Dataset 2017–2025</div>', unsafe_allow_html=True)
st.divider()

# ──────────────────────────────────────────
# FUNGSI LOAD & PROSES DATA
# ──────────────────────────────────────────
@st.cache_data
def load_and_process(file):
    df = pd.read_excel(file)

    # Forward fill kolom Bulan
    df['Bulan'] = df['Bulan'].ffill().astype(int)

    # Drop kolom tidak relevan & data leakage
    kolom_drop = ['NO', 'Kelompok Tani', 'Alokasi Vs RDKK (%)', 'Realisasi Vs Alokasi (%)']
    df = df.drop(columns=[c for c in kolom_drop if c in df.columns])

    # Konversi numerik
    for col in ['Realisasi (ton)', 'RDKK (ton)', 'Alokasi (ton)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Standardisasi teks
    df['Kecamatan']   = df['Kecamatan'].str.strip().str.title()
    df['Jenis Pupuk'] = df['Jenis Pupuk'].str.strip()

    # Encoding
    le_pupuk = LabelEncoder()
    le_kec   = LabelEncoder()
    df['Jenis Pupuk Enc'] = le_pupuk.fit_transform(df['Jenis Pupuk'])
    df['Kecamatan Enc']   = le_kec.fit_transform(df['Kecamatan'])

    return df, le_pupuk, le_kec


@st.cache_resource
def train_models(df):
    fitur  = ['Jenis Pupuk Enc', 'Kecamatan Enc', 'Bulan', 'Tahun', 'RDKK (ton)', 'Alokasi (ton)']
    target = 'Realisasi (ton)'

    X = df[fitur]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, min_samples_split=2, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    hasil = {
        'rf_model'  : rf,
        'xgb_model' : xgb,
        'X_test'    : X_test,
        'y_test'    : y_test,
        'rf_pred'   : rf_pred,
        'xgb_pred'  : xgb_pred,
        'fitur'     : fitur,
        'metrics': {
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
    return hasil

# ──────────────────────────────────────────
# SIDEBAR — UPLOAD FILE
# ──────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wheat.png", width=80)
    st.title("⚙️ Pengaturan")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset Excel",
        type=["xlsx"],
        help="Upload file Data-pupuk-bersubsidi.xlsx"
    )

    st.markdown("---")
    st.markdown("### ℹ️ Informasi")
    st.info("""
    **Fitur yang digunakan:**
    - Jenis Pupuk
    - Kecamatan
    - Bulan
    - Tahun
    - RDKK (ton)
    - Alokasi (ton)

    **Target:**
    - Realisasi (ton)
    """)
    st.markdown("---")
    st.caption("Pemrograman 5 · ML End-to-End")

# ──────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload file dataset Excel terlebih dahulu di sidebar kiri untuk memulai.")
    st.markdown("""
    ### 📋 Cara Penggunaan:
    1. Upload file `Data-pupuk-bersubsidi.xlsx` di sidebar kiri
    2. Aplikasi akan otomatis memproses data dan melatih model
    3. Jelajahi hasil EDA, evaluasi model, dan coba prediksi manual

    ### 🎯 Tujuan Aplikasi:
    Memprediksi realisasi penyaluran pupuk bersubsidi berdasarkan data historis
    menggunakan algoritma **Random Forest** dan **XGBoost**.
    """)
    st.stop()

# ── Proses data setelah upload ──
df, le_pupuk, le_kec = load_and_process(uploaded_file)
hasil = train_models(df)

# ──────────────────────────────────────────
# TAB NAVIGASI
# ──────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data & EDA",
    "🤖 Evaluasi Model",
    "📈 Feature Importance",
    "🔮 Prediksi Manual",
    "📋 Kesimpulan"
])

# ══════════════════════════════════════════
# TAB 1 — DATA & EDA
# ══════════════════════════════════════════
with tab1:
    st.subheader("📊 Eksplorasi Data (EDA)")

    # Ringkasan dataset
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Baris", f"{df.shape[0]:,}")
    col2.metric("Total Kolom", df.shape[1])
    col3.metric("Jenis Pupuk", df['Jenis Pupuk'].nunique())
    col4.metric("Kecamatan", df['Kecamatan'].nunique())

    st.markdown("---")

    # Tampilkan data
    with st.expander("🗂️ Lihat Data Mentah (10 Baris Pertama)"):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📐 Statistik Deskriptif"):
        st.dataframe(df.describe().round(3), use_container_width=True)

    st.markdown("---")

    # Visualisasi
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Total Realisasi per Jenis Pupuk**")
        fig, ax = plt.subplots(figsize=(6, 4))
        data_pupuk = df.groupby('Jenis Pupuk')['Realisasi (ton)'].sum().sort_values(ascending=False)
        sns.barplot(x=data_pupuk.index, y=data_pupuk.values, palette='viridis', ax=ax)
        ax.set_xlabel("Jenis Pupuk")
        ax.set_ylabel("Total Realisasi (ton)")
        ax.set_title("Realisasi per Jenis Pupuk")
        for i, v in enumerate(data_pupuk.values):
            ax.text(i, v + 50, f'{v:,.0f}', ha='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    with col_b:
        st.markdown("**Tren Realisasi per Tahun**")
        fig, ax = plt.subplots(figsize=(6, 4))
        tren = df.groupby('Tahun')['Realisasi (ton)'].sum()
        ax.plot(tren.index, tren.values, marker='o', color='steelblue', linewidth=2.5)
        ax.fill_between(tren.index, tren.values, alpha=0.15, color='steelblue')
        ax.set_xlabel("Tahun")
        ax.set_ylabel("Total Realisasi (ton)")
        ax.set_title("Tren Realisasi 2017–2025")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("**Heatmap Korelasi**")
        fig, ax = plt.subplots(figsize=(6, 5))
        cols_corr = ['Bulan', 'Tahun', 'RDKK (ton)', 'Alokasi (ton)',
                     'Jenis Pupuk Enc', 'Kecamatan Enc', 'Realisasi (ton)']
        sns.heatmap(df[cols_corr].corr(), annot=True, fmt='.2f',
                    cmap='coolwarm', ax=ax, linewidths=0.5)
        ax.set_title("Korelasi Antar Variabel")
        plt.tight_layout()
        st.pyplot(fig)

    with col_d:
        st.markdown("**RDKK vs Alokasi vs Realisasi per Kecamatan**")
        fig, ax = plt.subplots(figsize=(6, 5))
        kec_sum = df.groupby('Kecamatan')[['RDKK (ton)', 'Alokasi (ton)', 'Realisasi (ton)']].sum()
        kec_sum.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black', linewidth=0.5)
        ax.set_xlabel("Kecamatan")
        ax.set_ylabel("Total (ton)")
        ax.set_title("Perbandingan per Kecamatan")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("**Scatter Plot: Alokasi vs Realisasi**")
    fig, ax = plt.subplots(figsize=(8, 4))
    for pupuk in df['Jenis Pupuk'].unique():
        sub = df[df['Jenis Pupuk'] == pupuk]
        ax.scatter(sub['Alokasi (ton)'], sub['Realisasi (ton)'], label=pupuk, alpha=0.5, s=30)
    max_val = max(df['Alokasi (ton)'].max(), df['Realisasi (ton)'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Ideal')
    ax.set_xlabel("Alokasi (ton)")
    ax.set_ylabel("Realisasi (ton)")
    ax.set_title("Scatter: Alokasi vs Realisasi")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════
# TAB 2 — EVALUASI MODEL
# ══════════════════════════════════════════
with tab2:
    st.subheader("🤖 Evaluasi & Perbandingan Model")

    metrics = hasil['metrics']

    # Kartu metrik
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌲 Random Forest")
        m = metrics['RF']
        c1, c2 = st.columns(2)
        c1.metric("MAE",    f"{m['MAE']:.4f} ton")
        c2.metric("RMSE",   f"{m['RMSE']:.4f} ton")
        c1.metric("MSE",    f"{m['MSE']:.4f}")
        c2.metric("R² Score", f"{m['R2']:.4f}")

    with col2:
        st.markdown("### ⚡ XGBoost")
        m = metrics['XGBoost']
        c1, c2 = st.columns(2)
        c1.metric("MAE",    f"{m['MAE']:.4f} ton")
        c2.metric("RMSE",   f"{m['RMSE']:.4f} ton")
        c1.metric("MSE",    f"{m['MSE']:.4f}")
        c2.metric("R² Score", f"{m['R2']:.4f}")

    st.markdown("---")

    # Tabel perbandingan
    st.markdown("### 📋 Tabel Perbandingan")
    tabel = pd.DataFrame({
        'Model'   : ['Random Forest', 'XGBoost'],
        'MAE'     : [metrics['RF']['MAE'],  metrics['XGBoost']['MAE']],
        'MSE'     : [metrics['RF']['MSE'],  metrics['XGBoost']['MSE']],
        'RMSE'    : [metrics['RF']['RMSE'], metrics['XGBoost']['RMSE']],
        'R² Score': [metrics['RF']['R2'],   metrics['XGBoost']['R2']]
    }).round(4)
    st.dataframe(tabel, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Grafik perbandingan
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        models = ['Random Forest', 'XGBoost']
        r2_vals = [metrics['RF']['R2'], metrics['XGBoost']['R2']]
        bars = ax.bar(models, r2_vals, color=['#4C72B0', '#C44E52'], edgecolor='black', width=0.4)
        ax.set_ylim(0, 1.1)
        ax.set_title("Perbandingan R² Score")
        ax.set_ylabel("R² Score")
        for bar, val in zip(bars, r2_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        rmse_vals = [metrics['RF']['RMSE'], metrics['XGBoost']['RMSE']]
        bars = ax.bar(models, rmse_vals, color=['#4C72B0', '#C44E52'], edgecolor='black', width=0.4)
        ax.set_title("Perbandingan RMSE (Lebih Kecil = Lebih Baik)")
        ax.set_ylabel("RMSE (ton)")
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.2f}', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    # Actual vs Predicted
    st.markdown("---")
    st.markdown("### 📉 Actual vs Predicted")
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(hasil['y_test'], hasil['rf_pred'], alpha=0.5, color='steelblue', s=20)
        max_v = max(hasil['y_test'].max(), hasil['rf_pred'].max())
        ax.plot([0, max_v], [0, max_v], 'r--')
        ax.set_xlabel("Aktual (ton)")
        ax.set_ylabel("Prediksi (ton)")
        ax.set_title("RF: Aktual vs Prediksi")
        plt.tight_layout()
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(hasil['y_test'], hasil['xgb_pred'], alpha=0.5, color='darkorange', s=20)
        max_v = max(hasil['y_test'].max(), hasil['xgb_pred'].max())
        ax.plot([0, max_v], [0, max_v], 'r--')
        ax.set_xlabel("Aktual (ton)")
        ax.set_ylabel("Prediksi (ton)")
        ax.set_title("XGBoost: Aktual vs Prediksi")
        plt.tight_layout()
        st.pyplot(fig)

# ══════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ══════════════════════════════════════════
with tab3:
    st.subheader("📈 Feature Importance")
    st.markdown("Fitur mana yang paling berpengaruh terhadap prediksi realisasi pupuk.")

    fitur = hasil['fitur']
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**🌲 Random Forest**")
        rf_imp = pd.Series(hasil['rf_model'].feature_importances_, index=fitur).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        rf_imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title("Feature Importance — RF")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(
            rf_imp.sort_values(ascending=False).reset_index()
            .rename(columns={'index': 'Fitur', 0: 'Score'}).round(4),
            use_container_width=True, hide_index=True
        )

    with col_b:
        st.markdown("**⚡ XGBoost**")
        xgb_imp = pd.Series(hasil['xgb_model'].feature_importances_, index=fitur).sort_values()
        fig, ax = plt.subplots(figsize=(5, 4))
        xgb_imp.plot(kind='barh', ax=ax, color='darkorange', edgecolor='black')
        ax.set_title("Feature Importance — XGBoost")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(
            xgb_imp.sort_values(ascending=False).reset_index()
            .rename(columns={'index': 'Fitur', 0: 'Score'}).round(4),
            use_container_width=True, hide_index=True
        )

# ══════════════════════════════════════════
# TAB 4 — PREDIKSI MANUAL
# ══════════════════════════════════════════
with tab4:
    st.subheader("🔮 Prediksi Realisasi Pupuk")
    st.markdown("Masukkan data input untuk memprediksi berapa ton pupuk yang akan terealisasi.")

    col1, col2 = st.columns(2)

    with col1:
        jenis_pupuk = st.selectbox("🌱 Jenis Pupuk", sorted(df['Jenis Pupuk'].unique()))
        kecamatan   = st.selectbox("📍 Kecamatan",   sorted(df['Kecamatan'].unique()))
        bulan       = st.slider("📅 Bulan", min_value=1, max_value=12, value=6,
                                format="%d", help="1 = Januari, 12 = Desember")

    with col2:
        tahun   = st.number_input("📆 Tahun", min_value=2017, max_value=2030, value=2025)
        rdkk    = st.number_input("📦 RDKK (ton)", min_value=0.0, value=500.0, step=10.0)
        alokasi = st.number_input("🏷️ Alokasi (ton)", min_value=0.0, value=300.0, step=10.0)

    st.markdown("---")

    if st.button("🔮 Prediksi Sekarang", use_container_width=True, type="primary"):
        pupuk_enc = le_pupuk.transform([jenis_pupuk])[0]
        kec_enc   = le_kec.transform([kecamatan])[0]

        input_df = pd.DataFrame([{
            'Jenis Pupuk Enc': pupuk_enc,
            'Kecamatan Enc'  : kec_enc,
            'Bulan'          : bulan,
            'Tahun'          : tahun,
            'RDKK (ton)'     : rdkk,
            'Alokasi (ton)'  : alokasi
        }])

        pred_rf  = hasil['rf_model'].predict(input_df)[0]
        pred_xgb = hasil['xgb_model'].predict(input_df)[0]

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:1rem; color:#555; margin-bottom:5px;">🌲 Random Forest</div>
                <div class="result-value">{pred_rf:.2f} ton</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:1rem; color:#555; margin-bottom:5px;">⚡ XGBoost</div>
                <div class="result-value">{pred_xgb:.2f} ton</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📋 Ringkasan Input")
        input_summary = pd.DataFrame({
            'Parameter' : ['Jenis Pupuk', 'Kecamatan', 'Bulan', 'Tahun', 'RDKK (ton)', 'Alokasi (ton)'],
            'Nilai'     : [jenis_pupuk, kecamatan, bulan, tahun, rdkk, alokasi]
        })
        st.dataframe(input_summary, use_container_width=True, hide_index=True)

        # Gauge sederhana: persentase realisasi vs alokasi
        if alokasi > 0:
            pct_rf  = (pred_rf  / alokasi) * 100
            pct_xgb = (pred_xgb / alokasi) * 100
            st.markdown("### 📊 Estimasi Penyerapan Alokasi")
            col1, col2 = st.columns(2)
            col1.metric("RF — Penyerapan Alokasi",   f"{pct_rf:.1f}%",
                        delta=f"{pct_rf - 100:.1f}% dari alokasi")
            col2.metric("XGB — Penyerapan Alokasi",  f"{pct_xgb:.1f}%",
                        delta=f"{pct_xgb - 100:.1f}% dari alokasi")

# ══════════════════════════════════════════
# TAB 5 — KESIMPULAN
# ══════════════════════════════════════════
with tab5:
    st.subheader("📋 Kesimpulan")

    m_rf  = hasil['metrics']['RF']
    m_xgb = hasil['metrics']['XGBoost']
    best  = "XGBoost" if m_xgb['R2'] > m_rf['R2'] else "Random Forest"

    st.success(f"✅ **Model terbaik: {best}** dengan R² Score tertinggi.")

    st.markdown(f"""
    ### Ringkasan Alur Proyek
    1. **Load Data** → Dataset pupuk bersubsidi 2017–2025 ({df.shape[0]} baris)
    2. **EDA** → Eksplorasi distribusi, pola, dan korelasi antar variabel
    3. **Preprocessing** → Forward fill Bulan, drop leakage, encoding, imputasi
    4. **Visualisasi** → 5 chart: bar, line tren, heatmap, bar grouped, scatter
    5. **Modeling** → Random Forest & XGBoost dilatih dan dievaluasi
    6. **Evaluasi** → Perbandingan MAE, MSE, RMSE, R² Score
    7. **Feature Importance** → Identifikasi fitur paling berpengaruh
    8. **Prediksi Manual** → Simulasi prediksi input baru

    ### Hasil Evaluasi
    | Model | MAE | RMSE | R² Score |
    |---|---|---|---|
    | Random Forest | {m_rf['MAE']:.4f} ton | {m_rf['RMSE']:.4f} ton | {m_rf['R2']:.4f} |
    | XGBoost | {m_xgb['MAE']:.4f} ton | {m_xgb['RMSE']:.4f} ton | {m_xgb['R2']:.4f} |

    ### Temuan Utama
    - **Alokasi (ton)** adalah fitur paling berpengaruh terhadap realisasi
    - Gap besar antara RDKK dan Alokasi di semua kecamatan — kebutuhan petani belum terpenuhi sepenuhnya
    - Kedua model mencapai **R² > 0.99** — performa sangat baik untuk data tabular ini

    ### Saran
    - Model ini dapat digunakan dinas pertanian untuk estimasi realisasi sebelum musim tanam
    - Data yang lebih akurat (tanpa duplikasi antar tahun) akan meningkatkan performa lebih lanjut
    """)

    st.markdown("---")
    st.caption("Pemrograman 5 · Machine Learning End-to-End · Dataset Pupuk Bersubsidi Lombok Barat 2017–2025")
