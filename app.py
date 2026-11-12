from __future__ import annotations

import io
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from dashboard_logic import filter_dataset

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Pupuk Lobar",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Independent visual identity: agricultural field records, not a generic card dashboard.
DASHBOARD_CSS = Path(__file__).with_name("dashboard.css").read_text(encoding="utf-8")
st.markdown(f"<style>{DASHBOARD_CSS}</style>", unsafe_allow_html=True)

# Carbon-inspired chart palette keeps data series distinct and accessible.
INK = "#161616"
TEAL = "#0F62FE"
CITRON = "#78A9FF"
AMBER = "#525252"
MUTED = "#6F6F6F"
LINE = "#C6C6C6"
SURFACE = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": LINE,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "medium",
    "axes.labelsize": 10,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": LINE,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.65,
    "text.color": INK,
    "font.family": "sans-serif",
})

MODEL_FILES = {
    "rf": "model_rf_pupuk.pkl",
    "xgb": "model_xgb_pupuk.pkl",
    "le_pupuk": "encoder_pupuk.pkl",
    "le_kec": "encoder_kecamatan.pkl",
}
DEFAULT_FEATURES = [
    "Jenis Pupuk Enc", "Kecamatan Enc", "Bulan", "Tahun", "RDKK (ton)", "Alokasi (ton)"
]


def model_roots() -> list[Path]:
    """Likely repo roots for local runs and Streamlit Cloud."""
    roots: list[Path] = []
    for raw in (Path(__file__).resolve().parent, Path.cwd(), Path("/mount/src")):
        try:
            path = raw.resolve()
        except OSError:
            continue
        for candidate in (path, path / "models", path / "artifacts"):
            if candidate not in roots:
                roots.append(candidate)
    return roots


def model_fingerprint() -> tuple:
    """Include file mtimes in cache key so newly committed files reload."""
    signature = []
    for name in [*MODEL_FILES.values(), "feature_names.json"]:
        found = next((root / name for root in model_roots() if (root / name).is_file()), None)
        if found:
            try:
                signature.append((str(found), found.stat().st_mtime_ns, found.stat().st_size))
            except OSError:
                signature.append((str(found), 0, 0))
        else:
            signature.append((name, None, None))
    return tuple(signature)


@st.cache_resource(show_spinner="Memuat model…")
def load_models(fingerprint: tuple) -> dict:
    roots = model_roots()
    loaded: dict = {"errors": {}, "paths": {}}
    for key, filename in MODEL_FILES.items():
        path = next((root / filename for root in roots if (root / filename).is_file()), None)
        if path is None:
            loaded["errors"][key] = f"File tidak ditemukan: {filename}"
            continue
        try:
            loaded[key] = joblib.load(path)
            loaded["paths"][key] = str(path)
        except Exception as exc:
            loaded["errors"][key] = f"Gagal membaca {path}: {type(exc).__name__}: {exc}"

    feature_file = next((root / "feature_names.json" for root in roots if (root / "feature_names.json").is_file()), None)
    if feature_file:
        try:
            with feature_file.open(encoding="utf-8") as handle:
                loaded["features"] = json.load(handle)
            loaded["paths"]["features"] = str(feature_file)
        except Exception as exc:
            loaded["errors"]["features"] = f"Gagal membaca {feature_file}: {exc}"
            loaded["features"] = DEFAULT_FEATURES
    else:
        loaded["features"] = DEFAULT_FEATURES
    return loaded


def clean_dataset(file_bytes: bytes) -> pd.DataFrame:
    """Read and normalize source workbook columns without fitting new encoders."""
    frame = pd.read_excel(io.BytesIO(file_bytes))
    required = ["Bulan", "Tahun", "Jenis Pupuk", "Kecamatan", "RDKK (ton)", "Alokasi (ton)", "Realisasi (ton)"]
    missing_cols = [column for column in required if column not in frame.columns]
    if missing_cols:
        raise ValueError("Kolom wajib tidak ditemukan: " + ", ".join(missing_cols))

    frame = frame.copy()
    frame["Bulan"] = pd.to_numeric(frame["Bulan"].ffill(), errors="coerce")
    frame["Tahun"] = pd.to_numeric(frame["Tahun"], errors="coerce")
    for column in ("RDKK (ton)", "Alokasi (ton)", "Realisasi (ton)"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Jenis Pupuk"] = frame["Jenis Pupuk"].astype("string").str.strip()
    frame["Kecamatan"] = frame["Kecamatan"].astype("string").str.strip().str.title()
    frame = frame.dropna(subset=["Bulan", "Tahun", "Jenis Pupuk", "Kecamatan", "Realisasi (ton)"])
    frame["Bulan"] = frame["Bulan"].astype(int)
    frame["Tahun"] = frame["Tahun"].astype(int)
    return frame


@st.cache_data(show_spinner="Membaca dataset…")
def cached_dataset(file_bytes: bytes) -> pd.DataFrame:
    return clean_dataset(file_bytes)


@st.cache_data(show_spinner="Menerapkan filter…")
def filtered_dataset(
    frame: pd.DataFrame,
    districts: tuple[str, ...] = (),
    products: tuple[str, ...] = (),
    year_range: tuple[int, int] | None = None,
    search: str = "",
) -> pd.DataFrame:
    return filter_dataset(
        frame,
        districts=list(districts),
        products=list(products),
        year_range=year_range,
        search=search,
    )


def ton(value: float | int) -> str:
    return f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def show_chart(fig) -> None:
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def make_report_header(model_ready: bool, frame: pd.DataFrame | None) -> None:
    if model_ready:
        readiness = "Model prediksi siap"
    elif frame is None:
        readiness = "Unggah workbook untuk ringkasan; file model belum lengkap"
    else:
        readiness = "Ringkasan data aktif; file model belum lengkap"
    status_class = "ready" if model_ready else ""
    if frame is None:
        coverage = "Belum ada workbook"
        coverage_note = "Unggah data untuk membuka ringkasan"
    elif frame.empty:
        coverage = "Tidak ada baris"
        coverage_note = "Sesuaikan filter untuk melihat data"
    else:
        first_year, last_year = int(frame["Tahun"].min()), int(frame["Tahun"].max())
        coverage = str(first_year) if first_year == last_year else f"{first_year}–{last_year}"
        coverage_note = f"{len(frame):,} baris · {frame['Kecamatan'].nunique()} kecamatan"
    st.markdown(
        f"""
        <header class="report-header">
          <div class="report-copy">
            <p class="report-kicker">LAPORAN DISTRIBUSI · LOMBOK BARAT</p>
            <h1 class="report-heading">Pupuk, dari alokasi hingga penyaluran.</h1>
            <p class="report-deck">Rekapitulasi data pupuk bersubsidi menurut kecamatan, jenis, dan tahun. Angka di bawah mengikuti pilihan filter dan workbook yang sedang aktif.</p>
            <div class="report-status"><span class="report-status-dot {status_class}"></span>{readiness}</div>
          </div>
          <aside class="report-aside" aria-label="Cakupan dataset">
            <div class="report-aside-label">Cakupan data aktif</div>
            <div class="report-aside-place">{coverage}</div>
            <div class="report-aside-note">{coverage_note}</div>
          </aside>
        </header>
        """,
        unsafe_allow_html=True,
    )


def draw_trend(frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        st.markdown(
            '<div class="empty-panel"><strong>Belum ada seri realisasi</strong>'
            '<span>Unggah workbook dari sidebar. Tren tahunan dan komposisi pupuk akan dihitung dari baris data yang terbaca.</span></div>',
            unsafe_allow_html=True,
        )
        return
    yearly = frame.groupby("Tahun", as_index=False)["Realisasi (ton)"].sum().sort_values("Tahun")
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.plot(yearly["Tahun"], yearly["Realisasi (ton)"], color=TEAL, linewidth=2.8, marker="o", markersize=6)
    ax.fill_between(yearly["Tahun"], yearly["Realisasi (ton)"], color=TEAL, alpha=.10)
    ax.grid(axis="y")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_ylabel("Realisasi (ton)")
    ax.set_xlabel("")
    ax.margins(x=.04)
    fig.tight_layout(pad=1.3)
    show_chart(fig)


def draw_product_mix(frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        st.markdown(
            '<div class="empty-panel"><strong>Komposisi pupuk menunggu data</strong>'
            '<span>Ringkasan menunjukkan jenis pupuk dengan realisasi terbesar setelah dataset dimuat.</span></div>',
            unsafe_allow_html=True,
        )
        return
    grouped = frame.groupby("Jenis Pupuk")["Realisasi (ton)"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.3))
    colors = [TEAL if i == len(grouped) - 1 else "#AFC9BF" for i in range(len(grouped))]
    ax.barh(grouped.index, grouped.values, color=colors, height=.56)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8)
    ax.grid(axis="x", alpha=.55)
    ax.set_axisbelow(True)
    ax.set_xlabel("Realisasi (ton)")
    fig.tight_layout(pad=1.3)
    show_chart(fig)


def evaluate_models(frame: pd.DataFrame, models: dict) -> tuple[dict, dict] | None:
    if frame is None or frame.empty:
        return None
    if any(key not in models for key in ("rf", "xgb", "le_pupuk", "le_kec")):
        return None
    data = frame.copy()
    data["Jenis Pupuk Enc"] = models["le_pupuk"].transform(data["Jenis Pupuk"])
    data["Kecamatan Enc"] = models["le_kec"].transform(data["Kecamatan"])
    features = models["features"]
    x = data[features]
    y = data["Realisasi (ton)"]
    if len(data) < 5:
        return None
    _, x_test, _, y_test = train_test_split(x, y, test_size=.2, random_state=42)
    metrics = []
    for key in ("rf", "xgb"):
        pred = models[key].predict(x_test)
        metrics.append({
            "mae": mean_absolute_error(y_test, pred),
            "rmse": np.sqrt(mean_squared_error(y_test, pred)),
            "r2": r2_score(y_test, pred),
        })
    return metrics[0], metrics[1]


# Load models without blocking data exploration when repository files are absent.
models = load_models(model_fingerprint())
model_ready = all(key in models for key in MODEL_FILES)

with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-symbol"><i></i><i></i><i></i><i></i></div>
          <div><div class="brand-name">Pupuk Lobar</div><div class="brand-sub">Distribusi bersubsidi · Lombok Barat</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**Dataset**")
    uploaded = st.file_uploader("Pilih workbook Excel", type=["xlsx"], label_visibility="collapsed")
    st.caption("Data dipakai untuk ringkasan dan evaluasi langsung.")
    st.divider()
    st.markdown("**Kesiapan model**")
    st.markdown(
        f'<div class="model-pill"><span class="pill-dot {"ready" if model_ready else ""}"></span>'
        f'{"Siap membuat prediksi" if model_ready else "File model belum lengkap"}</div>',
        unsafe_allow_html=True,
    )
    if not model_ready:
        with st.expander("File yang belum terbaca"):
            for message in models.get("errors", {}).values():
                st.caption(message)
            st.caption("Lokasi yang diperiksa: folder app.py, folder models/artifacts, dan working directory.")
    st.divider()
    st.caption("Random Forest + XGBoost")
    st.caption("Prediksi realisasi dalam ton")

frame = None
if uploaded is not None:
    try:
        frame = cached_dataset(uploaded.getvalue())
    except Exception as exc:
        st.sidebar.error(f"Dataset tidak terbaca: {exc}")

filtered_frame = None
if frame is not None and not frame.empty:
    district_options = sorted(frame["Kecamatan"].dropna().unique().tolist())
    product_options = sorted(frame["Jenis Pupuk"].dropna().unique().tolist())
    years = sorted(frame["Tahun"].dropna().astype(int).unique().tolist())
    year_bounds = (years[0], years[-1])
    current_districts = st.session_state.get("filter_districts", [])
    st.session_state["filter_districts"] = [
        value for value in current_districts if value in district_options
    ]
    current_products = st.session_state.get("filter_products", [])
    st.session_state["filter_products"] = [
        value for value in current_products if value in product_options
    ]
    current_years = st.session_state.get("filter_years", year_bounds)
    if (
        not isinstance(current_years, (tuple, list))
        or len(current_years) != 2
        or current_years[0] < year_bounds[0]
        or current_years[1] > year_bounds[1]
    ):
        st.session_state["filter_years"] = year_bounds
    st.session_state.setdefault("filter_search", "")

    def reset_filters(bounds: tuple[int, int]) -> None:
        st.session_state["filter_districts"] = []
        st.session_state["filter_products"] = []
        st.session_state["filter_years"] = bounds
        st.session_state["filter_search"] = ""

    with st.sidebar:
        st.divider()
        st.markdown("**Jelajahi dataset**")
        selected_districts = st.multiselect(
            "Kecamatan", district_options, key="filter_districts", placeholder="Semua kecamatan"
        )
        selected_products = st.multiselect(
            "Jenis pupuk", product_options, key="filter_products", placeholder="Semua jenis"
        )
        if len(years) > 1:
            selected_years = st.slider(
                "Rentang tahun", years[0], years[-1], value=year_bounds, key="filter_years"
            )
        else:
            selected_years = year_bounds
            st.caption(f"Tahun data: {years[0]}")
        search_query = st.text_input(
            "Cari baris", key="filter_search", placeholder="Kecamatan, pupuk, tahun…"
        )
        st.button(
            "Reset semua filter",
            use_container_width=True,
            on_click=reset_filters,
            args=(year_bounds,),
            disabled=not (
                selected_districts or selected_products or search_query.strip()
                or tuple(selected_years) != year_bounds
            ),
        )

    filtered_frame = filtered_dataset(
        frame,
        districts=tuple(selected_districts),
        products=tuple(selected_products),
        year_range=tuple(selected_years),
        search=search_query,
    )
    st.sidebar.caption(
        f"Tampil {len(filtered_frame):,} dari {len(frame):,} baris"
    )

view_frame = filtered_frame if filtered_frame is not None else frame

make_report_header(model_ready, view_frame)

# Real values appear only when the user supplies the workbook. No placeholder metrics.
with st.container(key="summary_metrics"):
    row = st.columns(4)
    if view_frame is not None and not view_frame.empty:
        total_realization = view_frame["Realisasi (ton)"].sum()
        total_allocation = view_frame["Alokasi (ton)"].sum()
        absorption = (total_realization / total_allocation * 100) if total_allocation else np.nan
        latest_year = int(view_frame["Tahun"].max())
        summary_values = [
            ("Realisasi tercatat", f"{ton(total_realization)} ton", "sesuai filter aktif"),
            ("Alokasi tercatat", f"{ton(total_allocation)} ton", "sesuai filter aktif"),
            ("Serapan", f"{ton(absorption)}%" if np.isfinite(absorption) else "—", "realisasi ÷ alokasi"),
            ("Wilayah / jenis", f"{view_frame['Kecamatan'].nunique()} / {view_frame['Jenis Pupuk'].nunique()}", f"hingga {latest_year}"),
        ]
    elif view_frame is not None:
        summary_values = [
            ("Realisasi tercatat", "0,0 ton", "tidak ada baris sesuai filter"),
            ("Alokasi tercatat", "0,0 ton", "tidak ada baris sesuai filter"),
            ("Serapan", "—", "tidak ada alokasi untuk dihitung"),
            ("Wilayah / jenis", "0 / 0", "tidak ada baris sesuai filter"),
        ]
    else:
        summary_values = [
            ("Realisasi tercatat", "—", "unggah dataset untuk dihitung"),
            ("Alokasi tercatat", "—", "unggah dataset untuk dihitung"),
            ("Serapan", "—", "realisasi dibanding alokasi"),
            ("Wilayah / jenis", "—", "dihitung dari dataset"),
        ]
    for cell, (label, value, note) in zip(row, summary_values):
        with cell:
            st.metric(label, value, help=note)

st.write("")
overview_tab, predict_tab, data_tab, model_tab = st.tabs(["Ringkasan", "Prediksi", "Dataset", "Model"])

with overview_tab:
    st.markdown(
        '<div class="section-heading"><h3>Jejak realisasi</h3>'
        '<span>Jumlah per tahun</span></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.35, .9], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Realisasi tahunan</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-hint">Total tonase yang tercatat per tahun.</div>', unsafe_allow_html=True)
            draw_trend(view_frame)
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Komposisi jenis pupuk</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-hint">Perbandingan realisasi pada data sesuai filter.</div>', unsafe_allow_html=True)
            draw_product_mix(view_frame)
    if view_frame is None:
        st.markdown(
            '<div class="status-note">Mulai dengan unggah file <b>Data-pupuk-bersubsidi.xlsx</b> dari panel kiri. '
            'Dashboard mengisi angka dari data nyata; tidak ada metrik contoh.</div>',
            unsafe_allow_html=True,
        )

with predict_tab:
    st.markdown('<div class="section-heading"><h3>Buat estimasi</h3><span>Hasil dalam ton</span></div>', unsafe_allow_html=True)
    if not model_ready:
        st.warning("Prediksi belum bisa dijalankan karena satu atau lebih file model/encoder tidak terbaca. Ringkasan dan eksplorasi dataset tetap bisa digunakan.")
        with st.expander("Periksa lokasi file"):
            for message in models.get("errors", {}).values():
                st.write(f"- {message}")
            st.write("Pastikan file berada di folder yang sama dengan `app.py` pada repository Streamlit Cloud.")
    else:
        form_col, result_col = st.columns([1.05, .95], gap="large")
        with form_col:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Parameter distribusi</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-hint">Isi kebutuhan dan alokasi untuk satu wilayah.</div>', unsafe_allow_html=True)
                with st.form("prediction_form"):
                    c1, c2 = st.columns(2)
                    with c1:
                        kind = st.selectbox("Jenis pupuk", sorted(list(models["le_pupuk"].classes_)))
                        district = st.selectbox("Kecamatan", sorted(list(models["le_kec"].classes_)))
                        year = st.number_input("Tahun", min_value=2017, max_value=2035, value=2025, step=1)
                    with c2:
                        month_labels = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
                        month_name = st.selectbox("Bulan", month_labels, index=5)
                        rdkk = st.number_input("RDKK (ton)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
                        allocation = st.number_input("Alokasi (ton)", min_value=0.0, max_value=10000.0, value=300.0, step=10.0)
                    run = st.form_submit_button("Hitung estimasi", use_container_width=True)
        with result_col:
            with st.container(border=True):
                st.markdown('<div class="panel-title">Hasil model</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-hint">Prediksi muncul di sini setelah input dihitung.</div>', unsafe_allow_html=True)
                if run:
                    month_number = month_labels.index(month_name) + 1
                    values = {
                        "Jenis Pupuk Enc": int(models["le_pupuk"].transform([kind])[0]),
                        "Kecamatan Enc": int(models["le_kec"].transform([district])[0]),
                        "Bulan": month_number,
                        "Tahun": int(year),
                        "RDKK (ton)": rdkk,
                        "Alokasi (ton)": allocation,
                    }
                    try:
                        features = models["features"]
                        input_frame = pd.DataFrame([values])[features]
                        prediction_rf = float(models["rf"].predict(input_frame)[0])
                        prediction_xgb = float(models["xgb"].predict(input_frame)[0])
                        average = (prediction_rf + prediction_xgb) / 2
                        st.markdown(
                            f'<div class="result-panel"><div class="result-number">{ton(average)} ton</div>'
                            '<div class="result-caption">Rata-rata dua estimasi model</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.write("")
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Random Forest", f"{ton(prediction_rf)} ton")
                        with m2:
                            st.metric("XGBoost", f"{ton(prediction_xgb)} ton")
                        if allocation > 0:
                            st.caption(f"Perkiraan serapan dari alokasi: RF {ton(prediction_rf / allocation * 100)}% · XGBoost {ton(prediction_xgb / allocation * 100)}%")
                    except Exception as exc:
                        st.error(f"Prediksi gagal: {type(exc).__name__}: {exc}")
                else:
                    st.markdown(
                        '<div class="empty-panel"><strong>Siap menghitung</strong>'
                        '<span>Masukkan parameter di panel kiri. Estimasi RF dan XGBoost akan ditampilkan berdampingan.</span></div>',
                        unsafe_allow_html=True,
                    )

with data_tab:
    st.markdown('<div class="section-heading"><h3>Dataset sumber</h3><span>Data tampil setelah workbook diunggah</span></div>', unsafe_allow_html=True)
    if view_frame is None:
        st.markdown(
            '<div class="empty-panel"><strong>Workbook belum dipilih</strong>'
            '<span>Unggah file Excel dari panel kiri untuk melihat tabel, cakupan tahun, dan kualitas data.</span></div>',
            unsafe_allow_html=True,
        )
    elif view_frame.empty:
        st.info("Tidak ada baris yang cocok. Ubah filter atau gunakan Reset semua filter di panel kiri.")
    else:
        a, b, c = st.columns(3)
        a.metric("Baris tampil", f"{len(view_frame):,}", help=f"Dari {len(frame) if frame is not None else 0:,} baris valid")
        b.metric("Tahun tercakup", f"{view_frame['Tahun'].min()}–{view_frame['Tahun'].max()}")
        c.metric("Kecamatan", view_frame["Kecamatan"].nunique())
        st.dataframe(
            view_frame,
            width="stretch",
            hide_index=True,
            height=min(560, max(220, 42 + len(view_frame) * 35)),
            column_config={
                "Bulan": st.column_config.NumberColumn("Bulan", format="%d"),
                "Tahun": st.column_config.NumberColumn("Tahun", format="%d"),
                "RDKK (ton)": st.column_config.NumberColumn("RDKK (ton)", format="%.1f"),
                "Alokasi (ton)": st.column_config.NumberColumn("Alokasi (ton)", format="%.1f"),
                "Realisasi (ton)": st.column_config.NumberColumn("Realisasi (ton)", format="%.1f"),
            },
        )
        csv_bytes = view_frame.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Unduh hasil filter (CSV)", csv_bytes, "data_pupuk_filter.csv", "text/csv")
        st.caption("Urutkan tabel dengan klik judul kolom. Filter panel kiri mengubah tabel, ringkasan, dan grafik.")

with model_tab:
    st.markdown('<div class="section-heading"><h3>Kinerja model</h3><span>Evaluasi dari dataset yang diunggah</span></div>', unsafe_allow_html=True)
    if not model_ready:
        st.info("Kinerja tidak ditampilkan sampai model dan encoder berhasil dimuat.")
    elif frame is None:
        st.info("Unggah dataset untuk menghitung MAE, RMSE, R², dan feature importance pada data evaluasi.")
    else:
        try:
            evaluated = evaluate_models(view_frame, models)
            if evaluated is None:
                st.info("Dataset perlu memuat setidaknya 5 baris valid agar evaluasi bisa dihitung.")
            else:
                rf_metrics, xgb_metrics = evaluated
                m1, m2 = st.columns(2, gap="large")
                with m1:
                    with st.container(border=True):
                        st.markdown('<div class="panel-title">Random Forest</div>', unsafe_allow_html=True)
                        st.metric("MAE", f"{ton(rf_metrics['mae'])} ton")
                        st.metric("RMSE", f"{ton(rf_metrics['rmse'])} ton")
                        st.metric("R²", f"{rf_metrics['r2']:.4f}")
                with m2:
                    with st.container(border=True):
                        st.markdown('<div class="panel-title">XGBoost</div>', unsafe_allow_html=True)
                        st.metric("MAE", f"{ton(xgb_metrics['mae'])} ton")
                        st.metric("RMSE", f"{ton(xgb_metrics['rmse'])} ton")
                        st.metric("R²", f"{xgb_metrics['r2']:.4f}")
                st.caption("Evaluasi memakai random split 80/20 (random_state=42) pada dataset yang diunggah.")
        except Exception as exc:
            st.warning(f"Evaluasi tidak dapat dihitung: {type(exc).__name__}: {exc}")

        if hasattr(models["rf"], "feature_importances_") and hasattr(models["xgb"], "feature_importances_"):
            features = models["features"]
            imp_col1, imp_col2 = st.columns(2, gap="large")
            for col, key, title, color in (
                (imp_col1, "rf", "Random Forest", TEAL),
                (imp_col2, "xgb", "XGBoost", "#91B752"),
            ):
                with col:
                    with st.container(border=True):
                        st.markdown(f'<div class="panel-title">Fitur berpengaruh · {title}</div>', unsafe_allow_html=True)
                        importance = pd.Series(models[key].feature_importances_, index=features).sort_values()
                        fig, ax = plt.subplots(figsize=(6, 3.2))
                        ax.barh(importance.index, importance.values, color=color, height=.56)
                        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
                        ax.tick_params(axis="both", length=0, pad=7)
                        ax.grid(axis="x")
                        ax.set_axisbelow(True)
                        fig.tight_layout(pad=1.1)
                        show_chart(fig)
    if models.get("paths"):
        with st.expander("Lokasi model yang dimuat"):
            for key, path in models["paths"].items():
                st.code(f"{key}: {path}", language=None)

st.write("")
st.caption("Pupuk Lobar · Estimasi berbasis Random Forest dan XGBoost")
