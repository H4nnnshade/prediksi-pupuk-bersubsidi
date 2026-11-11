import io
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def sample_workbook() -> bytes:
    frame = pd.DataFrame(
        {
            "Bulan": [1, 2, 1, 2],
            "Tahun": [2023, 2023, 2024, 2024],
            "Jenis Pupuk": ["Urea", "NPK", "Urea", "NPK"],
            "Kecamatan": ["Gerung", "Sekotong", "Gerung", "Labuapi"],
            "RDKK (ton)": [100, 80, 120, 60],
            "Alokasi (ton)": [90, 75, 110, 55],
            "Realisasi (ton)": [85, 70, 95, 50],
        }
    )
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return buffer.getvalue()


def load_app_with_sample_data() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.file_uploader[0].set_value(("sample.xlsx", sample_workbook(), MIME_XLSX)).run()
    return app


def test_uploaded_workbook_exposes_live_filters_and_updates_metrics_table_and_csv():
    app = load_app_with_sample_data()

    assert not app.exception
    assert any("Ringkasan data aktif; file model belum lengkap" in item.value for item in app.markdown)
    assert len(app.slider) == 1
    assert len(app.multiselect) == 2
    assert app.metric[0].value == "300,0 ton"
    assert app.dataframe[0].value.shape[0] == 4

    app.multiselect[0].select("Gerung").run()
    app.slider[0].set_value((2024, 2024)).run()
    app.text_input[0].set_value("geru").run()

    assert not app.exception
    assert app.metric[0].value == "95,0 ton"
    assert app.metric[4].value == "1"
    assert app.dataframe[0].value[["Tahun", "Kecamatan", "Jenis Pupuk"]].to_dict("records") == [
        {"Tahun": 2024, "Kecamatan": "Gerung", "Jenis Pupuk": "Urea"}
    ]
    assert app.download_button[0].label == "Unduh hasil filter (CSV)"


def test_filter_that_matches_no_rows_shows_recovery_hint_without_exception():
    app = load_app_with_sample_data()
    app.text_input[0].set_value("not-a-real-district").run()

    assert not app.exception
    assert app.metric[0].value == "0,0 ton"
    assert any("Tidak ada baris yang cocok" in message.value for message in app.info)
    assert len(app.dataframe) == 0
