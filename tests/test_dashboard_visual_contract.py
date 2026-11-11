from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
CSS_PATH = ROOT / "dashboard.css"


def dashboard_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def test_page_uses_live_coverage_masthead_instead_of_previous_hero():
    source = APP_PATH.read_text(encoding="utf-8")
    css = dashboard_css()

    assert "make_report_header(model_ready, view_frame)" in source
    assert "def make_report_header(" in source
    assert ".report-heading" in css
    assert ".hero-band" not in css
    assert "make_hero_panel" not in source
    assert "Unggah workbook untuk ringkasan" in source


def test_summary_metrics_form_one_aligned_measurement_strip():
    css = dashboard_css().replace(" ", "")

    assert "grid-template-columns:repeat(4,minmax(0,1fr))" in css
    assert "background:transparent!important" in css
    assert "border-radius:0!important" in css
    assert "box-shadow:none!important" in css
    assert "--field-ground:#EFF4F1" in css
    assert "--field-dark:#153E38" in css
    assert "--field-leaf:#C7D86B" in css


def test_identity_is_field_station_not_previous_arc_like_hero():
    css = dashboard_css()

    assert ".report-heading" in css
    assert ".hero-band" not in css
    assert "repeating-linear-gradient" not in css
    assert "linear-gradient" not in css
    assert "Barlow Condensed" in css
    assert "[data-testid=\"stSidebar\"]{background:var(--field-dark)!important" in css


def test_report_layout_and_controls_cover_narrow_screen_and_accessibility():
    css = dashboard_css().replace(" ", "")

    assert "@media(max-width:900px)" in css
    assert "@media(max-width:720px)" in css
    assert "@media(max-width:390px)" in css
    assert "prefers-reduced-motion:reduce" in css
    assert "focus-visible" in css
    assert "color-scheme:light!important" in css
    assert "[data-testid=\"stDeployButton\"]{display:none!important}" in css
