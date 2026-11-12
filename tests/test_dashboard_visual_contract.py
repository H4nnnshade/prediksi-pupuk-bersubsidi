from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
CSS_PATH = ROOT / "dashboard.css"
CONFIG_PATH = ROOT / ".streamlit" / "config.toml"


def dashboard_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def test_dashboard_uses_mobbin_typefully_analytics_reference_palette():
    css = dashboard_css().replace(" ", "").lower()
    config = CONFIG_PATH.read_text(encoding="utf-8").lower()

    assert "--mobbin-canvas:#f7f8fa" in css
    assert "--mobbin-surface:#ffffff" in css
    assert "--mobbin-ink:#1f2937" in css
    assert "--mobbin-blue:#5578f6" in css
    assert "--mobbin-positive:#5b9674" in css
    assert 'primarycolor = "#5578f6"' in config.replace("\n", " ")
    assert "--pin-red" not in css
    assert "--cds-" not in css


def test_dashboard_uses_typefully_metric_and_chart_hierarchy():
    css = dashboard_css().replace(" ", "").lower()
    source = APP_PATH.read_text(encoding="utf-8")

    assert "mobbin-chart-heading" in source
    assert "metric_sparkline" in source
    assert ".mobbin-kpi-foot" in css
    assert "mobbin-chart-legend" in source
    assert "border-radius:12px" in css
    assert "grid-template-columns:repeat(4,minmax(0,1fr))" in css
    assert "border-right:1pxsolidvar(--mobbin-line)" in css
    assert "--mobbin-blue" in css
    assert "--mobbin-positive" in css
    assert "--pin-red" not in css


def test_dashboard_keeps_responsive_accessible_mobbin_layout():
    source = APP_PATH.read_text(encoding="utf-8")
    css = dashboard_css().replace(" ", "").lower()

    assert "make_report_header(model_ready, view_frame)" in source
    assert ".report-header" in css
    assert "@media(max-width:900px)" in css
    assert "@media(max-width:720px)" in css
    assert "@media(max-width:390px)" in css
    assert "prefers-reduced-motion:reduce" in css
    assert "focus-visible" in css
    assert "color-scheme:light" in css
    assert "border-radius:0!important" in css  # Tab strip and dividers stay flat as in reference.
    assert "#e60023" not in css
    assert "#0f62fe" not in css


def test_dashboard_keeps_source_attribution_for_reference():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Mobbin · Typefully Web Analytics Dashboard" in source
    assert "https://mobbin.com/explore/screens/07a5f279-ee9e-491d-8840-3f8014ec1e3f" in source


def test_dashboard_reference_description_matches_screen():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Typefully" in source
    assert "sparklines" in source
    assert "cumulative" in source.lower() or "akumulasi" in source.lower()
    assert "daily" in source.lower() or "tahunan" in source.lower() or "bulanan" in source.lower()
    assert "draw_trend" in source
    assert "metric_sparkline" in source
    assert "--mobbin-canvas" in (ROOT / "dashboard.css").read_text(encoding="utf-8")
    assert "empty-panel" in source  # Empty upload state remains available.
