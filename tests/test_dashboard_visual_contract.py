from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
CSS_PATH = ROOT / "dashboard.css"
CONFIG_PATH = ROOT / ".streamlit" / "config.toml"


def dashboard_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def test_dashboard_uses_pinterest_inspired_warm_palette():
    css = dashboard_css().replace(" ", "").lower()
    config = CONFIG_PATH.read_text(encoding="utf-8").lower()

    assert "--pin-red:#e60023" in css
    assert "--pin-plum:#211922" in css
    assert "--pin-sand:#e5e5e0" in css
    assert "--pin-olive:#62625b" in css
    assert 'primarycolor = "#e60023"' in config.replace("\n", " ")
    assert "#0f62fe" not in css
    assert "--cds-" not in css


def test_dashboard_replaces_rectangular_controls_with_rounded_pin_shapes():
    css = dashboard_css().replace(" ", "").lower()

    assert "font-family:'dmsans'" in css
    assert "border-radius:14px" in css
    assert "border-radius:24px" in css
    assert ".st-key-summary_metrics[data-testid=\"stcolumn\"]:nth-child(2)[data-testid=\"stmetric\"]" in css
    assert "#e60023" in css
    assert "--cds-interactive" not in css


def test_dashboard_keeps_report_content_responsive_and_accessible():
    source = APP_PATH.read_text(encoding="utf-8")
    css = dashboard_css().replace(" ", "").lower()

    assert "make_report_header(model_ready, view_frame)" in source
    assert ".report-header" in css
    assert "grid-template-columns:repeat(4,minmax(0,1fr))" in css
    assert "@media(max-width:900px)" in css
    assert "@media(max-width:720px)" in css
    assert "@media(max-width:390px)" in css
    assert "prefers-reduced-motion:reduce" in css
    assert "focus-visible" in css
    assert "color-scheme:light" in css
