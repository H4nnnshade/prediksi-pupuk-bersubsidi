from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
CSS_PATH = ROOT / "dashboard.css"


def dashboard_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def test_dashboard_adopts_carbon_inspired_blue_gray_design_tokens():
    css = dashboard_css().replace(" ", "").lower()

    assert "--cds-interactive:#0f62fe" in css
    assert "--cds-text:#161616" in css
    assert "--cds-layer:#f4f4f4" in css
    assert "--cds-border:#c6c6c6" in css
    assert "#216c5e" not in css
    assert "#c7d86b" not in css


def test_components_use_carbon_geometry_and_flat_layers():
    css = dashboard_css().replace(" ", "").lower()

    assert "border-radius:0" in css
    assert "box-shadow:none" in css
    assert "font-family:'ibmplexsans'" in css
    assert "font-family:'ibmplexmono'" in css
    assert "#0f62fe" in css
    assert "border-bottom:2pxsolidvar(--cds-interactive)" in css


def test_dashboard_keeps_report_content_responsive_and_accessible():
    source = APP_PATH.read_text(encoding="utf-8")
    css = dashboard_css().replace(" ", "").lower()

    assert "make_report_header(model_ready, view_frame)" in source
    assert ".report-heading" in css
    assert "grid-template-columns:repeat(4,minmax(0,1fr))" in css
    assert "@media(max-width:900px)" in css
    assert "@media(max-width:720px)" in css
    assert "@media(max-width:390px)" in css
    assert "prefers-reduced-motion:reduce" in css
    assert "focus-visible" in css
    assert "color-scheme:light" in css
