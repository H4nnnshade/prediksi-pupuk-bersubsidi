from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
CSS_PATH = ROOT / "dashboard.css"


def dashboard_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def test_dashboard_uses_new_hero_and_external_visual_system():
    source = APP_PATH.read_text(encoding="utf-8")

    assert 'with st.container(key="summary_metrics")' in source
    assert 'make_hero_panel(model_ready)' in source
    assert "dashboard.css" in source
    assert "make_flow_panel" not in source
    assert "Arc-inspired" not in source


def test_dashboard_visual_system_uses_equal_metric_tracks_and_compact_card_geometry():
    css = dashboard_css()

    assert "grid-template-columns:repeat(4,minmax(0,1fr))" in css
    assert "grid-template-columns:repeat(2,minmax(0,1fr))" in css
    assert "--field-radius:12px" in css
    assert "min-height:142px" in css


def test_dashboard_visual_system_has_explicit_small_screen_and_light_table_rules():
    css = dashboard_css()

    assert "@media(max-width:600px)" in css
    assert "@media(max-width:370px)" in css
    assert "color-scheme:light!important" in css
    assert "prefers-reduced-motion:reduce" in css


def test_dashboard_brand_and_surface_do_not_reuse_arc_visual_language():
    css = dashboard_css()

    assert "hero-band" in css
    assert "repeating-linear-gradient" in css
    assert "--field-green-deep:#214735" in css
    assert "Arc-inspired" not in css
    assert "--radius-hero" not in css
