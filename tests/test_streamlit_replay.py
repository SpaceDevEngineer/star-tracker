from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "Code" / "Streamlit_app" / "pipeline_app.py"


def test_successful_replay_reaches_final_attitude():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.slider[-1].set_value(0.0).run()
    app.button[0].click().run()

    assert not app.exception
    assert any("Pipeline complete" in message.value for message in app.success)


def test_quality_gate_refusal_renders_without_crashing():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    failed_image = next(
        path
        for path in sorted((ROOT / "Data" / "dataset_tess_test" / "images").glob("*.png"))
        if "cam3-ccd2" in path.name
    )

    app.selectbox[0].set_value(failed_image).run()
    app.slider[-1].set_value(0.0).run()
    app.button[0].click().run()

    assert not app.exception
    assert any("intended output" in warning.value for warning in app.warning)
