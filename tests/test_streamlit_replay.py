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

