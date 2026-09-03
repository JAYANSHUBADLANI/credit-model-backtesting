"""The dashboard script has to actually run, not just serve a page.

Streamlit returns the page shell before it executes the script, so an HTTP 200 proves nothing
about whether the page renders. These use Streamlit's own harness, which runs the script the
way the browser does. In the project this accompanies, that distinction caught an invalid
argument that would have raised on load and a cached loader that returned the wrong frame.
"""

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"
REPORTS = ROOT / "reports"


def test_the_dashboard_asks_for_a_run_rather_than_crashing_when_there_are_no_reports(tmp_path,
                                                                                    monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "config.yaml").write_text(
        (ROOT / "config" / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "config" / "config.yaml"))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert any("make demo" in str(w.value) for w in app.warning)


@pytest.mark.skipif(
    not (REPORTS / "headline.json").exists(),
    reason="no reports in the tree; run `make demo` first",
)
def test_the_dashboard_renders_a_populated_run_without_raising():
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "Credit model backtesting"
    # The four headline metrics, and at least the bridge and audit tables.
    assert len(app.metric) >= 4
    assert len(app.dataframe) >= 4
