from seis_interp.data import run_memory


def test_run_rss_peak_is_reset_for_each_run(monkeypatch):
    monkeypatch.setattr(run_memory, "current_rss_bytes", lambda: 1000)
    first = run_memory.RunRSSSampler()
    first.start()
    assert first.stop()["peak_cpu_rss_bytes"] == 1000
    monkeypatch.setattr(run_memory, "current_rss_bytes", lambda: 100)
    second = run_memory.RunRSSSampler()
    second.start()
    assert second.stop()["peak_cpu_rss_bytes"] == 100
