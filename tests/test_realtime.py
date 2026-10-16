import math

from respect.eval.realtime import latest_frame_schedule, rss_allowed_response, rss_min_distance, run_metrics


def test_latest_frame_schedule_drops_frames():
    # constant 0.25 s cycles: frames 0, 2 (arrived at 0.2 s), 5 (arrived at 0.5 s), ...
    sch = latest_frame_schedule(0, 10, lambda g, k: 0.25)
    assert [g for g, _, _ in sch][:4] == [0, 2, 5, 7]
    assert sch[-1][0] == 10


def test_latest_frame_schedule_waits_when_idle():
    sch = latest_frame_schedule(0, 3, lambda g, k: 0.05)
    assert [g for g, _, _ in sch] == [0, 1, 2, 3]
    assert math.isclose(sch[1][1], 0.1)


def test_rss_allowed_response_closed_form():
    v, gap, vf = 6.8, 20.8, 0.0
    rho = rss_allowed_response(v, gap, vf)
    assert math.isclose(rho, (gap - v * v / 8) / v, rel_tol=1e-9)
    assert math.isclose(rss_min_distance(v, vf, rho), gap, rel_tol=1e-9)
    assert rss_allowed_response(v, gap, vf, a_max=2.0) < rho


def test_run_metrics_reaction_and_age():
    rows = []
    for g, (ts, tp) in {0: (0.0, 1.0), 10: (1.0, 2.0), 20: (2.0, 3.0)}.items():
        rows.append(dict(g=g, g0=0, g_end=20, e=5, k=None if g == 0 else 10, t_arrival=g * 0.1,
                         t_pub=tp, t_decode=tp - 0.1, cycle_ms=1000.0, input_ms=1, prefill_ms=1, decode_ms=1, flow_ms=1))
    m = run_metrics(rows)
    assert math.isclose(m["reaction_s"], 2.0 - 0.5)       # frame 10 is the first at or after E = 5
    assert math.isclose(m["ages"][0], 1.0)                 # at t = 1.0 the trajectory of frame 0 is 1.0 s old
    assert m["expired"] == 0
