from stretch_dynamic_trajectory.server import duration_msg


def test_durations_near_a_whole_second_round_up_to_it():
    for seconds in (0.0, 0.5, 0.9999999999, 1.0, 2.9999999996, 12.25):
        d = duration_msg(seconds)
        assert 0 <= d.nanosec < 1_000_000_000
        assert abs(d.sec + d.nanosec * 1e-9 - seconds) < 1e-9
