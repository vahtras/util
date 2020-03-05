from unittest import mock

from util import timing


class TestTiming:
    @mock.patch("time.perf_counter")
    @mock.patch("time.time")
    def test_set(self, mock_time, mock_clock):
        r0 = timing.timing("set")
        r0.stop()
        assert mock_time.call_count == 2
        assert mock_clock.call_count == 2

    @mock.patch("time.perf_counter")
    @mock.patch("time.time")
    def test_get(self, mock_time, mock_clock):
        mock_time.side_effect = [0, 0.15]
        mock_clock.side_effect = [0, 0.25]
        r0 = timing.timing("dummy")
        info = str(r0)
        assert info == (
            "Time used in dummy               :" +
            "      0.25 (cpu)       0.15 (wall)"
        )
