import numpy as np

from scripts import prepare_mcc5_thu_gearbox, prepare_mcc5_thu_motor


def test_mcc5_preprocessors_take_exact_1024_point_raw_slices():
    signal = np.arange(4096 * 3, dtype=np.float32).reshape(4096, 3)
    starts = np.asarray([0, 2048], dtype=np.int64)

    for module in (prepare_mcc5_thu_gearbox, prepare_mcc5_thu_motor):
        windows = module.extract_windows(signal, starts)
        assert module.WINDOW_SIZE == 1024
        assert windows.shape == (2, 3, 1024)
        np.testing.assert_array_equal(windows[0], signal[:1024].T)
        np.testing.assert_array_equal(windows[1], signal[2048:3072].T)


def test_mcc5_preprocessors_define_two_source_candidates_per_recording():
    expected = (
        (prepare_mcc5_thu_gearbox, 36),
        (prepare_mcc5_thu_motor, 33),
    )
    for module, domain_count in expected:
        assert len(module.DOMAINS) == domain_count
        for offset in range(0, domain_count, 3):
            steady_1, steady_2, stream = module.DOMAINS[offset : offset + 3]
            assert steady_1.is_source and steady_2.is_source
            assert not stream.is_source
            if hasattr(steady_1, "file_condition"):
                assert (
                    steady_1.file_condition
                    == steady_2.file_condition
                    == stream.file_condition
                )
            else:
                assert (
                    steady_1.file_suffix
                    == steady_2.file_suffix
                    == stream.file_suffix
                )

