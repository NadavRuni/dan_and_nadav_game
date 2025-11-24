from analyzer_table.ball_type_score.run_scores import TestSuite
from analyzer_table.launcher_helper.score_helper.white_tests import (
    test_brightness,
    test_edge_runlength_penalty,
    test_lab_neutrality,
    test_white_mask_ratio,
    test_chroma_std_lab,
    test_white_chroma_texture_consistency,
)
from analyzer_table.launcher_helper.score_helper.black_tests import (
    test_black_low_dynamic_range,
    test_black_low_percentile,
    test_darkness,
    test_black_trimmed_darkness,
    test_black_dark_low_sat_ratio,
)


def build_white_suite() -> TestSuite:
    return TestSuite(
        [
            test_brightness.run,
            test_white_chroma_texture_consistency.run,
            test_white_mask_ratio.run,
            test_chroma_std_lab.run,
            test_edge_runlength_penalty.run,
        ]
    )


def build_black_suite() -> TestSuite:
    return TestSuite(
        [
            test_darkness.run,
            test_black_low_dynamic_range.run,
            test_black_low_percentile.run,
            test_black_trimmed_darkness.run,
            test_black_dark_low_sat_ratio.run,
        ]
    )
