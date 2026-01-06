"""
Factory functions for creating ball scoring test suites.

This module provides functions that assemble and return `TestSuite` objects
for classifying white and black balls. Each suite is a collection of specific
test functions imported from the scoring helper modules.
"""

from analyzer_table.ball_type_score.run_scores import TestSuite
from analyzer_table.launcher_helper.score_helper.black_tests import (
    test_black_dark_low_sat_ratio,
    test_black_low_dynamic_range,
    test_black_low_percentile,
    test_black_trimmed_darkness,
    test_darkness,
)
from analyzer_table.launcher_helper.score_helper.white_tests import (
    test_brightness,
    test_chroma_std_lab,
    test_edge_runlength_penalty,
    test_white_chroma_texture_consistency,
    test_white_mask_ratio,
)


def build_white_suite() -> TestSuite:
    """
    Builds the standard test suite for identifying white balls.

    The suite includes tests for brightness, chroma consistency, mask ratio,
    and edge characteristics.

    Returns:
        A TestSuite object containing the white ball tests.
    """
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
    """
    Builds the standard test suite for identifying black balls.

    The suite includes tests for darkness, low dynamic range, and percentile
    brightness.

    Returns:
        A TestSuite object containing the black ball tests.
    """
    return TestSuite(
        [
            test_darkness.run,
            test_black_low_dynamic_range.run,
            test_black_low_percentile.run,
            test_black_trimmed_darkness.run,
            test_black_dark_low_sat_ratio.run,
        ]
    )
