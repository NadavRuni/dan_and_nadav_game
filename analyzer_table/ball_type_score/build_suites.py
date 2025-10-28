# analyzer_table/ball_type_score/build_suites.py
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
    test_darkness ,
    test_black_trimmed_darkness,
    test_black_dark_low_sat_ratio,
)

#from analyzer_table.launcher_helper.score_helper.solid_tests import test_uniformity
#from analyzer_table.launcher_helper.score_helper.striped_tests import test_edge_contrast

def build_white_suite() -> TestSuite:
    """
    3 מבחנים שונים ל-WHITE + משלימים ל-5 עם brightness
    """
    return TestSuite([
        test_brightness.run,        # W1
        test_white_chroma_texture_consistency.run,    # W2
        test_white_mask_ratio.run,  # W3
        test_chroma_std_lab.run,    # W4
        test_edge_runlength_penalty.run,  # W5
    ])

def build_black_suite() -> TestSuite:
    return TestSuite([
        test_darkness.run,
        test_black_low_dynamic_range.run ,  
        test_black_low_percentile.run,
        test_black_trimmed_darkness.run,
        test_black_dark_low_sat_ratio.run,
        ])

# def build_solid_suite():
#     return TestSuite([test_uniformity.run]*5)

# def build_striped_suite():
#     return TestSuite([test_edge_contrast.run]*5)
