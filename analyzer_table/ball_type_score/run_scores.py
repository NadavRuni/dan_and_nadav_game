from typing import List, Callable
from analyzer_table.launcher_helper.json_models import Ball, Color_Score 


TestFunc = Callable[[Ball], float]

## make sure the score is bewtween 100 and 0
def _clamp_score(score: float) -> float:
    if score < 0:
        return 0.0
    if score > 100:
        return 100.0
    
    return score

class TestSuite: 
    def __init__(self, tests: List[TestFunc]):
        assert 1 <= len(tests) <= 5, "1 to 5 tests are required"
        self.tests = tests

def _run_suite_to_five(ball: Ball, suite: TestSuite) -> List[float]:
    scores = [_clamp_score(t(ball)) for t in suite.tests]
    while len(scores) < 5:
        scores.append(0.0)
    return scores
        

def run_white_suite(ball: Ball, suite: TestSuite):
     ## Run each test in the suite on the ball and clamp results between 0–100
    scores = _run_suite_to_five(ball, suite)
    # Access the 'white_score' section of the ball's color score data
    w = ball.color_score.white_score
    # Assign the computed scores to the respective fields
    w.white_score_test_1 = scores[0]
    w.white_score_test_2 = scores[1]
    w.white_score_test_3 = scores[2]
    w.white_score_test_4 = scores[3]
    w.white_score_test_5 = scores[4]

def run_black_suite(ball: Ball, suite: TestSuite):
     ## Run each test in the suite on the ball and clamp results between 0–100
    scores = _run_suite_to_five(ball, suite)
    # Access the 'white_score' section of the ball's color score data
    b = ball.color_score.black_score
    # Assign the computed scores to the respective fields
    b.black_score_test_1 = scores[0]
    b.black_score_test_2 = scores[1]
    b.black_score_test_3 = scores[2]
    b.black_score_test_4 = scores[3]
    b.black_score_test_5 = scores[4]


# def run_solid_suite(ball: Ball, suite: TestSuite):
#     ## Run each test in the suite on the ball and clamp results between 0–100
#     scores = [_clamp_score(t(ball)) for t in suite.tests]
#     # Access the 'white_score' section of the ball's color score data
#     s = ball.color_score.solid_score
#     # Assign the computed scores to the respective fields
#     s.solid_score_test_1 = scores[0]
#     s.solid_score_test_2 = scores[1]
#     s.solid_score_test_3 = scores[2]
#     s.solid_score_test_4 = scores[3]
#     s.solid_score_test_5 = scores[4]


# def run_striped_suite(ball: Ball, suite: TestSuite):
#     ## Run each test in the suite on the ball and clamp results between 0–100
#     scores = [_clamp_score(t(ball)) for t in suite.tests]
#     # Access the 'white_score' section of the ball's color score data
#     r = ball.color_score.striped_score
#     # Assign the computed scores to the respective fields
#     r.striped_score_test_1 = scores[0]
#     r.striped_score_test_2 = scores[1]
#     r.striped_score_test_3 = scores[2]
#     r.striped_score_test_4 = scores[3]
#     r.striped_score_test_5 = scores[4]


def score_balls(balls: List[Ball],
                white_suite: TestSuite,
                black_suite: TestSuite,
                #solid_suite: TestSuite,
                #striped_suite: TestSuite
                ):
    for ball in balls:
        run_white_suite(ball, white_suite)
        run_black_suite(ball, black_suite)
        #run_solid_suite(ball, solid_suite)
        #run_striped_suite(ball, striped_suite)