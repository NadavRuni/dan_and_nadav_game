"""
Runs test suites to score balls based on their color characteristics.

This module defines a framework for executing a series of test functions on each
ball to determine how well it matches the characteristics of a white or black ball.
"""

from typing import List, Callable

from analyzer_table.launcher_helper.json_models import Ball

# A type alias for a test function that takes a Ball and returns a float score.
TestFunc = Callable[[Ball], float]


def _clamp_score(score: float) -> float:
    """Clamps a score to the inclusive range [0, 100]."""
    if score < 0:
        return 0.0
    if score > 100:
        return 100.0
    return score


class TestSuite:
    """
    A collection of test functions to be run on a ball.

    Attributes:
        tests: A list of callable functions, where each function takes a Ball
               object and returns a score from 0 to 100.
    """

    def __init__(self, tests: List[TestFunc]):
        """
        Initializes the TestSuite.

        Args:
            tests: A list of 1 to 5 test functions.

        Raises:
            AssertionError: If the number of tests is not between 1 and 5.
        """
        assert 1 <= len(tests) <= 5, "A TestSuite requires 1 to 5 test functions."
        self.tests = tests


def _run_suite_and_get_scores(ball: Ball, suite: TestSuite) -> List[float]:
    """
    Executes all tests in a suite for a given ball and returns the scores.

    The scores are clamped to the range [0, 100] and the list is padded with
    zeros to ensure it always contains exactly 5 scores.

    Args:
        ball: The Ball object to be tested.
        suite: The TestSuite to run.

    Returns:
        A list containing exactly 5 float scores.
    """
    scores = [_clamp_score(test_func(ball)) for test_func in suite.tests]
    # Pad with zeros to ensure a consistent length of 5
    while len(scores) < 5:
        scores.append(0.0)
    return scores


def run_white_suite(ball: Ball, suite: TestSuite) -> None:
    """
    Runs the white ball test suite and assigns the scores to the ball's data.

    Note: This function mutates the 'ball' object's 'color_score' attribute.

    Args:
        ball: The Ball object to be scored.
        suite: The TestSuite containing tests for white ball characteristics.
    """
    scores = _run_suite_and_get_scores(ball, suite)
    white_scores = ball.color_score.white_score
    white_scores.white_score_test_1 = scores[0]
    white_scores.white_score_test_2 = scores[1]
    white_scores.white_score_test_3 = scores[2]
    white_scores.white_score_test_4 = scores[3]
    white_scores.white_score_test_5 = scores[4]


def run_black_suite(ball: Ball, suite: TestSuite) -> None:
    """
    Runs the black ball test suite and assigns the scores to the ball's data.

    Note: This function mutates the 'ball' object's 'color_score' attribute.

    Args:
        ball: The Ball object to be scored.
        suite: The TestSuite containing tests for black ball characteristics.
    """
    scores = _run_suite_and_get_scores(ball, suite)
    black_scores = ball.color_score.black_score
    black_scores.black_score_test_1 = scores[0]
    black_scores.black_score_test_2 = scores[1]
    black_scores.black_score_test_3 = scores[2]
    black_scores.black_score_test_4 = scores[3]
    black_scores.black_score_test_5 = scores[4]


def score_balls(
    balls: List[Ball], white_suite: TestSuite, black_suite: TestSuite
) -> None:
    """
    Runs the white and black scoring suites for every ball in a list.

    Args:
        balls: A list of Ball objects to be scored. The objects will be
               modified in-place with the scoring results.
        white_suite: The suite of tests for white ball characteristics.
        black_suite: The suite of tests for black ball characteristics.
    """
    for ball in balls:
        run_white_suite(ball, white_suite)
        run_black_suite(ball, black_suite)
