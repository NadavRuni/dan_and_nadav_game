"""
Defines the data structures and models for the pool table analysis.

This module contains dataclasses that represent the core entities of the
application, such as balls, pockets, scores, and geometric shapes. These models
are used to store and transfer data throughout the analysis pipeline.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any

from game_class.C_pocket import Pocket


@dataclass
class WhiteScore:
    """Represents the scoring results for tests determining if a ball is white."""

    white_score_test_1: float = 0.0
    white_score_test_2: float = 0.0
    white_score_test_3: float = 0.0
    white_score_test_4: float = 0.0
    white_score_test_5: float = 0.0


@dataclass
class BlackScore:
    """Represents the scoring results for tests determining if a ball is black."""

    black_score_test_1: float = 0.0
    black_score_test_2: float = 0.0
    black_score_test_3: float = 0.0
    black_score_test_4: float = 0.0
    black_score_test_5: float = 0.0


@dataclass
class SolidScore:
    """Represents the scoring results for tests determining if a ball is solid."""

    solid_score_test_1: float = 0.0
    solid_score_test_2: float = 0.0
    solid_score_test_3: float = 0.0
    solid_score_test_4: float = 0.0
    solid_score_test_5: float = 0.0


@dataclass
class StripedScore:
    """Represents the scoring results for tests determining if a ball is striped."""

    striped_score_test_1: float = 0.0
    striped_score_test_2: float = 0.0
    striped_score_test_3: float = 0.0
    striped_score_test_4: float = 0.0
    striped_score_test_5: float = 0.0


@dataclass
class ColorScore:
    """
    Aggregates the scoring results from all color tests for a single ball.
    """

    white_score: WhiteScore = field(default_factory=WhiteScore)
    black_score: BlackScore = field(default_factory=BlackScore)
    solid_score: SolidScore = field(default_factory=SolidScore)
    striped_score: StripedScore = field(default_factory=StripedScore)


@dataclass
class BallType:
    """
    Defines string constants for ball color classifications.

    Note:
        Using a StrEnum from Python's 'enum' library would be more idiomatic
        and provide better type safety.
    """

    WHITE = "white"
    BLACK = "black"
    SOLID = "solid"
    STRIPED = "striped"
    UNDEFINED = "undefined"


@dataclass
class Ball:
    """Represents a single pool ball detected on the table."""

    center: Tuple[int, int]
    radius: int
    color_score: ColorScore = field(default_factory=ColorScore)
    final_color: str = BallType.UNDEFINED
    single_ball_path: str = ""


@dataclass
class AnalyzerResult:
    """
    Represents the complete result of the image analysis, containing all
    detected balls and pockets.
    """

    black: Optional[Ball] = None
    white: Optional[Ball] = None
    balls: List[Ball] = field(default_factory=list)
    pockets: List[Pocket] = field(default_factory=list)


@dataclass
class Rectangle:
    """Represents a rectangle defined by its four corner coordinates."""

    top_left: Tuple[int, int]
    top_right: Tuple[int, int]
    bottom_left: Tuple[int, int]
    bottom_right: Tuple[int, int]


@dataclass
class Origin:
    """Represents an (x, y) coordinate point."""

    x: int
    y: int


@dataclass
class PhotoData:
    """
    Represents the data extracted from a cropped section of the table image.
    Includes geometry and a list of balls found within that section.
    """

    cut_name: str
    origin: Origin
    rectangle: Rectangle
    balls: List[Ball]

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the PhotoData object to a dictionary for JSON serialization.

        Returns:
            A dictionary representation of the instance.
        """
        return asdict(self)

    def save_json(self, path: str) -> None:
        """
        Saves the PhotoData object to a JSON file.

        Args:
            path: The file path where the JSON file will be saved.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_json(path: str) -> "PhotoData":
        """
        Loads a PhotoData object from a JSON file.

        Note:
            This method is lossy. It does not deserialize the 'color_score',
            'final_color', or 'single_ball_path' fields for the Ball objects.

        Args:
            path: The path to the JSON file.

        Returns:
            A new PhotoData instance with data from the file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        balls = [
            Ball(center=tuple(ball["center"]), radius=ball["radius"])
            for ball in data.get("balls", [])
        ]
        rect_data = data["rectangle"]
        rectangle = Rectangle(
            top_left=tuple(rect_data["top_left"]),
            top_right=tuple(rect_data["top_right"]),
            bottom_left=tuple(rect_data["bottom_left"]),
            bottom_right=tuple(rect_data["bottom_right"]),
        )
        origin = Origin(x=data["origin"]["x"], y=data["origin"]["y"])

        return PhotoData(
            cut_name=data["cut_name"],
            origin=origin,
            rectangle=rectangle,
            balls=balls,
        )
