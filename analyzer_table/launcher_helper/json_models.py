from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import json

@dataclass
class Ball:
    center: Tuple[int, int]
    radius: int

@dataclass
class Rectangle:
    top_left: Tuple[int, int]
    top_right: Tuple[int, int]
    bottom_left: Tuple[int, int]
    bottom_right: Tuple[int, int]

@dataclass
class Origin:
    x: int
    y: int

@dataclass
class PhotoData:
    cut_name: str
    origin: Origin
    rectangle: Rectangle
    balls: List[Ball]

    def to_dict(self) -> Dict:
        """המרת האובייקט למילון רגיל לצורך שמירה ל־JSON"""
        return asdict(self)

    def save_json(self, path: str):
        """שמירת האובייקט לקובץ JSON"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @staticmethod
    def load_json(path: str) -> "PhotoData":
        """טעינת קובץ JSON לאובייקט מסוג PhotoData"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        balls = [Ball(tuple(ball["center"]), ball["radius"]) for ball in data.get("balls", [])]
        rect = data["rectangle"]
        rectangle = Rectangle(
            tuple(rect["top_left"]),
            tuple(rect["top_right"]),
            tuple(rect["bottom_left"]),
            tuple(rect["bottom_right"]),
        )
        origin = Origin(data["origin"]["x"], data["origin"]["y"])

        return PhotoData(
            cut_name=data["cut_name"],
            origin=origin,
            rectangle=rectangle,
            balls=balls
        )
