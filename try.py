# client_confirm_rectangle.py

import requests
import os

url = "http://localhost:8000/api/confirm_rectangle"

picture_name = "test3_b2b.jpeg"
image_path = os.path.join("/Users/nadavhershkovitz/Downloads", picture_name)

data = {
    "image_path": image_path,
    "points": [
        {"x": 150, "y": 200},
        {"x": 160 , "y": 2900},
        {"x": 5500, "y": 2800},
        {"x": 5500, "y": 200}
    ],
    "display_width": 5712,
    "display_height": 3093,
    "original_width": 5712,
    "original_height": 3093
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())