"""
Generates synthetic sample runway images with realistic FOD targets
for testing the surveillance dashboard.
"""

from pathlib import Path
import cv2
import numpy as np

import config

def create_sample_runway_image() -> None:
    # 1280x720 asphalt runway canvas
    h, w = 720, 1280
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Runway asphalt background color (dark gray with texture)
    canvas[:, :] = (45, 48, 52)

    # Add noise / texture
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # White & Yellow Runway Markings
    # Centerline dashed white stripes
    stripe_w, stripe_h = 16, 120
    for y in range(40, h, 180):
        cv2.rectangle(canvas, (w // 2 - stripe_w // 2, y), (w // 2 + stripe_w // 2, y + stripe_h), (240, 240, 240), -1)

    # Runway edge lines
    cv2.line(canvas, (100, 0), (100, h), (255, 255, 255), 8)
    cv2.line(canvas, (w - 100, 0), (w - 100, h), (255, 255, 255), 8)

    # Add Synthetic Objects (Bottle, Metallic Tool, Stone)
    # 1. Plastic Bottle / Container
    cv2.rectangle(canvas, (450, 320), (490, 410), (220, 220, 220), -1)
    cv2.rectangle(canvas, (460, 305), (480, 320), (30, 180, 240), -1) # Cap

    # 2. Metal Wrench / Tool
    cv2.line(canvas, (650, 480), (740, 520), (180, 180, 190), 12)
    cv2.circle(canvas, (645, 477), 16, (160, 160, 170), 6)
    cv2.circle(canvas, (745, 523), 14, (160, 160, 170), 6)

    # 3. Stone / Rock
    cv2.ellipse(canvas, (320, 560), (25, 18), 30, 0, 360, (90, 100, 105), -1)

    # 4. Bird hazard
    cv2.ellipse(canvas, (820, 220), (35, 15), -15, 0, 360, (20, 20, 25), -1)
    cv2.line(canvas, (800, 215), (780, 195), (20, 20, 25), 4)
    cv2.line(canvas, (840, 215), (860, 195), (20, 20, 25), 4)

    output_path = config.IMAGES_DIR / "sample_runway.jpg"
    cv2.imwrite(str(output_path), canvas)
    print(f"Sample runway image generated at: {output_path}")

if __name__ == "__main__":
    create_sample_runway_image()
