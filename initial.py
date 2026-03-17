import cv2
import tkinter as tk
import numpy as np
from collections import Counter

GRID_SIZE = 270
SCREEN_WIDTH = 864
HISTORY_SIZE = 8   # increased for better stability

# ---------- COLOR MAP ----------
color_to_face = {
    "white": "U",
    "red": "R",
    "green": "F",
    "yellow": "D",
    "orange": "L",
    "blue": "B"
}

# ---------- REFERENCE COLORS (NEW) ----------
reference_colors = {
    "red":    np.array([0, 200, 150]),
    "orange": np.array([18, 220, 220]),
    "yellow": np.array([30, 220, 220]),
    "green":  np.array([60, 200, 150]),
    "blue":   np.array([110,200,150]),
    "white":  np.array([0, 0, 255])
}

def face_to_string(face):
    return "".join([color_to_face.get(c, "X") for row in face for c in row])

def get_screen_height():
    root = tk.Tk()
    h = root.winfo_screenheight()
    root.destroy()
    return h

def draw_grid(frame, grid_size):
    h, w = frame.shape[:2]

    sx = (w // 2) - (grid_size // 2)
    sy = (h // 2) - (grid_size // 2)
    ex = sx + grid_size
    ey = sy + grid_size

    cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 0, 0), 2)

    cell = grid_size // 3
    for i in range(1, 3):
        cv2.line(frame, (sx+i*cell, sy), (sx+i*cell, ey), (0,0,0), 2)
        cv2.line(frame, (sx, sy+i*cell), (ex, sy+i*cell), (0,0,0), 2)

    return frame, sx, sy, ex, ey

def process_roi(frame, sx, sy, ex, ey):
    roi = frame[sy:ey, sx:ex]
    return cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# ---------- IMPROVED CLASSIFIER ----------
def classify_cube_color(hsv):
    h, s, v = hsv

    if v < 50:
        return "unknown"

    if s < 40 and v > 150:
        return "white"

    hsv_vec = np.array([h, s, v])

    min_dist = float("inf")
    best = "unknown"

    for color, ref in reference_colors.items():
        dist = np.linalg.norm(hsv_vec - ref)
        if dist < min_dist:
            min_dist = dist
            best = color

    return best

def draw_labels(frame, sx, sy, cell, colors):
    for r in range(3):
        for c in range(3):
            text = colors[r][c]
            x = sx + c * cell + 10
            y = sy + r * cell + 30

            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 1)

# ---------- STABILIZATION ----------
def stabilize_colors(history):
    stable = [[None]*3 for _ in range(3)]

    for r in range(3):
        for c in range(3):
            vals = [h[r][c] for h in history]
            stable[r][c] = Counter(vals).most_common(1)[0][0]

    return stable

def build_cube_string(cube_faces):
    order = ["U","R","F","D","L","B"]
    return "".join([face_to_string(cube_faces[f]) for f in order])

# ---------- MAIN ----------
screen_height = get_screen_height()
cam = cv2.VideoCapture(0)

cube_faces = {}
history = []

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.resize(frame, (SCREEN_WIDTH, screen_height))
    frame = cv2.flip(frame, 1)

    frame, sx, sy, ex, ey = draw_grid(frame, GRID_SIZE)
    hsv_roi = process_roi(frame, sx, sy, ex, ey)

    cell = GRID_SIZE // 3
    cells_color = [[None]*3 for _ in range(3)]

    for r in range(3):
        for c in range(3):
            y1, y2 = r*cell, (r+1)*cell
            x1, x2 = c*cell, (c+1)*cell

            # 🔥 CENTER CROP (IMPORTANT)
            margin = 10
            cell_roi = hsv_roi[y1+margin:y2-margin, x1+margin:x2-margin]

            avg = np.mean(cell_roi, axis=(0,1))

            # 🔥 BRIGHTNESS NORMALIZATION
            h, s, v = avg
            v = min(v * 1.2, 255)

            color = classify_cube_color(np.array([h, s, v]))

            # 🔥 NOISE FILTER
            if s < 60 and v < 120:
                color = "unknown"

            cells_color[r][c] = color

    # ---------- STABILITY ----------
    history.append(cells_color)
    if len(history) > HISTORY_SIZE:
        history.pop(0)

    stable = stabilize_colors(history)

    draw_labels(frame, sx, sy, cell, stable)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # ---------- CAPTURE ----------
    if key == ord('c'):
        flat = [c for row in stable for c in row]

        if "unknown" in flat:
            print("❌ Not stable yet")
            continue

        center = stable[1][1]

        face_map = {
            "white": "U",
            "red": "R",
            "green": "F",
            "yellow": "D",
            "orange": "L",
            "blue": "B"
        }

        face_id = face_map[center]
        cube_faces[face_id] = stable

        print("✅ Stored:", face_id)
        print("Total:", len(cube_faces))

        # ---------- BUILD STRING ----------
        if len(cube_faces) == 6:
            cube_string = build_cube_string(cube_faces)

            print("\n🎯 Cube String:")
            print(cube_string)

            import kociemba
            solution = kociemba.solve(cube_string)

            print("\n🧠 Solution:", solution)
            break

cam.release()
cv2.destroyAllWindows()