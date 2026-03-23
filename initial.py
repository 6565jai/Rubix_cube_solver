import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import Counter
import kociemba

GRID_SIZE = 270

# ---------- COLOR ----------
color_to_face = {
    "white": "U","red": "R","green": "F",
    "yellow": "D","orange": "L","blue": "B"
}

reference_colors = {
    "red":    np.array([0, 200, 150]),
    "orange": np.array([18, 220, 220]),
    "yellow": np.array([30, 220, 220]),
    "green":  np.array([60, 200, 150]),
    "blue":   np.array([110,200,150]),
    "white":  np.array([0, 0, 255])
}

def classify(hsv):
    h,s,v = hsv
    if v < 50: return "unknown"
    if s < 40 and v > 150: return "white"

    hsv_vec = np.array([h,s,v])
    best, min_d = "unknown", 1e9

    for c, ref in reference_colors.items():
        d = np.linalg.norm(hsv_vec - ref)
        if d < min_d:
            min_d = d
            best = c
    return best

def stabilize(history):
    out = [[None]*3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            vals = [h[r][c] for h in history]
            out[r][c] = Counter(vals).most_common(1)[0][0]
    return out

# ---------- APP ----------
class CubeApp:
    def __init__(self, root):
        self.root = root
        self.cap = cv2.VideoCapture(0)

        self.label = tk.Label(root)
        self.label.pack()

        self.info = tk.Label(root, text="Show WHITE face", font=("Arial",14))
        self.info.pack()

        tk.Button(root, text="Capture", command=self.capture).pack()
        tk.Button(root, text="Solve", command=self.solve).pack()
        tk.Button(root, text="Next Move", command=self.next_move).pack()

        self.move_label = tk.Label(root, text="", font=("Arial",16))
        self.move_label.pack()

        self.history = []
        self.cube_faces = {}
        self.moves = []
        self.move_index = 0
        self.current_frame = None
        self.current_grid = None

        self.update()

    # ---------- DRAW GRID ----------
    def draw_grid(self, frame):
        h,w = frame.shape[:2]
        sx, sy = w//2-135, h//2-135
        ex, ey = sx+270, sy+270

        cv2.rectangle(frame,(sx,sy),(ex,ey),(0,0,0),2)
        cell = 90

        for i in range(1,3):
            cv2.line(frame,(sx+i*cell,sy),(sx+i*cell,ey),(0,0,0),2)
            cv2.line(frame,(sx,sy+i*cell),(ex,sy+i*cell),(0,0,0),2)

        return sx,sy,ex,ey,cell

    # ---------- ARROWS ----------
    def draw_arrow(self, frame, move):
        h,w = frame.shape[:2]
        cx,cy = w//2,h//2
        L=80

        if move=="R":
            cv2.arrowedLine(frame,(cx+140,cy),(cx+140,cy-L),(0,0,255),4)
        elif move=="R'":
            cv2.arrowedLine(frame,(cx+140,cy-L),(cx+140,cy),(0,0,255),4)
        elif move=="U":
            cv2.arrowedLine(frame,(cx,cy-140),(cx+L,cy-140),(0,0,255),4)
        elif move=="U'":
            cv2.arrowedLine(frame,(cx+L,cy-140),(cx,cy-140),(0,0,255),4)
        elif move=="L":
            cv2.arrowedLine(frame,(cx-140,cy),(cx-140,cy+L),(0,0,255),4)
        elif move=="L'":
            cv2.arrowedLine(frame,(cx-140,cy+L),(cx-140,cy),(0,0,255),4)

    # ---------- CAMERA LOOP ----------
    def update(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame,1)

        sx,sy,ex,ey,cell = self.draw_grid(frame)

        roi = frame[sy:ey, sx:ex]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        grid = [[None]*3 for _ in range(3)]

        for r in range(3):
            for c in range(3):
                y1,y2 = r*cell,(r+1)*cell
                x1,x2 = c*cell,(c+1)*cell

                crop = hsv[y1+10:y2-10, x1+10:x2-10]
                avg = np.mean(crop,(0,1))

                h,s,v = avg
                v = min(v*1.2,255)

                col = classify([h,s,v])

                if s<60 and v<120:
                    col="unknown"

                grid[r][c]=col

        self.history.append(grid)
        if len(self.history)>8:
            self.history.pop(0)

        stable = stabilize(self.history)
        self.current_grid = stable

        # draw labels
        for r in range(3):
            for c in range(3):
                cv2.putText(frame, stable[r][c],
                    (sx+c*cell+10, sy+r*cell+30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        # show arrow
        if self.move_index < len(self.moves):
            move = self.moves[self.move_index]
            self.draw_arrow(frame, move)
            self.move_label.config(text="Move: "+move)
        elif self.moves:
            self.move_label.config(text="SOLVED 🎉")

        # show in tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.imgtk = img
        self.label.configure(image=img)

        self.root.after(10, self.update)

    # ---------- CAPTURE ----------
    def capture(self):
        grid = self.current_grid
        flat = [c for row in grid for c in row]

        if "unknown" in flat:
            self.info.config(text="Not stable")
            return

        face_id = color_to_face[grid[1][1]]
        self.cube_faces[face_id] = grid

        self.info.config(text=f"{face_id} captured ({len(self.cube_faces)}/6)")

    # ---------- SOLVE ----------
    def solve(self):
        if len(self.cube_faces) < 6:
            self.info.config(text="Capture all faces")
            return

        order = ["U","R","F","D","L","B"]
        cube_string = ""

        for f in order:
            for row in self.cube_faces[f]:
                for col in row:
                    cube_string += color_to_face[col]

        sol = kociemba.solve(cube_string)
        self.moves = sol.split()
        self.move_index = 0

    def next_move(self):
        self.move_index += 1

# ---------- RUN ----------
root = tk.Tk()
root.title("Cube Solver GUI")

app = CubeApp(root)

root.mainloop()
