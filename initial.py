import cv2
import tkinter as tk

grid_size=270


def get_screen_height():
    root = tk.Tk()
    height = root.winfo_screenheight()
    root.destroy()
    return height

def draw_grid(frame, grid_size):
    h, w = frame.shape[:2]
    start_x = (w // 2) - (grid_size // 2)
    start_y = (h // 2) - (grid_size // 2)
    end_x = start_x + grid_size
    end_y = start_y + grid_size

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

    cell = grid_size // 3
    for i in range(1, 3):
        cv2.line(frame,
                 (start_x + i * cell, start_y),
                 (start_x + i * cell, end_y),
                 (255, 255, 255), 2)
        cv2.line(frame,
                 (start_x, start_y + i * cell),
                 (end_x, start_y + i * cell),
                 (255, 255, 255), 2)
    return frame

screen_height = get_screen_height()
cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        print("Could not find image")
        break
    frame = cv2.resize(frame, (screen_height, screen_height))
    frame = cv2.flip(frame, 1)
    frame = draw_grid(frame,grid_size)

    cv2.imshow("Grid", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
