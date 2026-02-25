import cv2
import tkinter as tk
import numpy as np

GRID_SIZE = 270
SCREEN_WIDTH = 864

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

    # Outer rectangle
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

    cell = grid_size // 3
    for i in range(1, 3):
        cv2.line(frame, (start_x + i * cell, start_y),
                 (start_x + i * cell, end_y), (255, 255, 255), 2)

        cv2.line(frame, (start_x, start_y + i * cell),
                 (end_x, start_y + i * cell), (255, 255, 255), 2)

    return frame, start_x, start_y, end_x, end_y

def process_roi(frame, start_x, start_y, end_x, end_y):
    roi = frame[start_y:end_y, start_x:end_x]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # avg_color = np.mean(roi, axis=(0, 1))
    # print("Average BGR:", avg_color)

    # cv2.imshow("ROI", roi)
    # cv2.imshow("Gray ROI", gray_roi)
    # cv2.imshow("HSV ROI", hsv_roi)
    return hsv_roi


screen_height = get_screen_height()

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Could not find image")
        break

    frame = cv2.resize(frame, (SCREEN_WIDTH, screen_height))
    frame = cv2.flip(frame, 1)

    frame, start_x, start_y, end_x, end_y = draw_grid(frame, GRID_SIZE)
    cell = GRID_SIZE // 3
    hsv_roi=process_roi(frame, start_x, start_y, end_x, end_y)
    cell = GRID_SIZE // 3
    cells_color = [[None for _ in range(3)] for _ in range(3)]

    for row in range(3):
        for col in range(3):
            y1 = row * cell
            y2 = (row + 1) * cell
            x1 = col * cell
            x2 = (col + 1) * cell
            cell_roi = hsv_roi[y1:y2, x1:x2]
            avg_hsv = np.mean(cell_roi, axis=(0,1))
            cells_color[row][col] = avg_hsv
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

