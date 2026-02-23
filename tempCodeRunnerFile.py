import cv2
cam=cv2.VideoCapture(0)
grid_size=300
while True:
    ret,frame=cam.read()
    screen_w = 1920   # change to your screen
    screen_h = 1080
    frame = cv2.resize(frame, (screen_w, screen_h))
    if(ret==False):
       print("Could not found image")
    frame_width,frame_height=frame.shape[:2]
    center_grid=(frame_width//2,frame_height//2)
    start_x=(frame_width//2)-(grid_size // 2)
    start_y=(frame_height//2)-(grid_size // 2)
    end_x=(frame_width//2)+(grid_size // 2)
    end_y=(frame_height//2)+(grid_size // 2)

    grid=cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),(255, 255, 255),thickness=2)
    
    for i in range(1,3):
        cv2.line(frame,(start_x+i*(grid_size//3),start_y),(start_x+i*(grid_size//3),start_y+grid_size),(255, 255, 255),thickness=2)
        cv2.line(frame,(start_x,start_y+i*(grid_size//3)),(start_x+grid_size,start_y+i*(grid_size//3)),(255, 255, 255),thickness=2)
    cv2.imshow("Grid",grid)
    if cv2.waitKey(1)& 0XFF==ord('q'):
        break
cv2.release
cv2.destroyAllWindows()
