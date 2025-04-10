import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def detect_intersection_frame(img):
    # filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 50])  # H=0, S=0, V=50
    upper_bound = np.array([180, 50, 150])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked_img = cv2.GaussianBlur(masked_img, (15, 15), 0)

    # edge detection
    edges = cv2.Canny(masked_img, 50, 150, apertureSize=3)

    # line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    # no lines found
    if lines is None:
        return img  

    y1_arr, y2_arr = [], []
    vals = 0
    max_x = -math.inf
    min_x = math.inf

    for (x1, y1, x2, y2) in np.squeeze(lines):
        # basically horizontal
        if abs(math.atan2((y2 - y1), (x2 - x1))) < 0.1:  
            y1_arr.append(y1)
            y2_arr.append(y2)
            vals += 1
            max_x = max(max_x, x1, x2)
            min_x = min(min_x, x1, x2)

    # not enough lines
    if vals < 4:
        return img  

    y1_arr = np.sort(y1_arr)
    y2_arr = np.sort(y2_arr)

    # middle 50%
    y1_middle = y1_arr[vals//4:3*vals//4]
    y2_middle = y2_arr[vals//4:3*vals//4]

    avg_y1 = sum(y1_middle) // len(y1_middle)
    avg_y2 = sum(y2_middle) // len(y2_middle)

    # draw the estimated intersection line
    cv2.line(img, (min_x, avg_y1), (max_x, avg_y2), (255, 0, 0), 3)

    return img

# for video testing

def process_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detect_intersection_frame(frame)

        if out:
            out.write(processed_frame)
        else:
            cv2.imshow('Intersection Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_video('../videos/espringsr_reencode.avi', output_path='intersection_detected.mp4')