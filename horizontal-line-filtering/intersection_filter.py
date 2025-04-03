import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def detect_intersection(im_path = None, img = None):
    if not im_path == None:
        img = cv2.imread('../images/intersection3.png')

    # filtering
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 50]) # H=0, S=0, V=50
    upper_bound = np.array([180, 50, 150])
    # Create a mask based on this range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    masked_img = cv2.GaussianBlur(masked_img, (15, 15), 0)

    # apply canny edge detection
    edges = cv2.Canny(masked_img, 50, 150, apertureSize=3)

    # apply line detection
    lines = cv2.HoughLinesP(
        edges, 
        rho=1,         # distance resolution in pixels
        theta=np.pi/180, # angle resolution in radians
        threshold=50,  # minimum number of votes (intersections in accumulator)
        minLineLength=50,  # minimum length of a line in pixels
        maxLineGap=10    # maximum gap between segments to consider them the same line
    )

    y1_arr = []
    y2_arr = []
    vals = 0
    max_x = -math.inf
    min_x = math.inf

    # get the horizontal lines
    for (x1, y1, x2, y2) in np.squeeze(lines):
        # keep only horizontalish lines
        if (abs(math.atan((y2 - y1) / (x2 - x1))) < 0.1):
            # get approx regression through the horizontal lines
            y1_arr.append(y1)
            y2_arr.append(y2)
            vals += 1
            max_x = max(max_x, x1, x2)
            min_x = min(min_x, x1, x2)
    np.sort(y1_arr)
    np.sort(y2_arr)

    # take middle 50%
    tot_y1 = sum(y1_arr[vals//4:3*vals//4])
    tot_y2 = sum(y2_arr[vals//4:3*vals//4])

    # visualize
    cv2.line(img, (min_x, tot_y1 // (vals//2)), (max_x, tot_y2 // (vals//2)), (255, 0, 0), 3)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.show()

    return (min_x, tot_y1 // (vals//2)), (max_x, tot_y2 // (vals//2))

if __name__ == '__main__':
    (x1, y1), (x2, y2) = detect_intersection('../images/intersection3.png')
    print(f'{x1} {y1} {x2} {y2}')