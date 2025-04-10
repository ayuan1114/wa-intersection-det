import cv2
import os
import sys

sys.path.append('horizontal-line-filtering')

from intersection_filter import detect_intersection, test_import
test_import()

def parse(video_path):
    frames = []

    # Open the AVI file
    video_path = 'videos/' + video_path
    print("Parsing from file " + video_path)
    
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Read and display frames
    while True:
        ret, frame = cap.read()  # Read a frame
        
        if not ret:
            print("End of video or error reading frame.")
            break
        
        frames.append(frame)
    cap.release()
    print('Finished parsing')
    return frames
        

def display(frames):
    print('Displaying...')
    for frame in frames:
        # Display the frame
        cv2.imshow('AVI Video', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cv2.destroyAllWindows()

frames = parse('clip1.avi')
annotated = []

print(frames[0].shape)
for frame in frames:
    print(frame.shape)
    annotated.append(detect_intersection(img=frame))

display(annotated)