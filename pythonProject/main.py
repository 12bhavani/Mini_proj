import cv2
import numpy as np
import pyautogui

# Define color range for green color
# Define a specific range for the green color you want to detect
low_green = np.array([35, 50, 50])  # Lower range for green color (adjust as needed)
high_green = np.array([85, 255, 255])  # Upper range for green color (adjust as needed)

cap = cv2.VideoCapture(0)

# Initialize previous y-coordinate and scrolling direction
prev_y = 0

# Initialize the moving_object flag
moving_object = False

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low_green, high_green)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize variables to track the largest green object
        largest_area = 0
        largest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_contour = contour

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the center of the detected object
            object_center_y = y + h // 2
            center_y = frame.shape[0] // 2

            # Calculate the distance between the object's center and the screen center
            distance = center_y - object_center_y

            # Adjust scrolling speed based on the distance
            scroll_speed = 0.3  # Adjust this value to control scrolling speed
            scroll_amount = int(distance * scroll_speed)

            # Perform scrolling action based on scroll direction and distance
            if not moving_object:
                moving_object = True
            else:
                # Use smaller scroll increments to make scrolling smoother
                scroll_increment = 35  # Adjust this value to control smoothness
                scroll_amount = min(abs(scroll_amount), scroll_increment) * (1 if scroll_amount >= 0 else -1)
                pyautogui.scroll(scroll_amount)

            prev_y = object_center_y

        # If no colored object detected, reset the moving_object flag
        if largest_contour is None:
            moving_object = False

        cv2.imshow('frame', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
