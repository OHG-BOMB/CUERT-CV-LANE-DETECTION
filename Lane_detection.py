import numpy as np
import cv2

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1 * 3.0 / 5)      
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: 
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(image, left_fit_average)
    else:
        left_line = None

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
    else:
        right_line = None

    averaged_lines = [left_line, right_line]
    return averaged_lines

# Calculate the middle line between two lane lines
def calculate_middle_line(left_line, right_line, image):
    if left_line is None or right_line is None:
        return None

    # Extract points from left and right lines
    left_x1, left_y1, left_x2, left_y2 = left_line[0]
    right_x1, right_y1, right_x2, right_y2 = right_line[0]

    # Calculate middle line points
    middle_x1 = (left_x1 + right_x1) // 2
    middle_x2 = (left_x2 + right_x2) // 2

    # Return middle line as [[x1, y1, x2, y2]]
    return [[middle_x1, left_y1, middle_x2, left_y2]]

# FILE PATH HERE: 

##########################################

vid = cv2.VideoCapture("FILE PATH")

##########################################

while(vid.isOpened()):
    isread, frame = vid.read()

    if not isread:
        print("Error reading frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (width // 4, height),
        (width // 1.7, height // 1.7),
        (width, height),
    ]], np.int32)

    filled_mask = cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, filled_mask)

    lines = cv2.HoughLinesP(masked_edges, rho=0.1, theta=np.pi / 360, threshold=20,
                            minLineLength=20, maxLineGap=300)

    line_image = np.zeros_like(frame)

    if lines is not None:
        averaged_lines = average_slope_intercept(frame, lines)
        if averaged_lines is not None:
            left_line = averaged_lines[0]
            right_line = averaged_lines[1]

            # Draw left and right lane lines
            if left_line is not None:
                x1, y1, x2, y2 = left_line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
            if right_line is not None:
                x1, y1, x2, y2 = right_line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

            # Calculate the middle line
            middle_line = calculate_middle_line(left_line, right_line, frame)
            if middle_line is not None:
                # Draw the middle line
                x1, y1, x2, y2 = middle_line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

                # Estimate car position (assume car is at the center of the frame)
                car_position_x = width // 2

                # Calculate offset from the middle line
                offset = car_position_x - x1  # x1 is the x-coordinate of the middle line at the bottom of the frame
                print(f"Car Offset from Middle Line: {offset} pixels")

    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    cv2.imshow('VIDEO', result)

vid.release()
cv2.destroyAllWindows()