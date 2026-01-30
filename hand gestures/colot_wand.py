import cv2
import numpy as np

# ------------------------------
# SETTINGS
# ------------------------------
# Define the color to track (HSV format).
# DEFAULT: Bright Blue (like a Pepsi cap or blue marker)
# If your object is not detecting, you will need to adjust these.
LOWER_COLOR = np.array([90, 100, 100])
UPPER_COLOR = np.array([130, 255, 255])

DRAWING_COLOR = (0, 255, 255)  # Yellow Line
THICKNESS = 4
CANVAS_SIZE = (1280, 720)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CANVAS_SIZE[0])
    cap.set(4, CANVAS_SIZE[1])

    # A separate black image to hold the drawing
    canvas = np.zeros((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)

    # Variables for drawing smooth lines
    x1, y1 = 0, 0

    print("---------------------------------------")
    print("ACTIVATED: COLOR WAND")
    print("---------------------------------------")
    print("1. Hold a BLUE object (pen, cap, paper) to draw.")
    print("2. Press 'c' to CLEAR the screen.")
    print("3. Press 'q' to QUIT.")
    print("---------------------------------------")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # 2. Convert to HSV color space (easier for computer to see colors)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3. Create a mask (Black & White image where White = Your Color)
        mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)

        # 4. Clean up the noise (removes tiny specks)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # 5. Find contours (the shapes of the colored blobs)
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If a color is found...
        if len(contours) > 0:
            # Get the biggest blob (closest object)
            c = max(contours, key=cv2.contourArea)

            # Find the center of that blob
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            # Only draw if the blob is big enough (reduces glitching)
            if M["m00"] > 0 and radius > 10:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # Draw a target circle on the video
                cv2.circle(frame, center, int(radius), (0, 255, 255), 2)

                # DRAWING LOGIC
                if x1 == 0 and y1 == 0:
                    x1, y1 = center  # Initialize start point
                else:
                    # Draw line on the canvas
                    cv2.line(canvas, (x1, y1), center,
                             DRAWING_COLOR, THICKNESS)

                # Update last position
                x1, y1 = center
            else:
                x1, y1 = 0, 0
        else:
            x1, y1 = 0, 0

        # 6. Combine the Camera Feed and the Canvas
        # Convert canvas to gray to create a mask
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)

        # Black out the area where the drawing is
        img_bg = cv2.bitwise_and(frame, frame, mask=img_inv)

        # Add the drawing on top
        img_final = cv2.bitwise_or(img_bg, canvas)

        cv2.imshow("Color Wand", img_final)
        # Show the mask window so you can debug what the camera sees
        # cv2.imshow("Mask Debug (White = Drawing)", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas = np.zeros(
                (CANVAS_SIZE[1], CANVAS_SIZE[0], 3), dtype=np.uint8)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
