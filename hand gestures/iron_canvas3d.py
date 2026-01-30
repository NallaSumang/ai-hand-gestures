import cv2
import numpy as np
import mediapipe as mp
import math

# ------------------------------
# CONFIGURATION
# ------------------------------
WIDTH, HEIGHT = 1280, 720
PINCH_THRESHOLD = 30
SMOOTHING = 5
BRUSH_THICKNESS = 8
NEON_GLOW_INTENSITY = 1.5

# ------------------------------
# 3D ENGINE (MATH HELPERS)
# ------------------------------


class Engine3D:
    def __init__(self):
        # Define the 8 corners of a cube (x, y, z)
        self.vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ])
        # Define which corners connect to each other
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back Face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front Face
            (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting Lines
        ]
        self.angle_x = 0
        self.angle_y = 0
        self.scale = 100  # Size of the cube

    def rotate(self, axis, angle):
        # Rotation Matrix Math
        if axis == 'x':
            rotation = np.array([
                [1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)]
            ])
        elif axis == 'y':
            rotation = np.array([
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)]
            ])
        return rotation

    def get_projected_points(self, center_pos, rot_x, rot_y):
        cx, cy = center_pos
        projected_points = []

        # Apply rotation matrices
        rx = self.rotate('x', rot_y)  # Hand Y controls X rotation (Pitch)
        ry = self.rotate('y', rot_x)  # Hand X controls Y rotation (Yaw)

        for v in self.vertices:
            # Rotate vertex
            rotated = np.dot(ry, v)
            rotated = np.dot(rx, rotated)

            # Simple Orthographic Projection (ignore Z depth for cleaner look)
            # x_2d = x * scale + center_x
            x = int(rotated[0] * self.scale + cx)
            y = int(rotated[1] * self.scale + cy)

            projected_points.append((x, y))

        return projected_points


# ------------------------------
# MEDIA PIPE SETUP
# ------------------------------
# Support both old (solutions) and new (tasks) MediaPipe APIs
USE_NEW_API = False
hands = None
MPImage = None  # Will be set if using new API
ImageFormat = None  # Will be set if using new API

# Try old API first (MediaPipe 0.10.8)
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)
    USE_NEW_API = False
    print("Using MediaPipe solutions API (legacy)")
except AttributeError:
    # Try new API (MediaPipe 0.10.30+)
    try:
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
        from mediapipe import Image as MPImage, ImageFormat
        from mediapipe.tasks import python as tasks_python
        import os
        import urllib.request
        
        # Download model if not exists
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(model_url, model_path)
            print("Model downloaded successfully!")
        
        base_options = HandLandmarkerOptions(
            base_options=tasks_python.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.8
        )
        hands = HandLandmarker.create_from_options(base_options)
        USE_NEW_API = True
        print("Using MediaPipe tasks API (new)")
    except Exception as e:
        import sys
        print("=" * 60)
        print("ERROR: Could not initialize MediaPipe")
        print("=" * 60)
        print(f"Error: {e}")
        print(f"Python version: {sys.version}")
        print(f"MediaPipe version: {mp.__version__ if hasattr(mp, '__version__') else 'unknown'}")
        print("\nSOLUTIONS:")
        print("1. Use Python 3.10 or 3.11 with: pip install mediapipe==0.10.8")
        print("2. For Python 3.13, ensure MediaPipe 0.10.30+ is installed")
        print("=" * 60)
        sys.exit(1)

# ------------------------------
# PALETTE CLASS
# ------------------------------


class Palette:
    def __init__(self):
        self.colors = [
            ((0, 0, 255), "RED"),
            ((0, 255, 0), "GREEN"),
            ((255, 255, 0), "CYAN"),
            ((255, 0, 255), "MAGENTA"),
            ((255, 255, 255), "WHITE"),
            ((200, 200, 200), "CUBE"),    # NEW 3D MODE
            ((0, 0, 0), "CLEAR")
        ]
        self.current_color = (255, 255, 0)
        self.mode = "DRAW"  # Modes: DRAW, CUBE
        self.center = (640, 0)
        self.radius = 120

    def draw_and_check(self, img, cursor_pos, is_pinching):
        overlay = img.copy()
        cx, cy = self.center

        # Draw Palette Arc
        num_colors = len(self.colors)
        sector_size = 180 / num_colors

        hovered_item = None

        for i, (color, name) in enumerate(self.colors):
            start_angle = i * sector_size
            end_angle = (i + 1) * sector_size

            # Check hover
            in_sector = False
            if cursor_pos:
                mx, my = cursor_pos
                dist = math.hypot(mx - cx, my - cy)
                if self.radius < dist < self.radius + 60:
                    angle = math.degrees(math.atan2(my - cy, mx - cx))
                    if angle < 0:
                        angle += 360
                    if start_angle <= angle <= end_angle:
                        in_sector = True
                        hovered_item = (color, name)

            # Draw Segment
            thickness = 40 if not in_sector else 60

            # Highlight Active Mode
            draw_col = color
            if name == "CUBE" and self.mode == "CUBE":
                draw_col = (255, 100, 100)  # Highlight Cube button if active

            cv2.ellipse(overlay, (cx, cy), (self.radius + 30, self.radius + 30),
                        0, start_angle, end_angle, draw_col, thickness)

            if in_sector:
                cv2.putText(overlay, name, (cx - 50, cy + 250),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

        # Handle Selection
        action = None
        if hovered_item and is_pinching:
            color, name = hovered_item
            if name == "CLEAR":
                action = "CLEAR"
            elif name == "CUBE":
                self.mode = "CUBE"
            else:
                self.mode = "DRAW"
                self.current_color = color

        modified_img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        return action, modified_img

# ------------------------------
# MAIN LOOP
# ------------------------------


def main():
    global USE_NEW_API, hands
    
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    palette = Palette()
    engine = Engine3D()

    prev_x, prev_y = 0, 0
    curr_x, curr_y = 0, 0

    # Debounce for stamping (prevents drawing 50 cubes in 1 second)
    can_stamp = True

    print("--------------------------------------")
    print("   IRON CANVAS 3D: SYSTEMS ONLINE")
    print("--------------------------------------")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cursor_pos = None
        hand_landmarks_list = None

        # Process with appropriate API
        if USE_NEW_API:
            # New API (MediaPipe 0.10.30+)
            mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_img)
            detection_result = hands.detect(mp_image)
            if detection_result.hand_landmarks:
                hand_landmarks_list = detection_result.hand_landmarks
        else:
            # Old API (MediaPipe 0.10.8)
            results = hands.process(rgb_img)
            if results.multi_hand_landmarks:
                hand_landmarks_list = results.multi_hand_landmarks

        if hand_landmarks_list:
            for hand_lms in hand_landmarks_list:
                # Handle different landmark structures
                if USE_NEW_API:
                    # New API: hand_lms is already a list of landmarks
                    lm_list = [(int(lm.x * WIDTH), int(lm.y * HEIGHT))
                               for lm in hand_lms]
                else:
                    # Old API: hand_lms.landmark contains the landmarks
                    lm_list = [(int(lm.x * WIDTH), int(lm.y * HEIGHT))
                               for lm in hand_lms.landmark]

                if lm_list:
                    # Index (8) and Thumb (4)
                    x1, y1 = lm_list[8]
                    x2, y2 = lm_list[4]

                    # Smoothing
                    curr_x = prev_x + (x1 - prev_x) / SMOOTHING
                    curr_y = prev_y + (y1 - prev_y) / SMOOTHING
                    cursor_pos = (int(curr_x), int(curr_y))

                    dist = math.hypot(x2 - x1, y2 - y1)
                    is_pinching = dist < PINCH_THRESHOLD

                    # Draw Cursor
                    cv2.circle(img, (x1, y1), 8, palette.current_color, -1)

                    # HUD: Palette
                    action, img = palette.draw_and_check(
                        img, cursor_pos, is_pinching)
                    if action == "CLEAR":
                        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

                    # LOGIC: Check Mode
                    if curr_y > 200:  # Below Palette

                        # --- 3D CUBE MODE ---
                        if palette.mode == "CUBE":
                            # Calculate Rotation based on Hand Position (screen center = 0 rot)
                            rot_x = (curr_x - WIDTH/2) / 200  # Yaw
                            rot_y = (curr_y - HEIGHT/2) / 200  # Pitch

                            # Get 3D points
                            points = engine.get_projected_points(
                                (curr_x, curr_y), rot_x, rot_y)

                            # Draw Wireframe on CAMERA (Preview)
                            for edge in engine.edges:
                                pt1 = points[edge[0]]
                                pt2 = points[edge[1]]
                                cv2.line(img, pt1, pt2, (255, 255, 255), 2)

                            # STAMP LOGIC (Draw on Canvas)
                            if is_pinching and can_stamp:
                                for edge in engine.edges:
                                    pt1 = points[edge[0]]
                                    pt2 = points[edge[1]]
                                    cv2.line(canvas, pt1, pt2,
                                             palette.current_color, 4)
                                can_stamp = False  # Lock stamping
                            elif not is_pinching:
                                can_stamp = True  # Reset lock

                        # --- NORMAL DRAW MODE ---
                        elif palette.mode == "DRAW" and is_pinching:
                            if prev_x == 0:
                                prev_x, prev_y = curr_x, curr_y
                            cv2.line(canvas, (int(prev_x), int(prev_y)),
                                     (int(curr_x), int(curr_y)),
                                     palette.current_color, BRUSH_THICKNESS)

                    prev_x, prev_y = curr_x, curr_y
        else:
            _, img = palette.draw_and_check(img, None, False)
            prev_x, prev_y = 0, 0

        # --- GLOW & RENDER ---
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, img_inv)
        img = cv2.add(img, canvas)

        canvas_blur = cv2.GaussianBlur(canvas, (21, 21), 0)
        img = cv2.addWeighted(img, 1.0, canvas_blur, NEON_GLOW_INTENSITY, 0)

        cv2.imshow("Iron Canvas 3D", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
