import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True)

# Read image (make sure person.jpg exists in same folder)
image = cv2.imread("person.jpg")

if image is None:
    print("Error: Image not found")
    exit()

# Convert to RGB
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process pose
results = pose.process(img_rgb)

# Check if pose is detected
if results.pose_landmarks:
    print("Pose detected!")

    # Draw landmarks on image
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    # Show image with landmarks
    cv2.imshow("Pose", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Print landmark coordinates
    for i, lm in enumerate(results.pose_landmarks.landmark):
        print(i, lm.x, lm.y)

else:
    print("No pose detected")
