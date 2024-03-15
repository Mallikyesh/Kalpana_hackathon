import cv2
import os

# Path to the Haar cascade file
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Create a cascade classifier object
clf = cv2.CascadeClassifier(cascade_path)

# Capture video from the default camera (built-in webcam)
camera = cv2.VideoCapture(0)

# Directory to save the captured face images
save_dir = "captured_faces"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Counter for naming the saved images
image_counter = 0

# Main loop for face detection
while True:
    # Read a frame from the camera
    _, frame = camera.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Draw rectangles around the detected faces and capture images
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
        
        # Capture the face region
        face_region = frame[y:y+height, x:x+width]
        
        # Save the captured face image
        image_path = os.path.join(save_dir, f"face_{image_counter}.jpg")
        cv2.imwrite(image_path, face_region)
        print(f"Saved image: {image_path}")
        
        # Increment image counter
        image_counter += 1
    
    # Display the frame with detected faces
    cv2.imshow("Faces", frame)
    
    # Check for 'q' key to quit the loop
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera
camera.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
