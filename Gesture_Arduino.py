import cv2

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Display the frame
        cv2.imshow('Camera', frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key != -1:
            break

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# import cv2
# import numpy as np
# import tensorflow as tf

# def load_model():
#     model = tf.keras.models.load_model('point_history_classifier.h5')
#     return model

# def preprocess_frame(frame):
#     # Preprocess the frame before passing it to the model
#     resized_frame = cv2.resize(frame, (224, 224))
#     normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
#     return normalized_frame

# def main():
#     # Load the pre-trained gesture recognition model
#     model = load_model()

#     # Open the default camera
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Couldn't open camera.")
#         return

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Couldn't capture frame.")
#             break

#         # Preprocess the frame
#         processed_frame = preprocess_frame(frame)

#         # Perform gesture recognition using the model
#         prediction = model.predict(np.expand_dims(processed_frame, axis=0))[0]

#         # Get the predicted gesture label
#         predicted_gesture = np.argmax(prediction)

#         # Display the predicted gesture label on the frame
#         cv2.putText(frame, f"Predicted Gesture: {predicted_gesture}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#         # Display the frame
#         cv2.imshow('Gesture Recognition', frame)

#         # Check for key press
#         key = cv2.waitKey(1)
#         if key != -1:
#             break

#     # Release the camera
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
