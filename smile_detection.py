# Read images, videos, etc
import cv2

# Load pre-trained data on face frontals from opencv (haar cascade algo)
# Classifiers are detectors (face detector in this case)
trained_face_data = cv2.CascadeClassifier('faceData.xml')
trained_smile_data = cv2.CascadeClassifier('smileData.xml')

# Use webcam, '0' will use default webcam
# Can also use videos for this
webcam = cv2.VideoCapture(0)

# Iterate over frames until video ends (webcam ends)

while True:

    # Read frames, first var is boolean, and second one is frame
    # Use frame instead of image
    successful_frame_read, frame = webcam.read()

    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # No matter size of face it will detect (multi scale)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)


    for (x, y, w, h) in face_coordinates:

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 5)

    # Show Frame
    cv2.imshow('Programming Face Detector', frame)

    # Pause code so image will show until key is pressed
    # Instead of waiting for key, we wait 1ms, so wont pause until key is pressed
    # Wait is needed so popup will stay
    key = cv2.waitKey(1)

    # If Q is pressed, stop loop (ASCII CODE FOR Q)
    if key == 81 or key == 113:
        break

# Clean up code
webcam.release()