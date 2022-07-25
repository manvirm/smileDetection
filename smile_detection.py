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
    
    # scalefactor is how much you want to blur image (therefore less data in image, easier to detect smile)
    # minNeighbors is how many rectangles need to be in the area for it to be a smile,
    # this is good with dealing with overlapping smiles (many squares in one area)


    for (x, y, w, h) in face_coordinates:

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 5)

        #slice frame to only show face, first argument slices the number of arrays within the array
        #second argument slices the actual numbers within the array for each sliced array
        #since frame is a 2D array
        face = frame[y:y+h, x:x+w]

        grayscaled_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # scalefactor is how much you want to blur image (therefore less data in image, easier to detect smile)
        # minNeighbors is how many rectangles need to be in the area for it to be a smile,
        # this is good with dealing with overlapping smiles (many squares in one area)
        smile_coordinates = trained_smile_data.detectMultiScale(grayscaled_face, scaleFactor=1.7, minNeighbors=20)

        for (xs, ys, ws, hs) in smile_coordinates:

            cv2.rectangle(face, (xs,ys), (xs+ws, ys+hs), (0, 255, 0), 5)
            #break so it only prints one rectangle per face
            break

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