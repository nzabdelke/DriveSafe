# Importing all the required packages
import cv2
import dlib
import time
from collections import OrderedDict
import numpy as np
from scipy import spatial

# Remove
from imutils.video import VideoStream


def shapeObjectToArr(shape, dtype="int"):
    coordinateTuple = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coordinateTuple[i] = (shape.part(i).x, shape.part(i).y)
    return coordinateTuple


def calculateEAR(eye):
    # abs(p2-p6)
    numeratorOne = spatial.distance.euclidean(eye[1], eye[5])
    # abs(p3-p5)
    numeratorTwo = spatial.distance.euclidean(eye[2], eye[4])
    # abs(p1-p4)
    denominator = spatial.distance.euclidean(eye[0], eye[3])
    # Calculating the eye aspect rario
    EAR = (numeratorOne + numeratorTwo) / (
        2.0 * denominator
    )  # Denominator is multiplied by 2 since we have 2 sets of vertical displacements and only 1 horizontal
    return EAR


def resize(image, width=None, height=None):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:

        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    newImage = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return newImage


FRAMES_BELOW_EAR_MAX = (
    25  # Number of frames where EAR is below the min and alert should sound
)
EAR_MIN = 0.25  # EAR min which indicates that EAR is below the minimum for an open eye
numOfFrames = 0  # Number of consecutive frames where EAR below min. If numOfFrames is more than FRAMES_BELOW_EAR_MAX, sound alert.
numOfFrames2 = 0  # Number of consecutive frames where there are no eyes detected
# Make instance of dlib's face detector (Histogram of Oriented Gradients) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)  # .dat file from http://dlib.net/face_landmark_detection.py.html


FACIAL_LANDMARKS = OrderedDict(
    [
        ("mouth", (48, 68)),
        ("inner_mouth", (60, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17)),
    ]
)

# Extracting left and right eye by slicing the list produced by the facial landmarks from dlib
(rightStart, rightEnd) = FACIAL_LANDMARKS["right_eye"]
(leftStart, leftEnd) = FACIAL_LANDMARKS["left_eye"]

# Capture stream from the webcam
videoStream = VideoStream(
    src=0
).start()  # Make instance of VideoStream using the webcam argument
time.sleep(0.5)  # Pause for 0.5 seconds to give buffer for webcam

# Iterate over the frames from the video stream
while True:
    # Read each frame
    frame = videoStream.read()
    # Resizing image to make it easier to preprocess
    frame = resize(frame, width=600)
    # Grayscaling the picture to improve performance
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    faces = detector(grayscale, 0)

    if len(faces) != 1:
        numOfFrames2 += 1
        if numOfFrames2 >= FRAMES_BELOW_EAR_MAX:
            cv2.putText(
                frame,
                "PAY ATTENTION!",
                (20, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
    else:
        numOfFrames2 = 0

    # Iterare over detected faces from the view stream. This allows for multiple face detection
    for face in faces:

        # Find the facial landmarks for a face and convert (x,y) tuples to an array
        facialLandmark = predictor(grayscale, face)
        facialLandmark = shapeObjectToArr(facialLandmark)
        print(facialLandmark)

        # Slice  array to get the coordinates of both eyes using indices
        rightEye = facialLandmark[rightStart:rightEnd]
        rightEAR = calculateEAR(rightEye)
        leftEye = facialLandmark[leftStart:leftEnd]
        leftEAR = calculateEAR(leftEye)

        # Find average EAR using distinct EAR for each eye
        EAR = (leftEAR + rightEAR) / 2.0

        # Find the conex hull(smallest convex polygon that contains all the points of it) for the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # Draw the convex hull
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        # If below the min EAR, increment the # of frames
        if EAR < EAR_MIN:
            numOfFrames += 1
            if numOfFrames >= FRAMES_BELOW_EAR_MAX:
                # Tell driver to pay attention
                cv2.putText(
                    frame,
                    "PAY ATTENTION!",
                    (20, 30),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

        # Reset numOfFrames
        else:
            numOfFrames = 0

        # Display the EAR
        cv2.putText(
            frame,
            "EAR: {:.2f}".format(EAR),
            (240, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    if len(faces) != 1:
        cv2.putText(
            frame,
            "PAY ATTENTION!",
            (20, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    # Display the program frame
    cv2.imshow("Driver Attentiveness Detector", frame)
    interruptKey = (
        cv2.waitKey(1) & 0xFF
    )  # Mask variable by leaving only last 8 bits as ASCII character set uses 8 bits

    # Exit if user presses q
    if interruptKey == ord("q"):
        break

# De-allocate any associated memory usage
cv2.destroyAllWindows()
videoStream.stop()

