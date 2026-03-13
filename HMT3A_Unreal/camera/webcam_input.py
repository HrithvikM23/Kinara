import cv2


def run_webcam():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()