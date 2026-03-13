import cv2
import tkinter as tk
from tkinter import filedialog


def choose_video():

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = ""

    while not path:
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )

    root.destroy()

    return path


def run_video():

    video_path = choose_video()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Video could not be opened")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()