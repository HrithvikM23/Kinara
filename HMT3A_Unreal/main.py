import sys

from camera.webcam_input import run_webcam
from camera.video_input import run_video


def choose_input():

    print("Select input source:")
    print("1. Webcam (very slow)")
    print("2. Video file")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        run_webcam()

    elif choice == "2":
        run_video()

    else:
        print("Invalid choice")
        sys.exit(1)


def main():
    choose_input()


if __name__ == "__main__":
    main()