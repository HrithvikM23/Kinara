import sys
import cv2

from camera.video_input import choose_video
from camera.webcam_input import run_webcam
from pose_server.pose_detector import PoseDetector
from network.packet_builder import build_packet
from network.udp_sender import UDPSender


def run_pipeline(source):
    """
    source: path to video file, or 0 for webcam
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # Read first frame to determine resolution for OpenPose
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    detector = PoseDetector(first_frame)
    sender   = UDPSender()

    # Process first frame immediately so it isn't skipped
    frame_index = 0
    for frame in [first_frame]:
        keypoints, rendered = detector.detect(frame)
        if keypoints is not None:
            sender.send(build_packet(keypoints, frame_index))
        cv2.imshow("Pose", rendered)
        frame_index += 1

    # Process remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, rendered = detector.detect(frame)

        if keypoints is not None:
            packet = build_packet(keypoints, frame_index)
            sender.send(packet)
            print(f"Frame {frame_index:04d} | sent {len(packet)} bytes")

        cv2.imshow("Pose", rendered)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

        frame_index += 1

    cap.release()
    sender.close()
    cv2.destroyAllWindows()


def choose_input():

    print("Select input source:")
    print("1. Webcam")
    print("2. Video file")

    choice = input("Enter choice: ").strip()

    if choice == "1":
        run_pipeline(0)

    elif choice == "2":
        from camera.video_input import choose_video
        run_pipeline(choose_video())

    else:
        print("Invalid choice.")
        sys.exit(1)


def main():
    choose_input()


if __name__ == "__main__":
    main()