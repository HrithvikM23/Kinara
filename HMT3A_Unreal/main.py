import sys
import cv2

from camera.video_input        import choose_video
from pose_server.pose_detector import PoseDetector
from network.packet_builder    import build_packet
from network.udp_sender        import UDPSender
import config
from utils.video_output import VideoWriter

def run_pipeline(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    detector = PoseDetector()
    sender   = UDPSender()

    frame_index = 0

    for frame in [first_frame]:
        people, rendered = detector.detect(frame)
        if people:
            sender.send(build_packet(people, frame_index))
        cv2.imshow("Pose", rendered)
        frame_index += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people, rendered = detector.detect(frame)

        if people:
            packet = build_packet(people, frame_index)
            sender.send(packet)
            print(f"Frame {frame_index:04d} | {len(people)} person(s) | {len(packet)} bytes")

        cv2.imshow("Pose", rendered)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_index += 1

    cap.release()
    detector.close()
    sender.close()
    cv2.destroyAllWindows()


def choose_input():
    print("=" * 50)
    print("Select input source:")
    print("1. Webcam")
    print("2. Video file")
    print("=" * 50)

    choice = int(input("Enter choice: "))

    if choice == 1:
        run_pipeline(0)
    elif choice == 2:
        run_pipeline(choose_video())
    else:
        print("Invalid choice.")
        sys.exit(1)


def main():
    config.MAX_PERSONS = int(input("Enter number of people to track: "))
    choose_input()


if __name__ == "__main__":
    main()