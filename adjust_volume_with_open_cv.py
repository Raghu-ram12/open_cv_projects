import mediapipe as mp
import cv2
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):  # type: ignore
    global latest_result
    latest_result = result


devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume  # type:ignore
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)

cap = cv2.VideoCapture(0)
time_stamp = 0
MAX_DISTANCE = 200
MIN_DISTANCE = 30

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        landmarker.detect_async(mp_image, time_stamp)
        time_stamp += 1

        if latest_result and latest_result.hand_landmarks:
            height, width, _ = frame.shape
            for hand_landmarks in latest_result.hand_landmarks:  # type:ignore
                for lm in hand_landmarks:
                    screen_x = int(lm.x * width)
                    screen_y = int(lm.y * height)
                    cv2.circle(frame, (screen_x, screen_y), 5, (0, 0, 225), -1)

                thumb = hand_landmarks[4]
                index = hand_landmarks[8]
                thumb_x = int(thumb.x * width)
                thumb_y = int(thumb.y * height)
                index_x = int(index.x * width)
                index_y = int(index.y * height)

                dist = distance(thumb_x, thumb_y, index_x, index_y)

                clamped_dist = max(MIN_DISTANCE, min(MAX_DISTANCE, dist))

                vol_percent = (clamped_dist - MIN_DISTANCE) / (
                    MAX_DISTANCE - MIN_DISTANCE
                )

                target_vol_db = min_vol + (vol_percent * (max_vol - min_vol))

                volume.SetMasterVolumeLevel(target_vol_db, None)

                cv2.circle(frame, (thumb_x, thumb_y), 8, (255, 0, 0), -1)
                cv2.circle(frame, (index_x, index_y), 8, (255, 0, 0), -1)
                cv2.line(
                    frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 255), 3
                )

                cv2.putText(
                    frame,
                    f"Dist: {dist:.1f} Vol: {vol_percent*100:.1f}%",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Hand Landmark Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
