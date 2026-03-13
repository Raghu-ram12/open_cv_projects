import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
latest_result = None


def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):  # type: ignore
    global latest_result
    latest_result = result


cap = cv2.VideoCapture(0)
time_stamp = 0
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=print_result,
)
current_stroke = []
strokes = []
write = True
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
            for hand_landmarks, handed in zip(
                latest_result.hand_landmarks, latest_result.handedness
            ):  # type:ignore
                # print(handed[0].display_name)
                for lm in hand_landmarks:
                    screen_x = int(lm.x * width)
                    screen_y = int(lm.y * height)
                    cv2.circle(frame, (screen_x, screen_y), 5, (0, 0, 225), -1)

                if handed[0].display_name == "Left":

                    tip_y = hand_landmarks[8].y * height
                    dip_y = hand_landmarks[7].y * height
                    middle_tip = hand_landmarks[12].y * height
                    middle_dip = hand_landmarks[11].y * height
                    
                    #drawing 
                    
                    if tip_y < dip_y and write:
                        point = (
                            int(hand_landmarks[8].x * width),
                            int(hand_landmarks[8].y * height),
                        )
                        current_stroke.append(point)

                    # erasing last stroke
                    
                    thumb_x = hand_landmarks[4].x * width

                    if hand_landmarks[8].x * width < thumb_x and strokes:
                        #print(thumb_x)
                        strokes=[]

                    # new strokes
                    
                    if middle_dip < middle_tip:
                        write = True
                    else:
                        write = False
                        strokes.append(current_stroke)
                        current_stroke = []
                
                #drawing previous strokes
                
                for stroke in strokes:
                    for i in range(1, len(stroke)):
                        cv2.line(frame, stroke[i - 1], stroke[i], (0, 225, 0), 5)
                
                # drawing current stroke
                
                if len(current_stroke) >= 2:
                    for i in range(1, len(current_stroke)):
                        cv2.line(
                            frame,
                            current_stroke[i - 1],
                            current_stroke[i],
                            (0, 225, 0),
                            5,
                        )

        cv2.imshow("Hand Landmark Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
