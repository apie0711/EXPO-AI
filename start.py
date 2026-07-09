from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time


# =========================
# 설정
# =========================

MODEL_CANDIDATES = [
    Path(r"C:\Users\kim\Desktop\EXPO딥러닝\runs\detect\runs\detect\road_ai_yolo11n_2class_jetson\weights\best.pt"),
    Path("models/selected_best.pt"),
    Path("runs/detect/road_ai_yolo11n_2class_jetson/weights/best.pt"),
    Path("runs/detect/train/weights/best.pt"),
    Path("yolo11n.pt")
]
VIDEO_PATH = Path("test_video.mp4")
CAMERA_INDEX = 0
LOOP_VIDEO = True

USE_ROAD_AREA_FILTER = False

CONF_THRESHOLD = 0.35
IMG_SIZE = 640

TARGET_DISPLAY_FPS = 60

# 정확성 우선이면 매 프레임 추론
INFER_EVERY_N_FRAMES = 1

# 영상 프레임 건너뛰지 않음
VIDEO_READ_STRIDE = 1
# 영상 파일일 때 프레임 건너뛰기
# 1 = 안 건너뜀
# 2 = 한 프레임씩 건너뜀
# 3 = 두 프레임씩 건너뜀

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

DEVICE_ID = "DEVICE_001"
VEHICLE_ID = "CAR_001"


# =========================
# 모델 선택
# =========================

def select_model_path():
    for path in MODEL_CANDIDATES:
        if path.exists() or str(path) == "yolo11n.pt":
            return str(path)
    return "yolo11n.pt"


# =========================
# 빠른 텍스트 출력
# =========================

def put_text(frame, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# =========================
# 도로 영역
# =========================

def get_road_area(width, height):
    return np.array([
        [int(width * 0.12), height],
        [int(width * 0.88), height],
        [int(width * 0.63), int(height * 0.38)],
        [int(width * 0.37), int(height * 0.38)]
    ], dtype=np.int32)


def is_inside_road_area(cx, cy, road_area):
    return cv2.pointPolygonTest(road_area, (cx, cy), False) >= 0
def is_road_marking_like(frame, x1, y1, x2, y2):
    """
    화살표, 차선, 실선, 도로 글자처럼
    흰색/노란색 도로 표시로 보이는 영역을 걸러냄.
    """
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 흰색 계열: 밝고 채도가 낮음
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 170]),
        np.array([180, 80, 255])
    )

    # 노란색 계열: 노란 차선/표시
    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 60, 120]),
        np.array([40, 255, 255])
    )

    marking_mask = cv2.bitwise_or(white_mask, yellow_mask)

    marking_ratio = cv2.countNonZero(marking_mask) / (roi.shape[0] * roi.shape[1])

    # 박스 안에서 흰색/노란색 비율이 높으면 도로 표시로 판단
    return marking_ratio > 0.35

# =========================
# 클래스 이름 정리
# =========================

def normalize_class_name(raw_name):
    name = str(raw_name).upper().replace(" ", "_").replace("-", "_")

    if "POTHOLE" in name:
        return "POTHOLE"

    if "OBSTACLE" in name:
        return "OBSTACLE"

    return "OBSTACLE"


# =========================
# 위험도 / 레이더 더미 / JSON
# =========================

def estimate_radar_from_bbox(x1, y1, x2, y2, frame_width, frame_height):
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    area_ratio = (bbox_w * bbox_h) / (frame_width * frame_height)

    if area_ratio >= 0.10:
        distance = 1.5
    elif area_ratio >= 0.05:
        distance = 2.5
    elif area_ratio >= 0.02:
        distance = 3.5
    else:
        distance = 5.0

    return {
        "radarDetected": True,
        "radarDistance": distance,
        "pointCount": int(area_ratio * 1000) + 5,
        "meanSnr": round(10 + area_ratio * 100, 2),
        "maxSnr": round(18 + area_ratio * 120, 2)
    }


def calculate_risk_level(obstacle_type, confidence, radar_distance):
    if obstacle_type == "POTHOLE":
        return "HIGH"

    if radar_distance is not None and radar_distance <= 2.0:
        return "HIGH"

    if confidence >= 0.60:
        return "MEDIUM"

    return "LOW"


def calculate_final_confidence(camera_confidence, radar_detected):
    if radar_detected:
        return min(camera_confidence + 0.10, 0.99)
    return camera_confidence * 0.85


def make_server_payload(obstacle_type, risk_level, final_confidence,
                        camera_confidence, radar_data):
    return {
        "deviceId": DEVICE_ID,
        "vehicleId": VEHICLE_ID,
        "obstacleType": obstacle_type,
        "riskLevel": risk_level,
        "confidence": round(final_confidence, 2),
        "latitude": 37.123456,
        "longitude": 127.123456,
        "detectedAt": datetime.now().isoformat(timespec="seconds"),
        "radarDistance": radar_data["radarDistance"],
        "radarDetected": radar_data["radarDetected"],
        "cameraConfidence": round(camera_confidence, 2),
        "finalConfidence": round(final_confidence, 2),
        "imagePath": None
    }


# =========================
# 메인
# =========================

def main():
    global USE_ROAD_AREA_FILTER

    model_path = select_model_path()
    print("사용 모델:", model_path)

    model = YOLO(model_path)
    print("모델 클래스 목록:", model.names)

    # 영상 파일이 있으면 영상 사용, 없으면 카메라 사용
    using_video = VIDEO_PATH.exists()

    if using_video:
        print("입력 소스: 영상 파일", VIDEO_PATH)
        cap = cv2.VideoCapture(str(VIDEO_PATH))
    else:
        print("입력 소스: USB 카메라", CAMERA_INDEX)
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

        # 카메라 속도용 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("입력 소스를 열 수 없습니다.")
        return

    window_name = "Smart Road Saver - Fast Mode"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)

    frame_count = 0
    last_results = None
    last_payload_print_time = 0

    delay = max(1, int(1000 / TARGET_DISPLAY_FPS))

    fps_time = time.time()
    fps_counter = 0
    display_fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            if using_video and LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                last_results = None
                continue
            break

        # 영상 파일은 답답하면 프레임을 건너뜀
        if using_video and VIDEO_READ_STRIDE > 1:
            for _ in range(VIDEO_READ_STRIDE - 1):
                cap.grab()

        frame_count += 1
        height, width = frame.shape[:2]
        road_area = get_road_area(width, height)

        # =========================
        # YOLO 추론
        # =========================

        if frame_count % INFER_EVERY_N_FRAMES == 0 or last_results is None:
            last_results = model(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                device=0,
                verbose=False
            )

        results = last_results
        boxes = results[0].boxes

        display_frame = frame.copy()

        if USE_ROAD_AREA_FILTER:
            cv2.polylines(display_frame, [road_area], True, (255, 255, 0), 2)

        current_detection = None

        # =========================
        # 박스 처리
        # =========================

        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            raw_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else model.names[cls_id]
            obstacle_type = normalize_class_name(raw_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if bbox_w <= 0 or bbox_h <= 0:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            aspect_ratio = bbox_w / bbox_h
            area_ratio = (bbox_w * bbox_h) / (width * height)

          # 클래스별 confidence 기준
            if obstacle_type == "POTHOLE":
                if conf < 0.45:
                    continue

                # 도로표시를 포트홀로 착각한 경우 제거
                if is_road_marking_like(frame, x1, y1, x2, y2):
                    continue

                # 포트홀 오탐 필터
                if aspect_ratio > 5.0 or aspect_ratio < 0.20:
                    continue

                if area_ratio > 0.25:
                    continue

                if area_ratio < 0.0005:
                    continue


            if obstacle_type == "OBSTACLE":
                if conf < 0.50:
                    continue

                # 차선/실선/화살표/도로 글자 오인식 제거
                if aspect_ratio > 8.0 or aspect_ratio < 0.10:
                    continue

                if area_ratio > 0.35:
                    continue

                if is_road_marking_like(frame, x1, y1, x2, y2):
                    continue
            inside_road = True
            if USE_ROAD_AREA_FILTER:
                inside_road = is_inside_road_area(cx, cy, road_area)

            if not inside_road:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (120, 120, 120), 1)
                continue

            radar_data = estimate_radar_from_bbox(x1, y1, x2, y2, width, height)
            final_conf = calculate_final_confidence(conf, radar_data["radarDetected"])
            risk_level = calculate_risk_level(
                obstacle_type,
                final_conf,
                radar_data["radarDistance"]
            )

            if current_detection is None or final_conf > current_detection["finalConfidence"]:
                current_detection = {
                    "obstacleType": obstacle_type,
                    "cameraConfidence": conf,
                    "finalConfidence": final_conf,
                    "riskLevel": risk_level,
                    "bbox": (x1, y1, x2, y2),
                    "radarData": radar_data
                }

            # 박스 표시
            if obstacle_type == "POTHOLE":
                box_color = (0, 0, 255)
            else:
                box_color = (0, 165, 255)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
            cv2.circle(display_frame, (cx, cy), 5, box_color, -1)

            label = f"{obstacle_type} {conf:.2f}"
            put_text(
                display_frame,
                label,
                (x1, max(y1 - 10, 20)),
                scale=0.7,
                color=box_color,
                thickness=2
            )

        # =========================
        # FPS 계산
        # =========================

        fps_counter += 1
        now = time.time()

        if now - fps_time >= 1.0:
            display_fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        # =========================
        # 상태 패널
        # =========================

        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (470, 260), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.45, display_frame, 0.55, 0)

        put_text(display_frame, "Smart Road Saver", (20, 45), scale=1.0, color=(255, 255, 255), thickness=2)

        if current_detection is not None:
            obstacle_type = current_detection["obstacleType"]
            risk_level = current_detection["riskLevel"]
            camera_conf = current_detection["cameraConfidence"]
            final_conf = current_detection["finalConfidence"]
            radar_data = current_detection["radarData"]

            put_text(display_frame, "STATUS: DANGER DETECTED", (20, 85), scale=0.8, color=(0, 0, 255), thickness=2)
            put_text(display_frame, f"TYPE: {obstacle_type}", (20, 120), scale=0.7, color=(255, 255, 255), thickness=2)
            put_text(display_frame, f"RISK: {risk_level}", (20, 150), scale=0.7, color=(255, 255, 255), thickness=2)
            put_text(display_frame, f"CAM CONF: {camera_conf:.2f}", (20, 180), scale=0.7, color=(255, 255, 255), thickness=2)
            put_text(display_frame, f"FINAL CONF: {final_conf:.2f}", (20, 210), scale=0.7, color=(255, 255, 255), thickness=2)
            put_text(display_frame, f"RADAR: {radar_data['radarDistance']}m", (20, 240), scale=0.7, color=(255, 255, 255), thickness=2)

            payload = make_server_payload(
                obstacle_type=obstacle_type,
                risk_level=risk_level,
                final_confidence=final_conf,
                camera_confidence=camera_conf,
                radar_data=radar_data
            )

            if now - last_payload_print_time >= 1.0:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                last_payload_print_time = now

        else:
            put_text(display_frame, "STATUS: MONITORING", (20, 85), scale=0.8, color=(0, 255, 0), thickness=2)
            put_text(display_frame, "TYPE: NONE", (20, 120), scale=0.7, color=(255, 255, 255), thickness=2)
            put_text(display_frame, "RISK: LOW", (20, 150), scale=0.7, color=(255, 255, 255), thickness=2)
            put_text(display_frame, "CAMERA: NORMAL", (20, 180), scale=0.7, color=(255, 255, 255), thickness=2)
            put_text(display_frame, "RADAR: STANDBY", (20, 210), scale=0.7, color=(255, 255, 255), thickness=2)

        # 하단 정보
        put_text(
            display_frame,
            f"FPS: {display_fps:.1f} | IMG: {IMG_SIZE} | INFER: every {INFER_EVERY_N_FRAMES} frames | ESC: exit | R: road filter",
            (20, height - 25),
            scale=0.6,
            color=(255, 255, 255),
            thickness=2
        )

        if USE_ROAD_AREA_FILTER:
            put_text(
                display_frame,
                "Road-area filter ON",
                (20, height - 55),
                scale=0.6,
                color=(255, 255, 0),
                thickness=2
            )

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(delay) & 0xFF

        if key == 27:
            break

        if key == ord("r"):
            USE_ROAD_AREA_FILTER = not USE_ROAD_AREA_FILTER
            print("도로 영역 필터:", USE_ROAD_AREA_FILTER)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()