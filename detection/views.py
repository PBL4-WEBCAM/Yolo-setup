import cv2
from django.http import StreamingHttpResponse
from django.shortcuts import render
from ultralytics import YOLO
from collections import Counter
import os

# Load model
model = YOLO("yolov8s.pt")

# Biến lưu trữ kết quả nhận diện qua các frame
detection_buffer = []
BUFFER_SIZE = 10  # Đủ 10 frame mới xử lý
last_spoken_object = None  # Tránh phát âm lặp lại liên tục

# Dictionary dịch sang tiếng Việt (thêm các từ bạn cần)
TRANSLATIONS = {
    'person': 'người',
    'laptop': 'máy tính xách tay',
    'phone': 'điện thoại',
    'cup': 'cốc',
    'bottle': 'chai',
    'book': 'sách',
    'keyboard': 'bàn phím',
    'mouse': 'chuột',
    'chair': 'ghế',
    'desk': 'bàn',
    'monitor': 'màn hình',
    'cell phone': 'điện thoại di động',
    'tv': 'tivi',
    'remote': 'điều khiển',
    'clock': 'đồng hồ',
    'backpack': 'ba lô',
    'handbag': 'túi xách',
    'tie': 'cà vạt',
    'umbrella': 'ô',
    'car': 'ô tô',
    'bicycle': 'xe đạp',
    'dog': 'chó',
    'cat': 'mèo',
    'bird': 'chim',
}


def speak_object(object_name, language='vi'):
    """Phát âm tên đồ vật qua loa"""
    if language == 'vi':
        # Dịch sang tiếng Việt
        text = TRANSLATIONS.get(object_name.lower(), object_name)
        # Dùng giọng tiếng Việt (nếu có cài)
        os.system(f'say -v "Thi" "{text}"')
    else:
        # Giọng Anh Mỹ
        text = object_name
        os.system(f'say -v "Samantha" "{text}"')


def gen_frames():
    global detection_buffer, last_spoken_object

    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Nhận diện với model (tắt verbose log)
        results = model(frame, stream=True, conf=0.5, verbose=False)

        # Lấy object có confidence cao nhất trong frame này
        best_detection = None
        best_conf = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])

                if conf > best_conf:
                    best_conf = conf
                    best_detection = label

        # Thêm vào buffer (None nếu không có gì)
        detection_buffer.append(best_detection)

        # Giới hạn buffer size
        if len(detection_buffer) > BUFFER_SIZE:
            detection_buffer.pop(0)

        # Khi đủ 10 frames, xử lý
        if len(detection_buffer) == BUFFER_SIZE:
            # Lọc bỏ None (frame không phát hiện gì)
            valid_detections = [d for d in detection_buffer if d is not None]

            # Chỉ xử lý nếu có ít nhất 6/10 frames phát hiện object
            if len(valid_detections) >= 6:
                # Đếm object xuất hiện nhiều nhất
                counter = Counter(valid_detections)
                most_common_object, count = counter.most_common(1)[0]

                # Nếu object này khác với object vừa nói
                if most_common_object != last_spoken_object:
                    print(f"\n🔍 DETECTED: {most_common_object.upper()}")
                    print(
                        f"   Confidence: {count}/{len(valid_detections)} frames ({count / len(valid_detections) * 100:.1f}%)")
                    print(f"   Frame: {frame_count}")
                    print("-" * 50)

                    # Phát âm (chọn 'vi' hoặc 'en')
                    speak_object(most_common_object, language='en')

                    last_spoken_object = most_common_object

            # Reset buffer
            detection_buffer = []

        # Hiển thị frame gốc (không vẽ gì)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def home(request):
    return render(request, "detect.html")