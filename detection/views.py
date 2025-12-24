import cv2
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
from collections import Counter
import os
import platform
import socket
import time
from unidecode import unidecode
import base64
from datetime import datetime

# ======================================================================
# 1. TỪ ĐIỂN DỊCH THUẬT (COCO 80 LỚP)
# ======================================================================
COCO_VIETNAMESE = {
    'person': 'Người', 'bicycle': 'Xe đạp', 'car': 'Ô tô', 'motorcycle': 'Xe máy',
    'airplane': 'Máy bay', 'bus': 'Xe buýt', 'train': 'Tàu hỏa', 'truck': 'Xe tải',
    'boat': 'Thuyền', 'traffic light': 'Đèn giao thông', 'fire hydrant': 'Trụ cứu hỏa',
    'stop sign': 'Biển báo dừng', 'parking meter': 'Máy đỗ xe', 'bench': 'Ghế dài',
    'bird': 'Chim', 'cat': 'Mèo', 'dog': 'Chó', 'horse': 'Ngựa', 'sheep': 'Cừu',
    'cow': 'Bò', 'elephant': 'Voi', 'bear': 'Gấu', 'zebra': 'Ngựa vằn',
    'giraffe': 'Hươu cao cổ', 'backpack': 'Ba lô', 'umbrella': 'Cái ô',
    'handbag': 'Túi xách', 'tie': 'Cà vạt', 'suitcase': 'Va li', 'frisbee': 'Đĩa ném',
    'skis': 'Ván trượt tuyết', 'snowboard': 'Ván trượt tuyết', 'sports ball': 'Bóng thể thao',
    'kite': 'Cái diều', 'baseball bat': 'Gậy bóng chày', 'baseball glove': 'Găng bóng chày',
    'skateboard': 'Ván trượt', 'surfboard': 'Ván lướt sóng', 'tennis racket': 'Vợt tennis',
    'bottle': 'Cái chai', 'wine glass': 'Ly rượu', 'cup': 'Cái cốc', 'fork': 'Cái nĩa',
    'knife': 'Con dao', 'spoon': 'Cái thìa', 'bowl': 'Cái bát', 'banana': 'Chuối',
    'apple': 'Táo', 'sandwich': 'Bánh sandwich', 'orange': 'Cam', 'broccoli': 'Bông cải xanh',
    'carrot': 'Cà rốt', 'hot dog': 'Xúc xích', 'pizza': 'Pizza', 'donut': 'Bánh donut',
    'cake': 'Bánh kem', 'chair': 'Cái ghế', 'couch': 'Ghế sofa', 'potted plant': 'Cây cảnh',
    'bed': 'Cái giường', 'dining table': 'Bàn ăn', 'toilet': 'Bồn cầu', 'tv': 'Ti vi',
    'laptop': 'Máy tính xách tay', 'mouse': 'Chuột máy tính', 'remote': 'Điều khiển',
    'keyboard': 'Bàn phím', 'cell phone': 'Điện thoại', 'microwave': 'Lò vi sóng',
    'oven': 'Lò nướng', 'toaster': 'Máy nướng bánh', 'sink': 'Bồn rửa',
    'refrigerator': 'Tủ lạnh', 'book': 'Quyển sách', 'clock': 'Đồng hồ',
    'vase': 'Lọ hoa', 'scissors': 'Cái kéo', 'teddy bear': 'Gấu bông',
    'hair drier': 'Máy sấy tóc', 'toothbrush': 'Bàn chải'
}

# ======================================================================
# 2. CẤU HÌNH KẾT NỐI WIFI (UDP)
# ======================================================================
ESP_IP = "172.20.10.2"  # <--- HÃY SỬA SỐ NÀY (Ví dụ: 172.20.10.3)
ESP_PORT = 4210


def send_to_esp(text):
    """Gửi dữ liệu qua mạng WiFi (UDP) đến ESP8266"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(text.encode('utf-8'), (ESP_IP, ESP_PORT))
        print(f"   📡 Đã bắn tín hiệu đến {ESP_IP}: {text.replace('|', '/')}")
        sock.close()
    except Exception as e:
        print(f"   ⚠️ Lỗi gửi WiFi: {e}")
        print(f"      -> Kiểm tra lại xem IP {ESP_IP} có đúng chưa?")


# ======================================================================
# 3. BIẾN TOÀN CỤC LƯU TRẠNG THÁI (CHO API)
# ======================================================================
model = YOLO("yolov8s.pt")
detection_buffer = []
BUFFER_SIZE = 10
last_spoken_object = None

# ⭐ BIẾN MỚI: Lưu thông tin detection cuối cùng để API lấy
last_detection_data = None


# ======================================================================
# 4. LOGIC CHÍNH (ĐÃ CẬP NHẬT ĐỂ LƯU DỮ LIỆU CHO API)
# ======================================================================
def gen_frames():
    global detection_buffer, last_spoken_object, last_detection_data

    cap = cv2.VideoCapture("http://172.20.10.7:81/stream")
    frame_count = 0
    last_saved_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        results = model(frame, stream=True, conf=0.5, verbose=False)

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

        detection_buffer.append(best_detection)
        if len(detection_buffer) > BUFFER_SIZE:
            detection_buffer.pop(0)

        if len(detection_buffer) == BUFFER_SIZE:
            valid_detections = [d for d in detection_buffer if d is not None]

            if len(valid_detections) >= 6:
                counter = Counter(valid_detections)
                most_common_object, count = counter.most_common(1)[0]

                if most_common_object != last_spoken_object:
                    print(f"\n🔍 DETECTED: {most_common_object.upper()}")

                    # Chuẩn bị dữ liệu
                    vietnamese_name = COCO_VIETNAMESE.get(most_common_object, "???")
                    vietnamese_unaccented = unidecode(vietnamese_name)
                    english_name = most_common_object.title()

                    # Lấy ID Class
                    class_id = -1
                    for k, v in model.names.items():
                        if v == most_common_object:
                            class_id = k
                            break

                    # ⭐ MỚI: Lưu frame hiện tại thành base64 (thumbnail)
                    _, buffer = cv2.imencode('.jpg', frame)
                    thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')

                    # ⭐ MỚI: Lưu dữ liệu detection vào biến toàn cục
                    last_detection_data = {
                        'label': english_name,
                        'vocabulary': vietnamese_name,
                        'confidence': best_conf,
                        'detected_time': datetime.now().isoformat(),
                        'thumbnail': f"data:image/jpeg;base64,{thumbnail_base64}",
                        'class_id': class_id
                    }

                    # Gửi qua WiFi (GIỮ NGUYÊN LOGIC CŨ)
                    if class_id != -1:
                        display_string = f"{english_name}|{vietnamese_unaccented}|{class_id}"
                        send_to_esp(display_string)

                    last_spoken_object = most_common_object

            detection_buffer = []

        # Mã hóa hình ảnh để hiển thị lên Web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


# ======================================================================
# 5. API ENDPOINTS (MỚI THÊM)
# ======================================================================

@csrf_exempt
def api_get_last_detection(request):
    """
    API GET: Lấy thông tin detection cuối cùng
    URL: http://localhost:8000/api/get-last-detection/
    """
    global last_detection_data

    if last_detection_data is None:
        return JsonResponse({
            'success': False,
            'message': 'Chưa có đối tượng nào được nhận diện',
            'detection': None
        })

    return JsonResponse({
        'success': True,
        'message': 'Đã có dữ liệu',
        'detection': last_detection_data
    })


@csrf_exempt
def api_reset_detection(request):
    """
    API POST: Reset trạng thái detection (để nhận diện đối tượng mới)
    URL: http://localhost:8000/api/reset-detection/
    """
    global last_spoken_object, last_detection_data, detection_buffer

    if request.method == 'POST':
        last_spoken_object = None
        last_detection_data = None
        detection_buffer = []

        print("\n🔄 API RESET: Đã xóa trạng thái, sẵn sàng nhận diện đối tượng mới!")

        return JsonResponse({
            'success': True,
            'message': 'Đã reset thành công. Hệ thống sẵn sàng nhận diện đối tượng mới.'
        })

    return JsonResponse({
        'success': False,
        'message': 'Chỉ chấp nhận POST request'
    }, status=405)


# ======================================================================
# 6. VIEWS GỐC (GIỮ NGUYÊN)
# ======================================================================
def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def home(request):
    return render(request, "detect.html")