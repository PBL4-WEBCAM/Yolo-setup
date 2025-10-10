import cv2
import numpy as np
import base64
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from ultralytics import YOLO
import json

# Load model OpenImages (đã train sẵn, 600+ classes)
model = YOLO("yolov8s.pt")

# Từ điển ánh xạ tiếng Anh -> Tiếng Việt
VOCABULARY_MAP = {
    'person': 'Người',
    'bicycle': 'Xe đạp',
    'car': 'Xe hơi',
    'motorcycle': 'Xe máy',
    'airplane': 'Máy bay',
    'bus': 'Xe buýt',
    'train': 'Tàu hỏa',
    'truck': 'Xe tải',
    'boat': 'Thuyền',
    'traffic light': 'Đèn giao thông',
    'fire hydrant': 'Vòi cứu hỏa',
    'stop sign': 'Biển báo dừng',
    'parking meter': 'Đồng hồ đỗ xe',
    'bench': 'Ghế dài',
    'bird': 'Chim',
    'cat': 'Con mèo',
    'dog': 'Con chó',
    'horse': 'Con ngựa',
    'sheep': 'Con cừu',
    'cow': 'Con bò',
    'elephant': 'Con voi',
    'bear': 'Con gấu',
    'zebra': 'Ngựa vằn',
    'giraffe': 'Hươu cao cổ',
    'backpack': 'Ba lô',
    'umbrella': 'Ô, dù',
    'handbag': 'Túi xách',
    'tie': 'Cà vạt',
    'suitcase': 'Vali',
    'frisbee': 'Đĩa bay',
    'skis': 'Ván trượt tuyết',
    'snowboard': 'Ván trượt tuyết',
    'sports ball': 'Bóng thể thao',
    'kite': 'Diều',
    'baseball bat': 'Gậy bóng chày',
    'baseball glove': 'Găng bóng chày',
    'skateboard': 'Ván trượt',
    'surfboard': 'Ván lướt sóng',
    'tennis racket': 'Vợt tennis',
    'bottle': 'Chai',
    'wine glass': 'Ly rượu',
    'cup': 'Cốc, chén',
    'fork': 'Nĩa',
    'knife': 'Dao',
    'spoon': 'Muỗng',
    'bowl': 'Bát, tô',
    'banana': 'Chuối',
    'apple': 'Táo',
    'sandwich': 'Bánh sandwich',
    'orange': 'Cam',
    'broccoli': 'Bông cải xanh',
    'carrot': 'Cà rốt',
    'hot dog': 'Xúc xích',
    'pizza': 'Pizza',
    'donut': 'Bánh donut',
    'cake': 'Bánh ngọt',
    'chair': 'Ghế',
    'couch': 'Ghế sofa',
    'potted plant': 'Cây cảnh',
    'bed': 'Giường',
    'dining table': 'Bàn ăn',
    'toilet': 'Nhà vệ sinh',
    'tv': 'Tivi',
    'laptop': 'Máy tính xách tay',
    'mouse': 'Chuột máy tính',
    'remote': 'Điều khiển từ xa',
    'keyboard': 'Bàn phím',
    'cell phone': 'Điện thoại di động',
    'microwave': 'Lò vi sóng',
    'oven': 'Lò nướng',
    'toaster': 'Máy nướng bánh mì',
    'sink': 'Bồn rửa',
    'refrigerator': 'Tủ lạnh',
    'book': 'Sách',
    'clock': 'Đồng hồ',
    'vase': 'Bình hoa',
    'scissors': 'Kéo',
    'teddy bear': 'Gấu bông',
    'hair drier': 'Máy sấy tóc',
    'toothbrush': 'Bàn chải đánh răng',
}
def gen_frames():
    """Hàm streaming video (giữ nguyên)"""
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model(frame, stream=True, conf=0.4)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request):
    """Streaming video feed (giữ nguyên)"""
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def home(request):
    """Trang chủ (giữ nguyên)"""
    return render(request, "detect.html")


# ============ API MỚI CHO C# ============

@csrf_exempt
@require_http_methods(["POST"])
def detect_api(request):
    """
    API nhận ảnh từ C# MAUI, phân tích bằng YOLO và trả về kết quả JSON
    """
    try:
        # Kiểm tra có file ảnh không
        if 'image' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No image file provided'
            }, status=400)

        # Đọc file ảnh
        image_file = request.FILES['image']
        image_bytes = image_file.read()

        # Convert sang numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JsonResponse({
                'success': False,
                'error': 'Invalid image format'
            }, status=400)

        # Nhận diện với YOLO
        results = model(img, conf=0.4)

        # Xử lý kết quả
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Lấy tọa độ bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]

                # Crop ảnh vùng detected
                cropped = img[int(y1):int(y2), int(x1):int(x2)]

                # Encode thành base64
                _, buffer = cv2.imencode('.jpg', cropped)
                thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')

                # Lấy từ vựng tiếng Việt
                vietnamese_word = VOCABULARY_MAP.get(label, label)

                detection = {
                    'label': label,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # [x, y, width, height]
                    'vocabulary': vietnamese_word,
                    'thumbnail': f'data:image/jpeg;base64,{thumbnail_base64}'
                }
                detections.append(detection)

        return JsonResponse({
            'success': True,
            'detections': detections,
            'total': len(detections)
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'ok',
        'model': 'yolov8s',
        'classes': len(model.names)
    })