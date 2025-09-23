import cv2
from django.http import StreamingHttpResponse
from django.shortcuts import render
from ultralytics import YOLO

# Load model OpenImages (đã train sẵn, 600+ classes)
model = YOLO("yolov8s.pt")

def gen_frames():
    cap = cv2.VideoCapture(0)  # webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Nhận diện với model OpenImages
            results = model(frame, stream=True, conf=0.4)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Encode JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')

def home(request):
    return render(request, "detect.html")
