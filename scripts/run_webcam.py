import time
import cv2
from ultralytics import YOLO

def main():
    models = {
        "YOLOv8n (CNN Baseline)": "yolov8n_voc.onnx",
        "RT-DETR (Vision Transformer)": "rtdetr_voc.onnx"
    }

    current_key = "YOLOv8n (CNN Baseline)"
    print(f"[INFO] Initializing with {current_key}...")
    active_model = YOLO(models[current_key], task="detect")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not found. Please verify webcam access.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()

    print("\n" + "=" * 55)
    print(" LIVE STREAM CONTROLS:")
    print("   Press 'm' -> Switch between CNN and Vision Transformer")
    print("   Press 'q' -> Exit")
    print("=" * 55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_inf = time.time()
        results = active_model.predict(source=frame, imgsz=416, conf=0.35, verbose=False)
        inf_latency = (time.time() - start_inf) * 1000

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        annotated_frame = results[0].plot()

        hud_text_1 = f"Model: {current_key}"
        hud_text_2 = f"FPS: {fps:.1f} | Latency: {inf_latency:.1f}ms | Res: 416x416"

        cv2.putText(annotated_frame, hud_text_1, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, hud_text_2, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Lightweight Edge Detection Benchmark", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            if current_key == "YOLOv8n (CNN Baseline)":
                current_key = "RT-DETR (Vision Transformer)"
            else:
                current_key = "YOLOv8n (CNN Baseline)"
            print(f"[SWITCH] Switching model to: {current_key}...")
            active_model = YOLO(models[current_key], task="detect")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()