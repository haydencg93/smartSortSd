"""Smart Sort host-only prototype using the EXACT YOLO11n family, not ESP-DL inference.

Requirements: pip install ultralytics opencv-python requests
Run dry: python yolo11n_pc_test.py --source 0
Run with API: python yolo11n_pc_test.py --source 0 --send
Run on video: python yolo11n_pc_test.py --source path/to/video.mp4

YOLO11n only detects 80 COCO classes and does not have a dedicated hand category.
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Desktop Smart Sort YOLO11n trigger test")
    parser.add_argument("--source", default="0", help="0 for webcam, or video file")
    parser.add_argument("--model", default="yolo11n.pt", help="Desktop Ultralytics .pt model")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/classify")
    parser.add_argument("--send", action="store_true", help="Actually POST JPEG to your ML API")
    parser.add_argument("--threshold", type=float, default=.55)
    parser.add_argument("--hold-ms", type=int, default=500)
    parser.add_argument("--rearm-ms", type=int, default=1000)
    parser.add_argument("--min-area", type=float, default=.02)
    parser.add_argument("--zone-width", type=float, default=.60,
                        help="Centered fraction of image used for the scan zone")
    args = parser.parse_args()

    if not (0 < args.zone_width <= 1):
        parser.error("--zone-width must be between 0 and 1")
    if args.hold_ms < 0 or args.rearm_ms < 0:
        parser.error("confirmation/rearm delays cannot be negative")
    try:
        import cv2
        import requests
        from ultralytics import YOLO
    except ImportError as exc:
        parser.error(f"Missing dependency: {exc}. Install: pip install ultralytics opencv-python requests")

    from auto_scan import AutoScan, Detection

    half = args.zone_width / 2
    zone = (.5 - half, .5 - half, .5 + half, .5 + half)
    gate = AutoScan(confirm_ms=args.hold_ms, rearm_ms=args.rearm_ms,
                    threshold=args.threshold, min_area=args.min_area, zone=zone)
    net = YOLO(args.model)  # If absent, ultralytics downloads its pretrained weights.
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        parser.error("Could not open camera/video source: " + args.source)

    print("Running YOLO11n on the PC. NOT a timing simulation of ESP32-S3.")
    print("Center-zone filter enabled; 'person' is excluded, but hands cannot be reliably excluded.")
    print("API uploads:", "ENABLED" if args.send else "DISABLED (dry run)")
    print("Press q to quit, r to manually reset state. To rearm, remove the eligible object.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No more frames / camera disconnected.")
                break
            h, w = frame.shape[:2]
            started = time.monotonic()
            output = net.predict(frame, imgsz=320, conf=args.threshold, verbose=False)[0]
            infer_ms = (time.monotonic() - started) * 1000
            seen = []
            for b in output.boxes:
                label = int(b.cls.item())
                score = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                seen.append(Detection(label, score, x1/w, y1/h, x2/w, y2/h))
                color = (0, 0, 255) if label == 0 else (0, 200, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                name = output.names.get(label, str(label))
                cv2.putText(frame, f"{name} {score:.2f}", (int(x1), max(12, int(y1)-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)

            triggered = gate.update(seen, int(time.monotonic() * 1000))
            x_left, y_top, x_right, y_bottom = zone
            cv2.rectangle(frame, (int(w*x_left), int(h*y_top)),
                          (int(w*x_right), int(h*y_bottom)), (0, 255, 255), 2)
            cv2.putText(frame, f"{gate.state.value} | PC inference: {infer_ms:.0f} ms",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            cv2.imshow("Smart Sort YOLO11n (PC simulator)", frame)

            if triggered:
                selected = gate.target
                print(f"[TRIGGER] class={output.names.get(selected.class_id, selected.class_id)}, confidence={selected.confidence:.2f}")
                # Acquire a fresh unannotated frame; never send the boxes or overlay to the API.
                ok, fresh = cap.read()
                if not ok:
                    fresh = None
                if fresh is not None:
                    # Send an object-centered crop; your API classifies material, not COCO object class.
                    pad = .08
                    fh, fw = fresh.shape[:2]
                    x1 = max(0, int((selected.x1-pad)*fw))
                    y1 = max(0, int((selected.y1-pad)*fh))
                    x2 = min(fw, int((selected.x2+pad)*fw))
                    y2 = min(fh, int((selected.y2+pad)*fh))
                    cropped = fresh[y1:y2, x1:x2]
                    if cropped.size:
                        success, jpg = cv2.imencode(".jpg", cropped)
                        if success and args.send:
                            try:
                                response = requests.post(
                                    args.api_url,
                                    files={"file": ("item.jpg", jpg.tobytes(), "image/jpeg")},
                                    timeout=7)
                                response.raise_for_status()
                                result = response.json()
                                print("[API] material:", result.get("material"),
                                      "disposal:", result.get("classification"),
                                      "confidence:", result.get("confidence"))
                            except requests.RequestException as exc:
                                print("[API ERROR]", exc)
                        elif success:
                            print("[DRY RUN] Would POST the cropped JPEG to", args.api_url)
                gate.capture_finished() # Failure must not cause upload storms.

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                gate.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
