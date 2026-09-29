"""
Smart Sort - PC Prototype (Autonomous / Hands-Free Cloud Client)

State machine per frame for autonomous checking:
    EMPTY     -> box is clear (this is the re-armed, ready state)
    SETTLING  -> something entered the box; waiting for N consecutive
                 still frames before it's confident it's not just a hand
                 passing through
    COOLDOWN  -> just fired a request; briefly ignores everything so it
                 doesn't double-scan the same item, then re-arms once the
                 box goes back to EMPTY (or the cooldown timer expires)

Press 'd' while running to toggle a debug overlay showing the live
presence/motion numbers -- use it to tune the constants below for your
camera and lighting.
"""

import cv2
import numpy as np
import requests
import time

API_URL = "http://127.0.0.1:8000/classify"
REQUEST_TIMEOUT_SECONDS = 5

PRESENCE_THRESHOLD = 35            # per-pixel intensity diff vs baseline to count as "changed"
PRESENCE_NOISE_KERNEL = 5          # morphological opening kernel -- erases isolated noise pixels
                                   # before they get counted, so sensor grain doesn't add up

PRESENCE_ENTER_FRACTION = 0.15     # fraction of ROI that must change to count as "item just entered"
PRESENCE_EXIT_FRACTION = 0.06      # once present, must drop below THIS (lower) fraction to count as
                                   # "removed" -- this gap (hysteresis) stops flickering right at the edge

STILLNESS_THRESHOLD = 8            # frame-to-frame mean diff below this = "not moving"
STILLNESS_FRAMES_REQUIRED = 10     # consecutive still frames before auto-triggering (~0.3-0.5s typical)
COOLDOWN_SECONDS = 2.0             # minimum time after a capture before it can fire again

STATE_EMPTY = "EMPTY"
STATE_SETTLING = "SETTLING"
STATE_COOLDOWN = "COOLDOWN"

_noise_kernel = np.ones((PRESENCE_NOISE_KERNEL, PRESENCE_NOISE_KERNEL), np.uint8)


def to_gray_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (9, 9), 0)


def presence_fraction_from(gray, baseline):
    """
    Difference vs. the startup baseline, with a morphological open pass to
    strip out isolated single-pixel noise (sensor grain, JPEG artifacts)
    before it gets counted -- that noise is the usual cause of "captures
    every little bit" false triggers.
    """
    diff = cv2.absdiff(gray, baseline)
    mask = (diff > PRESENCE_THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _noise_kernel)
    return float(np.count_nonzero(mask)) / mask.size


def send_for_classification(roi):
    roi_resized = cv2.resize(roi, (224, 224))
    success, encoded_image = cv2.imencode('.jpg', roi_resized)
    if not success:
        print("Failed to encode image, skipping this capture.")
        return

    try:
        start_t = time.time()
        response = requests.post(
            API_URL,
            files={"file": ("item.jpg", encoded_image.tobytes(), "image/jpeg")},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()
        round_trip = time.time() - start_t

        print(f"\n--- Classification Result ---")
        print(f"Detected Material: {result['material'].upper()} ({result['confidence']*100:.1f}%)")
        print(f"Iowa City Routing: {result['classification'].upper()}")
        print(f"Round-trip latency: {round_trip:.3f}s")
        if round_trip > 5:
            print("WARNING: exceeded the 5-second C1.1 latency requirement.")

    except requests.exceptions.Timeout:
        print(f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


cap = cv2.VideoCapture(0)
print("Smart Sort (Autonomous Mode) Active.")
print("Hold an item in the green box -- it scans automatically once it's still.")
print("Press 'r' to re-capture the empty-box baseline.")
print("Press 'd' to toggle the tuning overlay. Press 'q' to quit.")

baseline = None
prev_gray = None
still_count = 0
state = STATE_EMPTY
cooldown_until = 0.0
show_debug = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    box_size = int(min(h, w) * 0.6)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]
    gray = to_gray_blur(roi)

    if baseline is None:
        baseline = gray.copy()
        prev_gray = gray.copy()

    presence_fraction = presence_fraction_from(gray, baseline)

    # Hysteresis: which threshold applies depends on whether we currently
    # think something is there. Prevents flicker right at the boundary.
    if state == STATE_EMPTY:
        item_present = presence_fraction > PRESENCE_ENTER_FRACTION
    else:
        item_present = presence_fraction > PRESENCE_EXIT_FRACTION

    motion_score = float(np.mean(cv2.absdiff(gray, prev_gray)))
    is_still = motion_score < STILLNESS_THRESHOLD

    now = time.time()

    # --- resolve state transitions ---
    if state == STATE_COOLDOWN and now >= cooldown_until:
        state = STATE_SETTLING if item_present else STATE_EMPTY
        still_count = 0

    if state == STATE_EMPTY and item_present:
        state = STATE_SETTLING
        still_count = 0

    if state == STATE_SETTLING:
        if not item_present:
            state = STATE_EMPTY
            still_count = 0
        elif is_still:
            still_count += 1
        else:
            still_count = 0

    # --- render + trigger for the (possibly just-updated) current state ---
    display_frame = frame.copy()

    if state == STATE_EMPTY:
        box_color, status_text = (0, 255, 0), "Ready"

    elif state == STATE_SETTLING:
        box_color = (0, 255, 255)
        status_text = f"Hold still... ({still_count}/{STILLNESS_FRAMES_REQUIRED})"
        if still_count >= STILLNESS_FRAMES_REQUIRED:
            send_for_classification(roi)
            state = STATE_COOLDOWN
            cooldown_until = now + COOLDOWN_SECONDS
            box_color, status_text = (0, 165, 255), "Cooldown..."

    else:  # STATE_COOLDOWN
        box_color, status_text = (0, 165, 255), "Cooldown..."

    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(display_frame, status_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    if show_debug:
        debug_lines = [
            f"presence_fraction: {presence_fraction:.3f}  (enter>{PRESENCE_ENTER_FRACTION} exit<{PRESENCE_EXIT_FRACTION})",
            f"motion_score:      {motion_score:.1f}  (still<{STILLNESS_THRESHOLD})",
            f"state: {state}",
        ]
        for i, line in enumerate(debug_lines):
            cv2.putText(display_frame, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imshow('Smart Sort - Autonomous Prototype', display_frame)

    prev_gray = gray

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        baseline = gray.copy()
        print("Baseline re-captured.")
    elif key == ord('d'):
        show_debug = not show_debug
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()