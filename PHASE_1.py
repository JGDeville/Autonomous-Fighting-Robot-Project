"""
robot_vision_phase1.py
----------------------
Phase 1: Autonomous State Machine + Performance Metrics.

State machine behaviour:
  SEARCHING  - one or both markers not visible → STOP
  ALIGNING   - steering error > ALIGN_THRESHOLD → hard LEFT or RIGHT spin
  ATTACKING  - aligned, distance > RAM_THRESHOLD → FORWARD
  RAMMING    - distance <= RAM_THRESHOLD → FORWARD (contact)

Detection:
  Uses the same tuned DetectorParameters and 0.5x downscaled detection
  as Phase 2 and Phase 3 so detection quality is not a confound when
  comparing phases. Only the control algorithm differs between scripts.

Metrics logged per run:
  - Time to align            (s)   time until |steering_error| < threshold
  - Time to contact          (s)   time until RAM distance reached
  - Heading error at contact (°)   steering error at moment of contact
  - Mean absolute steer error(°)   average misalignment across whole run
  - Max absolute steer error (°)   worst misalignment during run
  - Path efficiency          (%)   straight-line displacement / actual path * 100
  - Detection reliability    (%)   frames with both markers / total frames

CSV files saved to ./logs/ next to this script.

Usage:
    python robot_vision_phase1.py
    Q = quit | S = stop | T = test | A = autonomous | R = record
"""

import numpy as np
import cv2
import socket
import time
import threading
import queue
import csv
import os
from datetime import datetime

# ── Log directory — always next to this script ────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# ── ESP32 ─────────────────────────────────────────────────────────────────────
ESP32_IP   = "192.168.4.1"
ESP32_PORT = 8888
TIMEOUT    = 5

# ── ArUco ─────────────────────────────────────────────────────────────────────
ARUCO_TYPE = "DICT_4X4_1000"
ARUCO_DICT_MAP = {
    "DICT_4X4_50":         cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":        cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250":        cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000":       cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50":         cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100":        cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250":        cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000":       cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50":         cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100":        cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250":        cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000":       cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50":         cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100":        cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250":        cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000":       cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

INTRINSIC_CAMERA = np.array(
    ((1026.40507, 0, 664.89667),
     (0, 1026.24251, 391.919086),
     (0, 0, 1)),
    dtype=np.float64
)
DISTORTION    = np.array((0.079, -0.27, 0, 0, 0), dtype=np.float64)
MARKER_LENGTH = 0.055

ROBOT_ID    = 0
OPPONENT_ID = 1

# ── Thresholds ────────────────────────────────────────────────────────────────
ALIGN_THRESHOLD_DEG = 20.0
ATTACK_THRESHOLD_PX = 300   # used for HUD ring only in Phase 1
RAM_THRESHOLD_PX    = 180
CMD_INTERVAL        = 0.15

# ── Detection scaling ─────────────────────────────────────────────────────────
# Detector runs on a half-resolution copy of the frame for speed and to reduce
# motion blur. Corners are scaled back up to full resolution before pose
# estimation so accuracy is unaffected. Identical to Phase 2 and Phase 3.
DETECT_SCALE = 0.5


# =============================================================================
#  States
# =============================================================================

class State:
    SEARCHING = "SEARCHING"
    ALIGNING  = "ALIGNING"
    ATTACKING = "ATTACKING"
    RAMMING   = "RAMMING"

STATE_COLOURS = {
    State.SEARCHING: (100, 100, 100),   # grey
    State.ALIGNING:  (0, 165, 255),     # orange
    State.ATTACKING: (0, 255, 0),       # green
    State.RAMMING:   (0, 0, 255),       # red
}


# =============================================================================
#  Performance Logger
# =============================================================================

class PerformanceLogger:
    """
    Records per-frame data and computes metrics at end of run.

    Metrics (command oscillations intentionally excluded for Phase 1):
      - time_to_align_s             first frame where |error| < ALIGN_THRESHOLD
      - time_to_contact_s           first frame where distance <= RAM_THRESHOLD
      - heading_at_contact_deg      steering error at moment of contact
      - mean_abs_steering_error_deg average |error| across all frames
      - max_abs_steering_error_deg  worst |error| seen during run
      - path_efficiency_pct         straight-line displacement / actual path * 100
      - detection_reliability_pct   % frames where both markers were visible
    """

    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[LOG] Log directory: {os.path.abspath(self.log_dir)}")
        self._reset()

    def _reset(self):
        self.running            = False
        self.start_time         = None
        self.frames             = []
        self.time_to_align      = None
        self.time_to_contact    = None
        self.heading_at_contact = None

    def start_run(self):
        self._reset()
        self.running    = True
        self.start_time = time.time()
        print(f"\n[LOG] ── Run started ───────────────────────────────")
        print(f"[LOG] Saving to: {os.path.abspath(self.log_dir)}")

    def record_frame(self, steering_error, distance, command,
                     both_detected, robot_center=None):
        """
        Call every camera frame while running is True.

        Parameters
        ----------
        steering_error : float or None   signed degrees
        distance       : float or None   pixels to opponent
        command        : str             command sent this frame
        both_detected  : bool            were both markers visible?
        robot_center   : (int,int) or None  pixel centre of robot marker
                         — used for true 2D path efficiency calculation
        """
        if not self.running:
            return

        elapsed = time.time() - self.start_time

        # Store robot pixel position for true path efficiency
        rx = robot_center[0] if robot_center is not None else None
        ry = robot_center[1] if robot_center is not None else None

        self.frames.append({
            'time':           elapsed,
            'steering_error': steering_error,
            'distance':       distance,
            'command':        command,
            'both_detected':  both_detected,
            'robot_x':        rx,
            'robot_y':        ry,
        })

        # ── Milestone: first alignment ────────────────────────────────────
        if (self.time_to_align is None
                and steering_error is not None
                and abs(steering_error) <= ALIGN_THRESHOLD_DEG):
            self.time_to_align = elapsed
            print(f"[LOG] Aligned   t={elapsed:.2f}s  error={steering_error:+.1f}°")

        # ── Milestone: contact ────────────────────────────────────────────
        if (self.time_to_contact is None
                and distance is not None
                and distance <= RAM_THRESHOLD_PX):
            self.time_to_contact    = elapsed
            self.heading_at_contact = steering_error
            print(f"[LOG] Contact   t={elapsed:.2f}s  "
                  f"dist={distance:.0f}px  error={steering_error:+.1f}°")

    def end_run(self, label="run"):
        """Finalise metrics, print summary, save CSVs. Returns metrics dict."""
        if not self.running or not self.frames:
            print("[LOG] No data to save.")
            return {}

        self.running = False
        total_time   = self.frames[-1]['time']
        n_frames     = len(self.frames)

        # ── Detection reliability ─────────────────────────────────────────
        detected      = sum(1 for f in self.frames if f['both_detected'])
        detection_pct = 100.0 * detected / n_frames

        # ── Steering error stats ──────────────────────────────────────────
        errors   = [abs(f['steering_error']) for f in self.frames
                    if f['steering_error'] is not None]
        mean_err = round(float(np.mean(errors)), 1) if errors else None
        max_err  = round(float(np.max(errors)),  1) if errors else None

        # ── Path efficiency (true 2D) ─────────────────────────────────────
        # Uses actual robot pixel positions logged each frame.
        # Efficiency = straight-line distance (start→end) / sum of
        # frame-to-frame step distances * 100
        # A perfectly straight path scores 100%. Wandering scores lower.
        coords = [(f['robot_x'], f['robot_y']) for f in self.frames
                  if f['robot_x'] is not None and f['robot_y'] is not None]

        path_efficiency_pct = None
        if len(coords) >= 2:
            # Straight-line distance from first to last position
            straight_line = np.sqrt(
                (coords[-1][0] - coords[0][0]) ** 2 +
                (coords[-1][1] - coords[0][1]) ** 2
            )
            # Actual path: sum of all frame-to-frame steps
            actual_path = sum(
                np.sqrt((coords[i][0] - coords[i-1][0]) ** 2 +
                        (coords[i][1] - coords[i-1][1]) ** 2)
                for i in range(1, len(coords))
            )
            if actual_path > 0:
                path_efficiency_pct = round(100.0 * straight_line / actual_path, 1)

        # ── Build metrics dict ────────────────────────────────────────────
        metrics = {
            'label':                       label,
            'total_time_s':                round(total_time, 3),
            'time_to_align_s':             round(self.time_to_align, 3)
                                           if self.time_to_align      is not None else None,
            'time_to_contact_s':           round(self.time_to_contact, 3)
                                           if self.time_to_contact    is not None else None,
            'heading_at_contact_deg':      round(self.heading_at_contact, 1)
                                           if self.heading_at_contact is not None else None,
            'mean_abs_steering_error_deg': mean_err,
            'max_abs_steering_error_deg':  max_err,
            'path_efficiency_pct':         path_efficiency_pct,
            'detection_reliability_pct':   round(detection_pct, 1),
            'total_frames':                n_frames,
        }

        # ── Print summary ─────────────────────────────────────────────────
        print("\n[LOG] ── Run Summary ────────────────────────────────")
        print(f"       Label                   : {metrics['label']}")
        print(f"       Total time              : {metrics['total_time_s']} s")
        print(f"       Time to align           : {metrics['time_to_align_s']} s")
        print(f"       Time to contact         : {metrics['time_to_contact_s']} s")
        print(f"       Heading error @ contact : {metrics['heading_at_contact_deg']} °")
        print(f"       Mean |steering error|   : {metrics['mean_abs_steering_error_deg']} °")
        print(f"       Max  |steering error|   : {metrics['max_abs_steering_error_deg']} °")
        print(f"       Path efficiency         : {metrics['path_efficiency_pct']} %")
        print(f"       Detection reliability   : {metrics['detection_reliability_pct']} %")
        print(f"       Frames recorded         : {metrics['total_frames']}")
        print("[LOG] ───────────────────────────────────────────────")

        # ── Save CSVs ─────────────────────────────────────────────────────
        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.log_dir, f"summary_{ts}_{label}.csv")
        frames_path  = os.path.join(self.log_dir, f"frames_{ts}_{label}.csv")

        try:
            with open(summary_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=metrics.keys())
                w.writeheader()
                w.writerow(metrics)
            print(f"\n[LOG] Summary saved  → {os.path.abspath(summary_path)}")
        except Exception as e:
            print(f"[LOG] ERROR saving summary: {e}")

        try:
            with open(frames_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.frames[0].keys())
                w.writeheader()
                w.writerows(self.frames)
            print(f"[LOG] Frame log saved → {os.path.abspath(frames_path)}\n")
        except Exception as e:
            print(f"[LOG] ERROR saving frame log: {e}")

        return metrics


# =============================================================================
#  Command Worker
# =============================================================================

class CommandWorker(threading.Thread):

    def __init__(self):
        super().__init__(daemon=True)
        self.queue      = queue.Queue()
        self.sock       = None
        self.connected  = False
        self.last_cmd   = None
        self._stop_flag = threading.Event()

    def connect(self):
        print(f"[NET] Connecting to ESP32 at {ESP32_IP}:{ESP32_PORT}...")
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(TIMEOUT)
            self.sock.connect((ESP32_IP, ESP32_PORT))
            self.connected = True
            print("[NET] Connected.")
        except Exception as e:
            print(f"[NET] Connection failed: {e}")
            self.connected = False

    def send(self, command):
        if not self.sock:
            return False
        try:
            self.sock.sendall((command.strip().upper() + "\n").encode())
            response = self.sock.recv(64).decode().strip()
            print(f"[NET] → {command.upper():<10}  ← {response}")
            return True
        except Exception as e:
            print(f"[NET] Send failed ({command}): {e}")
            self.connected = False
            return False

    def put(self, command, hold_seconds=None):
        self.queue.put((command, hold_seconds))

    def put_sequence(self, sequence):
        for cmd, duration in sequence:
            self.queue.put((cmd, duration))

    def stop(self):
        self._stop_flag.set()

    def run(self):
        self.connect()
        while not self._stop_flag.is_set():
            try:
                command, hold = self.queue.get(timeout=0.1)
                self.send(command)
                self.last_cmd = command
                if hold:
                    time.sleep(hold)
            except queue.Empty:
                continue
        if self.sock:
            self.send("STOP")
            self.sock.close()
            print("[NET] Socket closed.")


# =============================================================================
#  ArUco — tuned detector + downscaled detection (identical to Phase 2 & 3)
# =============================================================================

def make_detector():
    """
    Build an ArucoDetector with parameters tuned for reliable detection.
    Used by all three phases so detection quality is not a confound.

    Key parameter choices:
      adaptiveThreshWinSizeMin/Max  — wider range catches markers under
                                      uneven or changing lighting
      minMarkerPerimeterRate        — accepts smaller/more distant markers
      polygonalApproxAccuracyRate   — tolerates perspective warp from
                                      fast movement
      CORNER_REFINE_SUBPIX          — sub-pixel corner refinement reduces
                                      pose jitter on detected markers
    """
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[ARUCO_TYPE])
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin    = 3
    p.adaptiveThreshWinSizeMax    = 23
    p.adaptiveThreshWinSizeStep   = 4
    p.adaptiveThreshConstant      = 7
    p.minMarkerPerimeterRate      = 0.03
    p.maxMarkerPerimeterRate      = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.cornerRefinementMethod      = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


def pose_estimation(frame, detector):
    """
    Detect markers on a downscaled copy of the frame for speed, then scale
    corners back up to full resolution for drawing and pose estimation.

    Downscaling to DETECT_SCALE (0.5) means the detector processes a
    640x360 image rather than 1280x720 — roughly 4x faster, and motion
    blur has less impact on a smaller image. Pose accuracy is unaffected
    because corners are divided by DETECT_SCALE before being passed to
    estimatePoseSingleMarkers.
    """
    h, w = frame.shape[:2]

    # Detect on small frame
    small      = cv2.resize(frame, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    corners_small, ids, _ = detector.detectMarkers(gray_small)

    detections = {}

    if ids is None or len(corners_small) == 0:
        return frame, detections

    # Scale corners back to full resolution
    corners = [c / DETECT_SCALE for c in corners_small]

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    for i in range(len(ids)):
        marker_id = int(ids[i][0])
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[i], MARKER_LENGTH, INTRINSIC_CAMERA, DISTORTION
        )
        cv2.drawFrameAxes(frame, INTRINSIC_CAMERA, DISTORTION, rvec, tvec, 0.01)
        c  = corners[i].reshape((4, 2))
        cX = int((c[0][0] + c[2][0]) / 2.0)
        cY = int((c[0][1] + c[2][1]) / 2.0)
        detections[marker_id] = {
            'center': (cX, cY),
            'tvec':   tvec[0][0].tolist(),
            'rvec':   rvec[0][0].tolist(),
        }

    return frame, detections


# =============================================================================
#  Geometry
# =============================================================================

def get_robot_heading(rvec):
    R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
    return np.arctan2(R[1][0], R[0][0])

def get_steering_error(robot_center, opponent_center, robot_heading_rad):
    dx = opponent_center[0] - robot_center[0]
    dy = opponent_center[1] - robot_center[1]
    angle_to_opponent = np.arctan2(dy, dx)
    error = angle_to_opponent - robot_heading_rad
    error = (error + np.pi) % (2 * np.pi) - np.pi
    return np.degrees(error)

def get_pixel_distance(robot_center, opponent_center):
    dx = opponent_center[0] - robot_center[0]
    dy = opponent_center[1] - robot_center[1]
    return np.sqrt(dx**2 + dy**2)


# =============================================================================
#  State Machine — Phase 1: hard LEFT/RIGHT spin for alignment
# =============================================================================

def decide_state_and_command(detections):
    """
    Phase 1 state machine.

    Priority order:
      1. Can't see both markers         → SEARCHING, STOP
      2. Steering error > threshold     → ALIGNING,  LEFT or RIGHT (full spin)
      3. Distance <= RAM threshold      → RAMMING,   FORWARD
      4. Aligned and outside RAM zone   → ATTACKING, FORWARD

    Note: there is no proportional steering here. The robot stops
    all forward movement and spins in place until aligned, then drives
    straight forward. This is the baseline Phase 1 behaviour.
    """
    info = {'steering_error': None, 'distance': None}

    if ROBOT_ID not in detections or OPPONENT_ID not in detections:
        return State.SEARCHING, "STOP", info

    robot_center    = detections[ROBOT_ID]['center']
    opponent_center = detections[OPPONENT_ID]['center']
    robot_rvec      = detections[ROBOT_ID]['rvec']

    robot_heading  = get_robot_heading(robot_rvec)
    steering_error = get_steering_error(robot_center, opponent_center, robot_heading)
    distance       = get_pixel_distance(robot_center, opponent_center)

    info['steering_error'] = steering_error
    info['distance']       = distance

    # Priority 1: align before doing anything else
    if abs(steering_error) > ALIGN_THRESHOLD_DEG:
        command = "RIGHT" if steering_error > 0 else "LEFT"
        return State.ALIGNING, command, info

    # Priority 2: RAM if very close
    if distance <= RAM_THRESHOLD_PX:
        return State.RAMMING, "FORWARD", info

    # Priority 3: aligned and far enough away — drive forward
    return State.ATTACKING, "FORWARD", info


# =============================================================================
#  HUD
# =============================================================================

def draw_hud(frame, detections, state, command, info, worker, autonomous, logger):
    h, w = frame.shape[:2]

    # Connection status
    conn_col = (0, 255, 0) if worker.connected else (0, 0, 255)
    cv2.putText(frame, "ESP32: OK" if worker.connected else "ESP32: DISCONNECTED",
                (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, conn_col, 2)

    # Autonomous mode
    auto_col = (0, 255, 0) if autonomous else (0, 165, 255)
    cv2.putText(frame, "AUTO: ON  [A]" if autonomous else "AUTO: OFF [A]",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, auto_col, 2)

    # Phase label — always visible so recordings are unambiguous
    cv2.putText(frame, "PHASE 1 — State Machine",
                (w // 2 - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # State + command
    state_col = STATE_COLOURS.get(state, (255, 255, 255))
    cv2.putText(frame, f"STATE:   {state}",   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_col, 2)
    cv2.putText(frame, f"CMD:     {command}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Steering error
    if info['steering_error'] is not None:
        err     = info['steering_error']
        err_col = (0, 255, 0) if abs(err) <= ALIGN_THRESHOLD_DEG else (0, 165, 255)
        cv2.putText(frame, f"STEER:   {err:+.1f} deg", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, err_col, 2)

    # Distance
    if info['distance'] is not None:
        dist     = info['distance']
        dist_col = (0, 0, 255) if dist <= RAM_THRESHOLD_PX else (0, 255, 0)
        cv2.putText(frame, f"DIST:    {dist:.0f} px", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, dist_col, 2)

    # Recording indicator
    if logger.running:
        elapsed = time.time() - logger.start_time
        cv2.putText(frame, f"REC  {elapsed:.1f}s", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        if int(elapsed * 2) % 2 == 0:
            cv2.circle(frame, (170, 148), 8, (0, 0, 255), -1)
    else:
        cv2.putText(frame, "R = start recording", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

    # Robot marker + heading arrow + threshold rings
    if ROBOT_ID in detections:
        rc = detections[ROBOT_ID]['center']
        cv2.circle(frame, rc, 10, (0, 255, 0), -1)
        cv2.putText(frame, "ROBOT", (rc[0] + 12, rc[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if info['steering_error'] is not None:
            heading = get_robot_heading(detections[ROBOT_ID]['rvec'])
            ax = int(rc[0] + 60 * np.cos(heading))
            ay = int(rc[1] + 60 * np.sin(heading))
            cv2.arrowedLine(frame, rc, (ax, ay), (255, 255, 0), 2, tipLength=0.2)
        cv2.circle(frame, rc, RAM_THRESHOLD_PX,    (0, 0, 255),   1)  # red  = RAM zone
        cv2.circle(frame, rc, ATTACK_THRESHOLD_PX, (0, 255, 255), 1)  # cyan = attack zone

    # Opponent marker
    if OPPONENT_ID in detections:
        oc = detections[OPPONENT_ID]['center']
        cv2.circle(frame, oc, 10, (0, 0, 255), -1)
        cv2.putText(frame, "OPPT", (oc[0] + 12, oc[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Vector line robot → opponent
    if ROBOT_ID in detections and OPPONENT_ID in detections:
        rc       = detections[ROBOT_ID]['center']
        oc       = detections[OPPONENT_ID]['center']
        line_col = (0, 255, 0) if state in (State.ATTACKING, State.RAMMING) else (0, 165, 255)
        cv2.arrowedLine(frame, rc, oc, line_col, 2, tipLength=0.05)

    cv2.putText(frame, f"Queue:   {worker.queue.qsize()}", (10, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, "Q=quit  S=stop  T=test  A=auto  R=record",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


# =============================================================================
#  Main
# =============================================================================

def main():
    detector = make_detector()   # tuned parameters, identical to Phase 2 & 3

    worker = CommandWorker()
    worker.start()

    logger = PerformanceLogger()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("[CAM] Cannot open webcam.")
        worker.stop()
        return

    print("[CAM] Camera opened.")
    print("PHASE 1 — Autonomous State Machine (hard LEFT/RIGHT spin)")
    print("A = autonomous  |  R = record  |  S = stop  |  T = test  |  Q = quit")
    print("=" * 60)

    autonomous     = False
    current_state  = State.SEARCHING
    current_cmd    = "STOP"
    last_sent_cmd  = None
    last_send_time = 0.0
    run_counter    = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[CAM] Failed to grab frame.")
                break

            frame, detections = pose_estimation(frame, detector)

            if autonomous:
                current_state, current_cmd, info = decide_state_and_command(detections)
            else:
                current_state = State.SEARCHING
                current_cmd   = "STOP"
                info          = {'steering_error': None, 'distance': None}

            # Send command (rate limited)
            now = time.time()
            if current_cmd != last_sent_cmd or (now - last_send_time) > CMD_INTERVAL:
                worker.put(current_cmd)
                last_sent_cmd  = current_cmd
                last_send_time = now

            # Pass robot centre to logger for true path efficiency
            robot_center = detections[ROBOT_ID]['center'] if ROBOT_ID in detections else None
            both_detected = (ROBOT_ID in detections and OPPONENT_ID in detections)

            logger.record_frame(
                steering_error = info['steering_error'],
                distance       = info['distance'],
                command        = current_cmd,
                both_detected  = both_detected,
                robot_center   = robot_center,
            )

            # Auto-end run on contact
            if (logger.running
                    and logger.time_to_contact is not None
                    and current_state == State.RAMMING):
                logger.end_run(label=f"phase1_run{run_counter:02d}")

            frame = draw_hud(frame, detections, current_state,
                             current_cmd, info, worker, autonomous, logger)
            cv2.imshow("Robot Vision — Phase 1", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                if logger.running:
                    logger.end_run(label=f"phase1_run{run_counter:02d}_aborted")
                print("\n[SYS] Quitting.")
                break

            elif key == ord('a'):
                autonomous = not autonomous
                print(f"\n[SYS] Autonomous: {'ON' if autonomous else 'OFF'}")
                if not autonomous:
                    with worker.queue.mutex:
                        worker.queue.queue.clear()
                    worker.put("STOP")

            elif key == ord('s'):
                autonomous = False
                if logger.running:
                    logger.end_run(label=f"phase1_run{run_counter:02d}_stopped")
                with worker.queue.mutex:
                    worker.queue.queue.clear()
                worker.put("STOP")
                last_sent_cmd = "STOP"
                print("\n[SYS] Emergency STOP.")

            elif key == ord('r'):
                if not logger.running:
                    run_counter += 1
                    logger.start_run()
                    print(f"[LOG] Recording run {run_counter}. "
                          "Contact auto-stops the run, or press R again to stop early.")
                else:
                    logger.end_run(label=f"phase1_run{run_counter:02d}")

            elif key == ord('t'):
                if not autonomous:
                    print("\n[SYS] Running test sequence.")
                    worker.put_sequence([
                        ("FORWARD", 1.0),
                        ("STOP",    0.5),
                        ("BACK",    1.0),
                        ("STOP",    0.5),
                        ("LEFT",    0.8),
                        ("STOP",    0.5),
                        ("RIGHT",   0.8),
                        ("STOP",    0.5),
                    ])
                else:
                    print("\n[SYS] Turn off autonomous mode before running test.")

    except KeyboardInterrupt:
        print("\n[SYS] Interrupted.")
        if logger.running:
            logger.end_run(label=f"phase1_run{run_counter:02d}_interrupted")

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[SYS] Done.")


if __name__ == "__main__":
    main()