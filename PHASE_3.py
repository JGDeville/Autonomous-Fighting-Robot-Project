"""
robot_vision_phase3.py
----------------------
Phase 3: PID Steering Controller + Performance Metrics.

State machine behaviour:
  SEARCHING  - one or both markers not visible, hold last cmd or STOP
  HOLDING    - detection dropped, holding last valid command for HOLD_DURATION
  ATTACKING  - PID-controlled MOTOR curve toward opponent
  RAMMING    - distance <= RAM_THRESHOLD -> FORWARD (contact)

Key difference from Phase 2:
  steer_to_motor_command() (proportional-only) replaced by PIDController.
  The integral term corrects persistent offset; the derivative term damps
  overshoot. The robot converges more smoothly on the opponent heading.

PID tuning procedure (do in order):
  Step 1 - Set KI=0, KD=0. Raise KP until robot aligns but oscillates,
           then back off until oscillation just stops. Start: KP=0.8
  Step 2 - Raise KD slowly until overshoot is eliminated. Start: KD=0.05
  Step 3 - Only add KI if robot consistently stops a few degrees short.
           Keep very small - too much causes windup. Start: KI=0.01

Detection:
  Uses the same tuned DetectorParameters and 0.5x downscaled detection
  as Phase 1 and Phase 2 so detection quality is not a confound.

Metrics logged per run:
  - Time to align            (s)   time until |steering_error| < threshold
  - Time to contact          (s)   time until RAM distance reached
  - Heading error at contact (deg) steering error at moment of contact
  - Mean absolute steer error(deg) average misalignment across whole run
  - Max absolute steer error (deg) worst misalignment during run
  - Path efficiency          (%)   straight-line displacement / actual path * 100
                                   (true 2D pixel coords - identical to Phase 1/2)
  - Detection reliability    (%)   frames with both markers / total frames

  PID gains (KP, KI, KD) are recorded in the CSV filename and as constant
  columns in the frame CSV for tuning analysis. They are NOT in the summary
  CSV so its schema matches Phase 1 and Phase 2 for direct comparison.

CSV files saved to ./logs/ next to this script.

Usage:
    python robot_vision_phase3.py
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

# -- Log directory: always next to this script, never relative to cwd ---------
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# -- ESP32 --------------------------------------------------------------------
ESP32_IP   = "192.168.4.1"
ESP32_PORT = 8888
TIMEOUT    = 5

# -- ArUco --------------------------------------------------------------------
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
MARKER_LENGTH = 0.05

# -- Marker IDs: must match Phase 1 and Phase 2 -------------------------------
ROBOT_ID    = 0
OPPONENT_ID = 1

# -- Thresholds ---------------------------------------------------------------
ALIGN_THRESHOLD_DEG = 20.0
ATTACK_THRESHOLD_PX = 300   # used for HUD ring only
RAM_THRESHOLD_PX    = 180
CMD_INTERVAL        = 0.15
HOLD_DURATION       = 0.3

# -- Motor PWM ----------------------------------------------------------------
PWM_STOP = 1500
PWM_FWD  = 1650
PWM_REV  = 1350
PWM_MIN  = 1100
PWM_MAX  = 1900

# -- PID Gains ----------------------------------------------------------------
# Tune these constants. See docstring for procedure.
KP = 0.66    # proportional gain  -- main steering strength
KI = 0.0   # integral gain      -- corrects persistent offset
KD = 0.0278   # derivative gain    -- dampens overshoot

# Maximum PID output (maps to full wheel differential at this error magnitude)
PID_MAX_OUTPUT = 90.0

# -- Detection scaling: identical to Phase 1 and Phase 2 ---------------------
DETECT_SCALE = 0.5


# =============================================================================
#  States
# =============================================================================

class State:
    SEARCHING = "SEARCHING"
    HOLDING   = "HOLDING"
    ATTACKING = "ATTACKING"
    RAMMING   = "RAMMING"

STATE_COLOURS = {
    State.SEARCHING: (100, 100, 100),
    State.HOLDING:   (0, 165, 255),
    State.ATTACKING: (0, 255, 0),
    State.RAMMING:   (0, 0, 255),
}


# =============================================================================
#  PID Controller
# =============================================================================

class PIDController:
    """
    PID controller for steering error.

    Input:  steering error in degrees (negative = left, positive = right)
    Output: correction in [-PID_MAX_OUTPUT, +PID_MAX_OUTPUT]
            which is then mapped to a differential motor command.

    reset() must be called whenever autonomous mode is toggled off,
    the robot reaches RAM state, or detection is lost -- otherwise stale
    integral and derivative values will cause a jerk on re-engagement.
    """

    def __init__(self, kp, ki, kd, max_output=PID_MAX_OUTPUT,
                 integral_limit=50.0):
        self.kp             = kp
        self.ki             = ki
        self.kd             = kd
        self.max_output     = max_output
        self.integral_limit = integral_limit

        self._integral   = 0.0
        self._prev_error = None
        self._prev_time  = None

    def reset(self):
        """Clear all state. Call on stop, mode change, or detection dropout."""
        self._integral   = 0.0
        self._prev_error = None
        self._prev_time  = None

    def compute(self, error):
        """
        Compute PID output for the given steering error (degrees).
        Returns correction in [-max_output, +max_output].
        Also returns individual P, I, D terms for logging.
        """
        now = time.time()

        # First call after reset: only P term available (no dt or prev_error)
        if self._prev_time is None:
            self._prev_error = error
            self._prev_time  = now
            p_term = self.kp * error
            return float(np.clip(p_term, -self.max_output, self.max_output)), \
                   round(p_term, 3), 0.0, 0.0

        dt = now - self._prev_time
        if dt <= 0:
            dt = 1e-6

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup clamp
        self._integral += error * dt
        self._integral  = float(np.clip(self._integral,
                                        -self.integral_limit,
                                         self.integral_limit))
        i_term = self.ki * self._integral

        # Derivative (on error, not measurement)
        d_term = self.kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time  = now

        output = float(np.clip(p_term + i_term + d_term,
                               -self.max_output, self.max_output))
        return output, round(p_term, 3), round(i_term, 3), round(d_term, 3)


# =============================================================================
#  Performance Logger
# =============================================================================

class PerformanceLogger:
    """
    Records per-frame data and computes metrics at end of run.

    Summary CSV schema is IDENTICAL to Phase 1 and Phase 2:
      label, total_time_s, time_to_align_s, time_to_contact_s,
      heading_at_contact_deg, mean_abs_steering_error_deg,
      max_abs_steering_error_deg, path_efficiency_pct,
      detection_reliability_pct, total_frames

    command_oscillations excluded to match Phase 1 and Phase 2.
    kp/ki/kd NOT in summary CSV (would break cross-phase comparison) --
    they are recorded in the filename and in the frame CSV instead.

    Path efficiency uses true 2D pixel coordinates, identical to Phase 1/2.
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
        print(f"\n[LOG] -- Run started -------------------------------------------")
        print(f"[LOG] Saving to: {os.path.abspath(self.log_dir)}")

    def record_frame(self, steering_error, distance, command,
                     both_detected, robot_center=None,
                     pid_p=None, pid_i=None, pid_d=None):
        """
        Call every camera frame while running is True.

        robot_center : (int, int) or None
            Pixel centre of robot marker. Used for true 2D path efficiency,
            identical to Phase 1 and Phase 2 so values are comparable.

        pid_p/i/d : float or None
            Individual PID terms logged to frame CSV for tuning analysis.
            Not included in summary CSV.
        """
        if not self.running:
            return

        elapsed = time.time() - self.start_time

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
            'pid_p':          pid_p,
            'pid_i':          pid_i,
            'pid_d':          pid_d,
            'kp':             KP,
            'ki':             KI,
            'kd':             KD,
        })

        # Milestone: first alignment
        if (self.time_to_align is None
                and steering_error is not None
                and abs(steering_error) <= ALIGN_THRESHOLD_DEG):
            self.time_to_align = elapsed
            print(f"[LOG] Aligned   t={elapsed:.2f}s  error={steering_error:+.1f}deg")

        # Milestone: contact
        if (self.time_to_contact is None
                and distance is not None
                and distance <= RAM_THRESHOLD_PX):
            self.time_to_contact    = elapsed
            self.heading_at_contact = steering_error
            print(f"[LOG] Contact   t={elapsed:.2f}s  "
                  f"dist={distance:.0f}px  error={steering_error:+.1f}deg")

    def end_run(self, label="run"):
        """Finalise metrics, print summary, save CSVs."""
        if not self.running or not self.frames:
            print("[LOG] No data to save.")
            return {}

        self.running = False
        total_time   = self.frames[-1]['time']
        n_frames     = len(self.frames)

        # Detection reliability
        detected      = sum(1 for f in self.frames if f['both_detected'])
        detection_pct = 100.0 * detected / n_frames

        # Steering error stats
        errors   = [abs(f['steering_error']) for f in self.frames
                    if f['steering_error'] is not None]
        mean_err = round(float(np.mean(errors)), 1) if errors else None
        max_err  = round(float(np.max(errors)),  1) if errors else None

        # True 2D path efficiency -- identical calculation to Phase 1 and Phase 2
        coords = [(f['robot_x'], f['robot_y']) for f in self.frames
                  if f['robot_x'] is not None and f['robot_y'] is not None]

        path_efficiency_pct = None
        if len(coords) >= 2:
            straight_line = np.sqrt(
                (coords[-1][0] - coords[0][0]) ** 2 +
                (coords[-1][1] - coords[0][1]) ** 2
            )
            actual_path = sum(
                np.sqrt((coords[i][0] - coords[i-1][0]) ** 2 +
                        (coords[i][1] - coords[i-1][1]) ** 2)
                for i in range(1, len(coords))
            )
            if actual_path > 0:
                path_efficiency_pct = round(100.0 * straight_line / actual_path, 1)

        # Summary metrics -- schema MUST match Phase 1 and Phase 2 exactly
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

        # Print summary
        print("\n[LOG] -- Run Summary -------------------------------------------")
        print(f"       Label                   : {metrics['label']}")
        print(f"       KP={KP}  KI={KI}  KD={KD}")
        print(f"       Total time              : {metrics['total_time_s']} s")
        print(f"       Time to align           : {metrics['time_to_align_s']} s")
        print(f"       Time to contact         : {metrics['time_to_contact_s']} s")
        print(f"       Heading error @ contact : {metrics['heading_at_contact_deg']} deg")
        print(f"       Mean |steering error|   : {metrics['mean_abs_steering_error_deg']} deg")
        print(f"       Max  |steering error|   : {metrics['max_abs_steering_error_deg']} deg")
        print(f"       Path efficiency         : {metrics['path_efficiency_pct']} %")
        print(f"       Detection reliability   : {metrics['detection_reliability_pct']} %")
        print(f"       Frames recorded         : {metrics['total_frames']}")
        print("[LOG] -------------------------------------------------------")

        # Save CSVs
        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Gains encoded in filename for easy identification without opening the file
        gain_tag     = f"kp{KP}_ki{KI}_kd{KD}"
        summary_path = os.path.join(self.log_dir, f"summary_{ts}_{label}_{gain_tag}.csv")
        frames_path  = os.path.join(self.log_dir, f"frames_{ts}_{label}_{gain_tag}.csv")

        try:
            with open(summary_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=metrics.keys())
                w.writeheader()
                w.writerow(metrics)
            print(f"\n[LOG] Summary saved  -> {os.path.abspath(summary_path)}")
        except Exception as e:
            print(f"[LOG] ERROR saving summary: {e}")

        try:
            with open(frames_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.frames[0].keys())
                w.writeheader()
                w.writerows(self.frames)
            print(f"[LOG] Frame log saved -> {os.path.abspath(frames_path)}\n")
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
            print(f"[NET] -> {command.upper():<22}  <- {response}")
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
#  ArUco -- tuned detector + downscaled detection (identical to Phase 1 & 2)
# =============================================================================

def make_detector():
    """
    Build an ArucoDetector with parameters tuned for reliable detection.
    Identical to Phase 1 and Phase 2 so detection quality is not a confound.
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
    Detect on downscaled frame, scale corners back up for pose estimation.
    Identical pipeline to Phase 1 and Phase 2.
    """
    h, w = frame.shape[:2]
    small      = cv2.resize(frame, (int(w * DETECT_SCALE), int(h * DETECT_SCALE)))
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    corners_small, ids, _ = detector.detectMarkers(gray_small)

    detections = {}
    if ids is None or len(corners_small) == 0:
        return frame, detections

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
#  PID-based motor command
# =============================================================================

def pid_to_motor_command(pid_output, base_speed_us=None):
    """
    Convert PID output [-PID_MAX_OUTPUT, +PID_MAX_OUTPUT] to MOTOR command.

    Positive output -> turn right (slow/reverse left wheel).
    Negative output -> turn left  (slow/reverse right wheel).

    Mapping is identical to Phase 2 proportional version but now driven
    by PID output rather than raw error / MAX_STEER_DEG.
    """
    if base_speed_us is None:
        base_speed_us = PWM_FWD

    t         = np.clip(pid_output / PID_MAX_OUTPUT, -1.0, 1.0)
    fwd_range = PWM_FWD  - PWM_STOP
    rev_range = PWM_STOP - PWM_REV

    if t >= 0:
        left_us  = PWM_FWD - int(t * (fwd_range + rev_range))
        right_us = base_speed_us
    else:
        right_us = PWM_FWD - int((-t) * (fwd_range + rev_range))
        left_us  = base_speed_us

    left_us  = int(np.clip(left_us,  PWM_MIN, PWM_MAX))
    right_us = int(np.clip(right_us, PWM_MIN, PWM_MAX))
    return f"MOTOR {left_us} {right_us}"


# =============================================================================
#  State Machine
# =============================================================================

def decide_state_and_command(detections, pid, last_valid_cmd, last_detection_time):
    """
    Phase 3 state machine with PID steering and dropout hold.

    Returns (state, command, info, pid_terms, last_valid_cmd, last_detection_time)
    pid_terms dict contains p/i/d values for logging.
    """
    info      = {'steering_error': None, 'distance': None}
    pid_terms = {'p': None, 'i': None, 'd': None}
    now       = time.time()

    if ROBOT_ID in detections and OPPONENT_ID in detections:
        rc  = detections[ROBOT_ID]['center']
        oc  = detections[OPPONENT_ID]['center']
        hdg = get_robot_heading(detections[ROBOT_ID]['rvec'])
        err = get_steering_error(rc, oc, hdg)
        dst = get_pixel_distance(rc, oc)

        info['steering_error'] = err
        info['distance']       = dst

        if dst <= RAM_THRESHOLD_PX:
            # Close enough -- full forward, reset PID so it starts fresh
            pid.reset()
            return State.RAMMING, "FORWARD", info, pid_terms, "FORWARD", now

        # ATTACKING: compute PID correction
        pid_output, p, i, d = pid.compute(err)
        pid_terms = {'p': p, 'i': i, 'd': d}
        cmd = pid_to_motor_command(pid_output)
        return State.ATTACKING, cmd, info, pid_terms, cmd, now

    else:
        time_since = now - last_detection_time
        if last_valid_cmd is not None and time_since < HOLD_DURATION:
            return State.HOLDING, last_valid_cmd, info, pid_terms, last_valid_cmd, last_detection_time
        else:
            pid.reset()
            return State.SEARCHING, "STOP", info, pid_terms, None, last_detection_time


# =============================================================================
#  HUD
# =============================================================================

def draw_hud(frame, detections, state, command, info, pid_terms,
             worker, autonomous, logger):
    h, w = frame.shape[:2]

    # Connection status
    conn_col = (0, 255, 0) if worker.connected else (0, 0, 255)
    cv2.putText(frame, "ESP32: OK" if worker.connected else "ESP32: DISCONNECTED",
                (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, conn_col, 2)

    # Phase label -- always visible so recordings are unambiguous
    cv2.putText(frame, "PHASE 3 -- PID Steering",
                (w // 2 - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    # Autonomous mode
    auto_col = (0, 255, 0) if autonomous else (0, 165, 255)
    cv2.putText(frame, "AUTO: ON  [A]" if autonomous else "AUTO: OFF [A]",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, auto_col, 2)

    # State + command
    state_col = STATE_COLOURS.get(state, (255, 255, 255))
    cv2.putText(frame, f"STATE:   {state}",   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_col, 2)
    cv2.putText(frame, f"CMD:     {command}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

    # PID terms -- live display for tuning
    if pid_terms['p'] is not None:
        cv2.putText(frame,
                    f"PID  P:{pid_terms['p']:+.1f}  I:{pid_terms['i']:+.1f}  D:{pid_terms['d']:+.1f}",
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Gains reminder
    cv2.putText(frame, f"Kp={KP}  Ki={KI}  Kd={KD}",
                (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Recording indicator
    if logger.running:
        elapsed = time.time() - logger.start_time
        cv2.putText(frame, f"REC  {elapsed:.1f}s", (10, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        if int(elapsed * 2) % 2 == 0:
            cv2.circle(frame, (170, 178), 8, (0, 0, 255), -1)
    else:
        cv2.putText(frame, "R = start recording", (10, 185),
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
        cv2.circle(frame, rc, RAM_THRESHOLD_PX,    (0, 0, 255),   1)
        cv2.circle(frame, rc, ATTACK_THRESHOLD_PX, (0, 255, 255), 1)

    # Opponent marker
    if OPPONENT_ID in detections:
        oc = detections[OPPONENT_ID]['center']
        cv2.circle(frame, oc, 10, (0, 0, 255), -1)
        cv2.putText(frame, "OPPT", (oc[0] + 12, oc[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Vector line robot -> opponent
    if ROBOT_ID in detections and OPPONENT_ID in detections:
        rc       = detections[ROBOT_ID]['center']
        oc       = detections[OPPONENT_ID]['center']
        line_col = (0, 255, 0) if state in (State.ATTACKING, State.RAMMING) else (0, 165, 255)
        cv2.arrowedLine(frame, rc, oc, line_col, 2, tipLength=0.05)

    cv2.putText(frame, f"Queue:   {worker.queue.qsize()}", (10, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, "Q=quit  S=stop  T=test  A=auto  R=record",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return frame


# =============================================================================
#  Main
# =============================================================================

def main():
    detector = make_detector()

    worker = CommandWorker()
    worker.start()

    logger = PerformanceLogger()

    # One PID instance -- persists across frames, reset on stop/dropout
    pid = PIDController(kp=KP, ki=KI, kd=KD)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE,      -1)

    if not cap.isOpened():
        print("[CAM] Cannot open webcam.")
        worker.stop()
        return

    print("[CAM] Camera opened.")
    print("PHASE 3 -- PID Steering Controller")
    print(f"Gains: Kp={KP}  Ki={KI}  Kd={KD}")
    print("A = autonomous  |  R = record  |  S = stop  |  T = test  |  Q = quit")
    print("=" * 60)

    autonomous          = False
    current_state       = State.SEARCHING
    current_cmd         = "STOP"
    pid_terms           = {'p': None, 'i': None, 'd': None}
    last_sent_cmd       = None
    last_send_time      = 0.0
    run_counter         = 0
    last_valid_cmd      = None
    last_detection_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[CAM] Failed to grab frame.")
                break

            frame, detections = pose_estimation(frame, detector)

            if ROBOT_ID in detections and OPPONENT_ID in detections:
                last_detection_time = time.time()

            if autonomous:
                current_state, current_cmd, info, pid_terms, \
                    last_valid_cmd, last_detection_time = \
                    decide_state_and_command(
                        detections, pid, last_valid_cmd, last_detection_time)
            else:
                current_state       = State.SEARCHING
                current_cmd         = "STOP"
                info                = {'steering_error': None, 'distance': None}
                pid_terms           = {'p': None, 'i': None, 'd': None}
                last_valid_cmd      = None
                last_detection_time = time.time()

            # Send command (rate limited)
            now = time.time()
            if current_cmd != last_sent_cmd or (now - last_send_time) > CMD_INTERVAL:
                worker.put(current_cmd)
                last_sent_cmd  = current_cmd
                last_send_time = now

            # Pass robot centre for true 2D path efficiency
            robot_center  = detections[ROBOT_ID]['center'] if ROBOT_ID in detections else None
            both_detected = (ROBOT_ID in detections and OPPONENT_ID in detections)

            logger.record_frame(
                steering_error = info['steering_error'],
                distance       = info['distance'],
                command        = current_cmd,
                both_detected  = both_detected,
                robot_center   = robot_center,
                pid_p          = pid_terms['p'],
                pid_i          = pid_terms['i'],
                pid_d          = pid_terms['d'],
            )

            # Auto-end run on contact
            if (logger.running
                    and logger.time_to_contact is not None
                    and current_state == State.RAMMING):
                logger.end_run(label=f"phase3_run{run_counter:02d}")

            frame = draw_hud(frame, detections, current_state, current_cmd,
                             info, pid_terms, worker, autonomous, logger)
            cv2.imshow("Robot Vision -- Phase 3", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                if logger.running:
                    logger.end_run(label=f"phase3_run{run_counter:02d}_aborted")
                print("\n[SYS] Quitting.")
                break

            elif key == ord('a'):
                autonomous = not autonomous
                pid.reset()
                print(f"\n[SYS] Autonomous: {'ON' if autonomous else 'OFF'}")
                if not autonomous:
                    last_valid_cmd = None
                    with worker.queue.mutex:
                        worker.queue.queue.clear()
                    worker.put("STOP")

            elif key == ord('s'):
                autonomous     = False
                last_valid_cmd = None
                pid.reset()
                if logger.running:
                    logger.end_run(label=f"phase3_run{run_counter:02d}_stopped")
                with worker.queue.mutex:
                    worker.queue.queue.clear()
                worker.put("STOP")
                last_sent_cmd = "STOP"
                print("\n[SYS] Emergency STOP.")

            elif key == ord('r'):
                if not logger.running:
                    run_counter += 1
                    logger.start_run()
                    print(f"[LOG] Recording run {run_counter}  "
                          f"Kp={KP} Ki={KI} Kd={KD}")
                else:
                    logger.end_run(label=f"phase3_run{run_counter:02d}")

            elif key == ord('t'):
                if not autonomous:
                    print("\n[SYS] Running test sequence.")
                    worker.put_sequence([
                        ("FORWARD",                  1.0),
                        ("STOP",                     0.5),
                        ("BACK",                     1.0),
                        ("STOP",                     0.5),
                        (pid_to_motor_command( 45),  0.8),
                        ("STOP",                     0.5),
                        (pid_to_motor_command(-45),  0.8),
                        ("STOP",                     0.5),
                    ])
                else:
                    print("\n[SYS] Turn off autonomous mode before running test.")

    except KeyboardInterrupt:
        print("\n[SYS] Interrupted.")
        if logger.running:
            logger.end_run(label=f"phase3_run{run_counter:02d}_interrupted")

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[SYS] Done.")


if __name__ == "__main__":
    main()