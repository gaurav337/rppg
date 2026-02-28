"""
Aegis-X  ·  Live rPPG Heartbeat Monitor  v2.0
================================================
A premium Streamlit application that uses remote photoplethysmography (rPPG)
to measure heart rate in real-time from your webcam.

v2.0 — CHROM method, adaptive filtering, session stats, improved UI.
"""

import streamlit as st
import cv2
import dlib
import numpy as np
import time
import os
import urllib.request
import bz2
import pandas as pd
from collections import deque
from scipy.signal import butter, filtfilt, welch, detrend, find_peaks, medfilt

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DLIB_LANDMARK_PATH = "models/shape_predictor_68_face_landmarks.dat"
DLIB_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

BPM_LOW = 45
BPM_HIGH = 180
FREQ_LOW = BPM_LOW / 60.0
FREQ_HIGH = BPM_HIGH / 60.0

FOREHEAD_LANDMARKS = list(range(17, 27))
LEFT_CHEEK_LANDMARKS = [1, 2, 3, 4, 31, 36]
RIGHT_CHEEK_LANDMARKS = [12, 13, 14, 15, 35, 45]
NOSE_BRIDGE_LANDMARKS = [27, 28, 29, 30]

MAX_BUFFER = 450          # ~15s at 30fps — bigger buffer = stabler
MIN_FRAMES_FOR_BPM = 75   # ~2.5s of data — faster first reading
BPM_SMOOTH_WINDOW = 5     # median filter window for BPM smoothing
BPM_JUMP_THRESHOLD = 25   # reject BPM changes > this per reading

# ─────────────────────────────────────────────
# Page Config & Custom CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Aegis-X · Live rPPG",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #06060f 0%, #0c0c24 30%, #0a1628 70%, #060610 100%);
}
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Glass Card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 14px;
}
.glass-card-accent {
    background: linear-gradient(135deg, rgba(255,55,95,0.07) 0%, rgba(200,80,192,0.04) 100%);
    border: 1px solid rgba(255, 55, 95, 0.12);
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(255, 55, 95, 0.06), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 14px;
}
.glass-card-stats {
    background: linear-gradient(135deg, rgba(59,130,246,0.06) 0%, rgba(99,102,241,0.03) 100%);
    border: 1px solid rgba(59, 130, 246, 0.1);
    border-radius: 20px;
    padding: 22px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.05);
    margin-bottom: 14px;
}

/* ── Hero Title ── */
.hero-container {
    display: flex;
    align-items: center;
    gap: 14px;
}
.hero-logo {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #ff375f, #c850c0);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 15px rgba(255,55,95,0.3);
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ff375f, #ff6b8a, #c850c0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.95rem;
    color: rgba(255,255,255,0.35);
    font-weight: 400;
    margin-top: 2px;
    letter-spacing: 0.3px;
}
.hero-version {
    display: inline-block;
    padding: 2px 8px;
    background: rgba(255,55,95,0.12);
    border: 1px solid rgba(255,55,95,0.2);
    border-radius: 6px;
    font-size: 0.65rem;
    color: #ff6b8a;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-left: 8px;
    vertical-align: super;
}

/* ── BPM Display ── */
.bpm-container {
    text-align: center;
    padding: 16px 0;
}
.bpm-value {
    font-size: 6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ff375f, #ff8fa3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    letter-spacing: -4px;
}
.bpm-unit {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.3);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 5px;
    margin-top: 6px;
}

/* ── Pulse animation ── */
@keyframes pulse {
    0%   { transform: scale(1);   opacity: 1; }
    50%  { transform: scale(1.2); opacity: 0.8; }
    100% { transform: scale(1);   opacity: 1; }
}
@keyframes glow {
    0%   { box-shadow: 0 0 5px rgba(255,55,95,0.3); }
    50%  { box-shadow: 0 0 20px rgba(255,55,95,0.5), 0 0 40px rgba(255,55,95,0.2); }
    100% { box-shadow: 0 0 5px rgba(255,55,95,0.3); }
}
.heart-pulse {
    display: inline-block;
    font-size: 2.6rem;
    animation: pulse 0.9s ease-in-out infinite;
    filter: drop-shadow(0 0 8px rgba(255,55,95,0.4));
}

/* ── Session Stats ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 8px;
}
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    transition: all 0.3s ease;
}
.stat-card:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.1);
}
.stat-card.low {
    border-left: 3px solid #3b82f6;
}
.stat-card.avg {
    border-left: 3px solid #34d399;
}
.stat-card.high {
    border-left: 3px solid #f87171;
}
.stat-label {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.35);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.stat-val {
    font-size: 1.8rem;
    font-weight: 800;
    line-height: 1;
}
.stat-val.low-color  { color: #60a5fa; }
.stat-val.avg-color  { color: #34d399; }
.stat-val.high-color { color: #f87171; }
.stat-sub {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.25);
    margin-top: 4px;
    font-weight: 500;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 10px;
}
.metric-item {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 12px 8px;
    text-align: center;
}
.metric-label {
    font-size: 0.6rem;
    color: rgba(255,255,255,0.3);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 4px;
}
.metric-val {
    font-size: 1.3rem;
    font-weight: 700;
    color: rgba(255,255,255,0.85);
}
.metric-val.good { color: #34d399; }
.metric-val.warn { color: #fbbf24; }
.metric-val.bad  { color: #f87171; }

/* ── Signal Quality Bar ── */
.quality-bar-bg {
    width: 100%;
    height: 5px;
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    margin-top: 8px;
    overflow: hidden;
}
.quality-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease, background 0.5s ease;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.status-active {
    background: rgba(52,211,153,0.12);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.2);
}
.status-waiting {
    background: rgba(251,191,36,0.12);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.2);
}
.status-off {
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.25);
    border: 1px solid rgba(255,255,255,0.06);
}

/* ── Section header ── */
.section-label {
    font-size: 0.65rem;
    color: rgba(255,255,255,0.25);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 10px;
    padding-left: 2px;
}

/* ── Camera frame ── */
[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── Hide default chrome ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Button styling ── */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    padding: 14px 24px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.3px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: none;
    background: linear-gradient(135deg, #ff375f, #c850c0) !important;
    color: white !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255,55,95,0.35);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Divider ── */
.soft-divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    margin: 14px 0;
}

/* ── Info panel ── */
.info-panel {
    background: rgba(59, 130, 246, 0.04);
    border: 1px solid rgba(59, 130, 246, 0.08);
    border-radius: 14px;
    padding: 16px 20px;
    color: rgba(255,255,255,0.5);
    font-size: 0.82rem;
    line-height: 1.7;
}
.info-panel strong {
    color: rgba(255,255,255,0.7);
}

/* ── Idle camera card ── */
.idle-camera {
    text-align: center;
    padding: 80px 20px;
}
.idle-icon {
    font-size: 4.5rem;
    margin-bottom: 16px;
    filter: grayscale(0.3);
}
.idle-text {
    color: rgba(255,255,255,0.3);
    font-size: 0.95rem;
    line-height: 1.6;
}
.idle-text strong {
    color: rgba(255,255,255,0.5);
}

/* ── Tips ── */
.tips-container {
    margin-top: 12px;
}
.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    color: rgba(255,255,255,0.4);
    font-size: 0.8rem;
}
.tip-icon {
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 1px;
}
</style>
"""


# ─────────────────────────────────────────────
# Signal Processing (Enhanced)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading face detection models …")
def load_detector_and_predictor():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(DLIB_LANDMARK_PATH):
        st.info("⬇️ Downloading dlib shape-predictor model (~60 MB) …")
        urllib.request.urlretrieve(DLIB_URL, DLIB_LANDMARK_PATH + ".bz2")
        with bz2.BZ2File(DLIB_LANDMARK_PATH + ".bz2") as fr, \
             open(DLIB_LANDMARK_PATH, "wb") as fw:
            fw.write(fr.read())
        os.remove(DLIB_LANDMARK_PATH + ".bz2")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_LANDMARK_PATH)
    return detector, predictor


def get_forehead_roi(landmarks, frame_shape):
    brow_pts = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in FOREHEAD_LANDMARKS]
    )
    x_min, x_max = int(brow_pts[:, 0].min()), int(brow_pts[:, 0].max())
    y_brow = int(brow_pts[:, 1].min())
    h = int((x_max - x_min) * 0.45)  # taller forehead region for more pixels
    y_top = max(0, y_brow - h)
    x_pad = int((x_max - x_min) * 0.1)
    return y_top, y_brow, x_min + x_pad, x_max - x_pad


def get_cheek_rois(landmarks):
    rois = []
    for indices in [LEFT_CHEEK_LANDMARKS, RIGHT_CHEEK_LANDMARKS]:
        pts = np.array(
            [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]
        )
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        px, py = int((x_max - x_min) * 0.1), int((y_max - y_min) * 0.1)
        rois.append((y_min + py, y_max - py, x_min + px, x_max - px))
    return rois


def get_nose_roi(landmarks):
    pts = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in NOSE_BRIDGE_LANDMARKS]
    )
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    pad = int((x_max - x_min) * 0.3)
    return y_min, y_max, max(0, x_min - pad), x_max + pad


def draw_roi_overlay(frame, overlay, y1, y2, x1, x2, color=(52, 211, 153)):
    """Draw rectangle + semi-transparent fill on overlay."""
    h, w = frame.shape[:2]
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    if y2 <= y1 or x2 <= x1:
        return []
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    sub = overlay[y1:y2, x1:x2]
    fill = np.full_like(sub, color)
    cv2.addWeighted(fill, 0.12, sub, 0.88, 0, sub)
    return [frame[y1:y2, x1:x2]]


def extract_rgb_from_frame(frame, detector, predictor):
    """
    Extract mean R, G, B channel values from facial ROIs.
    Returns (rgb_means, annotated_frame, face_found).
    rgb_means is a (3,) array [R, G, B] or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Histogram equalization for better face detection in varied lighting
    gray_eq = cv2.equalizeHist(gray)
    faces = detector(gray_eq, 0)

    if len(faces) == 0:
        # Try on original grayscale
        faces = detector(gray, 0)
    if len(faces) == 0:
        return None, frame, False

    face = faces[0]
    landmarks = predictor(gray, face)
    roi_pixels_list = []

    overlay = frame.copy()
    # Face box (subtle)
    cv2.rectangle(
        overlay,
        (face.left(), face.top()),
        (face.right(), face.bottom()),
        (255, 55, 95), 2,
    )

    # Forehead
    y1, y2, x1, x2 = get_forehead_roi(landmarks, frame.shape)
    rois = draw_roi_overlay(frame, overlay, y1, y2, x1, x2)
    roi_pixels_list.extend(rois)

    # Cheeks
    for cy1, cy2, cx1, cx2 in get_cheek_rois(landmarks):
        rois = draw_roi_overlay(frame, overlay, cy1, cy2, cx1, cx2)
        roi_pixels_list.extend(rois)

    # Nose bridge (extra signal)
    ny1, ny2, nx1, nx2 = get_nose_roi(landmarks)
    rois = draw_roi_overlay(frame, overlay, ny1, ny2, nx1, nx2, color=(96, 165, 250))
    roi_pixels_list.extend(rois)

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    if roi_pixels_list:
        all_pixels = np.concatenate([r.reshape(-1, 3) for r in roi_pixels_list], axis=0)
        rgb_means = np.mean(all_pixels, axis=0).astype(np.float64)  # [R, G, B]
        return rgb_means, frame, True
    return None, frame, True


def chrom_rppg(rgb_buffer):
    """
    CHROM (Chrominance-based rPPG) method.
    De Haan & Jeanne (2013).
    Returns a 1D pulse signal.
    """
    rgb = np.array(rgb_buffer, dtype=np.float64)  # (N, 3)
    # Normalize each channel by its mean
    means = rgb.mean(axis=0, keepdims=True) + 1e-8
    rgb_norm = rgb / means

    r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]

    # CHROM signal
    xs = 3.0 * r - 2.0 * g
    ys = 1.5 * r + g - 1.5 * b

    # Bandpass each before combining (sliding window not needed for short buffers)
    # Use standard deviation ratio to combine
    alpha = np.std(xs) / (np.std(ys) + 1e-8)
    signal = xs - alpha * ys

    return signal


def pos_rppg(rgb_buffer):
    """
    POS (Plane Orthogonal to Skin) method.
    Wang et al. (2017).
    Returns a 1D pulse signal.
    """
    rgb = np.array(rgb_buffer, dtype=np.float64)
    means = rgb.mean(axis=0, keepdims=True) + 1e-8
    rgb_norm = rgb / means

    r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]

    xs = g - b
    ys = g + b - 2.0 * r

    alpha = np.std(xs) / (np.std(ys) + 1e-8)
    signal = xs + alpha * ys

    return signal


def interpolate_nans_rgb(rgb_buffer):
    """Interpolate NaN rows in RGB buffer."""
    arr = np.array(rgb_buffer, dtype=np.float64)
    for ch in range(3):
        col = arr[:, ch]
        nans = np.isnan(col)
        if nans.any() and not nans.all():
            x = np.arange(len(col))
            col[nans] = np.interp(x[nans], x[~nans], col[~nans])
        arr[:, ch] = col
    return arr


def bandpass_filter(signal, fps, order=4):
    nyquist = fps / 2.0
    low = FREQ_LOW / nyquist
    high = min(FREQ_HIGH / nyquist, 0.99)
    if low >= high or low <= 0 or len(signal) < 27:
        return signal
    b, a = butter(N=order, Wn=[low, high], btype="bandpass")
    try:
        return filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)), len(signal) - 1))
    except Exception:
        return signal


def estimate_bpm_welch(signal, fps):
    nperseg = min(512, len(signal))
    noverlap = nperseg * 3 // 4
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, noverlap=noverlap, nfft=2048)
    mask = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    vf, vp = freqs[mask], psd[mask]
    if len(vp) == 0:
        return 0.0, 0.0
    idx = np.argmax(vp)
    bpm = vf[idx] * 60.0
    total = np.sum(vp)
    if total > 0:
        w = max(2, len(vp) // 15)
        lo, hi = max(0, idx - w), min(len(vp), idx + w + 1)
        peak_power = np.sum(vp[lo:hi])
        conf = peak_power / total
    else:
        conf = 0.0
    return bpm, conf


def estimate_bpm_peaks(signal, fps):
    sig = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    min_dist = max(1, int(fps * 60.0 / BPM_HIGH))
    peaks, props = find_peaks(sig, distance=min_dist, prominence=0.25, height=0)
    if len(peaks) < 3:
        return 0.0, 0.0
    intervals = np.diff(peaks) / fps
    # Remove outlier intervals
    med = np.median(intervals)
    good = intervals[(intervals > med * 0.6) & (intervals < med * 1.5)]
    if len(good) < 2:
        good = intervals
    bpm = 60.0 / np.mean(good) if np.mean(good) > 0 else 0.0
    cv = np.std(good) / (np.mean(good) + 1e-8)
    conf = max(0.0, min(1.0, 1.0 - cv))
    return bpm, conf


def compute_bpm(rgb_buffer, fps):
    """
    Fuse CHROM + POS + Green-channel for robust BPM estimation.
    Returns (bpm, confidence).
    """
    rgb_arr = interpolate_nans_rgb(rgb_buffer)
    if np.isnan(rgb_arr).all():
        return 0.0, 0.0

    results = []

    # Method 1: CHROM
    try:
        sig_chrom = chrom_rppg(rgb_arr)
        sig_chrom = detrend(sig_chrom)
        sig_chrom = bandpass_filter(sig_chrom, fps)
        bpm_w, conf_w = estimate_bpm_welch(sig_chrom, fps)
        bpm_p, conf_p = estimate_bpm_peaks(sig_chrom, fps)
        if bpm_w > 0:
            results.append((bpm_w, conf_w * 1.2))  # slight boost for Welch
        if bpm_p > 0:
            results.append((bpm_p, conf_p))
    except Exception:
        pass

    # Method 2: POS
    try:
        sig_pos = pos_rppg(rgb_arr)
        sig_pos = detrend(sig_pos)
        sig_pos = bandpass_filter(sig_pos, fps)
        bpm_w, conf_w = estimate_bpm_welch(sig_pos, fps)
        bpm_p, conf_p = estimate_bpm_peaks(sig_pos, fps)
        if bpm_w > 0:
            results.append((bpm_w, conf_w * 1.1))
        if bpm_p > 0:
            results.append((bpm_p, conf_p * 0.9))
    except Exception:
        pass

    # Method 3: Green channel
    try:
        green = rgb_arr[:, 1].copy()
        green = detrend(green)
        green = bandpass_filter(green, fps)
        bpm_w, conf_w = estimate_bpm_welch(green, fps)
        if bpm_w > 0:
            results.append((bpm_w, conf_w * 0.8))
    except Exception:
        pass

    if not results:
        return 0.0, 0.0

    # Weighted average of all results
    bpms = np.array([r[0] for r in results])
    confs = np.array([r[1] for r in results])

    # Remove outliers: cluster around median
    median_bpm = np.median(bpms)
    mask = np.abs(bpms - median_bpm) < 20
    if mask.sum() == 0:
        mask = np.ones(len(bpms), dtype=bool)

    bpms, confs = bpms[mask], confs[mask]
    total_conf = np.sum(confs) + 1e-9
    bpm = np.sum(bpms * confs) / total_conf
    confidence = min(1.0, np.max(confs))

    if bpm < BPM_LOW or bpm > BPM_HIGH:
        return 0.0, 0.0

    return bpm, confidence


def get_display_signal(rgb_buffer, fps):
    """Get the best signal for waveform display."""
    rgb_arr = interpolate_nans_rgb(rgb_buffer)
    try:
        sig = chrom_rppg(rgb_arr)
        sig = detrend(sig)
        sig = bandpass_filter(sig, fps)
        return sig
    except Exception:
        return rgb_arr[:, 1]


# ─────────────────────────────────────────────
# UI Helpers
# ─────────────────────────────────────────────
def render_bpm_card(bpm, confidence):
    if bpm <= 0:
        val_html = '<span style="color:rgba(255,255,255,0.1);font-size:3.5rem;font-weight:300;">—</span>'
        heart = '<span style="font-size:2.6rem;opacity:0.2;">🤍</span>'
    else:
        val_html = f'<span class="bpm-value">{bpm:.0f}</span>'
        heart = '<span class="heart-pulse">❤️</span>'

    conf_pct = int(confidence * 100)
    if conf_pct > 60:
        bar_color = "#34d399"
        conf_label = "Strong"
    elif conf_pct > 35:
        bar_color = "#fbbf24"
        conf_label = "Moderate"
    else:
        bar_color = "#f87171"
        conf_label = "Weak"

    return f"""
    <div class="bpm-container">
        {heart}
        <div style="margin: 8px 0;">{val_html}</div>
        <div class="bpm-unit">beats per minute</div>
        <div class="soft-divider"></div>
        <div class="metric-label">Signal Confidence · {conf_label}</div>
        <div class="quality-bar-bg">
            <div class="quality-bar-fill" style="width:{conf_pct}%;background:linear-gradient(90deg, {bar_color}, {bar_color}aa);"></div>
        </div>
        <div style="font-size:0.75rem;color:rgba(255,255,255,0.25);margin-top:5px;">{conf_pct}%</div>
    </div>
    """


def render_session_stats(bpm_min, bpm_max, bpm_avg, reading_count):
    if reading_count < 1:
        lo = hi = av = "—"
        lo_sub = hi_sub = av_sub = "no data yet"
    else:
        lo = f"{bpm_min:.0f}"
        hi = f"{bpm_max:.0f}"
        av = f"{bpm_avg:.0f}"
        lo_sub = "bpm"
        hi_sub = "bpm"
        av_sub = f"over {reading_count} readings"

    return f"""
    <div class="section-label">Session Statistics</div>
    <div class="stats-grid">
        <div class="stat-card low">
            <div class="stat-label">Lowest</div>
            <div class="stat-val low-color">{lo}</div>
            <div class="stat-sub">{lo_sub}</div>
        </div>
        <div class="stat-card avg">
            <div class="stat-label">Average</div>
            <div class="stat-val avg-color">{av}</div>
            <div class="stat-sub">{av_sub}</div>
        </div>
        <div class="stat-card high">
            <div class="stat-label">Highest</div>
            <div class="stat-val high-color">{hi}</div>
            <div class="stat-sub">{hi_sub}</div>
        </div>
    </div>
    """


def render_metrics(fps, face_pct, buffer_len, elapsed):
    face_cls = "good" if face_pct > 80 else ("warn" if face_pct > 50 else "bad")
    mins, secs = divmod(int(elapsed), 60)
    time_str = f"{mins}:{secs:02d}" if mins > 0 else f"{secs}s"
    return f"""
    <div class="metric-row">
        <div class="metric-item">
            <div class="metric-label">FPS</div>
            <div class="metric-val">{fps:.0f}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Face</div>
            <div class="metric-val {face_cls}">{face_pct:.0f}%</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Buffer</div>
            <div class="metric-val">{buffer_len}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Time</div>
            <div class="metric-val">{time_str}</div>
        </div>
    </div>
    """


def render_status_badge(status):
    cls_map = {"active": "status-active", "waiting": "status-waiting", "off": "status-off"}
    label_map = {"active": "● Tracking", "waiting": "◐ Calibrating …", "off": "○ Idle"}
    return f'<span class="status-badge {cls_map[status]}">{label_map[status]}</span>'


def get_hr_zone(bpm):
    """Return heart rate zone label & color."""
    if bpm <= 0:
        return "—", "rgba(255,255,255,0.2)"
    elif bpm < 60:
        return "Resting (Low)", "#60a5fa"
    elif bpm < 80:
        return "Resting (Normal)", "#34d399"
    elif bpm < 100:
        return "Elevated", "#fbbf24"
    elif bpm < 140:
        return "Active", "#fb923c"
    else:
        return "Intense", "#f87171"


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Header ──
    header_l, header_r = st.columns([3, 1])
    with header_l:
        st.markdown(
            '<div class="hero-container">'
            '<div><p class="hero-title">Aegis-X <span class="hero-version">v2.0</span></p>'
            '<p class="hero-sub">Remote Photoplethysmography · Live Heart Rate Monitor</p></div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with header_r:
        st.markdown(
            '<div style="text-align:right;padding-top:22px;">'
            '<span class="status-badge status-off" id="global-badge">○ Idle</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # ── Load models ──
    detector, predictor = load_detector_and_predictor()

    # ── Layout ──
    col_cam, col_vitals = st.columns([1.3, 1], gap="large")

    with col_cam:
        st.markdown('<div class="section-label">Camera Feed</div>', unsafe_allow_html=True)

        if "running" not in st.session_state:
            st.session_state.running = False

        btn_label = "⏹  Stop Monitoring" if st.session_state.running else "▶  Start Monitoring"
        if st.button(btn_label, use_container_width=True, key="toggle_btn"):
            st.session_state.running = not st.session_state.running
            # Reset session stats on stop
            if not st.session_state.running:
                for k in ["sess_bpms"]:
                    st.session_state.pop(k, None)
            st.rerun()

        frame_placeholder = st.empty()

        st.markdown(
            '<div class="info-panel">'
            '<strong>🔬 How it works:</strong> Aegis-X v2.0 uses the <strong>CHROM</strong> and '
            '<strong>POS</strong> rPPG algorithms to extract your pulse from subtle color changes '
            'in facial skin. Multiple ROIs (forehead, cheeks, nose bridge) are fused for accuracy.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="glass-card" style="margin-top:12px;">'
            '<div class="section-label" style="margin-bottom:8px;">💡 Tips for Best Results</div>'
            '<div class="tips-container">'
            '<div class="tip-item"><span class="tip-icon">🔆</span> Ensure good, even lighting on your face</div>'
            '<div class="tip-item"><span class="tip-icon">🪑</span> Sit still — motion reduces accuracy</div>'
            '<div class="tip-item"><span class="tip-icon">📏</span> Keep your face 30–60 cm from the camera</div>'
            '<div class="tip-item"><span class="tip-icon">⏱️</span> Wait 5–10 seconds for a stable reading</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_vitals:
        st.markdown('<div class="section-label">Live Vitals</div>', unsafe_allow_html=True)
        status_placeholder = st.empty()
        bpm_placeholder = st.empty()
        zone_placeholder = st.empty()
        stats_placeholder = st.empty()
        metrics_placeholder = st.empty()

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">rPPG Waveform</div>', unsafe_allow_html=True)
        chart_placeholder = st.empty()

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">BPM Trend</div>', unsafe_allow_html=True)
        history_placeholder = st.empty()

    # ── Main Loop ──
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("🚫 Cannot access webcam. Check that your camera is connected.")
            st.session_state.running = False
            st.rerun()

        # Optimize camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        rgb_buffer = deque(maxlen=MAX_BUFFER)
        time_buffer = deque(maxlen=MAX_BUFFER)
        face_buffer = deque(maxlen=MAX_BUFFER)
        bpm_raw_history = deque(maxlen=120)
        bpm_smooth_history = deque(maxlen=120)
        bpm_time_history = deque(maxlen=120)

        # Session stats
        session_bpms = []
        last_valid_bpm = 0.0
        start_time = time.time()
        frame_count = 0

        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    frame_placeholder.error("❌ Lost camera feed.")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                now = time.time()
                frame_count += 1

                # Extract RGB
                rgb_means, annotated, face_found = extract_rgb_from_frame(
                    frame, detector, predictor
                )
                if rgb_means is not None:
                    rgb_buffer.append(rgb_means)
                else:
                    # Append last known or NaN
                    rgb_buffer.append(np.array([np.nan, np.nan, np.nan]))
                time_buffer.append(now)
                face_buffer.append(1 if face_found else 0)

                frame_placeholder.image(annotated, use_container_width=True)

                # Compute metrics
                elapsed = now - start_time
                n = len(time_buffer)
                if n > 1:
                    fps_est = (n - 1) / (time_buffer[-1] - time_buffer[0] + 1e-9)
                else:
                    fps_est = 30.0
                face_pct = 100.0 * sum(face_buffer) / len(face_buffer)

                # ── BPM estimation ──
                bpm, confidence = 0.0, 0.0
                display_signal = None

                if len(rgb_buffer) >= MIN_FRAMES_FOR_BPM and fps_est > 5:
                    bpm, confidence = compute_bpm(list(rgb_buffer), fps_est)

                    # Temporal smoothing: reject huge jumps
                    if bpm > 0 and last_valid_bpm > 0:
                        if abs(bpm - last_valid_bpm) > BPM_JUMP_THRESHOLD:
                            # Blend toward new value instead of jumping
                            bpm = last_valid_bpm * 0.7 + bpm * 0.3
                            confidence *= 0.6

                    if bpm > 0:
                        bpm_raw_history.append(bpm)
                        bpm_time_history.append(elapsed)

                        # Median filter for display
                        if len(bpm_raw_history) >= BPM_SMOOTH_WINDOW:
                            recent = list(bpm_raw_history)[-BPM_SMOOTH_WINDOW:]
                            smoothed = float(np.median(recent))
                        else:
                            smoothed = bpm
                        bpm_smooth_history.append(smoothed)
                        last_valid_bpm = smoothed
                        bpm = smoothed  # use smoothed for display

                        # Session stats
                        session_bpms.append(smoothed)

                    # Display signal
                    try:
                        display_signal = get_display_signal(list(rgb_buffer), fps_est)
                    except Exception:
                        pass

                # ── Waveform ──
                if display_signal is not None and len(display_signal) > 10:
                    display_len = min(200, len(display_signal))
                    wave = display_signal[-display_len:]
                    wave = (wave - np.mean(wave)) / (np.std(wave) + 1e-9)
                    chart_placeholder.line_chart(
                        pd.DataFrame({"rPPG": wave}),
                        height=160,
                        use_container_width=True,
                    )

                # ── Status ──
                if len(rgb_buffer) < MIN_FRAMES_FOR_BPM:
                    pct_done = int(len(rgb_buffer) / MIN_FRAMES_FOR_BPM * 100)
                    status_placeholder.markdown(
                        f'<div class="glass-card">{render_status_badge("waiting")}'
                        f' <span style="color:rgba(255,255,255,0.35);font-size:0.82rem;">'
                        f'Collecting frames … {pct_done}%</span></div>',
                        unsafe_allow_html=True,
                    )
                elif bpm > 0:
                    status_placeholder.markdown(
                        f'<div class="glass-card">{render_status_badge("active")}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    status_placeholder.markdown(
                        f'<div class="glass-card">{render_status_badge("waiting")}'
                        f' <span style="color:rgba(255,255,255,0.35);font-size:0.82rem;">'
                        f'Stabilising signal …</span></div>',
                        unsafe_allow_html=True,
                    )

                # ── BPM card ──
                bpm_placeholder.markdown(
                    f'<div class="glass-card-accent">{render_bpm_card(bpm, confidence)}</div>',
                    unsafe_allow_html=True,
                )

                # ── HR Zone ──
                zone_name, zone_color = get_hr_zone(bpm)
                zone_placeholder.markdown(
                    f'<div class="glass-card" style="text-align:center;padding:12px;">'
                    f'<div class="metric-label">Heart Rate Zone</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:{zone_color};margin-top:4px;">'
                    f'{zone_name}</div></div>',
                    unsafe_allow_html=True,
                )

                # ── Session Stats ──
                if session_bpms:
                    s_min = min(session_bpms)
                    s_max = max(session_bpms)
                    s_avg = np.mean(session_bpms)
                    s_count = len(session_bpms)
                else:
                    s_min = s_max = s_avg = 0
                    s_count = 0
                stats_placeholder.markdown(
                    f'<div class="glass-card-stats">{render_session_stats(s_min, s_max, s_avg, s_count)}</div>',
                    unsafe_allow_html=True,
                )

                # ── Metrics ──
                metrics_placeholder.markdown(
                    f'<div class="glass-card">{render_metrics(fps_est, face_pct, len(rgb_buffer), elapsed)}</div>',
                    unsafe_allow_html=True,
                )

                # ── BPM History ──
                if len(bpm_smooth_history) > 1:
                    hist_df = pd.DataFrame({"BPM": list(bpm_smooth_history)})
                    history_placeholder.line_chart(hist_df, height=120, use_container_width=True)

                time.sleep(0.005)

        finally:
            cap.release()
    else:
        # ── Idle state ──
        with col_vitals:
            status_placeholder.markdown(
                f'<div class="glass-card">{render_status_badge("off")}</div>',
                unsafe_allow_html=True,
            )
            bpm_placeholder.markdown(
                f'<div class="glass-card-accent">{render_bpm_card(0, 0)}</div>',
                unsafe_allow_html=True,
            )
            zone_placeholder.markdown(
                '<div class="glass-card" style="text-align:center;padding:12px;">'
                '<div class="metric-label">Heart Rate Zone</div>'
                '<div style="font-size:1.1rem;font-weight:700;color:rgba(255,255,255,0.15);margin-top:4px;">—</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            stats_placeholder.markdown(
                f'<div class="glass-card-stats">{render_session_stats(0, 0, 0, 0)}</div>',
                unsafe_allow_html=True,
            )
            metrics_placeholder.markdown(
                f'<div class="glass-card">{render_metrics(0, 0, 0, 0)}</div>',
                unsafe_allow_html=True,
            )

        with col_cam:
            frame_placeholder.markdown(
                '<div class="glass-card idle-camera">'
                '<div class="idle-icon">📷</div>'
                '<div class="idle-text">'
                'Press <strong>▶ Start Monitoring</strong> to begin<br>'
                'live heart rate tracking'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
