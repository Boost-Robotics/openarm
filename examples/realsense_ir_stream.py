#!/usr/bin/env python3
"""Stream stereo IR from a RealSense D405 at 1280x720 — before/after rectification
and live S2M2-L stereo depth estimation.

Displays three rows:
  Top:    raw IR1 + IR2 with epipolar lines
  Middle: rectified IR1 + IR2 with epipolar lines
  Bottom: disparity colormap (full + confidence-masked)

Same features should sit on the same green line in the middle row but not the top.

Requires: pip install pyrealsense2 opencv-python numpy pyyaml torch
          s2m2 package (pip install -e s2m2/)
The D405 must be connected via USB 3.0. Press 'q' to quit.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Force X11/XWayland for Qt (OpenCV) and GLFW (Open3D).
# Unsetting WAYLAND_DISPLAY makes GLFW fall back to X11 via XWayland,
# avoiding the "failed to create dri2 screen" / GLEW init failures.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.pop("WAYLAND_DISPLAY", None)  # force GLFW → X11
os.environ.setdefault("ROBOFLOW_API_KEY","KWtQdPM4NXlbyG7KMAiF")
os.environ.setdefault("XDG_SESSION_TYPE","x11")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
import pyrealsense2 as rs  # noqa: E402
import yaml  # noqa: E402
import torch  # noqa: E402
import supervision as sv  # noqa: E402
from inference import get_model


model_seg = get_model("pull-tag-segmentation/4")


# Ensure s2m2 is importable (handles case where package isn't pip-installed)
_s2m2_src = Path(__file__).resolve().parent.parent / "s2m2" / "src"
if _s2m2_src.is_dir() and str(_s2m2_src) not in sys.path:
    sys.path.insert(0, str(_s2m2_src))

from s2m2.core.utils.model_utils import load_model, run_stereo_matching  # noqa: E402

DEFAULT_CALIB = Path(__file__).parent / "d405_stereo_calib.yaml"
S2M2_WEIGHTS = Path(__file__).resolve().parent.parent / "s2m2" / "weights" / "pretrain_weights"

# Torch setup
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def load_calibration(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Load Kalibr-format stereo calibration. Returns (K1, D1, K2, D2, R, T, size)."""
    with path.open() as f:
        calib = yaml.safe_load(f)

    cam0, cam1 = calib["cam0"], calib["cam1"]
    fx0, fy0, cx0, cy0 = cam0["intrinsics"]
    fx1, fy1, cx1, cy1 = cam1["intrinsics"]

    k1 = np.array([[fx0, 0, cx0], [0, fy0, cy0], [0, 0, 1]], dtype=np.float64)
    k2 = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]], dtype=np.float64)
    d1 = np.array(cam0["distortion_coeffs"], dtype=np.float64)
    d2 = np.array(cam1["distortion_coeffs"], dtype=np.float64)

    t_mat = np.array(cam1["T_cn_cnm1"], dtype=np.float64)
    return k1, d1, k2, d2, t_mat[:3, :3], t_mat[:3, 3], tuple(cam0["resolution"])


def build_rectify_maps(
    k1: np.ndarray, d1: np.ndarray, k2: np.ndarray, d2: np.ndarray,
    r: np.ndarray, t: np.ndarray, size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo rectification remap tables + Q matrix + P1 (once at startup)."""
    r1, r2, p1, p2, Q, *_ = cv2.stereoRectify(k1, d1, k2, d2, size, r, t, alpha=0)
    m1x, m1y = cv2.initUndistortRectifyMap(k1, d1, r1, p1, size, cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(k2, d2, r2, p2, size, cv2.CV_32FC1)
    return m1x, m1y, m2x, m2y, Q, p1


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_epipolar_lines(img: np.ndarray, n: int = 18, color: tuple[int, int, int] = (0, 255, 0)) -> None:
    """Draw *n* evenly-spaced horizontal lines (scales with image size)."""
    h, w = img.shape[:2]
    step = max(h // n, 1)
    for y in range(step, h, step):
        cv2.line(img, (0, y), (w, y), color, 1)


def put_label(img: np.ndarray, text: str, x_frac: float, color: tuple[int, int, int]) -> None:
    """Draw a label at a position relative to image size."""
    h, w = img.shape[:2]
    scale = h / 360
    cv2.putText(img, text, (int(w * x_frac), int(h * 0.08)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(int(scale * 2), 1))


def make_row(left: np.ndarray, right: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Equalize, resize, hconcat two grayscale images into one BGR row."""
    left = cv2.equalizeHist(left)
    right = cv2.equalizeHist(right)
    half_w = out_w // 2
    left = cv2.resize(left, (half_w, out_h))
    right = cv2.resize(right, (out_w - half_w, out_h))
    return cv2.hconcat([cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)])


def gray_to_rgb_tensor(gray: np.ndarray) -> torch.Tensor:
    """Convert a uint8 grayscale image to a float32 [1,3,H,W] torch tensor."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # H,W,3 uint8
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)


def apply_colormap(x: np.ndarray) -> np.ndarray:
    """Normalize and apply JET colormap (same as s2m2 vis_utils)."""
    x_norm = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return cv2.applyColorMap(x_norm, cv2.COLORMAP_JET)


def make_depth_row(
    pred_disp: np.ndarray,
    pred_occ: np.ndarray,
    pred_conf: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Build a row: disparity colormap | confidence-masked disparity colormap."""
    disp_color = apply_colormap(pred_disp)
    valid = ((pred_conf > 0.1) * (pred_occ > 0.5)).astype(np.float32)
    disp_masked = (valid[:, :, np.newaxis] * disp_color).astype(np.uint8)

    half_w = out_w // 2
    left = cv2.resize(disp_color, (half_w, out_h))
    right = cv2.resize(disp_masked, (out_w - half_w, out_h))
    return cv2.hconcat([left, right])


def disp_to_pointcloud(
    disp: np.ndarray,
    Q: np.ndarray,
    gray: np.ndarray,
    conf: np.ndarray,
    occ: np.ndarray,
    depth_max: float = 0.5,
) -> o3d.geometry.PointCloud:
    """Reproject disparity to a coloured Open3D point cloud.

    Args:
        disp:  disparity map [H, W] float32
        Q:     4x4 reprojection matrix from stereoRectify
        gray:  rectified left IR image (uint8, single-channel)
        conf:  confidence map [H, W]
        occ:   occlusion map [H, W]
        depth_max: discard points farther than this (metres)
    """
    # reproject to XYZ  (units = whatever T was in, here metres)
    xyz = cv2.reprojectImageTo3D(disp.astype(np.float32), Q, handleMissingValues=True)

    # confidence + occlusion mask
    valid = (conf > 0.1) & (occ > 0.5) & (disp > 0.5)
    z = xyz[:, :, 2]
    valid &= np.isfinite(z) & (z > 0) & (z < depth_max)

    pts = xyz[valid]
    # Flip Y and Z so the cloud looks right when viewed from the default
    # Open3D camera (camera looks along +Z, Y points down in image coords).
    pts[:, 1] *= -1
    pts[:, 2] *= -1

    # colour from IR intensity (normalised to 0-1, replicated to RGB)
    intensity = gray[valid].astype(np.float64) / 255.0
    colours = np.stack([intensity, intensity, intensity], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colours)
    return pcd


# ---------------------------------------------------------------------------
# Depth from disparity
# ---------------------------------------------------------------------------


def disp_to_depth_mm(
    disp: np.ndarray,
    fx: float,
    baseline: float,
    conf: np.ndarray,
    occ: np.ndarray,
    depth_max_m: float = 1.0,
) -> np.ndarray:
    """Convert disparity to a uint16 depth image in millimetres.

    Pixels with low confidence, occlusion, or out-of-range depth are set to 0
    (matching the FoundationPose convention: ``depth[(depth<0.001)|…] = 0``).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_m = (baseline * fx) / disp  # metres
    valid = (
        np.isfinite(depth_m)
        & (disp > 0.5)
        & (depth_m > 0.001)
        & (depth_m < depth_max_m)
        & (conf > 0.1)
        & (occ > 0.5)
    )
    depth_mm = np.zeros_like(disp, dtype=np.uint16)
    depth_mm[valid] = np.clip(depth_m[valid] * 1000.0, 0, 65535).astype(np.uint16)
    return depth_mm


# ---------------------------------------------------------------------------
# FoundationPose recording helpers
# ---------------------------------------------------------------------------


def create_record_dir(base: Path) -> Path:
    """Create a timestamped output directory with rgb/ and depth/ sub-dirs."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = base / f"record_{stamp}"
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "color").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    return out


def save_cam_k(path: Path, p1: np.ndarray) -> None:
    """Write cam_K.txt (3x3 intrinsic from the rectified projection matrix P1)."""
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = p1[0, 0]  # fx
    K[1, 1] = p1[1, 1]  # fy
    K[0, 2] = p1[0, 2]  # cx
    K[1, 2] = p1[1, 2]  # cy
    np.savetxt(str(path / "cam_K.txt"), K, fmt="%.18e")


def save_frame(
    record_dir: Path,
    rect_gray: np.ndarray,
    depth_mm: np.ndarray,
    color_bgr: np.ndarray | None = None,
) -> str:
    """Save one frame pair.  Returns the id_str used as filename stem."""
    id_str = str(time.time_ns())
    cv2.imwrite(str(record_dir / "rgb" / f"{id_str}.png"),
                cv2.cvtColor(rect_gray, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(record_dir / "depth" / f"{id_str}.png"), depth_mm)
    if color_bgr is not None:
        cv2.imwrite(str(record_dir / "color" / f"{id_str}.png"), color_bgr)
    return id_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(calib_path: Path = DEFAULT_CALIB) -> int:
    width, height = 1280, 720
    fps = 15

    # ------------------------------------------------------------------
    # Load calibration + rectification maps
    # ------------------------------------------------------------------
    print(f"Loading calibration from {calib_path}")
    k1, d1, k2, d2, r, t, _sz = load_calibration(calib_path)
    m1x, m1y, m2x, m2y, Q, P1 = build_rectify_maps(k1, d1, k2, d2, r, t, (width, height))
    baseline = float(np.abs(t[0]))  # metres  (stereo X-translation)
    rect_fx = float(P1[0, 0])      # rectified focal length in pixels
    print(f"Rectification maps ready  (baseline={baseline*1e3:.2f} mm, fx={rect_fx:.1f} px).")

    # ------------------------------------------------------------------
    # Load S2M2-L stereo depth model
    # ------------------------------------------------------------------
    print(f"Loading S2M2-L model from {S2M2_WEIGHTS} on {DEVICE} ...")
    model = load_model(str(S2M2_WEIGHTS), model_type="L",
                       use_positivity=True, refine_iter=3, device=DEVICE)
    if model is None:
        print("Failed to load S2M2 model – exiting.", file=sys.stderr)
        return 1
    print("S2M2-L model ready.")

    # ------------------------------------------------------------------
    # Start RealSense pipeline
    # ------------------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start pipeline: {e}", file=sys.stderr)
        print("Ensure the D405 is connected via USB 3.0 and no other app is using it.", file=sys.stderr)
        return 1

    rs_device = profile.get_device()
    print(f"Device: {rs_device.get_info(rs.camera_info.name)}")
    depth_sensor = rs_device.first_depth_sensor()
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        print("IR emitter disabled (passive stereo).")

    win = "D405 stereo IR + S2M2 depth — 's' record, 'q' quit"
    cv2.namedWindow(win, cv2.WINDOW_KEEPRATIO)

    row_w, row_h = width, height // 3  # three rows in main window

    # Segmentation display in a separate window
    seg_win = "Segmentation"
    cv2.namedWindow(seg_win, cv2.WINDOW_KEEPRATIO)
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=4)

    # Recording state
    record_dir: Path | None = None
    record_count = 0
    record_base = Path(__file__).resolve().parent.parent / "FoundationPose" / "demo_data"

    # ------------------------------------------------------------------
    # Open3D non-blocking point cloud viewer
    # ------------------------------------------------------------------
    vis = o3d.visualization.Visualizer()
    try:
        vis.create_window(window_name="S2M2 point cloud (interactive)",
                          width=960, height=720)
        # Pump one event cycle so the GL context is fully ready
        vis.poll_events()
        vis.update_renderer()
    except Exception as e:
        print(f"[warn] Open3D window creation failed: {e}", file=sys.stderr)
        print("[warn] Continuing without 3-D point cloud window.", file=sys.stderr)
        vis = None

    pcd = o3d.geometry.PointCloud()
    if vis is not None:
        render_opt = vis.get_render_option()
        if render_opt is not None:
            render_opt.point_size = 2.0
            render_opt.background_color = np.array([0.05, 0.05, 0.1])
        vis.add_geometry(pcd)
    first_cloud = True  # need reset_view on first real cloud

    # Warm-up the model with a dummy forward pass
    print("Warming up model …")
    dummy_l = torch.zeros(1, 3, height, width, device=DEVICE)
    _ = run_stereo_matching(model, dummy_l, dummy_l, DEVICE)
    print("Warm-up done.")

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=10000)
            ir1, ir2 = frames.get_infrared_frame(1), frames.get_infrared_frame(2)
            color_frame = frames.get_color_frame()
            if not ir1 or not ir2:
                continue

            img1 = np.asanyarray(ir1.get_data())
            img2 = np.asanyarray(ir2.get_data())
            color_img = np.asanyarray(color_frame.get_data()) if color_frame else None

            # --- Row 1: raw ---
            raw = make_row(img1, img2, row_w, row_h)
            draw_epipolar_lines(raw, color=(0, 0, 255))
            put_label(raw, "RAW IR1", 0.01, (0, 0, 255))
            put_label(raw, "RAW IR2", 0.51, (0, 0, 255))

            # --- Row 2: rectified ---
            rect1 = cv2.remap(img1, m1x, m1y, cv2.INTER_LINEAR)
            rect2 = cv2.remap(img2, m2x, m2y, cv2.INTER_LINEAR)
            rect = make_row(rect1, rect2, row_w, row_h)
            draw_epipolar_lines(rect, color=(0, 255, 0))
            put_label(rect, "RECT IR1", 0.01, (0, 255, 0))
            put_label(rect, "RECT IR2", 0.51, (0, 255, 0))

            # --- Segmentation (separate window, uses color image) ---
            if color_img is not None:
                seg_results = model_seg.infer(image=color_img)
                detections = sv.Detections.from_inference(seg_results[0])
                seg_frame = color_img.copy()
                seg_frame = mask_annotator.annotate(scene=seg_frame, detections=detections)
                labels = [
                    f"{detections.data['class_name'][i]} {detections.confidence[i]:.2f}"
                    for i in range(len(detections))
                ] if len(detections) > 0 else []
                seg_frame = label_annotator.annotate(
                    scene=seg_frame, detections=detections, labels=labels,
                )
                cv2.imshow(seg_win, seg_frame)

            # --- Row 3: S2M2 depth ---
            left_t = gray_to_rgb_tensor(rect1)
            right_t = gray_to_rgb_tensor(rect2)
            pred_disp, pred_occ, pred_conf, avg_conf, run_ms = \
                run_stereo_matching(model, left_t, right_t, DEVICE)
            pred_disp_np = pred_disp.cpu().numpy()
            pred_occ_np = pred_occ.cpu().numpy()
            pred_conf_np = pred_conf.cpu().numpy()

            depth_row = make_depth_row(pred_disp_np, pred_occ_np, pred_conf_np,
                                       row_w, row_h)
            d_min, d_max = pred_disp_np.min(), pred_disp_np.max()
            put_label(depth_row, f"DISP [{d_min:.0f},{d_max:.0f}] {run_ms:.0f}ms",
                      0.01, (255, 255, 255))
            put_label(depth_row, f"CONF-MASKED  conf={avg_conf:.2f}",
                      0.51, (255, 255, 255))

            # --- Update Open3D point cloud ---
            if vis is not None:
                new_pcd = disp_to_pointcloud(pred_disp_np, Q, rect1,
                                             pred_conf_np, pred_occ_np,
                                             depth_max=0.5)
                pcd.points = new_pcd.points
                pcd.colors = new_pcd.colors
                vis.update_geometry(pcd)
                if first_cloud and len(pcd.points) > 0:
                    vis.reset_view_point(True)
                    first_cloud = False
                vis.poll_events()
                vis.update_renderer()

            # --- Save frame if recording ---
            if record_dir is not None:
                depth_mm = disp_to_depth_mm(pred_disp_np, rect_fx, baseline,
                                            pred_conf_np, pred_occ_np,
                                            depth_max_m=1.0)
                save_frame(record_dir, rect1, depth_mm, color_img)
                record_count += 1

            # --- Recording indicator on display ---
            if record_dir is not None:
                put_label(depth_row, f"REC {record_count}", 0.42, (0, 0, 255))

            cv2.imshow(win, cv2.vconcat([raw, rect, depth_row]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if record_dir is None:
                    record_dir = create_record_dir(record_base)
                    save_cam_k(record_dir, P1)
                    record_count = 0
                    print(f"Recording started → {record_dir}")
                else:
                    print(f"Recording stopped. {record_count} frames → {record_dir}")
                    record_dir = None
                    record_count = 0
    finally:
        pipeline.stop()
        if vis is not None:
            vis.destroy_window()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    calib = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CALIB
    sys.exit(main(calib))
