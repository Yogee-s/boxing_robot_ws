"""
Pose estimation wrappers for YOLO and MMPose.

Provides:
- BasePoseEstimator with predict() and predict_with_bbox()
- YOLOPoseEstimator (single-pass detection + pose)
- MMPoseEstimator (separate detector + pose model)
- MMPoseInferencerEstimator (high-level MMPose API)
- select_main_person() for multi-person disambiguation
- create_pose_estimator() factory
"""
import sys
from typing import List, Optional, Tuple

import numpy as np


def select_main_person(
    bboxes: np.ndarray,
    image_w: int,
    image_h: int,
) -> int:
    """Pick the boxer closest to the image center with the largest bbox.

    Uses a combined score: area / (1 + distance_to_center) so that a large,
    centered person always wins over a small person in the corner.

    Args:
        bboxes: (N, 4) array of [x1, y1, x2, y2] bounding boxes.
        image_w: Image width in pixels.
        image_h: Image height in pixels.

    Returns:
        Index of the best person (0-based).
    """
    if len(bboxes) == 0:
        return 0

    cx_img = image_w / 2.0
    cy_img = image_h / 2.0

    best_score = -1.0
    best_idx = 0

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox[:4]
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = ((cx - cx_img) ** 2 + (cy - cy_img) ** 2) ** 0.5
        score = area / (1.0 + dist)
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


class BasePoseEstimator:
    def predict(self, frame):
        """
        Predict pose for a single frame.
        Args:
            frame: RGB image (H, W, 3) numpy array
        Returns:
            keypoints: (K, 2) numpy array of pixel coordinates
            scores: (K,) numpy array of confidence scores
        """
        raise NotImplementedError

    def predict_with_bbox(self, frame):
        """
        Predict pose for a single frame, also returning the bounding box.
        Args:
            frame: RGB image (H, W, 3) numpy array
        Returns:
            keypoints: (K, 2) numpy array or None
            scores: (K,) numpy array or None
            bbox: (4,) array [x1, y1, x2, y2] or None
        """
        kp, scores = self.predict(frame)
        return kp, scores, None


class YOLOPoseEstimator(BasePoseEstimator):
    def __init__(self, weights, device='cuda:0', conf=0.15, imgsz=640):
        try:
            from ultralytics import YOLO
            import ultralytics
        except ImportError:
            raise ImportError("Missing YOLO dependencies. Install ultralytics first.")

        # Check version compatibility for YOLO26 models
        if 'yolo26' in weights.lower():
            try:
                from packaging import version
                if version.parse(ultralytics.__version__) < version.parse('8.4.0'):
                    raise RuntimeError(
                        f"YOLO26 models require ultralytics>=8.4.0, but you have {ultralytics.__version__}.\n"
                        f"Please upgrade: pip install --upgrade ultralytics\n"
                        f"Or use yolo11m-pose.pt which is compatible with your current version."
                    )
            except ImportError:
                pass

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        print(f"Loaded YOLO model: {weights} on {device}")

    def predict(self, frame):
        results = self.model(
            frame, device=self.device, conf=self.conf,
            imgsz=self.imgsz, verbose=False,
        )

        if not results:
            return None, None

        res = results[0]
        if (res.keypoints is None or res.keypoints.data is None
                or res.keypoints.data.shape[1] == 0):
            return None, None

        if len(res.keypoints.data) == 0:
            return None, None

        # Multi-person: select the main person (largest, most centered)
        h, w = frame.shape[:2]
        all_kps = res.keypoints.data.cpu().numpy()  # (N, K, 3)

        if res.boxes is not None and len(res.boxes.data) > 0:
            all_bboxes = res.boxes.data.cpu().numpy()[:, :4]  # (N, 4)
            best = select_main_person(all_bboxes, w, h)
        else:
            best = 0

        kps = all_kps[best]  # (K, 3) -> x, y, conf
        points = kps[:, :2]
        scores = kps[:, 2]
        return points, scores

    def predict_with_bbox(self, frame):
        """Single-pass YOLO: returns keypoints, scores, and bbox for main person."""
        results = self.model(
            frame, device=self.device, conf=self.conf,
            imgsz=self.imgsz, verbose=False,
        )

        if not results:
            return None, None, None

        res = results[0]
        if (res.keypoints is None or res.keypoints.data is None
                or res.keypoints.data.shape[1] == 0):
            return None, None, None

        if len(res.keypoints.data) == 0:
            return None, None, None

        h, w = frame.shape[:2]
        all_kps = res.keypoints.data.cpu().numpy()  # (N, K, 3)

        bbox = None
        best = 0
        if res.boxes is not None and len(res.boxes.data) > 0:
            all_bboxes = res.boxes.data.cpu().numpy()[:, :4]  # (N, 4)
            best = select_main_person(all_bboxes, w, h)
            bbox = all_bboxes[best].astype(np.float32)  # (4,)

        kps = all_kps[best]
        points = kps[:, :2]
        scores = kps[:, 2]
        return points, scores, bbox


class MMPoseEstimator(BasePoseEstimator):
    def __init__(self, config, checkpoint, device='cuda:0', detector_backend='yolo', detector_weights=None, detector_conf=0.25):
        try:
            from mmpose.apis import init_model, inference_topdown
            from mmpose.structures import merge_data_samples
        except ImportError:
            raise ImportError("Missing MMPose dependencies. Install mmpose first.")

        self.inference_topdown = inference_topdown
        self.merge_data_samples = merge_data_samples

        self.pose_model = init_model(config, checkpoint, device=device)
        print(f"Loaded MMPose model: {config}")

        self.detector = None
        if detector_backend == 'yolo' and detector_weights:
            from ultralytics import YOLO
            self.detector = YOLO(detector_weights)
            self.detector_conf = detector_conf
            print(f"Loaded YOLO detector: {detector_weights}")

    def predict(self, frame):
        bboxes = []
        if self.detector:
            det_results = self.detector(frame, conf=self.detector_conf, verbose=False)
            if det_results and det_results[0].boxes:
                boxes = det_results[0].boxes.data.cpu().numpy()
                person_boxes = boxes[boxes[:, 5] == 0]
                if len(person_boxes) > 0:
                    bboxes = person_boxes[:, :4]
                else:
                    return None, None
            else:
                return None, None
        else:
            h, w = frame.shape[:2]
            bboxes = [[0, 0, w, h]]

        if len(bboxes) == 0:
            return None, None

        curr_bboxes = bboxes[0:1]
        pose_results = self.inference_topdown(self.pose_model, frame, curr_bboxes)
        if not pose_results:
            return None, None

        sample = pose_results[0]
        data = sample.pred_instances
        kp = data.keypoints[0]
        scores = data.keypoint_scores[0]
        return kp, scores


class MMPoseInferencerEstimator(BasePoseEstimator):
    def __init__(self, pose2d='human', pose2d_weights=None, device='cuda:0',
                 det_model=None, det_weights=None, det_cat_ids=None):
        try:
            from mmpose.apis import MMPoseInferencer
        except ImportError:
            raise ImportError("Missing MMPose inferencer. Install mmpose>=1.0 first.")

        inferencer_kwargs = {
            'pose2d': pose2d,
            'device': device,
        }
        if pose2d_weights:
            inferencer_kwargs['pose2d_weights'] = pose2d_weights
        if det_model:
            inferencer_kwargs['det_model'] = det_model
        if det_weights:
            inferencer_kwargs['det_weights'] = det_weights
        if det_cat_ids is not None:
            inferencer_kwargs['det_cat_ids'] = det_cat_ids

        self.inferencer = MMPoseInferencer(**inferencer_kwargs)

    def _select_best_instance(self, instances):
        if not instances:
            return None
        best = None
        best_score = -1.0
        for inst in instances:
            scores = inst.get('keypoint_scores')
            if scores is None:
                continue
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best = inst
        return best

    def predict(self, frame):
        try:
            result = next(self.inferencer(frame, return_vis=False))
        except Exception:
            return None, None

        preds = result.get('predictions', [])
        if not preds:
            return None, None

        instances = preds[0] if isinstance(preds, list) and len(preds) > 0 else preds
        if isinstance(instances, dict):
            instances = [instances]

        best = self._select_best_instance(instances)
        if best is None:
            return None, None

        keypoints = np.array(best.get('keypoints', []), dtype=np.float32)
        scores = np.array(best.get('keypoint_scores', []), dtype=np.float32)
        if keypoints.size == 0:
            return None, None

        return keypoints, scores


def create_pose_estimator(backend, **kwargs):
    if backend == 'yolo':
        return YOLOPoseEstimator(
            weights=kwargs.get('yolo_weights'),
            device=kwargs.get('pose_device', 'cuda:0'),
            conf=kwargs.get('yolo_conf', 0.15),
            imgsz=kwargs.get('yolo_imgsz', 320)
        )
    elif backend == 'mmpose':
        return MMPoseEstimator(
            config=kwargs.get('pose_config'),
            checkpoint=kwargs.get('pose_checkpoint'),
            device=kwargs.get('pose_device', 'cuda:0'),
            detector_backend=kwargs.get('detector_backend', 'yolo'),
            detector_weights=kwargs.get('detector_weights'),
            detector_conf=kwargs.get('detector_conf', 0.25)
        )
    elif backend == 'rtmpose':
        det_cat_ids = kwargs.get('det_cat_ids')
        if isinstance(det_cat_ids, str):
            det_cat_ids = [int(x) for x in det_cat_ids.split(',') if x.strip()]
        return MMPoseInferencerEstimator(
            pose2d=kwargs.get('pose2d', 'human'),
            pose2d_weights=kwargs.get('pose2d_weights'),
            device=kwargs.get('pose_device', 'cuda:0'),
            det_model=kwargs.get('det_model'),
            det_weights=kwargs.get('det_weights'),
            det_cat_ids=det_cat_ids
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
