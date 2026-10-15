import logging
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ultralytics import YOLO

logger = logging.getLogger("AIDetector")

class SmartDetector:
    """YOLO Pose va Object Detection modellarini yuqori tezlikda boshqaruvchi klass."""

    def __init__(self, device: str = "cuda", conf_thresh: float = 0.20, imgsz: int = 960):
        # Qurilmani aniqlash
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA topilmadi, CPU ga o'tkazilmoqda.")
            self.device = "cpu"
        else:
            self.device = device

        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        logger.info(f"AI Model yuklanmoqda... Qurilma: {self.device}, ImgSize: {self.imgsz}")

        # CUDA cuDNN benchmark tezlatkichini faollashtirish
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        # Kichik va uzoqdagi obyektlar uchun yolo11s (Small) modeli
        try:
            self.pose_model = YOLO("yolo11s-pose.pt")
            self.obj_model = YOLO("yolo11s.pt")
        except Exception as e:
            logger.warning(f"yolo11s yuklashda xatolik, yolo11n ga o'tilmoqda: {e}")
            self.pose_model = YOLO("yolo11n-pose.pt")
            self.obj_model = YOLO("yolo11n.pt")

        # Modellarni qurilmaga ko'chirish
        self.pose_model.to(self.device)
        self.obj_model.to(self.device)

        # GPU TensorCores (FP16 Half precision) faollashtirish
        if self.device == "cuda":
            try:
                self.pose_model.model.half()
                self.obj_model.model.half()
                logger.info("YOLO modellari FP16 (TensorCore) rejimiga o'tkazildi.")
            except Exception as e:
                logger.warning(f"FP16 ga o'tishda ogohlantirish: {e}")

        logger.info("YOLO modellari muvaffaqiyatli yuklandi.")

        self._cam_frame_counts: Dict[str, int] = {}
        self._cached_phones: Dict[str, List[List[int]]] = {}
        self._total_inferences = 0

    def process_frame(
        self, frame: np.ndarray, cam_id: str = "default"
    ) -> Tuple[List[List[int]], List[np.ndarray], List[List[int]]]:
        """
        Kadrni tahlil qilib:
        - persons_boxes: [[x1, y1, x2, y2], ...]
        - persons_keypoints: [np.ndarray(17, 3), ...]
        - phones_boxes: [[x1, y1, x2, y2], ...]
        qaytaradi.
        """
        count = self._cam_frame_counts.get(cam_id, 0) + 1
        self._cam_frame_counts[cam_id] = count
        self._total_inferences += 1

        # Har 250 ta tahlildan so'ng GPU xotirasini tozalash (VRAM fragmentatsiyasining oldini oladi)
        if self.device == "cuda" and (self._total_inferences % 250 == 0):
            torch.cuda.empty_cache()

        persons_boxes: List[List[int]] = []
        persons_keypoints: List[np.ndarray] = []

        with torch.inference_mode():
            # 1. Pose Model: Talabalar va tana/bosh nuqtalari (har bir AI kadrida)
            pose_results = self.pose_model.predict(
                source=frame,
                device=self.device,
                imgsz=self.imgsz,
                conf=min(self.conf_thresh, 0.20),  # Partaga yotgan/uxlayotgan talabalarni o'tkazib yubormaslik uchun
                classes=[0],  # 0: person
                verbose=False
            )

            if pose_results and len(pose_results) > 0:
                res = pose_results[0]
                if res.boxes is not None and res.keypoints is not None:
                    boxes_coords = res.boxes.xyxy.cpu().numpy().astype(int)
                    kpts_data = res.keypoints.data.cpu().numpy()  # (N, 17, 3)

                    for i, box in enumerate(boxes_coords):
                        persons_boxes.append(box.tolist())
                        persons_keypoints.append(kpts_data[i])

            # 2. Object Model (Telefon va Qo'shimcha Odamlar):
            # Agar talaba partaga butunlay yotib olgan bo'lsa va pose modeli uni o'tkazib yuborsa,
            # standart detektor (obj_model) uni 'person' sifatida tutib beradi.
            cached = self._cached_phones.get(cam_id, [])
            if count % 2 == 1 or not cached:
                obj_results = self.obj_model.predict(
                    source=frame,
                    device=self.device,
                    imgsz=self.imgsz,
                    conf=0.20,
                    classes=[0, 67],  # 0: person, 67: cell phone
                    verbose=False
                )

                phones_boxes = []
                if obj_results and len(obj_results) > 0:
                    res_obj = obj_results[0]
                    if res_obj.boxes is not None:
                        boxes = res_obj.boxes.xyxy.cpu().numpy().astype(int)
                        confs = res_obj.boxes.conf.cpu().numpy()
                        cls_ids = res_obj.boxes.cls.cpu().numpy().astype(int)

                        for idx, b in enumerate(boxes):
                            c_id = cls_ids[idx]
                            c_conf = confs[idx]

                            if c_id == 67 and c_conf >= 0.22:
                                phones_boxes.append(b.tolist())
                            elif c_id == 0 and c_conf >= 0.22:
                                # Pose modeli o'tkazib yuborgan partadagi odam bormi?
                                is_duplicate = False
                                for p_b in persons_boxes:
                                    ix1 = max(b[0], p_b[0])
                                    iy1 = max(b[1], p_b[1])
                                    ix2 = min(b[2], p_b[2])
                                    iy2 = min(b[3], p_b[3])
                                    iw = max(0, ix2 - ix1)
                                    ih = max(0, iy2 - iy1)
                                    inter = iw * ih
                                    area1 = (b[2] - b[0]) * (b[3] - b[1])
                                    area2 = (p_b[2] - p_b[0]) * (p_b[3] - p_b[1])
                                    union = area1 + area2 - inter
                                    if union > 0 and (inter / union) > 0.30:
                                        is_duplicate = True
                                        break
                                if not is_duplicate:
                                    persons_boxes.append(b.tolist())
                                    persons_keypoints.append(np.zeros((17, 3), dtype=np.float32))

                self._cached_phones[cam_id] = phones_boxes
            else:
                phones_boxes = cached

        return persons_boxes, persons_keypoints, phones_boxes
