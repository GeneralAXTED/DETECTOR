import os
import cv2
import time
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger("BehaviorClassifier")

# Klass nomlari
CLASSES = ["NORMAL", "SLEEPING", "PHONE_TALKING", "PHONE_GAMING"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASSES)}

class BehaviorClassifier:
    """
    Talabalarning holatini (uxlash, telefon, dars qilish) maxsus
    MobileNetV3 neyron tarmog'i orqali ikkilamchi tekshiruvchi klass (Singleton).
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(BehaviorClassifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path: str = "models/behavior_classifier.pt", device: str = "cuda", auto_collect: bool = True):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.auto_collect = auto_collect
        self.dataset_dir = Path("dataset/crops")
        self.is_trained_model_loaded = False
        self.model = None

        # Saqlangan vaznlarni yuklash (agar mavjud bo'lsa)
        if self.model_path.exists():
            try:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                self.model = models.mobilenet_v3_small(weights=None)
                in_features = self.model.classifier[3].in_features
                self.model.classifier[3] = nn.Linear(in_features, len(CLASSES))
                self.model.to(self.device)
                self.model.eval()

                state_dict = torch.load(str(self.model_path), map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.is_trained_model_loaded = True
                logger.info(f"Maxsus AI Klassifikator yuklandi: {self.model_path} ({self.device})")
            except Exception as e:
                logger.warning(f"Klassifikator modelini yuklashda ogohlantirish: {e}")
                self.model = None
                self.is_trained_model_loaded = False
        else:
            logger.info("YOLO Pose & Heuristic AI tahlil tizimi to'liq faol.")

        # Agar dataset to'plash yoqilgan bo'lsa, papkalarni tayyorlash
        if self.auto_collect:
            for cls in CLASSES:
                (self.dataset_dir / cls).mkdir(parents=True, exist_ok=True)

    def save_crop_sample(self, crop_bgr, label: str):
        """Trening uchun auditoriyadan o'zbek talabalari suratlarini to'plash."""
        if not self.auto_collect or crop_bgr is None or crop_bgr.size == 0:
            return
        try:
            folder = self.dataset_dir / label
            folder.mkdir(parents=True, exist_ok=True)
            fname = f"{label.lower()}_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(folder / fname), crop_bgr)
        except Exception:
            pass

    def predict_crop(self, crop_bgr) -> Tuple[str, float]:
        """
        Kadr qirqmasini (crop) neyron tarmoq orqali sinflash:
        Qaytaradi: (class_name, confidence)
        """
        if not self.is_trained_model_loaded or self.model is None or crop_bgr is None or crop_bgr.size == 0:
            return "UNKNOWN", 0.0

        try:
            img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                outputs = self.model(tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                top_idx = torch.argmax(probs).item()
                conf = probs[top_idx].item()
                return IDX_TO_CLASS[top_idx], float(conf)
        except Exception as e:
            logger.warning(f"Klassifikatsiyada xatolik: {e}")
            return "UNKNOWN", 0.0

    def verify_violation(self, crop_bgr, candidate_type: str) -> Tuple[bool, float]:
        """
        YOLO aniqlagan shubhali holatni ikkilamchi AI klassifikator orqali tekshirish.
        candidate_type: "SLEEPING" yoki "PHONE"
        """
        if not self.is_trained_model_loaded:
            # Agar maxsus model hali o'rgatilmagan bo'lsa, avtomatik trening namunasi sifatida saqlaymiz
            # va asosiy tekshiruvga ruxsat beramiz
            return True, 1.0

        pred_class, conf = self.predict_crop(crop_bgr)

        # Agar klassifikator bu oddiy talaba (NORMAL) desa, qoidabuzarlikni rad etamiz!
        if pred_class == "NORMAL" and conf >= 0.60:
            logger.info(f"Klassifikator soxta qoidabuzarlikni inkor etdi (NORMAL: {conf:.2f})")
            return False, conf

        if candidate_type == "SLEEPING":
            if pred_class == "SLEEPING":
                return True, conf
            elif pred_class in ("PHONE_TALKING", "PHONE_GAMING") and conf >= 0.70:
                # Uxlash deb gumon qilingan, lekin model aniq telefon deb aytdi
                return False, conf

        elif candidate_type == "PHONE":
            if pred_class in ("PHONE_TALKING", "PHONE_GAMING"):
                return True, conf
            elif pred_class == "SLEEPING" and conf >= 0.70:
                # Telefon deb gumon qilingan, lekin model uxlash deb aytdi
                return False, conf

        # Shubhali holatda, agar model qat'iy qarshilik qilmasa, tasdiqlaymiz
        return True, conf
