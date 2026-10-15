import numpy as np
from typing import List, Optional, Tuple, Dict, Any

class SceneFilters:
    """
    Sinf xonasi kontekstini tahlil qiluvchi va noto'g'ri signallarni
    (o'qituvchi, eshik oldida o'tayotganlar, chaqaloqli talabalar) chiqarib tashlovchi filtrlar.
    """
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("detection", {})
        self.teacher_cfg = self.cfg.get("teacher_filter", {"enabled": True, "bottom_zone_ratio": 0.30})
        self.roi_cfg = self.cfg.get("roi", {"enabled": True, "margin_pct": 0.03})
        self.baby_cfg = self.cfg.get("baby_detection", {"enabled": True})

    def update_config(self, config: Dict[str, Any]):
        self.cfg = config.get("detection", {})
        self.teacher_cfg = self.cfg.get("teacher_filter", {"enabled": True, "bottom_zone_ratio": 0.30})
        self.roi_cfg = self.cfg.get("roi", {"enabled": True, "margin_pct": 0.03})
        self.baby_cfg = self.cfg.get("baby_detection", {"enabled": True})

    def is_teacher(self, box: List[int], kpts: Optional[np.ndarray], img_h: int, img_w: int) -> bool:
        """
        O'qituvchini aniqlash (Bosqichma-bosqich sinf oldi filtri):
        1. O'qituvchi doska oldida yoki minbarda (pastki zonada) turadi.
        2. Kameraga orqa o'girib talabalarga yuzlangan (yuz ko'rinmaydi, yelkalar aniq).
        3. Bounding box balandligi va o'lchami o'tirgan talabalarnikidan ancha katta.
        """
        if not self.teacher_cfg.get("enabled", True):
            return False

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        if bw < 100 or bh < 120:
            return False

        face_conf = 0.0
        sh_conf = 0.0
        if kpts is not None and len(kpts) >= 7:
            face_conf = max(kpts[0][2], kpts[1][2], kpts[2][2])
            sh_conf = max(kpts[5][2], kpts[6][2])

        # Agar yuzi kameraga to'g'ri ko'rinib turgan bo'lsa, bu o'tirgan talaba
        if face_conf >= 0.30:
            return False

        is_facing_away = (face_conf < 0.25 and sh_conf > 0.40)
        bottom_ratio = float(self.teacher_cfg.get("bottom_zone_ratio", 0.30))
        zone_top_y = img_h * (1.0 - bottom_ratio)
        in_teacher_zone = (y2 >= zone_top_y) or (y1 >= img_h * 0.50)

        if is_facing_away and in_teacher_zone and bh > (img_h * 0.22):
            return True

        return False

    def is_edge_person(self, box: List[int], img_w: int, img_h: int) -> bool:
        """
        Kadr chetiga tegib turgan, eshikdan kirib-chiqayotgan yoki kesilib qolgan siluetlarni
        qoidabuzarlik tahlilidan chiqarish (ROI filtri).
        """
        if not self.roi_cfg.get("enabled", True):
            return False

        margin_pct = float(self.roi_cfg.get("margin_pct", 0.03))
        mx = int(img_w * margin_pct)
        my = int(img_h * margin_pct)

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        # Kadr chegaralariga tegib turganlik
        if x1 <= mx or x2 >= img_w - mx or y1 <= my or y2 >= img_h - my:
            return True

        # Juda tor yoki kesilib qolgan siluet
        if bw < 55 or bh < 65:
            return True

        return False

    def check_student_with_baby(
        self,
        student_box: List[int],
        kpts: Optional[np.ndarray],
        all_boxes: List[List[int]]
    ) -> Tuple[bool, Optional[List[int]]]:
        """
        Chaqaloqli talabalarni aniqlash:
        Talaba bag'rida/quchog'ida (ko'krak yoki tizza sohasida) kichik chaqaloq b-boxi mavjudligini tekshiradi.
        Bunday talabalar barcha qoidabuzarliklardan ozod qilinadi.
        """
        if not self.baby_cfg.get("enabled", True):
            return False, None

        s_w = student_box[2] - student_box[0]
        s_h = student_box[3] - student_box[1]

        for other_b in all_boxes:
            if other_b == student_box:
                continue
            ocx = (other_b[0] + other_b[2]) / 2.0
            ocy = (other_b[1] + other_b[3]) / 2.0
            ow = other_b[2] - other_b[0]
            oh = other_b[3] - other_b[1]

            in_lap_x = (student_box[0] + s_w * 0.10 <= ocx <= student_box[2] - s_w * 0.10)
            in_lap_y = (student_box[1] + s_h * 0.25 <= ocy <= student_box[3] - s_h * 0.05)
            is_much_smaller = (ow < s_w * 0.55) and (oh < s_h * 0.55) and ((ow * oh) < (s_w * s_h) * 0.30)

            if in_lap_x and in_lap_y and is_much_smaller:
                return True, other_b

        return False, None
