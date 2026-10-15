import math
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from core.analyzers.base import TrackedStudent

class SleepingAnalyzer:
    """
    4-Bosqichli Ko'p Darajali Uxlash Tahlilchisi:
    Bosqich 1: Biometrik Geometriya (Bosh, Quloq, Yelka nisbatlari)
    Bosqich 2: Vaqtinchalik Barqarorlik (3-5 soniya uzluksiz bosh harakatsizligi / Dispersiya)
    Bosqich 3: Vizual Ko'rinish va Neyron Tarmoq Tasnifi
    Bosqich 4: Parta va Yozish Holatlarini Filtrlash (Soxta signallardan 100% himoya)
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("detection", {}).get("sleeping", {})
        self.enabled = self.cfg.get("enabled", True)
        self.min_duration = float(self.cfg.get("min_duration_seconds", 6.0))

    def update_config(self, config: Dict[str, Any]):
        self.cfg = config.get("detection", {}).get("sleeping", {})
        self.enabled = self.cfg.get("enabled", True)
        self.min_duration = float(self.cfg.get("min_duration_seconds", 6.0))

    def check_sleeping_posture(
        self,
        kpts: Optional[np.ndarray],
        box: List[int],
        img_h: int,
        img_w: int
    ) -> Tuple[bool, float, Optional[Tuple[float, float]], Dict[str, float]]:
        """
        Bosqich 1 & 4: Geometrik va anatomik holatni tekshirish.
        Qaytaradi: (is_sleeping_posture, geom_confidence, head_pos, stage_scores)
        """
        stage_scores = {"geom": 0.0, "ears_upright": 0.0, "face_visible": 0.0, "head_drop": 0.0}
        if not self.enabled or kpts is None or len(kpts) < 7:
            return False, 0.0, None, stage_scores

        bh = box[3] - box[1]
        bw = box[2] - box[0]

        # 1. Yelka nuqtalari
        l_sh = kpts[5]
        r_sh = kpts[6]
        valid_sh_y = [p[1] for p in (l_sh, r_sh) if p[2] > 0.15]
        avg_sh_y = sum(valid_sh_y) / len(valid_sh_y) if valid_sh_y else (box[1] + bh * 0.30)

        # 2. Bosh va yuz nuqtalari
        head_pts = [kpts[idx] for idx in range(5) if kpts[idx][2] > 0.15]
        nose = kpts[0]
        l_eye = kpts[1]
        r_eye = kpts[2]
        l_ear = kpts[3]
        r_ear = kpts[4]

        valid_ears_y = [p[1] for p in (l_ear, r_ear) if p[2] > 0.18]
        face_conf = max(nose[2], l_eye[2], r_eye[2])

        if head_pts:
            avg_hx = sum(p[0] for p in head_pts) / len(head_pts)
            avg_hy = sum(p[1] for p in head_pts) / len(head_pts)
            head_pos = (avg_hx, avg_hy)
            head_is_explicit = True
        else:
            avg_hx = (box[0] + box[2]) / 2.0
            avg_hy = avg_sh_y + 10.0
            head_pos = (avg_hx, avg_hy)
            head_is_explicit = False

        avg_head_y = head_pos[1]
        head_diff = avg_sh_y - avg_head_y  # > 0: bosh yelkadan balandda, < 0: bosh yelkadan pastda

        wrists = [kpts[idx] for idx in (9, 10) if len(kpts) > idx and kpts[idx][2] > 0.15]

        # --- QAT'IY FILTR 1: Quloqlar yelkadan balandda bo'lsa -> UXLASH EMAS! ---
        # Tepadan qarovchi kamerada talaba daftarga qaraganida ham quloqlari yelkasidan yuqorida bo'ladi
        if valid_ears_y:
            min_ear_y = min(valid_ears_y)
            if (avg_sh_y - min_ear_y) >= 4.0:
                stage_scores["ears_upright"] = 1.0
                return False, 0.0, head_pos, stage_scores

        # --- QAT'IY FILTR 2: Yuz kameraga aniq ko'rinsa va bosh tik bo'lsa -> UXLASH EMAS! ---
        if face_conf >= 0.22 and head_diff >= -6.0:
            stage_scores["face_visible"] = 1.0
            return False, 0.0, head_pos, stage_scores

        # --- QAT'IY FILTR 3: Bosh tana balandligining yuqori qismida joylashgan bo'lsa -> UXLASH EMAS! ---
        if head_is_explicit and (avg_head_y - box[1]) < bh * 0.28 and (avg_sh_y - box[1]) >= bh * 0.14:
            if head_diff >= -4.0:
                return False, 0.0, head_pos, stage_scores

        # --- QAT'IY FILTR 4: Qo'llardan balandda bo'lsa (yozish / o'qish masofasi) -> UXLASH EMAS! ---
        if wrists and head_is_explicit:
            min_wrist_dist = min(math.hypot(w[0] - head_pos[0], w[1] - head_pos[1]) for w in wrists)
            if min_wrist_dist > max(45, bh * 0.22):
                return False, 0.0, head_pos, stage_scores

        # =========================================================================
        # HAQIQIY UXLASH MEZONLARI
        # =========================================================================
        # 1. BOSHNI PARTAGA QO'YIB UXLASH (Yuzi ko'rinmaydi, yelka/qo'llar partada)
        if not head_is_explicit or face_conf < 0.18:
            ears_not_upright = (not valid_ears_y) or (min(valid_ears_y) >= avg_sh_y - 2.0)
            if ears_not_upright:
                stage_scores["geom"] = 0.95
                stage_scores["head_drop"] = 1.0
                return True, 0.95, head_pos, stage_scores

        # 2. BOSHNI QO'LLARGA / BILAKLARGA QO'YIB UXLASH
        if wrists:
            min_wrist_dist = min(math.hypot(w[0] - head_pos[0], w[1] - head_pos[1]) for w in wrists)
            if min_wrist_dist < max(35, bh * 0.18) and face_conf < 0.22:
                stage_scores["geom"] = 0.92
                stage_scores["head_drop"] = 0.9
                return True, 0.92, head_pos, stage_scores

        # 3. PARTA USTIGA BUTUNLAY YOTIB OLISH (Bosh chuqur tushgan)
        if head_diff < -12.0 and face_conf < 0.20:
            stage_scores["geom"] = 0.90
            stage_scores["head_drop"] = 0.95
            return True, 0.90, head_pos, stage_scores

        return False, 0.0, head_pos, stage_scores

    def verify_student_sleeping(
        self,
        student: TrackedStudent,
        kpts: Optional[np.ndarray],
        box: List[int],
        img_h: int,
        img_w: int,
        student_crop: Optional[np.ndarray],
        classifier,
        now: float
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Barcha 4 bosqichni birlashtiruvchi to'liq tekshirish funksiyasi.
        Qaytaradi: (is_confirmed_violation, final_confidence, stage_scores)
        """
        is_posture, geom_conf, head_pos, stage_scores = self.check_sleeping_posture(kpts, box, img_h, img_w)
        if head_pos is not None:
            student.record_head_pos(now, head_pos[0], head_pos[1])

        if not is_posture:
            student.sleep_start_time = None
            return False, 0.0, stage_scores

        # Bosqich 2: Harakat tahlili (talaba daftarga yozayotganda boshi muntazam qimirlaydi)
        head_moving = student.is_head_active(now, window_sec=5.0)
        if head_moving:
            stage_scores["temporal_motion_rejection"] = 1.0
            student.sleep_start_time = None
            return False, 0.0, stage_scores

        # Vaqtinchalik uzluksizlik taymeri
        if student.sleep_start_time is None:
            student.sleep_start_time = now
            return False, 0.0, stage_scores

        elapsed = now - student.sleep_start_time
        temporal_score = min(1.0, elapsed / self.min_duration)
        stage_scores["temporal"] = temporal_score

        if elapsed < self.min_duration:
            return False, 0.0, stage_scores

        # Bosqich 3: Maxsus AI klassifikator tekshiruvi (agar mavjud bo'lsa)
        verified, ai_conf = classifier.verify_violation(student_crop, "SLEEPING")
        stage_scores["classifier"] = ai_conf if verified else 0.5

        if not verified:
            return False, 0.0, stage_scores

        # Bosqich 1, 2, 3 birlashgan kompozit ishonchlilik bali
        final_conf = 0.50 * geom_conf + 0.30 * temporal_score + 0.20 * stage_scores["classifier"]
        return True, round(final_conf, 2), stage_scores
