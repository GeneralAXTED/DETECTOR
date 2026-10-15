import math
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from core.analyzers.base import TrackedStudent

class PhoneAnalyzer:
    """
    4-Bosqichli Ko'p Darajali Smartfon va Telefon Ishlatish Tahlilchisi:
    Bosqich 1: YOLO Obyekt + Biometrik Qo'l/Tana Geometriyasi
    Bosqich 2: Vaqtinchalik Barqarorlik (3 soniyalik uzluksiz tutib turish)
    Bosqich 3: Neyron Tarmoq va Vizual Qirqma Tasnifi
    Bosqich 4: Iyak, Quloq Qashlash va Orqa Qator To'siqlaridan Himoya
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("detection", {}).get("phone", {})
        self.enabled = self.cfg.get("enabled", True)
        self.min_duration = float(self.cfg.get("min_duration_seconds", 3.0))

    def update_config(self, config: Dict[str, Any]):
        self.cfg = config.get("detection", {}).get("phone", {})
        self.enabled = self.cfg.get("enabled", True)
        self.min_duration = float(self.cfg.get("min_duration_seconds", 3.0))

    def check_phone_posture(
        self,
        person_box: List[int],
        kpts: Optional[np.ndarray],
        phones_boxes: List[List[int]],
        is_head_active: bool = False
    ) -> Tuple[bool, float, str, Dict[str, float]]:
        """
        Bosqich 1 & 4: Telefon bilan bog'liq geometrik belgilarni tekshirish.
        Qaytaradi: (has_phone_posture, geom_conf, description, stage_scores)
        """
        stage_scores = {"geom": 0.0, "object_detected": 0.0, "two_hand": 0.0, "ear_call": 0.0}
        if not self.enabled or kpts is None or len(kpts) < 11:
            return False, 0.0, "", stage_scores

        px1, py1, px2, py2 = person_box
        bw = px2 - px1
        bh = py2 - py1

        lw = kpts[9]
        rw = kpts[10]
        l_el = kpts[7] if len(kpts) > 7 else [0, 0, 0]
        r_el = kpts[8] if len(kpts) > 8 else [0, 0, 0]
        nose = kpts[0]
        l_ear = kpts[3] if len(kpts) > 3 else [0, 0, 0]
        r_ear = kpts[4] if len(kpts) > 4 else [0, 0, 0]
        l_sh = kpts[5] if len(kpts) > 5 else [0, 0, 0]
        r_sh = kpts[6] if len(kpts) > 6 else [0, 0, 0]

        lw_val = lw[2] > 0.20
        rw_val = rw[2] > 0.20

        valid_sh_y = [p[1] for p in (l_sh, r_sh) if p[2] > 0.15]
        avg_sh_y = sum(valid_sh_y) / len(valid_sh_y) if valid_sh_y else (py1 + bh * 0.30)
        head_is_upright = (nose[2] > 0.15 and (avg_sh_y - nose[1]) > 4.0)

        # -------------------------------------------------------------------------
        # 1. TELEFONDA GAPLASHISH (Qo'ng'iroq - Call to Ear)
        # -------------------------------------------------------------------------
        if head_is_upright and phones_boxes:
            for ear, wrist, is_w_val in [(l_ear, lw, lw_val), (r_ear, rw, rw_val)]:
                if ear[2] > 0.20 and is_w_val:
                    dist_ear = math.hypot(wrist[0] - ear[0], wrist[1] - ear[1])
                    if dist_ear < max(35, bh * 0.18):
                        is_at_ear_level = abs(wrist[1] - ear[1]) < max(25, bh * 0.12)
                        if is_at_ear_level:
                            for ph in phones_boxes:
                                phcx = (ph[0] + ph[2]) / 2.0
                                phcy = (ph[1] + ph[3]) / 2.0
                                if math.hypot(wrist[0] - phcx, wrist[1] - phcy) < max(45, bh * 0.22):
                                    stage_scores["geom"] = 0.95
                                    stage_scores["ear_call"] = 1.0
                                    stage_scores["object_detected"] = 1.0
                                    return True, 0.95, "Telefonda gaplashayotgan talaba (Quloqqa qo'yilgan)", stage_scores

        # -------------------------------------------------------------------------
        # 2. PARTA USTIDA IKKI QO'LDA TELEFON USHLAB O'YNASH / YOZISH
        # -------------------------------------------------------------------------
        if lw_val and rw_val and nose[2] > 0.15:
            w_dist = math.hypot(lw[0] - rw[0], lw[1] - rw[1])
            hands_cx = (lw[0] + rw[0]) / 2.0
            hands_cy = (lw[1] + rw[1]) / 2.0

            wrists_together = w_dist < max(45, bw * 0.40)
            hands_in_center = (px1 + bw * 0.08 <= hands_cx <= px2 - bw * 0.08)

            # A) Qo'llar yelkadan pastda (parta taxtasi tekisligida) bo'lishi shart
            hands_down_on_desk = (lw[1] > avg_sh_y + max(25, bh * 0.18)) and (rw[1] > avg_sh_y + max(25, bh * 0.18))

            # B) Qo'llar iyak/burundan pastda, partada bo'lishi shart (iyak yonida qo'l tutishdan 100% himoya)
            vertical_drop_from_nose = hands_cy - nose[1]
            not_at_chin_level = vertical_drop_from_nose >= max(60, bh * 0.26)

            # C) Tirsaklar bilan taqqoslash (iyakda ushlanganda bilaklar tirsakdan tik yuqoriga qaragan bo'ladi)
            valid_el_y = [p[1] for p in (l_el, r_el) if p[2] > 0.15]
            avg_el_y = sum(valid_el_y) / len(valid_el_y) if valid_el_y else 0
            elbow_posture_ok = True
            if valid_el_y and hands_cy < avg_el_y - 20:
                elbow_posture_ok = False

            # D) Bosh to'g'ridan-to'g'ri qo'llarga (telefonga) qaratilgan
            dist_nose_to_hands = math.hypot(nose[0] - hands_cx, nose[1] - hands_cy)
            looking_at_hands = (dist_nose_to_hands < max(80, bh * 0.60)) and (abs(nose[0] - hands_cx) < bw * 0.35)

            # E) Iyakka va yuzga tegilmaganligi
            dist_chin_hands = math.hypot(nose[0] - hands_cx, hands_cy - nose[1])
            not_touching_face = dist_chin_hands >= max(55, bh * 0.24)

            # F) Quloqqa tegilmaganligi
            not_touching_ears = True
            for ear in (l_ear, r_ear):
                if ear[2] > 0.18:
                    if math.hypot(lw[0] - ear[0], lw[1] - ear[1]) < 35 or math.hypot(rw[0] - ear[0], rw[1] - ear[1]) < 35:
                        not_touching_ears = False
                        break

            # G) Telefon qutisi qo'llar yaqinida ko'rinsa
            phone_near_hands = False
            if phones_boxes:
                for ph in phones_boxes:
                    phcx = (ph[0] + ph[2]) / 2.0
                    phcy = (ph[1] + ph[3]) / 2.0
                    if math.hypot(hands_cx - phcx, hands_cy - phcy) < max(55, bh * 0.30):
                        phone_near_hands = True
                        break

            if phone_near_hands and wrists_together and hands_in_center and looking_at_hands:
                stage_scores["geom"] = 0.98
                stage_scores["two_hand"] = 1.0
                stage_scores["object_detected"] = 1.0
                return True, 0.98, "Smartfon / Telefon ishlatayotgan talaba (Ikki qo'lda)", stage_scores

            # Telefon kaftlar ichida berkitilgan bo'lsa (faqat va faqat partada bo'lgandagina!)
            if (wrists_together and hands_in_center and hands_down_on_desk and 
                not_at_chin_level and elbow_posture_ok and looking_at_hands and 
                not_touching_face and not_touching_ears):
                stage_scores["geom"] = 0.92
                stage_scores["two_hand"] = 1.0
                return True, 0.92, "Parta ustida ikki qo'lda telefon o'ynayotgan talaba", stage_scores

        # -------------------------------------------------------------------------
        # 3. KADRDA ANIKLANGAN TELEFON OBYEKTI BILAN ISHLASH (Bir qo'lda yoki partada)
        # -------------------------------------------------------------------------
        if phones_boxes:
            for ph in phones_boxes:
                phx1, phy1, phx2, phy2 = ph
                phw = phx2 - phx1
                phh = phy2 - phy1

                aspect = max(phw, phh) / max(1, min(phw, phh))
                if aspect < 1.15 or aspect > 3.5:
                    continue
                if phw > bw * 0.45 or phh > bh * 0.40:
                    continue

                phcx = (phx1 + phx2) / 2.0
                phcy = (phy1 + phy2) / 2.0

                if not ((px1 - bw * 0.18 <= phcx <= px2 + bw * 0.18) and (py1 <= phcy <= py2 + bh * 0.20)):
                    continue

                dist_lw = math.hypot(lw[0] - phcx, lw[1] - phcy) if lw_val else 9999
                dist_rw = math.hypot(rw[0] - phcx, rw[1] - phcy) if rw_val else 9999
                min_wrist = min(dist_lw, dist_rw)
                dist_nose = math.hypot(nose[0] - phcx, nose[1] - phcy) if nose[2] > 0.15 else 9999

                # A) Qo'llardan biri telefon yonida / ushlab turibdi
                if min_wrist <= max(45, min(bw, bh) * 0.32):
                    if dist_nose < bh * 0.55:
                        stage_scores["geom"] = 0.95
                        stage_scores["object_detected"] = 1.0
                        return True, 0.95, "Smartfon / Telefon ishlatayotgan talaba", stage_scores

                # B) Parta ostida / tizzada telefon va talaba unga qarab o'tiribdi
                if phcy > (py1 + bh * 0.38) and dist_nose < bh * 0.65:
                    stage_scores["geom"] = 0.92
                    stage_scores["object_detected"] = 1.0
                    return True, 0.92, "Parta ostida telefon ishlatayotgan talaba", stage_scores

        return False, 0.0, "", stage_scores

    def verify_student_phone(
        self,
        student: TrackedStudent,
        person_box: List[int],
        kpts: Optional[np.ndarray],
        phones_boxes: List[List[int]],
        student_crop: Optional[np.ndarray],
        classifier,
        now: float,
        is_sleeping: bool
    ) -> Tuple[bool, float, str, Dict[str, float]]:
        """
        Barcha 4 bosqichni birlashtiruvchi to'liq tekshirish funksiyasi.
        Qaytaradi: (is_confirmed_violation, final_confidence, description, stage_scores)
        """
        # Mutual Exclusion: Agar uxlash deb belgilangan bo'lsa, telefon tekshirilmaydi
        if is_sleeping:
            student.phone_start_time = None
            return False, 0.0, "", {}

        head_is_moving = student.is_head_active(now, window_sec=5.0)
        has_posture, geom_conf, desc, stage_scores = self.check_phone_posture(
            person_box, kpts, phones_boxes, is_head_active=head_is_moving
        )

        if not has_posture:
            student.phone_start_time = None
            return False, 0.0, "", stage_scores

        # Bosqich 2: Vaqtinchalik uzluksizlik (3 soniyadan ortiq davom etishi shart)
        if student.phone_start_time is None:
            student.phone_start_time = now
            return False, 0.0, desc, stage_scores

        elapsed = now - student.phone_start_time
        temporal_score = min(1.0, elapsed / self.min_duration)
        stage_scores["temporal"] = temporal_score

        if elapsed < self.min_duration:
            return False, 0.0, desc, stage_scores

        # Bosqich 3: Maxsus AI klassifikator tekshiruvi
        verified, ai_conf = classifier.verify_violation(student_crop, "PHONE")
        stage_scores["classifier"] = ai_conf if verified else 0.5

        if not verified:
            return False, 0.0, desc, stage_scores

        final_conf = 0.55 * geom_conf + 0.25 * temporal_score + 0.20 * stage_scores["classifier"]
        return True, round(final_conf, 2), desc, stage_scores
