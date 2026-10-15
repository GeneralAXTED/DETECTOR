import math
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from core.analyzers.base import TrackedStudent, ViolationEvent

class TalkingAnalyzer:
    """
    4-Bosqichli Ko'p Darajali Orqaga O'girilib Gaplashish Tahlilchisi:
    Bosqich 1: Bosh/Gaze Yo'nalishi va Yelka Asimmetriyasi
    Bosqich 2: Parta Qatorlari Tahlili (Yonidagi sherik ruxsat etiladi, faqat orqadagilar taqiqlanadi)
    Bosqich 3: Suhbatdoshning Aniq Reaksiyasi (Orqadagi talaba ham suhbatdami?)
    Bosqich 4: Vaqtinchalik Barqarorlik (4-5 soniya uzluksiz orqaga qarab turish)
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("detection", {}).get("talking", {})
        self.enabled = self.cfg.get("enabled", True)
        self.min_duration = float(self.cfg.get("min_duration_seconds", 4.5))

    def update_config(self, config: Dict[str, Any]):
        self.cfg = config.get("detection", {}).get("talking", {})
        self.enabled = self.cfg.get("enabled", True)
        self.min_duration = float(self.cfg.get("min_duration_seconds", 4.5))

    def is_facing_front(self, kpts: Optional[np.ndarray]) -> bool:
        """Doskaga / kameraga to'g'ri qarab turgan talabani aniqlash (100% orqaga qaramagan)."""
        if kpts is None or len(kpts) < 5:
            return False
        nose = kpts[0]
        l_eye = kpts[1]
        r_eye = kpts[2]
        if nose[2] >= 0.20 and l_eye[2] >= 0.20 and r_eye[2] >= 0.20:
            if l_eye[0] > r_eye[0]:
                return True
        return False

    def is_looking_down_at_desk(self, kpts: Optional[np.ndarray], box: List[int]) -> bool:
        """Daftarga konspekt yozayotgan yoki kitob o'qiyotgan talabani aniqlash."""
        if kpts is None or len(kpts) < 7:
            return False
        bh = box[3] - box[1]
        l_sh = kpts[5]
        r_sh = kpts[6]
        valid_sh_y = [p[1] for p in (l_sh, r_sh) if p[2] > 0.15]
        if not valid_sh_y:
            return False
        avg_sh_y = sum(valid_sh_y) / len(valid_sh_y)

        nose = kpts[0]
        l_eye = kpts[1]
        r_eye = kpts[2]

        wrists_y = [kpts[idx][1] for idx in (9, 10) if len(kpts) > idx and kpts[idx][2] > 0.15]
        wrists_on_desk = any(wy > (box[1] + bh * 0.40) for wy in wrists_y)

        if nose[2] > 0.15:
            eye_y = [p[1] for p in (l_eye, r_eye) if p[2] > 0.15]
            if eye_y and (nose[1] - sum(eye_y) / len(eye_y)) > 8:
                if wrists_on_desk or (avg_sh_y - nose[1]) < bh * 0.15:
                    return True
        else:
            head_pts = [kpts[idx] for idx in range(5) if kpts[idx][2] > 0.15]
            if not head_pts:
                return True
            avg_hy = sum(p[1] for p in head_pts) / len(head_pts)
            if (avg_sh_y - avg_hy) < bh * 0.12:
                return True

        return False

    def is_turned_towards_rear_peer(
        self,
        kpts1: np.ndarray,
        box1: List[int],
        box2: List[int]
    ) -> Tuple[bool, float]:
        """Oldindagi talaba (S1) orqadagi partadagi talaba (S2) tomon orqaga o'girilganligini geometrik tekshirish."""
        if kpts1 is None or len(kpts1) < 7:
            return False, 0.0

        bh1 = box1[3] - box1[1]
        cx1 = (box1[0] + box1[2]) / 2.0
        cy1 = (box1[1] + box1[3]) / 2.0
        cx2 = (box2[0] + box2[2]) / 2.0
        cy2 = (box2[1] + box2[3]) / 2.0

        vert_dist = cy1 - cy2
        if vert_dist < bh1 * 0.28 or vert_dist > bh1 * 1.80:
            return False, 0.0

        dx = cx2 - cx1
        if abs(dx) > bh1 * 1.25:
            return False, 0.0

        nose = kpts1[0]
        l_eye = kpts1[1]
        r_eye = kpts1[2]
        l_ear = kpts1[3]
        r_ear = kpts1[4]
        l_sh = kpts1[5]
        r_sh = kpts1[6]

        if nose[2] >= 0.22 and l_eye[2] >= 0.22 and r_eye[2] >= 0.22:
            return False, 0.0

        valid_sh_y = [p[1] for p in (l_sh, r_sh) if p[2] > 0.15]
        if not valid_sh_y:
            return False, 0.0
        avg_sh_y = sum(valid_sh_y) / len(valid_sh_y)

        head_pts = [kpts1[idx] for idx in range(5) if kpts1[idx][2] > 0.15]
        avg_hy = sum(p[1] for p in head_pts) / len(head_pts) if head_pts else float(box1[1] + 15)

        if (avg_sh_y - avg_hy) < 4.0:
            return False, 0.0

        face_conf = max(nose[2], l_eye[2], r_eye[2])
        sh_conf = min(l_sh[2], r_sh[2])

        # Tanasi bilan butunlay orqaga o'girilish
        if face_conf < 0.20 and sh_conf > 0.35:
            if (avg_sh_y - avg_hy) >= 10.0:
                if l_ear[2] > 0.20 and r_ear[2] > 0.20 and l_ear[0] < r_ear[0]:
                    return True, 0.95
                return True, 0.90

        # Boshni yelka ustidan orqadagi S2 tomon burish
        if dx > bh1 * 0.15:
            sh_right = max(l_sh[0], r_sh[0])
            if nose[2] >= 0.20 and (nose[0] >= sh_right - 10) and (l_eye[2] < 0.20):
                return True, 0.92
            if r_ear[2] >= 0.25 and l_ear[2] < 0.15 and l_eye[2] < 0.15:
                return True, 0.90
        elif dx < -bh1 * 0.15:
            sh_left = min(l_sh[0], r_sh[0])
            if nose[2] >= 0.20 and (nose[0] <= sh_left + 10) and (r_eye[2] < 0.20):
                return True, 0.92
            if l_ear[2] >= 0.25 and r_ear[2] < 0.15 and r_eye[2] < 0.15:
                return True, 0.90
        else:
            if face_conf < 0.20 and (avg_sh_y - avg_hy) >= 10.0:
                return True, 0.90

        return False, 0.0

    def is_rear_peer_attentive(
        self,
        kpts2: Optional[np.ndarray],
        box2: List[int],
        s2: TrackedStudent,
        now: float
    ) -> bool:
        """Orqadagi talaba (S2) suhbatda ishtirok etyaptimi?"""
        if kpts2 is None or len(kpts2) < 7:
            return True
        if s2.sleep_start_time is not None:
            return False
        if self.is_looking_down_at_desk(kpts2, box2) and s2.is_head_active(now, 5.0):
            return False
        face_conf2 = max(kpts2[0][2], kpts2[1][2], kpts2[2][2])
        if face_conf2 < 0.15:
            return False
        return True

    def analyze_talking_pairs(
        self,
        active_students: List[Tuple[TrackedStudent, np.ndarray, List[int]]],
        now: float
    ) -> List[ViolationEvent]:
        """Barcha talabalar juftliklarini 4 bosqichda tahlil qilish."""
        violations: List[ViolationEvent] = []
        n = len(active_students)
        if not self.enabled or n < 2:
            return violations

        for i in range(n):
            s1, kpts1, box1 = active_students[i]

            if self.is_facing_front(kpts1):
                s1.talking_start_time = None
                s1.talking_partner_id = None
                continue

            if self.is_looking_down_at_desk(kpts1, box1):
                s1.talking_start_time = None
                s1.talking_partner_id = None
                continue

            best_peer = None
            best_conf = 0.0

            for j in range(n):
                if i == j:
                    continue
                s2, kpts2, box2 = active_students[j]

                # Yonma-yon bitta partada o'tirganlar RUXSAT ETILGAN
                bh1 = box1[3] - box1[1]
                bh2 = box2[3] - box2[1]
                avg_h = (bh1 + bh2) / 2.0
                cy1 = (box1[1] + box1[3]) / 2.0
                cy2 = (box2[1] + box2[3]) / 2.0
                if abs(cy1 - cy2) <= avg_h * 0.28:
                    continue

                is_turned, turn_conf = self.is_turned_towards_rear_peer(kpts1, box1, box2)
                if not is_turned:
                    continue

                if not self.is_rear_peer_attentive(kpts2, box2, s2, now):
                    continue

                if turn_conf > best_conf:
                    best_conf = turn_conf
                    best_peer = (s2, box2)

            if best_peer is not None:
                s2_match, box2_match = best_peer
                if s1.talking_start_time is None or s1.talking_partner_id != s2_match.track_id:
                    s1.talking_start_time = now
                    s1.talking_partner_id = s2_match.track_id
                else:
                    duration = now - s1.talking_start_time
                    if duration >= self.min_duration:
                        temporal_conf = min(1.0, duration / (self.min_duration * 1.5))
                        composite_conf = round(0.65 * best_conf + 0.35 * temporal_conf, 2)
                        violations.append(ViolationEvent(
                            violation_type="TALKING",
                            description_uz="Orqaga o'girilib gaplashayotgan talaba",
                            confidence=composite_conf,
                            box=box1,
                            peer_box=box2_match,
                            keypoints=kpts1,
                            stage_scores={"geom": best_conf, "temporal": temporal_conf}
                        ))
            else:
                s1.talking_start_time = None
                s1.talking_partner_id = None

        return violations
