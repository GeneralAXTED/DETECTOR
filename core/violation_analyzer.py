import time
import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from core.analyzers.base import ViolationEvent, TrackedStudent
from core.analyzers.filters import SceneFilters
from core.analyzers.sleeping_analyzer import SleepingAnalyzer
from core.analyzers.phone_analyzer import PhoneAnalyzer
from core.analyzers.talking_analyzer import TalkingAnalyzer
from core.classifier import BehaviorClassifier

# Orqaga moslik uchun eksportlar
__all__ = ["ViolationEvent", "TrackedStudent", "ViolationAnalyzer"]

class ViolationAnalyzer:
    """
    Sinf xonasidagi barcha qoidabuzarliklarni (uxlash, telefon, gaplashish)
    4 ta bosqichda to'liq tekshiruvchi va boshqaruvchi markaziy tahlilchi (Facade).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.filters = SceneFilters(config)
        self.sleeping_analyzer = SleepingAnalyzer(config)
        self.phone_analyzer = PhoneAnalyzer(config)
        self.talking_analyzer = TalkingAnalyzer(config)

        # Kuzatilayotgan talabalar xotirasi (ID -> TrackedStudent)
        self.students: Dict[int, TrackedStudent] = {}
        self.next_id = 1
        self.last_cleanup = time.time()

        # Ikkilamchi maxsus AI klassifikator
        cls_cfg = config.get("classifier", {})
        self.classifier = BehaviorClassifier(
            model_path=cls_cfg.get("model_path", "models/behavior_classifier.pt"),
            device=config.get("system", {}).get("device", "cuda"),
            auto_collect=cls_cfg.get("auto_collect", True)
        )

    def update_config(self, config: Dict[str, Any]):
        """Konfiguratsiyani dinamik yangilash."""
        self.config = config
        self.filters.update_config(config)
        self.sleeping_analyzer.update_config(config)
        self.phone_analyzer.update_config(config)
        self.talking_analyzer.update_config(config)
        cls_cfg = config.get("classifier", {})
        self.classifier.auto_collect = cls_cfg.get("auto_collect", True)

    # Moslik uchun filtr delegatsiyalari
    def is_teacher(self, box: List[int], kpts: Optional[np.ndarray], img_h: int, img_w: int) -> bool:
        return self.filters.is_teacher(box, kpts, img_h, img_w)

    def is_edge_person(self, box: List[int], img_w: int, img_h: int) -> bool:
        return self.filters.is_edge_person(box, img_w, img_h)

    def check_student_with_baby(self, student_box: List[int], kpts: Optional[np.ndarray], all_boxes: List[List[int]]):
        return self.filters.check_student_with_baby(student_box, kpts, all_boxes)

    def _check_sleeping(self, kpts: np.ndarray, box: List[int], img_h: int, img_w: int):
        is_p, conf, head_pos, _ = self.sleeping_analyzer.check_sleeping_posture(kpts, box, img_h, img_w)
        return is_p, conf, head_pos

    def _check_phone_usage(self, person_box: List[int], kpts: Optional[np.ndarray], phones_boxes: List[List[int]], is_head_active: bool = False):
        has_p, conf, desc, _ = self.phone_analyzer.check_phone_posture(person_box, kpts, phones_boxes, is_head_active)
        return has_p, conf, desc

    def _match_or_create_student(self, box: List[int]) -> TrackedStudent:
        """Kadrlar o'rtasida b-box masofa/IoU orqali talabani bog'lash."""
        bx1, by1, bx2, by2 = box
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        bw, bh = bx2 - bx1, by2 - by1

        best_student = None
        min_dist = float("inf")

        for s in self.students.values():
            sx1, sy1, sx2, sy2 = s.box
            scx, scy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0
            dist = math.hypot(bcx - scx, bcy - scy)
            if dist < min_dist and dist < (bw + bh) * 0.40:
                min_dist = dist
                best_student = s

        if best_student is not None:
            best_student.update_box(box)
            return best_student

        student = TrackedStudent(self.next_id, box)
        self.students[self.next_id] = student
        self.next_id += 1
        return student

    def _cleanup_old_tracks(self):
        """Ko'rinmay qolgan talabalarni xotiradan tozalash."""
        now = time.time()
        if now - self.last_cleanup < 5.0:
            return
        self.last_cleanup = now
        dead_ids = [sid for sid, s in self.students.items() if now - s.last_seen > 10.0]
        for did in dead_ids:
            del self.students[did]

    def analyze_frame(
        self,
        persons_boxes: List[List[int]],
        persons_keypoints: List[np.ndarray],
        phones_boxes: List[List[int]],
        frame_shape: Tuple[int, int],
        raw_frame: Optional[np.ndarray] = None
    ) -> Tuple[List[ViolationEvent], List[List[int]], List[List[int]], List[Tuple[List[int], Optional[List[int]]]]]:
        """
        Kadr bo'yicha qoidabuzarliklarni 4 ta bosqichda tahlil qilish:
        1. Kontekst filtrlash (O'qituvchi, Chaqaloq, Chegara)
        2. Geometrik va anatomik tahlil
        3. Vaqtinchalik dispersiya va uzluksizlik (3-4 soniya)
        4. Vizual qirqma va neyron klassifikator tekshiruvi
        """
        self._cleanup_old_tracks()
        violations: List[ViolationEvent] = []
        teacher_boxes: List[List[int]] = []
        edge_boxes: List[List[int]] = []
        baby_students: List[Tuple[List[int], Optional[List[int]]]] = []

        now = time.time()
        h, w = frame_shape[:2]

        active_students: List[Tuple[TrackedStudent, np.ndarray, List[int]]] = []

        # 1. Shaxslarni ajratish: Chegara (ROI), O'qituvchi va Talabalar
        for i, box in enumerate(persons_boxes):
            kpts = persons_keypoints[i] if i < len(persons_keypoints) else None

            # Chegara / Kadr chetidagi to'liq ko'rinmayotgan shaxslar
            if self.filters.is_edge_person(box, w, h):
                edge_boxes.append(box)
                continue

            # O'qituvchi
            if self.filters.is_teacher(box, kpts, h, w):
                teacher_boxes.append(box)
                continue

            student = self._match_or_create_student(box)

            # Yo'lakda / eshik oldida yurib ketayotgan odam bo'lsa
            if student.is_walking(now):
                edge_boxes.append(box)
                continue

            active_students.append((student, kpts, box))

        # 2. Chaqaloqli talabalarni aniqlash
        active_student_boxes = [box for _, _, box in active_students]
        baby_student_boxes = []
        baby_child_boxes = []
        for student, kpts, box in active_students:
            has_baby, baby_b = self.filters.check_student_with_baby(box, kpts, active_student_boxes)
            if has_baby:
                baby_students.append((box, baby_b))
                baby_student_boxes.append(box)
                if baby_b is not None:
                    baby_child_boxes.append(baby_b)

        # 3. O'qituvchi va chaqaloq hududidagi telefonlarni chiqarib tashlash
        bottom_ratio = float(self.filters.teacher_cfg.get("bottom_zone_ratio", 0.30))
        zone_top_y = h * (1.0 - bottom_ratio)
        student_phones: List[List[int]] = []
        for ph in phones_boxes:
            phcx = (ph[0] + ph[2]) / 2.0
            phcy = (ph[1] + ph[3]) / 2.0

            in_excluded_area = False
            for tb in teacher_boxes:
                if (tb[0] - 80 <= phcx <= tb[2] + 80) and (tb[1] - 50 <= phcy <= tb[3] + 80):
                    in_excluded_area = True
                    break
                if (tb[0] - 60 <= phcx <= tb[2] + 60) and (phcy >= zone_top_y):
                    in_excluded_area = True
                    break

            for bb in baby_student_boxes:
                if (bb[0] - 40 <= phcx <= bb[2] + 40) and (bb[1] - 40 <= phcy <= bb[3] + 40):
                    in_excluded_area = True
                    break

            if not in_excluded_area:
                student_phones.append(ph)

        # 4. UXLASH VA TELEFON TAHLILI (Mutual Exclusion bilan)
        for student, kpts, box in active_students:
            if box in baby_student_boxes or box in baby_child_boxes:
                continue

            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1

            # Talabaning kadr qirqmasi (crop) - Ikkilamchi AI Klassifikator uchun
            student_crop = None
            if raw_frame is not None:
                pad_y = int(bh * 0.05)
                pad_x = int(bw * 0.05)
                cy1 = max(0, y1 - pad_y)
                cy2 = min(h, y2 + pad_y)
                cx1 = max(0, x1 - pad_x)
                cx2 = min(w, x2 + pad_x)
                student_crop = raw_frame[cy1:cy2, cx1:cx2]

            # --- A. UXLASH TAHLILI ---
            is_sleeping, sleep_conf, stage_scores_sleep = self.sleeping_analyzer.verify_student_sleeping(
                student=student,
                kpts=kpts,
                box=box,
                img_h=h,
                img_w=w,
                student_crop=student_crop,
                classifier=self.classifier,
                now=now
            )

            if is_sleeping:
                violations.append(ViolationEvent(
                    violation_type="SLEEPING",
                    description_uz="Uxlayotgan talaba (Partaga yotgan / harakatsiz)",
                    confidence=sleep_conf,
                    box=box,
                    keypoints=kpts,
                    stage_scores=stage_scores_sleep
                ))
                self.classifier.save_crop_sample(student_crop, "SLEEPING")

            # --- B. TELEFON TAHLILI (Faqat uxlab yotmagan talabada) ---
            is_phone, phone_conf, phone_desc, stage_scores_phone = self.phone_analyzer.verify_student_phone(
                student=student,
                person_box=box,
                kpts=kpts,
                phones_boxes=student_phones,
                student_crop=student_crop,
                classifier=self.classifier,
                now=now,
                is_sleeping=is_sleeping
            )

            if is_phone:
                violations.append(ViolationEvent(
                    violation_type="PHONE",
                    description_uz=phone_desc if phone_desc else "Smartfon / Telefon ishlatayotgan talaba",
                    confidence=phone_conf,
                    box=box,
                    keypoints=kpts,
                    stage_scores=stage_scores_phone
                ))
                self.classifier.save_crop_sample(student_crop, "PHONE_TALKING")

            # Oddiy holatdagi talabalardan avto-dataset to'plash
            if not is_sleeping and not is_phone and student_crop is not None:
                if int(now * 10) % 60 == 0:
                    self.classifier.save_crop_sample(student_crop, "NORMAL")

        # 5. ORQAGA O'GIRILIB GAPLASHISH TAHLILI
        if self.talking_analyzer.enabled and len(active_students) >= 2:
            eligible_students = [
                item for item in active_students
                if item[2] not in baby_student_boxes and item[2] not in baby_child_boxes
            ]
            if len(eligible_students) >= 2:
                talking_events = self.talking_analyzer.analyze_talking_pairs(eligible_students, now)
                violations.extend(talking_events)

        return violations, teacher_boxes, edge_boxes, baby_students
