import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from core.classifier import BehaviorClassifier

@dataclass
class ViolationEvent:
    violation_type: str        # "SLEEPING", "PHONE", "TALKING"
    description_uz: str        # O'zbekcha tavsif
    confidence: float          # 0.0 - 1.0
    box: List[int]             # [x1, y1, x2, y2]
    keypoints: Optional[np.ndarray] = None
    peer_box: Optional[List[int]] = None
    timestamp: float = field(default_factory=time.time)

class TrackedStudent:
    """Talabaning kadrlar davomidagi holatini kuzatish va vaqtinchalik filtrlash."""
    def __init__(self, track_id: int, initial_box: List[int]):
        self.track_id = track_id
        self.box = initial_box
        self.last_seen = time.time()
        
        # Holat taymerlari (soniyalarda hisoblash uchun)
        self.sleep_start_time: Optional[float] = None
        self.phone_start_time: Optional[float] = None
        self.talking_start_time: Optional[float] = None
        self.talking_partner_id: Optional[int] = None

        # Bosh harakati tarixi: [(timestamp, head_x, head_y), ...]
        self.head_history: List[Tuple[float, float, float]] = []

        # Tana harakati (yurish tezligi): [(timestamp, cx, cy), ...]
        cx = (initial_box[0] + initial_box[2]) / 2
        cy = (initial_box[1] + initial_box[3]) / 2
        self.pos_history: List[Tuple[float, float, float]] = [(time.time(), cx, cy)]

    def update_box(self, new_box: List[int]):
        now = time.time()
        self.box = new_box
        self.last_seen = now
        cx = (new_box[0] + new_box[2]) / 2
        cy = (new_box[1] + new_box[3]) / 2
        self.pos_history.append((now, cx, cy))
        if len(self.pos_history) > 30:
            self.pos_history = [(t, x, y) for t, x, y in self.pos_history if now - t <= 5.0]

    def is_walking(self, now: float, window_sec: float = 2.0) -> bool:
        """Kadrda yurib ketayotgan odamni aniqlash (o'tirgan talaba emas)."""
        recent = [(t, cx, cy) for t, cx, cy in self.pos_history if now - t <= window_sec]
        if len(recent) < 3:
            return False
        total_dist = 0.0
        for k in range(1, len(recent)):
            total_dist += math.hypot(recent[k][1] - recent[k-1][1], recent[k][2] - recent[k-1][2])
        dt = recent[-1][0] - recent[0][0]
        if dt > 0.3 and (total_dist / dt) > 70.0:  # 70 px/sec dan tez harakatlanayotgan bo'lsa
            return True
        return False

    def record_head_pos(self, now: float, hx: float, hy: float):
        """Bosh koordinatalari tarixini saqlash."""
        self.head_history.append((now, hx, hy))
        # Oxirgi 15 soniyani saqlash (ortiqchasini tozalash)
        if len(self.head_history) > 60:
            self.head_history = [(t, x, y) for t, x, y in self.head_history if now - t <= 15.0]

    def is_head_active(self, now: float, window_sec: float = 6.0) -> bool:
        """
        Talaba daftarga qarab konspekt yozayotgan yoki kitob o'qiyotgan paytda
        boshi muntazam qimirlab turadi (harakatda bo'ladi).
        Haqiqiy uxlab yotgan talabada esa bosh mutlaqo harakatsiz turadi.
        """
        recent = [(x, y) for t, x, y in self.head_history if now - t <= window_sec]
        if len(recent) < 5:
            return False
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        # Agar bosh 12 pikseldan ko'p qimirlagan bo'lsa, bu yozayotgan/harakatdagi faol talaba
        return bool(dx > 12 or dy > 12)


class ViolationAnalyzer:
    """Pose va Object detection natijalaridan qoidabuzarliklarni aniqlovchi tahlilchi."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config.get("detection", {})
        self.sleep_cfg = self.cfg.get("sleeping", {})
        self.phone_cfg = self.cfg.get("phone", {})
        self.talking_cfg = self.cfg.get("talking", {})
        self.teacher_cfg = self.cfg.get("teacher_filter", {"enabled": True, "bottom_zone_ratio": 0.30})
        self.roi_cfg = self.cfg.get("roi", {"enabled": True, "margin_pct": 0.03})
        self.baby_cfg = self.cfg.get("baby_detection", {"enabled": True})

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
        self.cfg = config.get("detection", {})
        self.sleep_cfg = self.cfg.get("sleeping", {})
        self.phone_cfg = self.cfg.get("phone", {})
        self.talking_cfg = self.cfg.get("talking", {})
        self.teacher_cfg = self.cfg.get("teacher_filter", {"enabled": True, "bottom_zone_ratio": 0.30})
        self.roi_cfg = self.cfg.get("roi", {"enabled": True, "margin_pct": 0.03})
        self.baby_cfg = self.cfg.get("baby_detection", {"enabled": True})
        cls_cfg = config.get("classifier", {})
        self.classifier.auto_collect = cls_cfg.get("auto_collect", True)

    def is_teacher(self, box: List[int], kpts: Optional[np.ndarray], img_h: int, img_w: int) -> bool:
        """
        O'qituvchini aniqlash:
        1. Kamera o'qituvchi orqasida joylashgan: O'qituvchi auditoriyaga (talabalarga) qarab turibdi,
           ya'ni orqa tarafi kamera tomonda:
           - Yuzi kameraga ko'rinmaydi (face_conf < 0.25)
           - Yelka va tana orqadan yaqqol ko'rinadi (sh_conf > 0.40)
        2. O'qituvchi oldingi planda (minbar, doska oldi, pastki zonada):
           - Bounding box balandligi katta: bh > img_h * 0.22
           - Pastki chegarasi y2 >= zone_top_y yoki y1 >= img_h * 0.50
        3. Talabalar kameraga yuzlanib o'tiradi (face_conf >= 0.30) - ular hech qachon o'qituvchi deb olinmaydi!
        """
        if not self.teacher_cfg.get("enabled", True):
            return False

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        # Kadr chetidagi kesilgan kichik bo'laklarni chetlatish
        if bw < 100:
            return False

        # Yuz va yelka ishonchliligi
        face_conf = 0.0
        sh_conf = 0.0
        if kpts is not None and len(kpts) >= 7:
            face_conf = max(kpts[0][2], kpts[1][2], kpts[2][2])
            sh_conf = max(kpts[5][2], kpts[6][2])

        # Agar yuzi kameraga ko'rinib turgan bo'lsa, bu talaba!
        if face_conf >= 0.30:
            return False

        # Orqa tarafi bilan turganlik sharti
        is_facing_away = (face_conf < 0.25 and sh_conf > 0.40)

        # O'qituvchi hududida (pastki minbar zonasi) va katta bo'lishi
        bottom_ratio = self.teacher_cfg.get("bottom_zone_ratio", 0.30)
        zone_top_y = img_h * (1.0 - bottom_ratio)
        in_teacher_zone = (y2 >= zone_top_y) or (y1 >= img_h * 0.50)

        if is_facing_away and in_teacher_zone and bh > (img_h * 0.22):
            return True

        return False

    def is_edge_person(self, box: List[int], img_w: int, img_h: int) -> bool:
        """
        Tasvir chetida turgan yoki yurib o'tayotgan shaxsni aniqlash.
        Kadr chetida tanasi to'liq ko'rinmayotganligi sababli, ular qoidabuzarlikdan
        (uxlash, telefon, gaplashish) to'liq ozod qilinadi.
        """
        if not self.roi_cfg.get("enabled", True):
            return False

        margin_pct = float(self.roi_cfg.get("margin_pct", 0.03))
        mx = int(img_w * margin_pct)
        my = int(img_h * margin_pct)

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        # Kadr chegaralariga tegib turganlik (kesilgan siluet)
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
        Chaqalog'i bilan o'tirgan talabalarni aniqlash:
        Talaba bag'rida/quchog'ida (ko'krak yoki tizza sohasida) kichik chaqaloq/bola
        shaxsi (kichik person bounding box) mavjudligini qat'iy tekshiradi.
        
        Qo'llar holatidan asossiz taxmin QILINMAYDI (chunki har qanday yozayotgan yoki
        partada o'tirgan talaba qo'llarini ko'krak/parta oldida ushlaydi).
        """
        if not self.baby_cfg.get("enabled", True):
            return False, None

        s_w = student_box[2] - student_box[0]
        s_h = student_box[3] - student_box[1]

        # Talaba ichida/quchog'ida haqiqiy kichik chaqaloq (person) b-boxi bormi?
        for other_b in all_boxes:
            if other_b == student_box:
                continue
            ocx = (other_b[0] + other_b[2]) / 2
            ocy = (other_b[1] + other_b[3]) / 2
            ow = other_b[2] - other_b[0]
            oh = other_b[3] - other_b[1]

            # Kichik chaqaloq qutisi talabaning ko'kragi/tizzasi sohasida joylashgan bo'lishi:
            # 1. Gorizontal markaz talaba enining ichida (chekkalarida emas)
            # 2. Vertikal markaz talabaning ko'krak yoki quchoq hududida
            # 3. Chaqaloq o'lchami kattalar siluetiga qaraganda ancha kichik bo'lishi
            in_lap_x = (student_box[0] + s_w * 0.10 <= ocx <= student_box[2] - s_w * 0.10)
            in_lap_y = (student_box[1] + s_h * 0.25 <= ocy <= student_box[3] - s_h * 0.05)
            is_much_smaller = (ow < s_w * 0.55) and (oh < s_h * 0.55) and ((ow * oh) < (s_w * s_h) * 0.30)

            if in_lap_x and in_lap_y and is_much_smaller:
                return True, other_b

        return False, None

    def _match_or_create_student(self, box: List[int]) -> TrackedStudent:
        """Kadrlar o'rtasida b-box IoU / masofa orqali talabani bog'lash."""
        bx1, by1, bx2, by2 = box
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        bw, bh = bx2 - bx1, by2 - by1

        best_student = None
        min_dist = float("inf")

        for s in self.students.values():
            sx1, sy1, sx2, sy2 = s.box
            scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
            dist = math.hypot(bcx - scx, bcy - scy)
            # Agar masofa talaba o'lchamining 40% idan kam bo'lsa, o'sha talaba
            if dist < min_dist and dist < (bw + bh) * 0.4:
                min_dist = dist
                best_student = s

        if best_student is not None:
            best_student.update_box(box)
            return best_student

        # Yangi talaba
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
        persons_keypoints: List[np.ndarray],  # Har biri (17, 3) [x, y, conf]
        phones_boxes: List[List[int]],
        frame_shape: Tuple[int, int],
        raw_frame: Optional[np.ndarray] = None
    ) -> Tuple[List[ViolationEvent], List[List[int]], List[List[int]], List[Tuple[List[int], Optional[List[int]]]]]:
        """
        Kadr bo'yicha qoidabuzarliklarni tahlil qilish.
        Qaytaradi: (violations, teacher_boxes, edge_boxes, baby_students)
        """
        self._cleanup_old_tracks()
        violations: List[ViolationEvent] = []
        teacher_boxes: List[List[int]] = []
        edge_boxes: List[List[int]] = []
        baby_students: List[Tuple[List[int], Optional[List[int]]]] = []

        now = time.time()
        h, w = frame_shape[:2]

        active_students: List[Tuple[TrackedStudent, np.ndarray, List[int]]] = []

        # 1. Shaxslarni ajratish: Chegara (Edge / ROI), O'qituvchi va Talabalar
        for i, box in enumerate(persons_boxes):
            kpts = persons_keypoints[i] if i < len(persons_keypoints) else None

            # Chegara / Kadr chetidagi to'liq ko'rinmayotgan shaxslar
            if self.is_edge_person(box, w, h):
                edge_boxes.append(box)
                continue

            # O'qituvchi
            if self.is_teacher(box, kpts, h, w):
                teacher_boxes.append(box)
                continue

            student = self._match_or_create_student(box)

            # Yo'lakda / eshik oldida yurib ketayotgan odam bo'lsa (o'tirgan talaba emas)
            if student.is_walking(now):
                edge_boxes.append(box)
                continue

            active_students.append((student, kpts, box))

        # 2. Chaqalog'i bilan o'tirgan talabalarni aniqlash
        active_student_boxes = [box for _, _, box in active_students]
        baby_student_boxes = []
        baby_child_boxes = []
        for student, kpts, box in active_students:
            has_baby, baby_b = self.check_student_with_baby(box, kpts, active_student_boxes)
            if has_baby:
                baby_students.append((box, baby_b))
                baby_student_boxes.append(box)
                if baby_b is not None:
                    baby_child_boxes.append(baby_b)

        # 3. O'qituvchi va chaqaloq hududidagi telefonlarni talabalar tahlilidan chiqarish
        bottom_ratio = self.teacher_cfg.get("bottom_zone_ratio", 0.30)
        zone_top_y = h * (1.0 - bottom_ratio)
        student_phones: List[List[int]] = []
        for ph in phones_boxes:
            phx1, phy1, phx2, phy2 = ph
            phcx = (phx1 + phx2) / 2
            phcy = (phy1 + phy2) / 2

            # Telefon o'qituvchi yoki chaqaloq atrofida bo'lsa chiqarib tashlash
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

        # 4. UXLASH VA TELEFON TAHLILI (O'zaro ziddiyatni bartaraf etish va AI Tasniflagich)
        for student, kpts, box in active_students:
            # Chaqalog'i bor talaba yoki chaqaloq qoidabuzarlikdan ozod qilinadi
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

            # --- 1. UXLASHNI TEKSHIRISH ---
            is_sleeping = False
            sleep_conf = 0.0
            if self.sleep_cfg.get("enabled", True) and kpts is not None and len(kpts) >= 7:
                is_sleeping, sleep_conf, head_pos = self._check_sleeping(kpts, box, h, w)
                if head_pos is not None:
                    student.record_head_pos(now, head_pos[0], head_pos[1])

                head_is_moving = student.is_head_active(now, window_sec=6.0)

                # Agar bosh harakatda bo'lsa (yozayotgan bo'lsa), uxlash emas
                if is_sleeping and not head_is_moving:
                    if student.sleep_start_time is None:
                        student.sleep_start_time = now
                    elif (now - student.sleep_start_time) >= float(self.sleep_cfg.get("min_duration_seconds", 6.0)):
                        # Ikkilamchi AI klassifikator orqali tekshirish
                        verified, ai_conf = self.classifier.verify_violation(student_crop, "SLEEPING")
                        if verified:
                            violations.append(ViolationEvent(
                                violation_type="SLEEPING",
                                description_uz="Uxlayotgan talaba (Partaga yotgan / harakatsiz)",
                                confidence=sleep_conf,
                                box=box,
                                keypoints=kpts
                            ))
                            self.classifier.save_crop_sample(student_crop, "SLEEPING")
                else:
                    student.sleep_start_time = None

            # --- 2. TELEFONNI TEKSHIRISH (Faqat agar talaba partada uxlab yotmagan bo'lsa!) ---
            # Mutual Exclusion: Partaga yotgan talabada quloq/qo'l yaqinligi tufayli soxta telefon berilmaydi!
            if self.phone_cfg.get("enabled", True) and not is_sleeping:
                head_is_moving = student.is_head_active(now, window_sec=6.0)
                has_phone, phone_conf, phone_desc = self._check_phone_usage(
                    box, kpts, student_phones, is_head_active=head_is_moving
                )
                if has_phone:
                    if student.phone_start_time is None:
                        student.phone_start_time = now
                    elif (now - student.phone_start_time) >= float(self.phone_cfg.get("min_duration_seconds", 3.0)):
                        # Ikkilamchi AI klassifikator orqali tekshirish
                        verified, ai_conf = self.classifier.verify_violation(student_crop, "PHONE")
                        if verified:
                            violations.append(ViolationEvent(
                                violation_type="PHONE",
                                description_uz=phone_desc if phone_desc else "Smartfon / Telefon ishlatayotgan talaba",
                                confidence=phone_conf,
                                box=box,
                                keypoints=kpts
                            ))
                            self.classifier.save_crop_sample(student_crop, "PHONE_TALKING")
                else:
                    student.phone_start_time = None
            else:
                student.phone_start_time = None

            # Oddiy dars qilayotgan talabalardan avto-dataset to'plash (NORMAL)
            if not is_sleeping and student.phone_start_time is None and student_crop is not None:
                if int(now * 10) % 60 == 0:
                    self.classifier.save_crop_sample(student_crop, "NORMAL")

        # 5. SHERIGI BILAN GAPLASHISH TAHLILI (Faqat oddiy talabalar juftliklari uchun)
        if self.talking_cfg.get("enabled", True) and len(active_students) >= 2:
            eligible_students = [
                item for item in active_students
                if item[2] not in baby_student_boxes and item[2] not in baby_child_boxes
            ]
            if len(eligible_students) >= 2:
                talking_events = self._check_talking_pairs(eligible_students, now, w, h)
                violations.extend(talking_events)

        return violations, teacher_boxes, edge_boxes, baby_students

    def _check_sleeping(
        self,
        kpts: np.ndarray,
        box: List[int],
        img_h: int,
        img_w: int
    ) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """
        Yuqoridagi (shift / burchak) kameralar uchun moslashtirilgan uxlash tekshiruvi:
        1. Bosh partaga to'liq qo'yilganda (yuz ko'milganda) 100% aniqlaydi.
        2. Salgina pastga qarab daftarga yozayotgan yoki kitob o'qiyotgan talabalarni (quloqlari
           yelkadan baland bo'lgani uchun) xato uxlash deb berilishidan to'liq himoya qiladi.
        """
        bh = box[3] - box[1]
        bw = box[2] - box[0]

        # 1. Yelka nuqtalari
        l_sh = kpts[5]
        r_sh = kpts[6]
        valid_sh_y = [p[1] for p in (l_sh, r_sh) if p[2] > 0.15]
        if valid_sh_y:
            avg_sh_y = sum(valid_sh_y) / len(valid_sh_y)
        else:
            # Agar yelkalar aniqlanmagan bo'lsa (b-box yuqori qismidan taxmin)
            avg_sh_y = box[1] + bh * 0.30

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
            avg_head_conf = sum(p[2] for p in head_pts) / len(head_pts)
            head_is_explicit = True
        else:
            # Yuz ko'rinmayapti (bosh partaga ko'milgan bo'lishi mumkin)
            avg_hx = (box[0] + box[2]) / 2
            avg_hy = avg_sh_y + 10.0
            head_pos = (avg_hx, avg_hy)
            avg_head_conf = 0.0
            head_is_explicit = False

        avg_head_y = head_pos[1]
        head_diff = avg_sh_y - avg_head_y  # > 0: bosh yelkadan balandda, < 0: bosh yelkadan pastda

        wrists = [kpts[idx] for idx in (9, 10) if kpts[idx][2] > 0.15]
        elbows = [kpts[idx] for idx in (7, 8) if kpts[idx][2] > 0.15]

        # --- QAT'IY QOIDA 1: Quloqlar yelkadan balandda bo'lsa -> UXLASH EMAS! ---
        # Tepadan qarovchi kamerada talaba daftarga yoki partaga qaraganida ham
        # uning quloqlari yelkasidan yuqorida (tasvirda kichikroq Y da) ko'rinadi.
        # Bu bosh bo'yin mushaklari orqali tik tutilganini, partaga yotib olinmaganini 100% isbotlaydi.
        if valid_ears_y:
            min_ear_y = min(valid_ears_y)
            if (avg_sh_y - min_ear_y) >= 4.0:
                return False, 0.0, head_pos

        # --- QAT'IY QOIDA 2: Yuz xususiyatlari kameraga aniq ko'rinib turgan bo'lsa -> UXLASH EMAS! ---
        if face_conf >= 0.22 and head_diff >= -6.0:
            return False, 0.0, head_pos

        # --- QAT'IY QOIDA 3: Bosh tana balandligining yuqori qismida joylashgan bo'lsa -> UXLASH EMAS! ---
        if head_is_explicit and (avg_head_y - box[1]) < bh * 0.28 and (avg_sh_y - box[1]) >= bh * 0.14:
            if head_diff >= -4.0:
                return False, 0.0, head_pos

        # --- QAT'IY QOIDA 4: Qo'llardan balandda bo'lsa (yozish / o'qish masofasi) -> UXLASH EMAS! ---
        if wrists and head_is_explicit:
            min_wrist_dist = min(math.hypot(w[0] - head_pos[0], w[1] - head_pos[1]) for w in wrists)
            if min_wrist_dist > max(45, bh * 0.22):
                return False, 0.0, head_pos

        # =========================================================================
        # HAQIQIY UXLASH MEZONLARI (Faqat partaga yotib olingandagina ishlaydi!)
        # =========================================================================

        # 1. BOSHNI PARTAGA QO'YIB UXLASH (Yuzi ko'rinmaydi, yelka va qo'llar partada):
        # Foydalanuvchi muammosi: "bittasi partaga boshini qo'ygan uxlayabdi"
        if not head_is_explicit or face_conf < 0.18:
            # Quloqlar yelkadan balandda EMAS (bosh tik emas)
            ears_not_upright = (not valid_ears_y) or (min(valid_ears_y) >= avg_sh_y - 2.0)
            if ears_not_upright:
                return True, 0.95, head_pos

        # 2. BOSHNI QO'LLARGA QO'YIB UXLASH (Qo'llar ustiga yotish):
        if wrists:
            min_wrist_dist = min(math.hypot(w[0] - head_pos[0], w[1] - head_pos[1]) for w in wrists)
            if min_wrist_dist < max(35, bh * 0.18) and face_conf < 0.22:
                return True, 0.92, head_pos

        # 3. PARTA USTIGA BUTUNLAY YOTIB OLISH (Bosh yelkadan pastga parta taxtasiga tushgan):
        if head_diff < -12.0 and face_conf < 0.20:
            return True, 0.90, head_pos

        return False, 0.0, head_pos

    def _check_phone_usage(
        self,
        person_box: List[int],
        kpts: Optional[np.ndarray],
        phones_boxes: List[List[int]],
        is_head_active: bool = False
    ) -> Tuple[bool, float, str]:
        """
        Telefon ishlatish holatini aniqlash:
        1. Qo'ng'iroq (Call to Ear): Bosh tik, bilak quloq sathida va telefon bilan birga ushlangan.
        2. Parta ustida o'yin / matn terish: Telefon ushlab turilgan va diqqat telefonda.
        3. Parta tagida telefon ishlatish (Lap / Under Desk phone):
           Qo'llar parta tagida (yashirilgan yoki tizzada), talaba esa diqqat bilan
           to'g'ridan-to'g'ri parta ostidagi telefoniga qarab o'tiribdi.
        """
        px1, py1, px2, py2 = person_box
        bw = px2 - px1
        bh = py2 - py1

        lw = kpts[9] if (kpts is not None and len(kpts) > 9) else [0, 0, 0]
        rw = kpts[10] if (kpts is not None and len(kpts) > 10) else [0, 0, 0]
        l_el = kpts[7] if (kpts is not None and len(kpts) > 7) else [0, 0, 0]
        r_el = kpts[8] if (kpts is not None and len(kpts) > 8) else [0, 0, 0]
        nose = kpts[0] if (kpts is not None and len(kpts) > 0) else [0, 0, 0]
        l_eye = kpts[1] if (kpts is not None and len(kpts) > 1) else [0, 0, 0]
        r_eye = kpts[2] if (kpts is not None and len(kpts) > 2) else [0, 0, 0]
        l_ear = kpts[3] if (kpts is not None and len(kpts) > 3) else [0, 0, 0]
        r_ear = kpts[4] if (kpts is not None and len(kpts) > 4) else [0, 0, 0]
        l_sh = kpts[5] if (kpts is not None and len(kpts) > 5) else [0, 0, 0]
        r_sh = kpts[6] if (kpts is not None and len(kpts) > 6) else [0, 0, 0]

        lw_val = lw[2] > 0.20
        rw_val = rw[2] > 0.20

        valid_sh_y = [p[1] for p in (l_sh, r_sh) if p[2] > 0.15]
        avg_sh_y = sum(valid_sh_y) / len(valid_sh_y) if valid_sh_y else (py1 + bh * 0.30)
        head_is_upright = (nose[2] > 0.15 and (avg_sh_y - nose[1]) > 4.0)

        # 1. TELEFONDA GAPLASHISH (Qo'ng'iroq - Call to Ear):
        # Quloqqa telefon qo'yilganda, telefon qutisi quloq/yuz yonida ko'rinishi shart.
        # Bu iyakka qo'l tirash yoki quloq qashlash bilan adashishdan 100% asraydi.
        if head_is_upright and phones_boxes:
            for ear, wrist, is_w_val in [(l_ear, lw, lw_val), (r_ear, rw, rw_val)]:
                if ear[2] > 0.20 and is_w_val:
                    dist_ear = math.hypot(wrist[0] - ear[0], wrist[1] - ear[1])
                    if dist_ear < max(35, bh * 0.18):
                        is_at_ear_level = abs(wrist[1] - ear[1]) < max(25, bh * 0.12)
                        if is_at_ear_level:
                            for ph in phones_boxes:
                                phcx = (ph[0] + ph[2]) / 2
                                phcy = (ph[1] + ph[3]) / 2
                                if math.hypot(wrist[0] - phcx, wrist[1] - phcy) < max(45, bh * 0.22):
                                    return True, 0.95, "Telefonda gaplashayotgan talaba (Quloqqa qo'yilgan)"

        # 2. PARTA USTIDA IKKI QO'LDA TELEFON USHLAB O'YNASH / YOZISH (Two-handed phone gaming / texting):
        # Foydalanuvchi talabi: "talaba parta ustida ikki qo'li bilan telefon ushlab o'ynab o'tiribdi"
        # Ikkala bilak ham aniq ko'rinishi shart (bu orqa qatordagi qo'llari to'silgan talabalarni 100% himoya qiladi)
        if lw_val and rw_val and nose[2] > 0.15:
            w_dist = math.hypot(lw[0] - rw[0], lw[1] - rw[1])
            hands_cx = (lw[0] + rw[0]) / 2
            hands_cy = (lw[1] + rw[1]) / 2

            # Ikki bilak birlashgan (telefonni ikki qo'llab ushlash):
            wrists_together = w_dist < max(45, bw * 0.40)
            hands_in_center = (px1 + bw * 0.08 <= hands_cx <= px2 - bw * 0.08)

            # --- QAT'IY FARQLASH: IYAK YONIDA QO'L TUTISH (Thinking / Listening) EMAS, AYNAN PARTA USTIDA BO'LISHI SHART! ---
            # 1. Qo'llar yelkadan ancha pastda (parta taxtasi sathida) bo'lishi shart:
            hands_down_on_desk = (lw[1] > avg_sh_y + max(25, bh * 0.18)) and (rw[1] > avg_sh_y + max(25, bh * 0.18))

            # 2. Qo'llar iyak/burundan pastda, partada bo'lishi shart (iyak yonida qo'l tutishdan 100% himoya):
            vertical_drop_from_nose = hands_cy - nose[1]
            not_at_chin_level = vertical_drop_from_nose >= max(60, bh * 0.26)

            # 3. Tirsaklar bilan taqqoslash (iyakda ushlanganda qo'llar tirsakdan balandda bo'ladi):
            valid_el_y = [p[1] for p in (l_el, r_el) if p[2] > 0.15]
            avg_el_y = sum(valid_el_y) / len(valid_el_y) if valid_el_y else 0
            elbow_posture_ok = True
            if valid_el_y:
                # Partada telefon o'ynaganda qo'llar tirsakdan yuqoriga (yuzga) ko'tarilmagan bo'ladi:
                if hands_cy < avg_el_y - 20:
                    elbow_posture_ok = False

            # Bosh to'g'ridan-to'g'ri qo'llarga (telefonga) qaratilgan:
            dist_nose_to_hands = math.hypot(nose[0] - hands_cx, nose[1] - hands_cy)
            looking_at_hands = (dist_nose_to_hands < max(80, bh * 0.60)) and (abs(nose[0] - hands_cx) < bw * 0.35)

            # Iyakka / yuzga tegilmaganligi:
            dist_chin_hands = math.hypot(nose[0] - hands_cx, hands_cy - nose[1])
            not_touching_face = dist_chin_hands >= max(55, bh * 0.24)

            # Quloqqa tegilmaganligi:
            not_touching_ears = True
            for ear in (l_ear, r_ear):
                if ear[2] > 0.18:
                    if math.hypot(lw[0] - ear[0], lw[1] - ear[1]) < 35 or math.hypot(rw[0] - ear[0], rw[1] - ear[1]) < 35:
                        not_touching_ears = False
                        break

            # Agar kadrda haqiqiy telefon aniqlangan bo'lsa:
            phone_near_hands = False
            if phones_boxes:
                for ph in phones_boxes:
                    phcx = (ph[0] + ph[2]) / 2
                    phcy = (ph[1] + ph[3]) / 2
                    if math.hypot(hands_cx - phcx, hands_cy - phcy) < max(55, bh * 0.30):
                        phone_near_hands = True
                        break

            # Agar telefon qutisi qo'llar orasida ko'rinsa:
            if phone_near_hands:
                if wrists_together and hands_in_center and looking_at_hands:
                    return True, 0.98, "Smartfon / Telefon ishlatayotgan talaba (Ikki qo'lda)"

            # Agar telefon qo'llar ichida berkitilgan bo'lsa (faqat va faqat partada bo'lgandagina!):
            if (wrists_together and hands_in_center and hands_down_on_desk and 
                not_at_chin_level and elbow_posture_ok and looking_at_hands and 
                not_touching_face and not_touching_ears):
                return True, 0.92, "Parta ustida ikki qo'lda telefon o'ynayotgan talaba"

        # 3. KADRDA ANIKLANGAN TELEFON OBYEKTI BILAN ISHLASH (Bir qo'lda, partada yoki tizzada):
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

                phcx = (phx1 + phx2) / 2
                phcy = (phy1 + phy2) / 2

                # Telefon talabaning hududida joylashganmi?
                if not ((px1 - bw * 0.18 <= phcx <= px2 + bw * 0.18) and (py1 <= phcy <= py2 + bh * 0.20)):
                    continue

                dist_lw = math.hypot(lw[0] - phcx, lw[1] - phcy) if lw_val else 9999
                dist_rw = math.hypot(rw[0] - phcx, rw[1] - phcy) if rw_val else 9999
                min_wrist = min(dist_lw, dist_rw)
                dist_nose = math.hypot(nose[0] - phcx, nose[1] - phcy) if nose[2] > 0.15 else 9999

                # A) Qo'llardan biri telefon yonida / ushlab turibdi:
                if min_wrist <= max(45, min(bw, bh) * 0.32):
                    if dist_nose < bh * 0.55:
                        return True, 0.95, "Smartfon / Telefon ishlatayotgan talaba"

                # B) Parta ostida / tizzada telefon va talaba unga qarab o'tiribdi:
                if phcy > (py1 + bh * 0.38) and dist_nose < bh * 0.65:
                    return True, 0.92, "Parta ostida telefon ishlatayotgan talaba"

        return False, 0.0, ""

    def _is_facing_front(self, kpts: Optional[np.ndarray]) -> bool:
        """
        Talaba kamerasiga (sinf oldiga / doskaga) yuzlanib o'tirganini aniqlash.
        Agar yuz (burun va ko'zlar) kameraga aniq ko'rinib turgan bo'lsa,
        talaba 100% orqaga qaramagan!
        """
        if kpts is None or len(kpts) < 5:
            return False
        nose = kpts[0]
        l_eye = kpts[1]
        r_eye = kpts[2]

        # Burun va ikkala ko'z kameraga ko'rinib turgan bo'lsa (frontal yoki biroz doskaga qaragan):
        if nose[2] >= 0.20 and l_eye[2] >= 0.20 and r_eye[2] >= 0.20:
            # Chap ko'z o'ng ko'zdan o'ngroqda (kamera to'g'ri frontal ko'rinishi)
            if l_eye[0] > r_eye[0]:
                return True
        return False

    def _is_looking_down_at_desk(self, kpts: Optional[np.ndarray], box: List[int]) -> bool:
        """Talaba partaga/daftarga qarab konspekt yozayotganini yoki o'qiyotganini aniqlash."""
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

        # Qo'llar (bilaklar) partada pastda joylashganligi
        wrists_y = [kpts[idx][1] for idx in (9, 10) if len(kpts) > idx and kpts[idx][2] > 0.15]
        wrists_on_desk = any(wy > (box[1] + bh * 0.40) for wy in wrists_y)

        # 1. Bosh pastga engashgan: Burun ko'zlardan pastda yoki yelka balandligiga yaqin
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

    def _is_turned_towards_rear_peer(
        self,
        kpts1: np.ndarray,
        box1: List[int],
        box2: List[int]
    ) -> Tuple[bool, float]:
        """
        Oldindagi talaba (S1) orqadagi partadagi talaba (S2) tomon
        HAQIQATDAN HAM orqaga o'girilganligini geometrik tekshirish.
        
        Qaytaradi: (is_turned, confidence)
        """
        if kpts1 is None or len(kpts1) < 7:
            return False, 0.0

        bh1 = box1[3] - box1[1]
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2

        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2

        # 1. S1 oldinda, S2 orqada bo'lishi shart:
        # Frontal kamerada orqadagi partalar tasvirning yuqorisida (kichikroq Y),
        # oldingi partalar esa pastrog'ida (kattaroq Y).
        vert_dist = cy1 - cy2
        if vert_dist < bh1 * 0.28 or vert_dist > bh1 * 1.80:
            return False, 0.0

        # Gorizontal oraliq
        dx = cx2 - cx1  # dx > 0 bo'lsa S2 o'ng-orqada, dx < 0 bo'lsa chap-orqada
        if abs(dx) > bh1 * 1.25:
            return False, 0.0

        # 2. S1 ning oldinga qarab turganligini tekshirish:
        # Agar burun va ikkala ko'z kameraga to'g'ri ko'rinib turgan bo'lsa, u orqaga qaramagan!
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

        # Bosh partaga egilmagan, balki orqadagi odamga qarash uchun tik turgan bo'lishi shart
        head_pts = [kpts1[idx] for idx in range(5) if kpts1[idx][2] > 0.15]
        if head_pts:
            avg_hy = sum(p[1] for p in head_pts) / len(head_pts)
        else:
            avg_hy = float(box1[1] + 15)

        # Agar bosh yelkadan pastda bo'lsa (partaga yotish yoki uxlash), bu orqaga qarash emas
        if (avg_sh_y - avg_hy) < 4.0:
            return False, 0.0

        # --- HOLAT 1: Tanasi bilan butunlay orqaga o'girilish (Orqasi kameraga qaratilgan) ---
        face_conf = max(nose[2], l_eye[2], r_eye[2])
        sh_conf = min(l_sh[2], r_sh[2])

        if face_conf < 0.20 and sh_conf > 0.35:
            if (avg_sh_y - avg_hy) >= 10.0:
                if l_ear[2] > 0.20 and r_ear[2] > 0.20:
                    if l_ear[0] < r_ear[0]:  # Inverted ears = orqa tarafi ko'rinmoqda
                        return True, 0.95
                return True, 0.90

        # --- HOLAT 2: Boshni yelka ustidan orqadagi S2 tomon keskin burish (Profil / Orqaga qarash) ---
        # S2 o'ng-orqada joylashgan bo'lsa (dx > 0):
        if dx > bh1 * 0.15:
            sh_right = max(l_sh[0], r_sh[0])
            if nose[2] >= 0.20 and (nose[0] >= sh_right - 10) and (l_eye[2] < 0.20):
                return True, 0.92
            if r_ear[2] >= 0.25 and l_ear[2] < 0.15 and l_eye[2] < 0.15:
                return True, 0.90

        # S2 chap-orqada joylashgan bo'lsa (dx < 0):
        elif dx < -bh1 * 0.15:
            sh_left = min(l_sh[0], r_sh[0])
            if nose[2] >= 0.20 and (nose[0] <= sh_left + 10) and (r_eye[2] < 0.20):
                return True, 0.92
            if l_ear[2] >= 0.25 and r_ear[2] < 0.15 and r_eye[2] < 0.15:
                return True, 0.90

        # S2 to'g'ridan-to'g'ri orqada bo'lsa (|dx| <= bh1 * 0.15):
        else:
            if face_conf < 0.20 and (avg_sh_y - avg_hy) >= 10.0:
                return True, 0.90

        return False, 0.0

    def _is_rear_peer_attentive(
        self,
        kpts2: Optional[np.ndarray],
        box2: List[int],
        s2: TrackedStudent,
        now: float
    ) -> bool:
        """
        Orqadagi talaba (S2) haqiqatdan ham oldingi talaba bilan suhbatda
        ishtirok etyaptimi yoki boshqa ish bilan bandmi (uxlash, yozish)?
        """
        if kpts2 is None or len(kpts2) < 7:
            return True

        if s2.sleep_start_time is not None:
            return False

        if self._is_looking_down_at_desk(kpts2, box2) and s2.is_head_active(now, 5.0):
            return False

        face_conf2 = max(kpts2[0][2], kpts2[1][2], kpts2[2][2])
        if face_conf2 < 0.15:
            return False

        return True

    def _check_talking_pairs(
        self,
        active_students: List[Tuple[TrackedStudent, np.ndarray, List[int]]],
        now: float,
        img_w: int,
        img_h: int
    ) -> List[ViolationEvent]:
        """
        Orqaga o'girilib gaplashayotgan talabalarni aniqlash:
        1. Yonma-yon bir partada o'tirganlar (abs(dy) <= 0.28 * avg_h) - RUXSAT ETILGAN (qoidabuzarlik EMAS).
           Foydalanuvchi talabi: "yonidagi bilan dars mobaynida gaplashishi mumkin".
        2. Faqat orqaga o'girilgan (oldindagi talaba orqadagi partadagi talaba tomon o'girilishi):
           Oldindagi talaba (S1) orqadagi suhbatdosh (S2) tomon boshini yoki tanasini
           haqiqatdan ham orqaga burgan bo'lishi shart!
        3. Doskaga yoki oldinga qarab turgan, daftarga yozayotgan talabalar MUTLAQO qoidabuzar deb olinmaydi!
        4. Orqadagi aniq suhbatdoshsiz yakka o'zi turgan talabaga soxta 'gaplashyapti' xatosi berilmaydi!
        """
        violations = []
        n = len(active_students)
        if n < 2:
            return violations

        min_dur = float(self.talking_cfg.get("min_duration_seconds", 4.5))

        for i in range(n):
            s1, kpts1, box1 = active_students[i]

            # 1. Frontal kameraga (doskaga/oldinga) qarab turgan talaba 100% orqaga qaramagan!
            if self._is_facing_front(kpts1):
                s1.talking_start_time = None
                s1.talking_partner_id = None
                continue

            # 2. Daftarga konspekt yozayotgan talaba orqaga qaramagan!
            if self._is_looking_down_at_desk(kpts1, box1):
                s1.talking_start_time = None
                s1.talking_partner_id = None
                continue

            # 3. Orqadagi sherikni qidirish
            best_peer = None
            best_conf = 0.0

            for j in range(n):
                if i == j:
                    continue
                s2, kpts2, box2 = active_students[j]

                # Yonma-yon bitta partada o'tirganlar qoidabuzarlik EMAS:
                bh1 = box1[3] - box1[1]
                bh2 = box2[3] - box2[1]
                avg_h = (bh1 + bh2) / 2
                cy1 = (box1[1] + box1[3]) / 2
                cy2 = (box2[1] + box2[3]) / 2
                if abs(cy1 - cy2) <= avg_h * 0.28:
                    continue  # Bitta partada o'tirgan partadoshlar (RUXSAT ETILGAN)

                # S1 haqiqatdan ham S2 tomon orqaga o'girilganmi?
                is_turned, turn_conf = self._is_turned_towards_rear_peer(kpts1, box1, box2)
                if not is_turned:
                    continue

                # Orqadagi talaba (S2) suhbatda ishtirok etyaptimi?
                if not self._is_rear_peer_attentive(kpts2, box2, s2, now):
                    continue

                if turn_conf > best_conf:
                    best_conf = turn_conf
                    best_peer = (s2, box2)

            # Agar haqiqiy orqadagi suhbatdosh topilsa
            if best_peer is not None:
                s2_match, box2_match = best_peer
                if s1.talking_start_time is None or s1.talking_partner_id != s2_match.track_id:
                    s1.talking_start_time = now
                    s1.talking_partner_id = s2_match.track_id
                else:
                    duration = now - s1.talking_start_time
                    if duration >= min_dur:
                        violations.append(ViolationEvent(
                            violation_type="TALKING",
                            description_uz="Orqaga o'girilib gaplashayotgan talaba",
                            confidence=best_conf,
                            box=box1,
                            peer_box=box2_match,
                            keypoints=kpts1
                        ))
            else:
                # Orqaga o'girilmagan yoki suhbatdosh yo'q -> taymerni bekor qilish
                s1.talking_start_time = None
                s1.talking_partner_id = None

        return violations

