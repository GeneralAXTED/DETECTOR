import cv2
import time
import logging
import threading
import numpy as np
from typing import Dict, Any, Optional
from core.detector import SmartDetector
from core.violation_analyzer import ViolationAnalyzer
from core.stream_manager import StreamManager
from core.alert_manager import AlertManager

logger = logging.getLogger("Engine")

class DetectionEngine:
    """Barcha oqimlarni davriy tahlil qilib boruvchi asosiy AI dvigateli."""

    def __init__(
        self,
        config: Dict[str, Any],
        stream_manager: StreamManager,
        detector: SmartDetector,
        alert_manager: AlertManager
    ):
        self.config = config
        self.stream_manager = stream_manager
        self.detector = detector
        self.alert_manager = alert_manager

        # Har bir kamera uchun alohida holat tahlilchisi (Trackers)
        self.analyzers: Dict[str, ViolationAnalyzer] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("AI Tahlil Dvigateli ishga tushirildi.")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("AI Tahlil Dvigateli to'xtatildi.")

    def _get_analyzer(self, cam_id: str) -> ViolationAnalyzer:
        if cam_id not in self.analyzers:
            self.analyzers[cam_id] = ViolationAnalyzer(self.config)
        return self.analyzers[cam_id]

    def _loop(self):
        """Asosiy sikl: Barcha faol kameralarni GPU va CPU ni to'ldirmasdan, adaptiv tahlil qilish."""
        last_ai_time: Dict[str, float] = {}
        cached_overlay: Dict[str, tuple] = {}
        last_visual_seq: Dict[str, int] = {}

        while self._running:
            camera_items = list(self.stream_manager.cameras.items())
            if not camera_items:
                time.sleep(0.3)
                continue

            num_cams = len(camera_items)
            # Optimal adaptiv AI tahlil oralig'i (GPU yukini 60-70% oralig'ida barqaror ushlab turadi):
            # 1 ta kamera: 0.20 soniya (~5 FPS)
            # 2 ta kamera: 0.25 soniya (~4 FPS)
            # 3-4 ta: 0.28 soniya (~3.5 FPS)
            # 5-8 ta: 0.35 soniya (~2.8 FPS)
            # 9+ ta: min(0.60, 0.045 * num_cams)
            if num_cams <= 1:
                min_ai_interval = 0.20
            elif num_cams == 2:
                min_ai_interval = 0.25
            elif num_cams <= 4:
                min_ai_interval = 0.28
            elif num_cams <= 8:
                min_ai_interval = 0.35
            else:
                min_ai_interval = min(0.60, 0.045 * num_cams)

            processed_any = False
            now = time.time()

            for cam_id, stream in camera_items:
                if not self._running:
                    break

                raw_frame, frame_seq = stream.get_latest_frame_with_seq()
                if raw_frame is None:
                    continue

                last_t = last_ai_time.get(cam_id, 0.0)
                is_ai_due = (now - last_t >= min_ai_interval)

                if is_ai_due:
                    last_ai_time[cam_id] = now
                    processed_any = True

                    # 1. AI Deteksiya (YOLO Pose + Phone Object)
                    try:
                        p_boxes, p_kpts, ph_boxes = self.detector.process_frame(raw_frame, cam_id=cam_id)
                    except Exception as e:
                        logger.error(f"Kamera [{stream.name}] tahlilida xatolik: {e}")
                        continue

                    # 2. Qoidabuzarliklar tahlili
                    analyzer = self._get_analyzer(cam_id)
                    violations, teacher_boxes, edge_boxes, baby_students = analyzer.analyze_frame(
                        p_boxes, p_kpts, ph_boxes, raw_frame.shape, raw_frame=raw_frame
                    )

                    # 3. Ogohlantirishlarni ishga tushirish (Cooldown va Telegram)
                    for v in violations:
                        self.alert_manager.trigger_violation(
                            camera_id=cam_id,
                            camera_name=stream.name,
                            violation=v,
                            raw_frame=raw_frame
                        )

                    # 4. Keshni saqlash
                    cached_overlay[cam_id] = (p_boxes, p_kpts, ph_boxes, violations, teacher_boxes, edge_boxes, baby_students)

                    # 5. Jonli kadr ustiga vizualizatsiya chizish (Web Dashboard uchun)
                    vis_frame = raw_frame.copy()
                    self._draw_overlay(vis_frame, p_boxes, p_kpts, ph_boxes, violations, teacher_boxes, edge_boxes, baby_students, stream.fps_actual)
                    stream.set_annotated_frame(vis_frame)
                    last_visual_seq[cam_id] = frame_seq

                elif stream.is_viewed() and (cam_id in cached_overlay) and (frame_seq != last_visual_seq.get(cam_id, -1)):
                    # AI oraliq kadrida: agar veb foydalanuvchi streamni tomosha qilayotgan bo'lsa,
                    # keshdagi chiziqlar bilan yengil yangilash (silliq 10-15 FPS video oqimi uchun)
                    last_visual_seq[cam_id] = frame_seq
                    p_boxes, p_kpts, ph_boxes, violations, teacher_boxes, edge_boxes, baby_students = cached_overlay[cam_id]
                    vis_frame = raw_frame.copy()
                    self._draw_overlay(vis_frame, p_boxes, p_kpts, ph_boxes, violations, teacher_boxes, edge_boxes, baby_students, stream.fps_actual)
                    stream.set_annotated_frame(vis_frame)

            # Agar bu siklda og'ir AI ishlamagan bo'lsa, CPU ga nafas berish
            if not processed_any:
                time.sleep(0.005)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        p_boxes,
        p_kpts,
        ph_boxes,
        violations,
        teacher_boxes,
        edge_boxes,
        baby_students,
        fps: float
    ):
        """Jonli kadr ustiga chiroyli ma'lumotlarni chizish."""
        h, w = frame.shape[:2]

        violation_boxes = [v.box for v in violations]
        peer_boxes = [v.peer_box for v in violations if v.peer_box is not None]
        baby_student_boxes = [b[0] for b in baby_students]
        baby_child_boxes = [b[1] for b in baby_students if b[1] is not None]

        # 1. O'qituvchi b-boxlari (Moviy / Ko'k rang)
        for t_box in teacher_boxes:
            cv2.rectangle(frame, (t_box[0], t_box[1]), (t_box[2], t_box[3]), (255, 180, 0), 2)
            cv2.rectangle(frame, (t_box[0], max(0, t_box[1] - 24)), (t_box[0] + 130, t_box[1]), (255, 180, 0), -1)
            cv2.putText(frame, "O'QITUVCHI", (t_box[0] + 5, max(15, t_box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 2. Chegaradagi / O'tayotgan shaxslar (Kulrang, nofaol)
        for e_box in edge_boxes:
            cv2.rectangle(frame, (e_box[0], e_box[1]), (e_box[2], e_box[3]), (120, 120, 120), 1)
            cv2.putText(frame, "CHEGARA", (e_box[0] + 3, max(15, e_box[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # 3. Chaqalog'i bilan o'tirgan talabalar (Binafsharang / Pushti)
        for s_box, ch_box in baby_students:
            cv2.rectangle(frame, (s_box[0], s_box[1]), (s_box[2], s_box[3]), (180, 105, 255), 2)
            tag = "CHAQALOQLI TALABA"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (s_box[0], max(0, s_box[1] - 24)), (s_box[0] + tw + 8, s_box[1]), (180, 105, 255), -1)
            cv2.putText(frame, tag, (s_box[0] + 4, max(15, s_box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if ch_box:
                cv2.rectangle(frame, (ch_box[0], ch_box[1]), (ch_box[2], ch_box[3]), (220, 150, 255), 2)
                cv2.putText(frame, "CHAQALOQ", (ch_box[0], max(15, ch_box[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 150, 255), 1)

        # 4. Talabalar b-boxlari (Yashil/Oddiy holat)
        special_boxes = violation_boxes + peer_boxes + teacher_boxes + edge_boxes + baby_student_boxes + baby_child_boxes
        for box in p_boxes:
            if box not in special_boxes:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 100), 2)

        # 5. Telefonlar b-boxlari (To'q sariq)
        for ph in ph_boxes:
            cv2.rectangle(frame, (ph[0], ph[1]), (ph[2], ph[3]), (0, 140, 255), 2)
            cv2.putText(frame, "TELEFON", (ph[0], max(15, ph[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        # 6. Qoidabuzarliklarni ajratib ko'rsatish va Suhbatdoshni aniq bog'lash
        for v in violations:
            x1, y1, x2, y2 = v.box

            if v.violation_type == "TALKING":
                amber = (0, 215, 255)
                # 1-talaba (Orqaga qaragan talaba)
                cv2.rectangle(frame, (x1, y1), (x2, y2), amber, 3)
                tag1 = "! ORQAGA QARAGAN TALABA !"
                (tw1, th1), _ = cv2.getTextSize(tag1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + tw1 + 8, y1), amber, -1)
                cv2.putText(frame, tag1, (x1 + 4, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # 2-talaba (Orqadagi talaba)
                if v.peer_box is not None:
                    px1, py1, px2, py2 = v.peer_box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), amber, 3)
                    tag2 = "! ORQADAGI TALABA !"
                    (tw2, th2), _ = cv2.getTextSize(tag2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (px1, max(0, py1 - 25)), (px1 + tw2 + 8, py1), amber, -1)
                    cv2.putText(frame, tag2, (px1 + 4, max(15, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                    # Bog'lash chizig'i (Connecting line)
                    cx1, cy1 = (x1 + x2) // 2, (y1 + y2) // 2
                    cx2, cy2 = (px1 + px2) // 2, (py1 + py2) // 2
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), amber, 3)
                    cv2.circle(frame, (cx1, cy1), 7, amber, -1)
                    cv2.circle(frame, (cx2, cy2), 7, amber, -1)

                    # O'rtadagi nishon (ORQAGA O'GIRILISH)
                    mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                    badge = "ORQAGA O'GIRILISH"
                    (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (mx - bw // 2 - 6, my - 13), (mx + bw // 2 + 6, my + 9), (0, 0, 0), -1)
                    cv2.rectangle(frame, (mx - bw // 2 - 6, my - 13), (mx + bw // 2 + 6, my + 9), amber, 2)
                    cv2.putText(frame, badge, (mx - bw // 2, my + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, amber, 2)

            else:
                color = (0, 0, 255)  # Qizil
                if v.violation_type == "SLEEPING":
                    color = (255, 120, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                tag = f"! {v.description_uz} !"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, tag, (x1 + 3, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Yuqori panel (Status bar)
        student_count = max(0, len(p_boxes) - len(teacher_boxes) - len(edge_boxes))
        status_bar = f"AI MONITOR | Talabalar: {student_count} | O'qituvchi: {len(teacher_boxes)} | Chaqaloqli: {len(baby_students)} | Qoidabuzarliklar: {len(violations)} | FPS: {fps:.1f}"
        cv2.rectangle(frame, (0, 0), (w, 32), (30, 30, 30), -1)
        cv2.putText(frame, status_bar, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
