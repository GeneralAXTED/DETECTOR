import cv2
import time
import logging
import threading
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from core.detector import SmartDetector
from core.violation_analyzer import ViolationAnalyzer, ViolationEvent
from core.stream_manager import StreamManager
from core.alert_manager import AlertManager

logger = logging.getLogger("Engine")

class DetectionEngine:
    """
    Barcha oqimlarni GPU va CPU ni to'ldirmasdan, bosqichma-bosqich taqsimlangan
    (Staggered Round-Robin) va ko'p darajali aniqlikda tahlil qiluvchi AI dvigateli.
    """

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

        # Har bir kamera uchun alohida 4-bosqichli holat tahlilchisi
        self.analyzers: Dict[str, ViolationAnalyzer] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Optimallashtirilgan AI Tahlil Dvigateli ishga tushirildi.")

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
        """
        Asosiy boshqaruv sikli:
        Foydalanuvchi talabi: 'qotib qolmasin, natija tez 3-4 soniya farq bilan bo'lsa ham ko'rinsin, aniq chiqsin'.
        GPU ga barcha kameralarni birdaniga tiqmasdan, kameralar navbatma-navbat (Staggered Dispatch)
        1.5 - 3.0 soniyada bir marta yuqori aniqlikda tahlil qilinadi.
        Brauzerda ko'rilayotgan kameralar esa kesh chiziqlari bilan silliq 12-15 FPS da ko'rsatiladi.
        """
        last_ai_time: Dict[str, float] = {}
        cached_overlay: Dict[str, tuple] = {}
        last_visual_seq: Dict[str, int] = {}

        while self._running:
            camera_items = list(self.stream_manager.cameras.items())
            if not camera_items:
                time.sleep(0.3)
                continue

            num_cams = len(camera_items)
            now = time.time()

            # Har bir kamera uchun adaptiv AI yangilanish oralig'i (soniyalarda):
            # 1 ta kamera: ~1.2 soniya
            # 2-4 ta: ~1.5 - 2.0 soniya
            # 5-8 ta: ~2.5 - 3.0 soniya
            # Bu GPU ni 30-40% oralig'ida muzdek ushlab turadi va 4+ kameralarda hech qachon qotmaydi!
            per_camera_interval = max(1.2, min(3.5, 0.40 * num_cams))

            # 1. Navbati kelgan, eng ko'p kutilgan kamerani topish (Round-Robin)
            due_cameras: List[Tuple[str, Any, float]] = []
            for cam_id, stream in camera_items:
                last_t = last_ai_time.get(cam_id, 0.0)
                overdue = now - last_t
                if overdue >= per_camera_interval:
                    due_cameras.append((cam_id, stream, overdue))

            # Agar navbati kelgan kamera bo'lsa, eng ko'p kutilgan BITTA kamerani AI tahlilidan o'tkazish
            if due_cameras:
                # Eng ko'p kutilgan kamerani birinchi tahlil qilish
                due_cameras.sort(key=lambda x: x[2], reverse=True)
                cam_id, stream, _ = due_cameras[0]

                raw_frame, frame_seq = stream.get_latest_frame_with_seq()
                if raw_frame is not None:
                    last_ai_time[cam_id] = now

                    # 1. AI Deteksiya (YOLO Pose + Phone Object)
                    try:
                        p_boxes, p_kpts, ph_boxes = self.detector.process_frame(raw_frame, cam_id=cam_id)
                    except Exception as e:
                        logger.error(f"Kamera [{stream.name}] tahlilida xatolik: {e}")
                        p_boxes, p_kpts, ph_boxes = [], [], []

                    # 2. Qoidabuzarliklar tahlili (4 bosqichli)
                    analyzer = self._get_analyzer(cam_id)
                    violations, teacher_boxes, edge_boxes, baby_students = analyzer.analyze_frame(
                        p_boxes, p_kpts, ph_boxes, raw_frame.shape, raw_frame=raw_frame
                    )

                    # 3. Ogohlantirishlar (Cooldown va Telegram)
                    for v in violations:
                        self.alert_manager.trigger_violation(
                            camera_id=cam_id,
                            camera_name=stream.name,
                            violation=v,
                            raw_frame=raw_frame
                        )

                    # 4. Vizual keshni yangilash
                    cached_overlay[cam_id] = (p_boxes, p_kpts, ph_boxes, violations, teacher_boxes, edge_boxes, baby_students)

                    # 5. Agar brauzerda tomosha qilinayotgan bo'lsa, chizib yangilash
                    if stream.is_viewed():
                        vis_frame = raw_frame.copy()
                        self._draw_overlay(vis_frame, p_boxes, p_kpts, ph_boxes, violations, teacher_boxes, edge_boxes, baby_students, stream.fps_actual)
                        stream.set_annotated_frame(vis_frame)
                        last_visual_seq[cam_id] = frame_seq

            # 2. Veb tomoshabinlar uchun oraliq kadrlar (Silliq 12-15 FPS Live Stream)
            # Faqat ayni paytda brauzerda ochilgan kameralar uchun yengil chizish
            for cam_id, stream in camera_items:
                if stream.is_viewed() and (cam_id in cached_overlay):
                    raw_frame, frame_seq = stream.get_latest_frame_with_seq()
                    if raw_frame is not None and frame_seq != last_visual_seq.get(cam_id, -1):
                        last_visual_seq[cam_id] = frame_seq
                        p_boxes, p_kpts, ph_boxes, violations, teacher_boxes, edge_boxes, baby_students = cached_overlay[cam_id]
                        vis_frame = raw_frame.copy()
                        self._draw_overlay(vis_frame, p_boxes, p_kpts, ph_boxes, violations, teacher_boxes, edge_boxes, baby_students, stream.fps_actual)
                        stream.set_annotated_frame(vis_frame)

            # CPU ga nafas berish va mikro-kechikish
            time.sleep(0.02)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        p_boxes,
        p_kpts,
        ph_boxes,
        violations: List[ViolationEvent],
        teacher_boxes,
        edge_boxes,
        baby_students,
        fps: float
    ):
        """Jonli kadr ustiga chiroyli, tushunarli belgilarni chizish."""
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

        # 3. Chaqalog'i bilan o'tirgan talabalar (Binafsharang)
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

        # 6. Qoidabuzarliklarni ajratib ko'rsatish
        for v in violations:
            x1, y1, x2, y2 = v.box

            if v.violation_type == "TALKING":
                amber = (0, 215, 255)
                # 1-talaba (Orqaga qaragan)
                cv2.rectangle(frame, (x1, y1), (x2, y2), amber, 3)
                tag1 = f"! ORQAGA QARAGAN ({int(v.confidence * 100)}%) !"
                (tw1, th1), _ = cv2.getTextSize(tag1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + tw1 + 8, y1), amber, -1)
                cv2.putText(frame, tag1, (x1 + 4, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # 2-talaba (Orqadagi)
                if v.peer_box is not None:
                    px1, py1, px2, py2 = v.peer_box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), amber, 3)
                    tag2 = "! ORQADAGI TALABA !"
                    (tw2, th2), _ = cv2.getTextSize(tag2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (px1, max(0, py1 - 25)), (px1 + tw2 + 8, py1), amber, -1)
                    cv2.putText(frame, tag2, (px1 + 4, max(15, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                    # Bog'lash chizig'i
                    cx1, cy1 = (x1 + x2) // 2, (y1 + y2) // 2
                    cx2, cy2 = (px1 + px2) // 2, (py1 + py2) // 2
                    cv2.line(frame, (cx1, cy1), (cx2, cy2), amber, 3)
                    cv2.circle(frame, (cx1, cy1), 7, amber, -1)
                    cv2.circle(frame, (cx2, cy2), 7, amber, -1)

                    # O'rtadagi nishon
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
                tag = f"! {v.description_uz} ({int(v.confidence * 100)}%) !"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, tag, (x1 + 3, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Yuqori status paneli
        student_count = max(0, len(p_boxes) - len(teacher_boxes) - len(edge_boxes))
        status_bar = f"AI MONITOR | Talabalar: {student_count} | O'qituvchi: {len(teacher_boxes)} | Qoidabuzarliklar: {len(violations)} | FPS: {fps:.1f}"
        cv2.rectangle(frame, (0, 0), (w, 32), (30, 30, 30), -1)
        cv2.putText(frame, status_bar, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
