import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import time
import logging
import threading
import numpy as np
from typing import Dict, Optional, Tuple, Any, List

logger = logging.getLogger("StreamManager")

class CameraStream:
    """Alohida kamera oqimini (RTSP/Webcam/Video) xavfsiz va kechikishsiz o'quvchi klass."""

    def __init__(self, cam_id: str, name: str, source: str, fps_limit: int = 15):
        self.cam_id = cam_id
        self.name = name
        self.source = int(source) if str(source).isdigit() else source
        self.fps_limit = fps_limit

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_seq = 0
        self._annotated_frame: Optional[np.ndarray] = None
        self._annotated_jpeg: Optional[bytes] = None
        self._last_frame_time = 0.0
        self.last_viewed_time = 0.0
        self.is_connected = False
        self.fps_actual = 0.0

    def start(self):
        """Oqimni orqa fonda ishga tushirish."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"[{self.name}] Oqim o'quvchi ishga tushdi: {self.source}")

    def stop(self):
        """Oqimni to'xtatish."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_cap()
        logger.info(f"[{self.name}] Oqim to'xtatildi.")

    def _release_cap(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.is_connected = False

    def _init_capture(self) -> bool:
        """VideoCapture obyektini ochish."""
        self._release_cap()
        try:
            # RTSP oqimlari uchun past kechikish, kengaytirilgan bufer va barqaror TCP uzatish
            if isinstance(self.source, str) and self.source.startswith("rtsp://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    "rtsp_transport;tcp|"
                    "buffer_size;2048000|"
                    "max_delay;500000|"
                    "reorder_queue_size;100|"
                    "stimeout;5000000"
                )
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:
                self._cap = cv2.VideoCapture(self.source)

            if self._cap.isOpened():
                self.is_connected = True
                logger.info(f"[{self.name}] Kamera muvaffaqiyatli ulandi.")
                return True
            else:
                logger.warning(f"[{self.name}] Kameraga ulanib bo'lmadi: {self.source}")
                return False
        except Exception as e:
            logger.error(f"[{self.name}] VideoCapture ochishda xatolik: {e}")
            return False

    def _capture_loop(self):
        """RTSP buferini to'ldirmasdan va HEVC kadrlarini uzmasdan eng so'nggi kadrni olish."""
        is_rtsp = isinstance(self.source, str) and self.source.startswith("rtsp://")
        fps_interval = 1.0 / max(1, self.fps_limit)
        frame_counter = 0
        fps_timer = time.time()
        seq = 0

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._init_capture():
                    time.sleep(3.0)  # Qayta ulanish oralig'i
                    continue

            # RTSP bo'lmagan manbalar (masalan, fayl) uchun FPS tezligini cheklash
            if not is_rtsp:
                now = time.time()
                elapsed = now - self._last_frame_time
                if elapsed < fps_interval:
                    time.sleep(min(fps_interval - elapsed, 0.01))

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning(f"[{self.name}] Kadr olinmadi, qayta ulanish kutilmoqda...")
                self._release_cap()
                time.sleep(2.0)
                continue

            now = time.time()
            self._last_frame_time = now
            frame_counter += 1
            seq += 1

            if now - fps_timer >= 1.0:
                self.fps_actual = frame_counter / (now - fps_timer)
                frame_counter = 0
                fps_timer = now

            with self._lock:
                self._latest_frame = frame
                self._latest_frame_seq = seq

    def touch_viewer(self):
        """Veb mijoz ushbu streamni ko'rayotganini belgilash."""
        self.last_viewed_time = time.time()

    def is_viewed(self) -> bool:
        """Oxirgi 4 soniya ichida stream ko'rilganmi?"""
        return (time.time() - self.last_viewed_time) < 4.0

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Tahlil uchun asl kadr nusxasini olish."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_latest_frame_with_seq(self) -> Tuple[Optional[np.ndarray], int]:
        """Tahlil uchun asl kadr va uning tartib raqamini olish (nusxasiz, tezkor)."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0
            return self._latest_frame, self._latest_frame_seq

    def set_annotated_frame(self, frame: np.ndarray):
        """Veb interfeysda ko'rsatish uchun chizilgan kadrni saqlash va faqat ko'rilayotgan bo'lsa JPEG kodlash."""
        jpeg_bytes = None
        # Faqat ayni paytda brauzerda tomosha qilinayotgan bo'lsagina JPEG ga kodlash (CPU ni 80% tejaydi!)
        if self.is_viewed():
            h, w = frame.shape[:2]
            if w > 1280:
                preview = cv2.resize(frame, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_LINEAR)
            else:
                preview = frame
            ret, buf = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                jpeg_bytes = buf.tobytes()

        with self._lock:
            self._annotated_frame = frame
            if jpeg_bytes is not None:
                self._annotated_jpeg = jpeg_bytes

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Veb interfeys uchun chizilgan kadrni olish."""
        with self._lock:
            if self._annotated_frame is not None:
                return self._annotated_frame
            return self._latest_frame

    def get_annotated_jpeg(self) -> Optional[bytes]:
        """Veb mijozlar uchun tayyor kodlangan JPEG baytlarini qaytarish."""
        self.touch_viewer()
        with self._lock:
            if self._annotated_jpeg is not None:
                return self._annotated_jpeg
            # Agar hali pre-encode qilinmagan bo'lsa, bir marta on-demand kodlash
            frame = self._annotated_frame if self._annotated_frame is not None else self._latest_frame
            if frame is not None:
                h, w = frame.shape[:2]
                preview = cv2.resize(frame, (1280, int(h * 1280 / w))) if w > 1280 else frame
                ret, buf = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    self._annotated_jpeg = buf.tobytes()
                    return self._annotated_jpeg
            return None


class StreamManager:
    """Barcha kameralarni boshqaruvchi yagona menejer."""

    def __init__(self, cameras_config: List[Dict[str, Any]]):
        self.cameras: Dict[str, CameraStream] = {}
        for cam in cameras_config:
            if cam.get("enabled", True):
                self.add_camera(cam)

    def add_camera(self, cam_cfg: Dict[str, Any]):
        cid = cam_cfg["id"]
        if cid in self.cameras:
            self.cameras[cid].stop()
        stream = CameraStream(
            cam_id=cid,
            name=cam_cfg.get("name", f"Kamera {cid}"),
            source=cam_cfg.get("source", "0"),
            fps_limit=cam_cfg.get("fps_limit", 15)
        )
        self.cameras[cid] = stream
        stream.start()

    def remove_camera(self, cam_id: str):
        if cam_id in self.cameras:
            self.cameras[cam_id].stop()
            del self.cameras[cam_id]

    def stop_all(self):
        for stream in self.cameras.values():
            stream.stop()
        self.cameras.clear()
