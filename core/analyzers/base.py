import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

@dataclass
class ViolationEvent:
    """Qoidabuzarlik hodisasi ma'lumotlari."""
    violation_type: str                  # "SLEEPING", "PHONE", "TALKING"
    description_uz: str                  # O'zbekcha aniq tavsif
    confidence: float                    # 0.0 - 1.0 (Birlashgan ko'p bosqichli ishonchlilik darajasi)
    box: List[int]                       # Qoidabuzar b-box [x1, y1, x2, y2]
    keypoints: Optional[np.ndarray] = None  # Pose nuqtalari
    peer_box: Optional[List[int]] = None # Suhbatdosh b-box [x1, y1, x2, y2] (TALKING uchun)
    stage_scores: Dict[str, float] = field(default_factory=dict) # 4 ta bosqich baholari
    timestamp: float = field(default_factory=time.time)

class TrackedStudent:
    """
    Talabaning kadrlar davomidagi holatini uzluksiz kuzatish,
    vaqtinchalik filtrlash va harakat dispersiyasini tahlil qilish klassi.
    """
    def __init__(self, track_id: int, initial_box: List[int]):
        self.track_id = track_id
        self.box = initial_box
        self.last_seen = time.time()
        
        # Holat taymerlari (soniyalarda uzluksizlikni tekshirish)
        self.sleep_start_time: Optional[float] = None
        self.phone_start_time: Optional[float] = None
        self.talking_start_time: Optional[float] = None
        self.talking_partner_id: Optional[int] = None

        # Bosh harakati tarixi: [(timestamp, head_x, head_y), ...]
        self.head_history: List[Tuple[float, float, float]] = []

        # Tana pozitsiyasi tarixi: [(timestamp, cx, cy), ...] (yurishni aniqlash)
        cx = (initial_box[0] + initial_box[2]) / 2.0
        cy = (initial_box[1] + initial_box[3]) / 2.0
        self.pos_history: List[Tuple[float, float, float]] = [(time.time(), cx, cy)]

        # Qo'l pozitsiyalari tarixi (telefon ushlash uzluksizligini tekshirish)
        self.hand_history: List[Tuple[float, float, float]] = []

    def update_box(self, new_box: List[int]):
        """Yangi kadrda talaba joylashuvini yangilash."""
        now = time.time()
        self.box = new_box
        self.last_seen = now
        cx = (new_box[0] + new_box[2]) / 2.0
        cy = (new_box[1] + new_box[3]) / 2.0
        self.pos_history.append((now, cx, cy))
        if len(self.pos_history) > 30:
            self.pos_history = [(t, x, y) for t, x, y in self.pos_history if now - t <= 5.0]

    def is_walking(self, now: float, window_sec: float = 2.0) -> bool:
        """Kadrda yo'lak bo'ylab yurib ketayotgan odamni aniqlash (o'tirgan talaba emas)."""
        recent = [(t, cx, cy) for t, cx, cy in self.pos_history if now - t <= window_sec]
        if len(recent) < 3:
            return False
        total_dist = 0.0
        for k in range(1, len(recent)):
            total_dist += math.hypot(recent[k][1] - recent[k-1][1], recent[k][2] - recent[k-1][2])
        dt = recent[-1][0] - recent[0][0]
        if dt > 0.3 and (total_dist / dt) > 70.0:  # 70 px/sec dan tezroq ko'chsa
            return True
        return False

    def record_head_pos(self, now: float, hx: float, hy: float):
        """Bosh koordinatalari tarixini saqlash."""
        self.head_history.append((now, hx, hy))
        if len(self.head_history) > 60:
            self.head_history = [(t, x, y) for t, x, y in self.head_history if now - t <= 12.0]

    def record_hand_pos(self, now: float, hx: float, hy: float):
        """Kaftlar markazi koordinatasini saqlash."""
        self.hand_history.append((now, hx, hy))
        if len(self.hand_history) > 40:
            self.hand_history = [(t, x, y) for t, x, y in self.hand_history if now - t <= 8.0]

    def get_head_motion_span(self, now: float, window_sec: float = 5.0) -> Tuple[float, float]:
        """Oxirgi N soniyadagi bosh harakatining gorizontal va vertikal tebranish diapazoni (dx, dy)."""
        recent = [(x, y) for t, x, y in self.head_history if now - t <= window_sec]
        if len(recent) < 4:
            return 0.0, 0.0
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        return dx, dy

    def is_head_active(self, now: float, window_sec: float = 5.0) -> bool:
        """
        Konspekt yozayotgan, kitob o'qiyotgan talaba boshini muntazam harakatlantiradi.
        Haqiqiy uxlab yotgan talabada esa bosh deyarli qimirlamaydi (dx < 10, dy < 10).
        """
        dx, dy = self.get_head_motion_span(now, window_sec)
        return bool(dx > 12.0 or dy > 12.0)
