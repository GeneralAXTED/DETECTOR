import os
import cv2
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from core.violation_analyzer import ViolationEvent
from telegram.bot import TelegramNotifier

logger = logging.getLogger("AlertManager")

class AlertManager:
    """Qoidabuzarliklar haqida xabar berish, rasmga olish va Telegramga yuborishni boshqaruvchi klass."""

    def __init__(self, config: Dict[str, Any], telegram_notifier: TelegramNotifier):
        self.cfg = config
        self.telegram = telegram_notifier
        self.captures_dir = Path(config.get("system", {}).get("captures_dir", "captures"))
        self.captures_dir.mkdir(parents=True, exist_ok=True)

        # Anti-spam cooldown xotirasi: (camera_id, violation_type) -> float (last_alert_time)
        self.cooldowns: Dict[str, float] = {}
        self.cooldown_seconds = config.get("telegram", {}).get("alert_cooldown_seconds", 60)
        self.include_status = config.get("telegram", {}).get("include_status", True)

        # Oxirgi qoidabuzarliklar ro'yxati (Dashboard uchun)
        self.history: List[Dict[str, Any]] = []

        # Tizim ishga tushganda yoki xotira to'lib ketmasligi uchun eski rasmlarni tozalash
        self.cleanup_captures()

    def cleanup_captures(self):
        """Eski yoki qolib ketgan rasmlarni diskdan tozalash (disk to'lib ketishini oldini oladi)."""
        try:
            for f in self.captures_dir.glob("*.jpg"):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Captures tozalashda xatolik: {e}")

    def update_config(self, config: Dict[str, Any]):
        self.cfg = config
        self.cooldown_seconds = config.get("telegram", {}).get("alert_cooldown_seconds", 60)
        self.include_status = config.get("telegram", {}).get("include_status", True)

    def trigger_violation(
        self,
        camera_id: str,
        camera_name: str,
        violation: ViolationEvent,
        raw_frame
    ):
        """Qoidabuzarlikni qayd qilish, rasmga chizish va Telegramga jo'natish."""
        now = time.time()
        cooldown_key = f"{camera_id}_{violation.violation_type}"
        last_time = self.cooldowns.get(cooldown_key, 0.0)

        # Cooldown tekshiruvi (spamdan himoya)
        if (now - last_time) < self.cooldown_seconds:
            return  # Hali cooldown vaqti tugamagan

        self.cooldowns[cooldown_key] = now
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Rasmni chizish (Annotated Snapshot)
        annotated_snapshot = raw_frame.copy()
        h, w = annotated_snapshot.shape[:2]

        x1, y1, x2, y2 = violation.box
        color = (0, 0, 255)  # Qizil
        if violation.violation_type == "SLEEPING":
            color = (255, 100, 0)
        elif violation.violation_type == "TALKING":
            color = (0, 215, 255)  # Amber

        if not self.include_status:
            # Foydalanuvchi talabi: "o'chirsa qoidabuzarni belgilab jo'nataversin holat haqida hechnima yozmay"
            cv2.rectangle(annotated_snapshot, (x1, y1), (x2, y2), color, 3)
            if violation.violation_type == "TALKING" and violation.peer_box:
                px1, py1, px2, py2 = violation.peer_box
                cv2.rectangle(annotated_snapshot, (px1, py1), (px2, py2), color, 3)
                cx1, cy1 = (x1 + x2) // 2, (y1 + y2) // 2
                cx2, cy2 = (px1 + px2) // 2, (py1 + py2) // 2
                cv2.line(annotated_snapshot, (cx1, cy1), (cx2, cy2), color, 3)
                cv2.circle(annotated_snapshot, (cx1, cy1), 8, color, -1)
                cv2.circle(annotated_snapshot, (cx2, cy2), 8, color, -1)
        elif violation.violation_type == "TALKING":
            # 1-talaba (Orqaga qaragan talaba)
            cv2.rectangle(annotated_snapshot, (x1, y1), (x2, y2), color, 3)
            tag1 = f"! ORQAGA QARAGAN TALABA ({int(violation.confidence * 100)}%) !"
            (tw1, th1), _ = cv2.getTextSize(tag1, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated_snapshot, (x1, max(0, y1 - 32)), (x1 + tw1 + 10, y1), color, -1)
            cv2.putText(annotated_snapshot, tag1, (x1 + 5, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            # 2-talaba (Orqadagi talaba)
            if violation.peer_box:
                px1, py1, px2, py2 = violation.peer_box
                cv2.rectangle(annotated_snapshot, (px1, py1), (px2, py2), color, 3)
                tag2 = "! ORQADAGI TALABA !"
                (tw2, th2), _ = cv2.getTextSize(tag2, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(annotated_snapshot, (px1, max(0, py1 - 32)), (px1 + tw2 + 10, py1), color, -1)
                cv2.putText(annotated_snapshot, tag2, (px1 + 5, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

                # Bog'lash chizig'i
                cx1, cy1 = (x1 + x2) // 2, (y1 + y2) // 2
                cx2, cy2 = (px1 + px2) // 2, (py1 + py2) // 2
                cv2.line(annotated_snapshot, (cx1, cy1), (cx2, cy2), color, 3)
                cv2.circle(annotated_snapshot, (cx1, cy1), 8, color, -1)
                cv2.circle(annotated_snapshot, (cx2, cy2), 8, color, -1)

                # O'rtadagi nishon (ORQAGA O'GIRILISH)
                mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                badge = "ORQAGA O'GIRILISH"
                (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated_snapshot, (mx - bw // 2 - 8, my - 16), (mx + bw // 2 + 8, my + 12), (0, 0, 0), -1)
                cv2.rectangle(annotated_snapshot, (mx - bw // 2 - 8, my - 16), (mx + bw // 2 + 8, my + 12), color, 2)
                cv2.putText(annotated_snapshot, badge, (mx - bw // 2, my + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            cv2.rectangle(annotated_snapshot, (x1, y1), (x2, y2), color, 3)
            label = f"{violation.description_uz} ({int(violation.confidence * 100)}%)"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(annotated_snapshot, (x1, max(0, y1 - 30)), (x1 + lw + 10, y1), color, -1)
            cv2.putText(annotated_snapshot, label, (x1 + 5, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Pastki qismga kamera va vaqt bannerini joylash
        footer_text = f"{camera_name} | {timestamp_str} | SMART CLASSROOM DETECTOR"
        cv2.rectangle(annotated_snapshot, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.putText(annotated_snapshot, footer_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        # 2. Diskka saqlash
        filename = f"{violation.violation_type.lower()}_{camera_id}_{file_timestamp}.jpg"
        save_path = self.captures_dir / filename
        cv2.imwrite(str(save_path), annotated_snapshot)
        logger.info(f"Qoidabuzarlik surati saqlandi: {save_path}")

        # 3. Tarixga qo'shish
        event_dict = {
            "id": f"{int(now * 1000)}",
            "camera_id": camera_id,
            "camera_name": camera_name,
            "type": violation.violation_type,
            "title": "Orqaga o'girilib gaplashayotgan talaba" if violation.violation_type == "TALKING" else violation.description_uz,
            "confidence": f"{int(violation.confidence * 100)}%",
            "time": timestamp_str,
            "image": filename
        }
        self.history.insert(0, event_dict)
        if len(self.history) > 100:
            self.history.pop()

        # 4. Telegramga yuborish
        if not self.include_status:
            caption = (
                f"🚨 <b>QOIDABUZARLIK ANIQLANDI!</b>\n\n"
                f"📍 <b>Kamera / Auditoriya:</b> {camera_name}\n"
                f"⏰ <b>Sana va vaqt:</b> {timestamp_str}\n\n"
                f"<i>Tizim: Smart Classroom AI Detector</i>"
            )
        else:
            icon = "🚨"
            if violation.violation_type == "SLEEPING":
                icon = "😴"
            elif violation.violation_type == "PHONE":
                icon = "📱"
            elif violation.violation_type == "TALKING":
                icon = "🗣️"

            if violation.violation_type == "TALKING":
                caption = (
                    f"🗣️ <b>ORQAGA O'GIRILIB GAPLASHAYOTGAN TALABA!</b>\n\n"
                    f"📍 <b>Kamera / Auditoriya:</b> {camera_name}\n"
                    f"⚠️ <b>Holat:</b> Talaba orqadagi partadagilar bilan gaplashmoqda\n"
                    f"👥 <b>Ishtirokchilar:</b> Orqaga qaragan talaba va orqadagi suhbatdosh\n"
                    f"📊 <b>Ishonchlilik darajasi:</b> {int(violation.confidence * 100)}%\n"
                    f"⏰ <b>Sana va vaqt:</b> {timestamp_str}\n\n"
                    f"<i>Tizim: Smart Classroom AI Detector</i>"
                )
            else:
                caption = (
                    f"{icon} <b>QOIDABUZARLIK ANIQLANDI!</b>\n\n"
                    f"📍 <b>Kamera / Auditoriya:</b> {camera_name}\n"
                    f"⚠️ <b>Holat:</b> {violation.description_uz}\n"
                    f"📊 <b>Ishonchlilik darajasi:</b> {int(violation.confidence * 100)}%\n"
                    f"⏰ <b>Sana va vaqt:</b> {timestamp_str}\n\n"
                    f"<i>Tizim: Smart Classroom AI Detector</i>"
                )
        self.telegram.send_violation_alert(str(save_path), caption)
