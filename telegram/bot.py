import os
import time
import queue
import logging
import threading
import requests
from typing import Optional, Tuple

logger = logging.getLogger("TelegramBot")

class TelegramNotifier:
    """Telegram orqali bildirishnomalarni asinxron/navbat bilan jo'natuvchi xizmat."""

    def __init__(self, bot_token: str = "", chat_id: str = "", enabled: bool = False):
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.enabled = enabled
        self._queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def update_credentials(self, bot_token: str, chat_id: str, enabled: bool):
        """Token va Chat ID larni yangilash."""
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.enabled = enabled
        logger.info(f"Telegram sozlamalari yangilandi: enabled={enabled}, chat_id={chat_id}")

    @staticmethod
    def test_credentials(bot_token: str, chat_id: str) -> Tuple[bool, str]:
        """Kiritilgan Bot Token va Chat ID to'g'riligini sinovdan o'tkazish."""
        if not bot_token or not chat_id:
            return False, "Bot Token yoki Chat ID kiritilmagan!"

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": "✅ <b>DETECTOR Tizimi</b>\n\nTelegram bot muvaffaqiyatli ulandi! Tizim qoidabuzarliklarni ushbu kanalga yuboradi.",
            "parse_mode": "HTML"
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            data = resp.json()
            if data.get("ok"):
                return True, "Xabar muvaffaqiyatli yuborildi!"
            else:
                return False, f"Telegram API xatosi: {data.get('description', 'Nomaʼlum xatolik')}"
        except Exception as e:
            return False, f"Ulanishda xatolik: {str(e)}"

    def send_violation_alert(self, photo_path: str, caption: str):
        """Qoidabuzarlik rasmi va ma'lumotini navbatga qo'shish."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            # Telegram yoqilmagan bo'lsa, xotira to'lib ketmasligi uchun rasm faylini o'chirish
            try:
                if os.path.exists(photo_path):
                    os.remove(photo_path)
            except Exception:
                pass
            return

        try:
            self._queue.put_nowait((photo_path, caption))
        except queue.Full:
            logger.warning("Telegram jo'natish navbati to'lib ketdi, eski xabarlar kutilmoqda.")
            try:
                if os.path.exists(photo_path):
                    os.remove(photo_path)
            except Exception:
                pass

    def _worker_loop(self):
        """Orqa fonda xabarlarni jo'natib turuvchi oqim."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            photo_path, caption = item
            self._send_photo_request(photo_path, caption)
            self._queue.task_done()
            time.sleep(0.5)  # Telegram API tezlik cheklovidan himoya

    def _send_photo_request(self, photo_path: str, caption: str):
        if not self.bot_token or not self.chat_id:
            try:
                if os.path.exists(photo_path):
                    os.remove(photo_path)
            except Exception:
                pass
            return

        if not os.path.exists(photo_path):
            logger.error(f"Telegramga jo'natish uchun rasm topilmadi: {photo_path}")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        try:
            with open(photo_path, "rb") as img_file:
                files = {"photo": img_file}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": "HTML"
                }
                resp = requests.post(url, data=data, files=files, timeout=20)
                res_data = resp.json()
                if not res_data.get("ok"):
                    logger.error(f"Telegramga rasm jo'natib bo'lmadi: {res_data.get('description')}")
                else:
                    logger.info(f"Qoidabuzarlik rasmi Telegramga muvaffaqiyatli yuborildi: {photo_path}")
        except Exception as e:
            logger.error(f"Telegram so'rovida xatolik: {str(e)}")
        finally:
            # Telegramga tashlanganidan so'ng disk to'lib ketmasligi uchun fayl darhol o'chiriladi
            try:
                if os.path.exists(photo_path):
                    os.remove(photo_path)
                    logger.info(f"Diskni tejash uchun rasm fayli o'chirildi: {photo_path}")
            except Exception as del_err:
                logger.warning(f"Rasmni o'chirishda xatolik ({photo_path}): {del_err}")

    def stop(self):
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)
