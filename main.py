import os
import sys
import logging

# Windows konsolida ANSI rang kodlari [32m xatosiz toza chiqishi uchun
if sys.platform == "win32":
    os.system("")

# FFmpeg va OpenCV xatoliklarini (masalan, HEVC POC ogohlantirishlarini) terminalga chiqmasligi uchun cheklash
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# CPU yuklanishini kamaytirish: OpenCV va PyTorch uchun oqimlar sonini optimallash
import cv2
import torch
cv2.setNumThreads(2)
if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(4)

import uvicorn
from pathlib import Path

# Log sozlamalari
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MAIN")

def main():
    logger.info("=" * 60)
    logger.info("   SMART CLASSROOM DETECTOR - AI PROCTORING SYSTEM   ")
    logger.info("=" * 60)
    
    # Loyiha ildiz yo'li
    project_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_dir))

    from config.settings import load_config
    cfg = load_config()
    system_cfg = cfg.get("system", {})
    host = system_cfg.get("host", "0.0.0.0")
    port = system_cfg.get("port", 8000)

    logger.info(f"Veb boshqaruv paneli ishga tushmoqda: http://127.0.0.1:{port}")
    logger.info(f"Lokal tarmoqdagi barcha qurilmalar uchun: http://<KOMPYUTER_IP>:{port}")

    # Uvicorn orqali veb-serverni ishga tushirish
    uvicorn.run("web.app:app", host=host, port=port, reload=False, log_level="info")

if __name__ == "__main__":
    main()
