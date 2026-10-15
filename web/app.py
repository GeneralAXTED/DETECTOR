import os
import cv2
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config.settings import load_config, save_config
from core.stream_manager import StreamManager
from core.detector import SmartDetector
from core.alert_manager import AlertManager
from core.engine import DetectionEngine
from telegram.bot import TelegramNotifier

BASE_DIR = Path(__file__).resolve().parent.parent
CAPTURES_DIR = BASE_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

app = FastAPI(title="Smart Classroom Detector")

# Global obyektlar
config: Dict[str, Any] = load_config()

# Telegram
tg_cfg = config.get("telegram", {})
telegram_notifier = TelegramNotifier(
    bot_token=tg_cfg.get("bot_token", ""),
    chat_id=tg_cfg.get("chat_id", ""),
    enabled=tg_cfg.get("enabled", False)
)

# Stream & Detector & Engine
stream_manager = StreamManager(config.get("cameras", []))
detector = SmartDetector(
    device=config.get("system", {}).get("device", "cuda"),
    conf_thresh=config.get("detection", {}).get("confidence_threshold", 0.25),
    imgsz=config.get("system", {}).get("imgsz", 960)
)
alert_manager = AlertManager(config, telegram_notifier)
engine = DetectionEngine(config, stream_manager, detector, alert_manager)

# Dvigatelni ishga tushirish
engine.start()


def generate_next_camera_id(cameras: list) -> str:
    """Yangi kamera uchun tartibli avtomatik ID yaratish (cam_1, cam_2, ...)."""
    existing_ids = {str(c.get("id", "")).strip() for c in cameras}
    idx = 1
    while f"cam_{idx}" in existing_ids:
        idx += 1
    return f"cam_{idx}"


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Asosiy boshqaruv paneli."""
    cfg = load_config()
    cameras = cfg.get("cameras", [])
    next_cam_id = generate_next_camera_id(cameras)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "config": cfg,
            "cameras": cameras,
            "cameras_json": json.dumps(cameras),
            "next_cam_id": next_cam_id,
            "history": alert_manager.history[:20]
        }
    )


@app.get("/api/cameras/{cam_id}/stream")
def video_feed(cam_id: str):
    """Kamera oqimini MJPEG formatida yuqori tezlikda (pre-encoded JPEG) uzatish."""
    stream = stream_manager.cameras.get(cam_id)
    if not stream:
        return JSONResponse({"error": "Kamera topilmadi"}, status_code=404)

    def frame_generator():
        while True:
            stream.touch_viewer()
            # 1. Tayyor pre-encoded yengil JPEG baytlarini olish (CPU 0% sarflanadi)
            jpeg_bytes = stream.get_annotated_jpeg()
            if jpeg_bytes is None:
                frame = stream.get_annotated_frame()
                if frame is not None:
                    h, w = frame.shape[:2]
                    preview = cv2.resize(frame, (1280, int(h * 1280 / w))) if w > 1280 else frame
                    ret, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        jpeg_bytes = buffer.tobytes()

            if jpeg_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            time.sleep(0.08)  # ~12 FPS: monitoring uchun optimal va yengil

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/captures/{filename}")
def get_capture_image(filename: str):
    """Saqlangan qoidabuzarlik rasmini qaytarish."""
    file_path = CAPTURES_DIR / filename
    if file_path.exists():
        return FileResponse(str(file_path))

    # Agar rasm Telegramga yuborilgach xotirani tejash uchun o'chirilgan bo'lsa, chiroyli SVG placeholder qaytariladi
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
        <rect width="100%" height="100%" fill="#111827"/>
        <circle cx="320" cy="135" r="42" fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>
        <path d="M302 137 L340 119 L323 153 L318 140 Z" fill="#38bdf8"/>
        <text x="320" y="210" fill="#f3f4f6" font-family="system-ui, -apple-system, sans-serif" font-size="16" font-weight="bold" text-anchor="middle">Telegramga yuborildi</text>
        <text x="320" y="235" fill="#9ca3af" font-family="system-ui, -apple-system, sans-serif" font-size="12" text-anchor="middle">Xotirani tejash maqsadida rasm fayli diskdan tozalangan</text>
    </svg>"""
    return Response(content=svg_content, media_type="image/svg+xml")


@app.get("/api/history")
def get_history():
    """Qoidabuzarliklar tarixini qaytarish."""
    return JSONResponse(alert_manager.history)


@app.post("/api/telegram/save")
async def save_telegram_settings(
    bot_token: str = Form(...),
    chat_id: str = Form(...),
    enabled: bool = Form(False),
    cooldown: int = Form(60),
    include_status: bool = Form(True)
):
    """Telegram parametrlarini saqlash va yangilash."""
    global config
    config = load_config()
    config["telegram"]["bot_token"] = bot_token.strip()
    config["telegram"]["chat_id"] = chat_id.strip()
    config["telegram"]["enabled"] = enabled
    config["telegram"]["alert_cooldown_seconds"] = cooldown
    config["telegram"]["include_status"] = include_status
    save_config(config)

    telegram_notifier.update_credentials(bot_token, chat_id, enabled)
    alert_manager.update_config(config)

    return JSONResponse({"success": True, "message": "Telegram sozlamalari muvaffaqiyatli saqlandi!"})


@app.post("/api/telegram/test")
async def test_telegram_settings(bot_token: str = Form(...), chat_id: str = Form(...)):
    """Telegram ulanishini sinab ko'rish."""
    ok, msg = TelegramNotifier.test_credentials(bot_token.strip(), chat_id.strip())
    return JSONResponse({"success": ok, "message": msg})


@app.get("/api/telegram/detect-chat-id")
async def detect_chat_id(bot_token: str):
    """getUpdates orqali oxirgi xabar kelgan kanal yoki guruh ID sini avtomatik aniqlash."""
    import requests
    if not bot_token:
        return JSONResponse({"success": False, "message": "Bot token kiritilmadi!"})

    url = f"https://api.telegram.org/bot{bot_token.strip()}/getUpdates"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if not data.get("ok"):
            return JSONResponse({"success": False, "message": f"Telegram API xatosi: {data.get('description')}"})

        results = data.get("result", [])
        if not results:
            return JSONResponse({
                "success": False,
                "message": "Hozircha yangi xabar topilmadi. Iltimos, Telegramdagi kanalingizga biror xabar (masalan: 'test') yuboring va qayta bosing!"
            })

        # Oxirgi yangilanishdan chat id ni qidirish
        detected_id = None
        chat_title = ""
        for update in reversed(results):
            if "channel_post" in update:
                detected_id = str(update["channel_post"]["chat"]["id"])
                chat_title = update["channel_post"]["chat"].get("title", "Kanal")
                break
            elif "message" in update:
                detected_id = str(update["message"]["chat"]["id"])
                chat_title = update["message"]["chat"].get("title") or update["message"]["chat"].get("first_name", "Guruh/Chat")
                break

        if detected_id:
            return JSONResponse({
                "success": True,
                "chat_id": detected_id,
                "title": chat_title,
                "message": f"Kanal ID topildi: {detected_id} ({chat_title})"
            })
        else:
            return JSONResponse({"success": False, "message": "Kanal yoki guruh ID topilmadi."})

    except Exception as e:
        return JSONResponse({"success": False, "message": f"Ulanish xatosi: {str(e)}"})


@app.post("/api/cameras/add")
async def add_camera(
    name: str = Form(...),
    source: str = Form(...),
    cam_id: Optional[str] = Form(None),
    fps_limit: int = Form(15)
):
    """Yangi kamera qo'shish (ID avtomatik beriladi)."""
    global config
    config = load_config()
    cameras = config.get("cameras", [])

    # Agar cam_id ko'rsatilmagan yoki bo'sh bo'lsa, avtomatik ID berish
    if not cam_id or not cam_id.strip():
        final_cam_id = generate_next_camera_id(cameras)
    else:
        final_cam_id = cam_id.strip()
        # Agar bu ID allaqachon mavjud bo'lsa, avtomatik takrorlanmas ID ga almashtirish
        if any(c["id"] == final_cam_id for c in cameras):
            final_cam_id = generate_next_camera_id(cameras)

    new_cam = {
        "id": final_cam_id,
        "name": name.strip(),
        "source": source.strip(),
        "enabled": True,
        "fps_limit": fps_limit
    }
    cameras.append(new_cam)
    config["cameras"] = cameras
    save_config(config)

    stream_manager.add_camera(new_cam)
    return JSONResponse({"success": True, "message": f"Kamera '{name}' (ID: {final_cam_id}) muvaffaqiyatli qo'shildi!"})


@app.post("/api/cameras/edit")
async def edit_camera(
    cam_id: str = Form(...),
    name: str = Form(...),
    source: str = Form(...),
    fps_limit: int = Form(15),
    enabled: Optional[bool] = Form(None)
):
    """Mavjud kamerani tahrirlash."""
    global config
    config = load_config()
    cameras = config.get("cameras", [])

    target_cam = None
    for c in cameras:
        if c["id"] == cam_id:
            c["name"] = name.strip()
            c["source"] = source.strip()
            c["fps_limit"] = fps_limit
            if enabled is not None:
                c["enabled"] = enabled
            target_cam = c
            break

    if not target_cam:
        return JSONResponse({"success": False, "message": "Kamera topilmadi!"}, status_code=404)

    config["cameras"] = cameras
    save_config(config)

    # Oqimni yangilash
    if target_cam.get("enabled", True):
        stream_manager.add_camera(target_cam)
    else:
        stream_manager.remove_camera(cam_id)

    return JSONResponse({"success": True, "message": f"Kamera '{name}' muvaffaqiyatli tahrirlandi!"})


@app.post("/api/cameras/delete")
async def delete_camera(cam_id: str = Form(...)):
    """Kamerani o'chirish."""
    global config
    config = load_config()
    cameras = [c for c in config.get("cameras", []) if c["id"] != cam_id]
    config["cameras"] = cameras
    save_config(config)

    stream_manager.remove_camera(cam_id)
    return JSONResponse({"success": True, "message": "Kamera o'chirildi!"})


@app.post("/api/cameras/toggle")
async def toggle_camera(cam_id: str = Form(...), enabled: bool = Form(...)):
    """Kamerani yoqish yoki o'chirish (faollashtirish)."""
    global config
    config = load_config()
    target_cam = None
    for c in config.get("cameras", []):
        if c["id"] == cam_id:
            c["enabled"] = enabled
            target_cam = c
            break

    save_config(config)

    if target_cam:
        if enabled:
            stream_manager.add_camera(target_cam)
        else:
            stream_manager.remove_camera(cam_id)
        state_str = "yoqildi" if enabled else "o'chirildi"
        return JSONResponse({"success": True, "message": f"Kamera {state_str}!"})
    return JSONResponse({"success": False, "message": "Kamera topilmadi!"}, status_code=404)


@app.post("/api/cameras/bulk-toggle")
async def bulk_toggle_cameras(request: Request):
    """Tanlangan kameralarni ommaviy yoqish yoki o'chirish."""
    global config
    config = load_config()
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        cam_ids = body.get("cam_ids") or body.get("ids", [])
        if isinstance(cam_ids, str):
            target_ids = set([x.strip() for x in cam_ids.split(",") if x.strip()])
        else:
            target_ids = set([str(x).strip() for x in cam_ids if str(x).strip()])
        enabled = bool(body.get("enabled", False))
    else:
        form = await request.form()
        ids = form.get("ids") or form.get("cam_ids", "")
        target_ids = set([x.strip() for x in str(ids).split(",") if x.strip()])
        enabled = str(form.get("enabled", "true")).lower() in ("true", "1", "yes")

    count = 0
    for c in config.get("cameras", []):
        if c["id"] in target_ids:
            c["enabled"] = enabled
            count += 1
            if enabled:
                stream_manager.add_camera(c)
            else:
                stream_manager.remove_camera(c["id"])

    save_config(config)
    state_str = "yoqildi" if enabled else "o'chirildi"
    return JSONResponse({"success": True, "message": f"{count} ta kamera muvaffaqiyatli {state_str}!"})


@app.post("/api/cameras/bulk-delete")
async def bulk_delete_cameras(request: Request):
    """Tanlangan kameralarni ommaviy o'chirish."""
    global config
    config = load_config()
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        cam_ids = body.get("cam_ids") or body.get("ids", [])
        if isinstance(cam_ids, str):
            target_ids = set([x.strip() for x in cam_ids.split(",") if x.strip()])
        else:
            target_ids = set([str(x).strip() for x in cam_ids if str(x).strip()])
    else:
        form = await request.form()
        ids = form.get("ids") or form.get("cam_ids", "")
        target_ids = set([x.strip() for x in str(ids).split(",") if x.strip()])

    new_cameras = []
    deleted_count = 0
    for c in config.get("cameras", []):
        if c["id"] in target_ids:
            stream_manager.remove_camera(c["id"])
            deleted_count += 1
        else:
            new_cameras.append(c)

    config["cameras"] = new_cameras
    save_config(config)
    return JSONResponse({"success": True, "message": f"{deleted_count} ta kamera o'chirildi!"})


@app.post("/api/cameras/batch-generate")
async def batch_generate_cameras(
    name_prefix: str = Form("Auditoriya "),
    url_pattern: str = Form(...),
    start_num: int = Form(1),
    end_num: int = Form(16),
    fps_limit: int = Form(15),
    enabled: bool = Form(False)
):
    """NVR kanal yoki IP diapazoni bo'yicha kameralarni ommaviy generatsiya qilish."""
    global config
    config = load_config()
    cameras = config.get("cameras", [])
    existing_ids = {str(c.get("id", "")).strip() for c in cameras}
    existing_sources = {str(c.get("source", "")).strip() for c in cameras}

    if start_num > end_num:
        start_num, end_num = end_num, start_num

    # Bir vaqtning o'zida maksimal 128 ta
    end_num = min(end_num, start_num + 128)

    added = 0
    for num in range(start_num, end_num + 1):
        url = url_pattern.replace("{CH}", str(num)).replace("{ch}", str(num)).replace("{N}", str(num)).replace("{n}", str(num)).replace("{num}", str(num)).replace("{IP}", str(num)).replace("{ip}", str(num))
        if url in existing_sources:
            continue

        cid = generate_next_camera_id(cameras)
        cam = {
            "id": cid,
            "name": f"{name_prefix.strip()} {num}".strip(),
            "source": url.strip(),
            "enabled": enabled,
            "fps_limit": fps_limit
        }
        cameras.append(cam)
        existing_ids.add(cid)
        existing_sources.add(url)
        if enabled:
            stream_manager.add_camera(cam)
        added += 1

    config["cameras"] = cameras
    save_config(config)
    return JSONResponse({"success": True, "message": f"{added} ta yangi kamera muvaffaqiyatli generatsiya qilindi!"})


@app.post("/api/detection/save")
async def save_detection_settings(
    conf_thresh: float = Form(0.25),
    sleep_enabled: bool = Form(True),
    sleep_dur: float = Form(10.0),
    phone_enabled: bool = Form(True),
    phone_dur: float = Form(2.0),
    talking_enabled: bool = Form(True),
    talking_dur: float = Form(3.0),
    teacher_filter_enabled: bool = Form(True),
    teacher_bottom_zone: float = Form(0.30),
    roi_enabled: bool = Form(True),
    roi_margin: float = Form(0.03),
    baby_enabled: bool = Form(True)
):
    """Aniqlash parametrlarini yangilash."""
    global config
    config = load_config()

    config["detection"]["confidence_threshold"] = conf_thresh
    config["detection"]["sleeping"]["enabled"] = sleep_enabled
    config["detection"]["sleeping"]["min_duration_seconds"] = sleep_dur
    config["detection"]["phone"]["enabled"] = phone_enabled
    config["detection"]["phone"]["min_duration_seconds"] = phone_dur
    config["detection"]["talking"]["enabled"] = talking_enabled
    config["detection"]["talking"]["min_duration_seconds"] = talking_dur

    if "teacher_filter" not in config["detection"]:
        config["detection"]["teacher_filter"] = {}
    config["detection"]["teacher_filter"]["enabled"] = teacher_filter_enabled
    config["detection"]["teacher_filter"]["bottom_zone_ratio"] = teacher_bottom_zone

    if "roi" not in config["detection"]:
        config["detection"]["roi"] = {}
    config["detection"]["roi"]["enabled"] = roi_enabled
    config["detection"]["roi"]["margin_pct"] = roi_margin

    if "baby_detection" not in config["detection"]:
        config["detection"]["baby_detection"] = {}
    config["detection"]["baby_detection"]["enabled"] = baby_enabled

    save_config(config)

    detector.conf_thresh = conf_thresh
    for analyzer in engine.analyzers.values():
        analyzer.update_config(config)

    return JSONResponse({"success": True, "message": "Aniqlash parametrlari yangilandi!"})
