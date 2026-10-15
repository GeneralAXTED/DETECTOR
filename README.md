# 🎓 SMART CLASSROOM DETECTOR — Aqlli Auditoriya va Qoidabuzarlik Detektori

Ushbu tizim lokal tarmoqdagi (LAN) IP kameralar va NVR qurilmalaridan video oqimni (RTSP) qabul qilib, sun'iy intellekt (YOLO Pose + Object Detection) yordamida talabalarning dars va imtihon paytidagi qoidabuzarliklarini real vaqt rejimida aniqlaydi va Telegram kanalga tezkor foto-hisobot jo'natadi.

---

## ⚡ Asosiy Imkoniyatlar

1. **Lokal Tarmoq (LAN / Offline):**
   - Lokal NVR (Hikvision, Dahua, Uniview) va har qanday RTSP/ONVIF kameralardan video qabul qilish.
   - Internet talab qilmaydi (faqat Telegramga xabar jo'natish uchun tashqi tarmoq kerak).
   - Kameralar soni chegaralanmagan — har bir kamera uchun alohida oqim va tahlil.

2. **AI Qoidabuzarlik Turlari:**
   - 😴 **Uxlayotgan talabalar (Sleeping / Head-Down):** Boshini partaga qo'yib yotish, yelkadan pastga haddan ortiq egilish holatini kadrlar davomida (masalan, 3 soniya) aniqlash.
   - 📱 **Telefon o'ynayotgan talabalar (Cell phone usage):** Parta ostida yoki yuzga yaqin tutib smartfon o'ynash, matn terish holatini aniqlash.
   - 🗣️ **Orqaga o'girilib gaplashish (Talking backward):** Yonidagi partadoshi bilan dars mobaynida gaplashish ruxsat etiladi, faqat orqaga o'girilib orqadagi partadagilar bilan gaplashish holatini aniqlash.
   - 🛡️ **Anti-spam Cooldown:** Bir xil talaba uchun Telegram kanalni rasmlar bilan to'ldirib yubormaslik uchun vaqt oralig'i (standart: 60 soniya).

3. **Telegram Kanal Integratsiyasi:**
   - Qoidabuzarlik aniqlangach, talaba atrofiga qizil/ko'k ramka chiziladi, qoidabuzarlik turi va auditoriya nomi yoziladi.
   - Rasm darhol Telegram kanal yoki guruhga jo'natiladi.

4. **Zamonaviy Veb Boshqaruv Paneli (Web Dashboard):**
   - Jonli videoni real vaqt rejimida brauzerda ko'rish (Live Stream).
   - Kameralarni qo'shish, o'chirish, tahrirlash.
   - Telegram sozlamalari va "Test Xabar" jo'natish.
   - Yopiq kanal ID sini 1 ta tugma orqali avtomatik aniqlash.
   - Aniqlash vaqt chegaralari va sezgirlikni sozlash.
   - Qoidabuzarliklar arxivi va suratlar galereyasi.

---

## 🚀 Tizimni Ishga Tushirish

### 1-usul: Tayyor skript orqali (Eng osoni)
Papka ichidagi **`run.bat`** faylini ikki marta bosing.

### 2-usul: Terminal orqali
```powershell
cd C:\Users\fyvyf\Desktop\DETECTOR
.\.venv\Scripts\python.exe main.py
```

Ishga tushgach, brauzeringizda quyidagi manzilni oching:
👉 **http://localhost:8000** (yoki lokal tarmoqdagi boshqa kompyuterlardan: `http://<KOMPYUTER_IP>:8000`)

---

## 📲 Telegram Yopiq Kanal ID sini Aniqlash

1. Telegramda yopiq kanalingizga kiring.
2. Botni (**@detector_qoidabuzar_bot**) kanalga **Administrator** qilib qo'shing.
3. Kanalingizga istalgan bitta xabar yozing (masalan: `test` yoki `salom`).
4. Veb-panelimizdagi **"Telegram Bot"** bo'limiga kiring va **"🔍 Avtomatik ID ni aniqlash"** tugmasini bosing! Tizim kanal ID sini (masalan: `-1002345678901`) avtomatik topib, saqlab qo'yadi.
5. **"Test Xabar Yuborish"** tugmasi orqali tekshirib ko'ring!

---

## 📹 Kameralar va NVR RTSP Havolalari Misollari

- **Hikvision NVR / IP Kamera:**
  `rtsp://admin:parol@192.168.1.100:554/Streaming/Channels/101`
- **Dahua NVR / IP Kamera:**
  `rtsp://admin:parol@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0`
- **Uniview NVR / IP Kamera:**
  `rtsp://admin:parol@192.168.1.100:554/unicast/c1/s0/live`
- **Noutbuk / USB Vebkamera:**
  `0` (yoki `1`)
- **Sinov Video Fayli (MP4):**
  `C:\Users\fyvyf\Desktop\DETECTOR\test_video.mp4`

---

## 📂 Loyiha Tuzilmasi

```
C:\Users\fyvyf\Desktop\DETECTOR\
├── .venv/                     # Python 3.12 + PyTorch (CUDA 12.4 RTX 3060)
├── config/
│   ├── config.example.yaml    # Sozlamalar namunasi (template)
│   ├── config.yaml            # Kameralar, Telegram va AI sozlamalari
│   └── settings.py            # Konfiguratsiyani o'qish/saqlash
├── core/
│   ├── stream_manager.py      # Ko'p oqimli kechikishsiz RTSP kadr tutuvchi
│   ├── detector.py            # Ultralytics YOLO11/YOLOv8 Pose + Object
│   ├── violation_analyzer.py  # Uxlash, Telefon, Gaplashish matematik algoritmlari
│   ├── alert_manager.py       # Anti-spam cooldown, rasm chizish va Telegram jo'natish
│   └── engine.py              # Asosiy AI tahlil sikli
├── telegram/
│   └── bot.py                 # Telegram Bot API asinxron integratsiyasi
├── web/
│   ├── app.py                 # FastAPI veb-serveri va API lari
│   └── templates/
│       └── index.html         # Zamonaviy TailwindCSS boshqaruv paneli
├── captures/                  # Qayd etilgan qoidabuzarliklar rasmlari
├── requirements.txt           # O'rnatilgan paketlar ro'yxati
├── run.bat                    # 1-klik bilan ishga tushiruvchi fayl
├── main.py                    # Kirish nuqtasi
└── README.md                  # Hujjatlar
```
