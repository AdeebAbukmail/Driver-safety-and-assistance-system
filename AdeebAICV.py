                      
                       
"""
ADEEB AI -- SMART DRIVER MONITORING SYSTEM
==========================================
مشروع أديب ابوكمييل.

ملف واحد فقط، يحتوي على:
  - تحليل الوجه (عين مغمضة/مفتوحة، تثاؤب، اتجاه الرأس) بمحرّكين حقيقيين:
        1) MediaPipe FaceMesh   (دقيق: EAR / MAR / solvePnP)
        2) OpenCV Haar Cascade  (احتياطي حقيقي يعمل بدون mediapipe إطلاقًا)
    هذا هو إصلاح المشكلة التي واجهتك: سابقًا إن لم تُهيّأ mediapipe كانت
    الحالة تصبح VISION ERROR ويبقى الفيديو فقط بلا أي تحليل. الآن النظام
    ينتقل تلقائيًا إلى محرك OpenCV ويستمر بالعمل ويعرض البيانات فعليًا.

  - كشف انشغال السائق بالهاتف بمصدرين:
        1) YOLO (ultralytics) إن كان مثبتًا -- يعطي صندوق + نسبة ثقة.
        2) مسار بدون YOLO: MediaPipe Hands + إطراق الرأس للأسفل (Heuristic).
    المسار الثاني معلن بوضوح كتقدير سلوكي (HAND+LOOK-DOWN) وليس تعرّفًا على
    جهاز هاتف، حتى لا تُعرض عليك نتيجة موهمة بدقّة لا تملكها.

  - جرس إنذار حقيقي: على الحاسوب (winsound/BEL) + على الهاتف (WebAudio
    Oscillator بصوت عالٍ) + على Arduino (Serial) + على ESP32 (HTTP).

  - وضع الهاتف: الحاسوب يعالج الصورة ويرجّعها للهاتف مرسومًا عليها الوجه
    والبيانات، مع عرض عنوان IP للحاسوب داخل الصفحة، وحجم صورة مصغّر (50%).

  - تواصل صوتي في الاتجاهين: الحاسوب -> الهاتف (نص يُنطق بـ TTS في المتصفح)،
    والهاتف -> الحاسوب (تسجيل صوتي فعلي يُرسل ويُشغَّل على الحاسوب).

  - واجهة تشغيل رسومية لاختيار الوضع (بديل عن قائمة التيرمنال)، مع بقاء
    قائمة التيرمنال كخيار احتياطي.

لا توجد بيانات وهمية: أي ميزة غير متوفرة تُعلن حالتها بوضوح بدل تلفيق نتيجة.
تصدير أكواد Arduino/ESP32 كملفات .ino:
    python adeeb_ai.py --export-firmware ./firmware_export
"""

import os
import sys
import time
import math
import queue
import shutil
import socket
import base64
import logging
import platform
import tempfile
import threading
import datetime
import io
import wave
from collections import deque
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

                                                                              
                  
                                                                              
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TK = True
except ImportError:
    HAS_TK = False

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

                                                                              
                    
                                                                              
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except Exception:
    HAS_PYTTSX3 = False

try:
    import serial
    import serial.tools.list_ports as list_ports
    HAS_PYSERIAL = True
except Exception:
    HAS_PYSERIAL = False

try:
    import winsound
    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

try:
    from flask import Flask, render_template_string
    from flask_socketio import SocketIO
    HAS_FLASK = True
except Exception:
    HAS_FLASK = False

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

try:
    import ollama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except Exception:
    HAS_SPEECH_RECOGNITION = False

try:
    import sounddevice as _sd_probe                                          
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ADEEB")

try:
    _log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adeeb_ai.log")
    _fh = logging.FileHandler(_log_file, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_fh)
    log.info(f"سجل الأخطاء التفصيلي: {_log_file}")
except Exception:
    pass


APP_NAME = "ADEEB AI"

                                                                                  
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(_THIS_DIR, "face_landmarker.task")
FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
MODEL_DOWNLOAD_RETRIES = 3


def ensure_model_file(path: str, url: str, label: str = "النموذج") -> bool:
    """ينزّل ملف النموذج إن لم يكن موجودًا. يعيد True عند الجاهزية."""
    if os.path.isfile(path) and os.path.getsize(path) > 1024 * 1024:
        return True
    import urllib.request
    last = None
    for attempt in range(1, MODEL_DOWNLOAD_RETRIES + 1):
        try:
            log.info(f"جاري تنزيل {label}... ({attempt}/{MODEL_DOWNLOAD_RETRIES})")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) < 1024 * 1024:
                raise IOError("حجم الملف المنزَّل غير صحيح.")
            os.replace(tmp, path)
            log.info(f"تم تنزيل {label} بنجاح.")
            return True
        except Exception as exc:
            last = exc
            log.warning(f"فشل التنزيل ({attempt}): {exc}")
            time.sleep(1.0)
    log.error(f"تعذر تنزيل {label}: {last}")
    return False


WELCOME_TEXT = "أهلاً بك في مشروع أديب ابوكمييل، أنت متصل الآن"
WELCOME_TEXT_EN = "Welcome to Adeeb Abu Kmail project. You are connected now."
PHONE_ALERT_TEXT = "تنبيه، يرجى التركيز أثناء القيادة، اترك الهاتف"
PHONE_ALERT_TEXT_EN = "Warning. Focus on driving and leave the phone now."
DROWSY_ALERT_TEXT = "انتبه، عيناك مغلقتان، ركّز في الطريق"
DROWSY_ALERT_TEXT_EN = "Warning. Your eyes are closed. Focus on the road."
HEAD_ALERT_TEXT = "التفت إلى الأمام من فضلك"
HEAD_ALERT_TEXT_EN = "Please look forward."
ADEEB_AI_GREETING = "مرحباً، أنا أسترو إيه آي، نموذج ذكاء اصطناعي صممني أديب ابوكمييل. أنا هنا للمساعدة. وظيفتي هي مساعدة السائق."
ADEEB_AI_SYSTEM_PROMPT = "أنت ADEEB AI، مساعد ذكاء اصطناعي لمساعدة السائق. صمّمك أديب ابوكمييل. إذا سُئلت من صممك أو من أنت أو من أنشأك، أجب بوضوح أنك ADEEB AI وأن أديب ابوكمييل صمماك. مهمتك الأساسية مساعدة السائق وتقديم إرشادات عامة وآمنة مرتبطة بالقيادة وشرح حالة النظام عند توفرها. لا تدّع أنك تقود السيارة أو تتحكم بها. كن مختصراً وواضحاً وبالعربية ما لم يطلب المستخدم لغة أخرى."
ADEEB_AI_MODEL = os.environ.get("ADEEB_OLLAMA_MODEL", "llama3.2:3b")


                                                                                
        
                                                                                
class Config:
                              
    EAR_THRESHOLD = 0.205
    EYE_WARNING_SEC = 1
    EYE_DANGER_SEC = 2                                        
    EAR_SMOOTHING_WINDOW = 5
    BLINK_MAX_SEC = 0.35
    GAZE_DEVIATION_THRESHOLD = 0.12
    PHONE_GAZE_HOLD_SEC = 1.0
    PHONE_ALERT_COOLDOWN_SEC = 5.0
    HAAR_EYE_MISS_FRAMES = 3                                                   

                              
    MAR_THRESHOLD = 0.55

                     
    HEAD_YAW_THRESHOLD_DEG = 22.0
    HEAD_PITCH_DOWN_THRESHOLD_DEG = 15.0
    HEAD_TURN_HOLD_SEC = 2.0
    INVERT_YAW = False                                                        

                      
    PHONE_HOLD_SEC = 3.0                                                  
    PHONE_CONF_THRESHOLD = 0.40
    PHONE_MEMORY_SEC = 0.8                                              
    YOLO_MODEL_PATH = "yolov8n.pt"
    YOLO_CELL_PHONE_CLASS_NAME = "cell phone"
    YOLO_EVERY_N_FRAMES = 3                                           

                     
    VOICE_COOLDOWN_SEC = 4.0                                        

                      
    PHONE_SERVER_PORT = 5000
    PHONE_RESULT_FPS = 12                                              
    PHONE_RESULT_WIDTH = 360
    ESP32_STREAM_PATH_AI_THINKER = ":81/stream"

                      
    ARDUINO_BAUDRATE = 9600
    SERIAL_TIMEOUT = 1.0

                       
    COLOR_BG = "#05070a"
    COLOR_GREEN = "#19ff8a"
    COLOR_RED = "#ff274d"
    COLOR_YELLOW = "#ffd84d"
    COLOR_TEXT_DIM = "#6f8f8a"
    COLOR_PANEL = "#0b0f14"

                        
    CAMERA_INDEX = 0
    FRAME_WIDTH = 960
    FRAME_HEIGHT = 540


LANGUAGE_AR = "ar"
LANGUAGE_EN = "en"

def localized(ar: str, en: str, language: str) -> str:
    return ar if language == LANGUAGE_AR else en


class SystemState:
    WAITING = "WAITING"
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    DANGER = "DANGER"
    PHONE_DANGER = "PHONE DANGER"
    NO_FACE = "NO FACE"
    VISION_ERROR = "VISION ERROR"


STATE_COLORS = {
    SystemState.WAITING: Config.COLOR_TEXT_DIM,
    SystemState.NORMAL: Config.COLOR_GREEN,
    SystemState.WARNING: Config.COLOR_YELLOW,
    SystemState.DANGER: Config.COLOR_RED,
    SystemState.PHONE_DANGER: Config.COLOR_RED,
    SystemState.NO_FACE: Config.COLOR_YELLOW,
    SystemState.VISION_ERROR: Config.COLOR_RED,
}


                                                                                
                                                                 
                                                                                
class VoiceManager:
    def __init__(self, cooldown: float = Config.VOICE_COOLDOWN_SEC,
                 on_speak_callback=None):
        self.cooldown = cooldown
        self.available = HAS_PYTTSX3
        self._last_spoken: Dict[str, float] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop_flag = threading.Event()
        self._engine = None
        self.on_speak_callback = on_speak_callback

        if not self.available:
            log.warning("pyttsx3 غير مثبت -- نطق الحاسوب معطل (الهاتف سينطق بدلاً عنه).")
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @staticmethod
    def _pick_arabic_voice(engine):
        try:
            for v in engine.getProperty("voices"):
                vid = (getattr(v, "id", "") or "").lower()
                name = (getattr(v, "name", "") or "").lower()
                langs = " ".join(str(x) for x in (getattr(v, "languages", []) or [])).lower()
                if "arabic" in name or "ar-" in vid or "ar_" in vid or "ar" in langs:
                    return v.id
        except Exception:
            pass
        return None

    def _make_engine(self):
        """تهيئة المحرك داخل نفس الـ Thread الذي سينطق (مهم على Windows)."""
        try:
            eng = pyttsx3.init()
            ar = self._pick_arabic_voice(eng)
            if ar:
                eng.setProperty("voice", ar)
            eng.setProperty("rate", 165)
            eng.setProperty("volume", 1.0)
            return eng
        except Exception as exc:
            log.error(f"فشل تهيئة pyttsx3: {exc}")
            return None

    def _worker(self):
        self._engine = self._make_engine()
        if self._engine is None:
            self.available = False
        while not self._stop_flag.is_set():
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._engine is None:
                self._engine = self._make_engine()
                if self._engine is None:
                    continue
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                                                                                 
                log.warning(f"خطأ TTS ({exc}) -- إعادة تهيئة المحرك.")
                try:
                    self._engine.stop()
                except Exception:
                    pass
                self._engine = self._make_engine()

    def speak(self, key: str, text: str, force: bool = False):
        now = time.time()
        if not force and (now - self._last_spoken.get(key, 0.0)) < self.cooldown:
            return
        self._last_spoken[key] = now

        if self.on_speak_callback:
            try:
                self.on_speak_callback(text)
            except Exception as exc:
                log.error(f"فشل بث النص للهاتف: {exc}")

        if self.available:
            self._queue.put(text)
        else:
            log.info(f"[TTS معطل على الحاسوب] النص: {text}")

    def speak_now(self, text: str):
        """نطق فوري على الحاسوب فقط، بدون تبريد (cooldown) وبدون بث الحدث العام
        'speak' إلى الهاتف -- تُستخدم لردود ADEEB AI التي تُبث للهاتف عبر
        قناتها الخاصة (ai_message) لتفادي نطق مزدوج لنفس النص."""
        if self.available:
            self._queue.put(text)
        else:
            log.info(f"[TTS معطل على الحاسوب] النص: {text}")

    def stop(self):
        self._stop_flag.set()


                                                                                
                                                           
                                                                                
class AlarmManager:
    def __init__(self, arduino_controller=None, esp_controller=None,
                 phone_broadcast=None):
        self.arduino = arduino_controller
        self.esp = esp_controller
        self.phone_broadcast = phone_broadcast                   
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self.platform = platform.system()

    def set_phone_broadcast(self, fn):
        self.phone_broadcast = fn

    def set_esp(self, esp):
        self.esp = esp

    @property
    def is_active(self) -> bool:
        return self._active

    def start(self):
        if self._active:
            return
        self._active = True
        self._stop_flag.clear()
        if self.arduino:
            self.arduino.send_command("ALARM_ON")
        if self.esp:
            self.esp.send_command("ALARM_ON")
        if self.phone_broadcast:
            try:
                self.phone_broadcast(True)
            except Exception:
                pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.warning("ALARM ON")

    def stop(self):
        if not self._active:
            return
        self._active = False
        self._stop_flag.set()
        if self.arduino:
            self.arduino.send_command("ALARM_OFF")
        if self.esp:
            self.esp.send_command("ALARM_OFF")
        if self.phone_broadcast:
            try:
                self.phone_broadcast(False)
            except Exception:
                pass
        log.info("ALARM OFF")

    def _loop(self):
        while not self._stop_flag.is_set():
            if HAS_WINSOUND:
                try:
                    winsound.Beep(2400, 300)
                except Exception:
                    self._fallback_beep()
            else:
                self._fallback_beep()
            time.sleep(0.12)

    def _fallback_beep(self):
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass
        time.sleep(0.2)


                                                                                
                                                                             
                                                                                
class AudioPlayer:
    CANDIDATES = [
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("mpv", ["--no-video", "--really-quiet"]),
        ("vlc", ["--intf", "dummy", "--play-and-exit"]),
        ("afplay", []),
        ("mpg123", ["-q"]),
    ]

    def __init__(self):
        self.player = None
        self.args: List[str] = []
        for name, args in self.CANDIDATES:
            path = shutil.which(name)
            if path:
                self.player = path
                self.args = args
                break
        if not self.player:
            log.warning("لا يوجد مشغل صوت خارجي (ffplay/mpv/vlc) -- "
                        "لن يُسمع صوت الهاتف على الحاسوب. ثبّت ffmpeg لتفعيله.")

    @property
    def available(self) -> bool:
        return self.player is not None

    def play_wav_bytes(self, data: bytes):
        """
        تشغيل WAV مباشرة بدون أي برنامج خارجي:
        Windows -> winsound.PlaySound(SND_MEMORY) ، غير ذلك -> sounddevice ،
        وإن لم يتوفر أي منهما نعود للمشغّل الخارجي.
        """
        if not data:
            return False
        if HAS_WINSOUND:
            try:
                threading.Thread(
                    target=lambda: winsound.PlaySound(data, winsound.SND_MEMORY),
                    daemon=True).start()
                return True
            except Exception as exc:
                log.warning(f"winsound فشل في تشغيل صوت الهاتف: {exc}")
        try:
            import io
            import wave
            import sounddevice as sd
            with wave.open(io.BytesIO(data), "rb") as wf:
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            threading.Thread(target=lambda: sd.play(audio, rate), daemon=True).start()
            return True
        except Exception:
            pass
        if self.available:
            self.play_bytes(data, ".wav")
            return True
        log.warning("لا توجد وسيلة لتشغيل صوت الهاتف على هذا الجهاز "
                    "(جرّب: pip install sounddevice).")
        return False

    def play_bytes(self, data: bytes, suffix: str = ".webm"):
        if not self.available or not data:
            return
        try:
            fd, path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            subprocess.Popen([self.player] + self.args + [path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            log.error(f"فشل تشغيل صوت الهاتف: {exc}")


                                                                                
                   
                                                                                
class ArduinoController:
    def __init__(self, port: Optional[str] = None,
                 baudrate: int = Config.ARDUINO_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.connected = False
        self._serial = None
        self._lock = threading.Lock()

        if not HAS_PYSERIAL:
            log.warning("pyserial غير مثبتة -- Arduino معطل.")
            return
        if not port:
            return
        self._connect()

    @staticmethod
    def list_available_ports() -> List[str]:
        if not HAS_PYSERIAL:
            return []
        try:
            return [p.device for p in list_ports.comports()]
        except Exception as exc:
            log.error(f"فشل سرد المنافذ: {exc}")
            return []

    def _connect(self):
        try:
            self._serial = serial.Serial(self.port, self.baudrate,
                                         timeout=Config.SERIAL_TIMEOUT)
            time.sleep(2.0)
            self.connected = True
            log.info(f"Arduino متصل على {self.port} @ {self.baudrate}")
        except Exception as exc:
            self.connected = False
            log.error(f"فشل الاتصال بـ Arduino ({self.port}): {exc}")

    def send_command(self, command: str) -> bool:
        if not self.connected or not self._serial:
            return False
        try:
            with self._lock:
                self._serial.write((command.strip() + "\n").encode("utf-8"))
            return True
        except Exception as exc:
            log.error(f"فشل إرسال '{command}': {exc}")
            self.connected = False
            return False

    def close(self):
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
        self.connected = False


                                                                                
       
                                                                                
class ESPBoardType:
    ESP32_S3_CAM = "ESP32-S3-CAM"
    ESP32_CAM_AI_THINKER = "ESP32-CAM AI Thinker"
    ESP32_S3_EYE = "ESP32-S3-EYE"
    ESP32_S3_DEVKIT = "ESP32-S3 DevKit"
    ESP32_S2 = "ESP32-S2"
    ESP32_WROOM_32 = "ESP32-WROOM-32"
    ESP32_WROVER = "ESP32-WROVER"
    OTHER_UNKNOWN = "Other / Unknown"

    CAMERA_CAPABLE = {ESP32_S3_CAM, ESP32_CAM_AI_THINKER, ESP32_S3_EYE, ESP32_WROVER}
    CONTROLLER_ONLY = {ESP32_S2, ESP32_WROOM_32, ESP32_S3_DEVKIT}

    ALL = [ESP32_S3_CAM, ESP32_CAM_AI_THINKER, ESP32_S3_EYE, ESP32_S3_DEVKIT,
           ESP32_S2, ESP32_WROOM_32, ESP32_WROVER, OTHER_UNKNOWN]


class ESPCamera:
    def __init__(self, ip_address: str, board_type: str):
        self.ip_address = (ip_address or "").strip()
        self.board_type = board_type
        self.has_camera = board_type in ESPBoardType.CAMERA_CAPABLE
        self.connected = False
        self.init_error: Optional[str] = None
        self._stream = None
        self._stream_bytes = b""

        if not HAS_REQUESTS:
            self.init_error = "requests غير مثبتة -- ESP32 معطل."
            log.warning(self.init_error)
            return
        if not self.ip_address:
            self.init_error = "لم يتم إدخال عنوان IP لـ ESP32."
            return

        if self.has_camera:
            self._connect_camera_stream()
        else:
            self._check_controller_reachable()

    def _connect_camera_stream(self):
        url = f"http://{self.ip_address}{Config.ESP32_STREAM_PATH_AI_THINKER}"
        try:
            self._stream = requests.get(url, stream=True, timeout=5)
            if self._stream.status_code == 200:
                self.connected = True
                log.info(f"كاميرا ESP32 متصلة: {url}")
            else:
                self.init_error = f"استجابة ESP32 غير متوقعة: HTTP {self._stream.status_code}"
                log.error(self.init_error)
        except Exception as exc:
            self.init_error = f"فشل الاتصال بكاميرا ESP32 ({url}): {exc}"
            log.error(self.init_error)

    def _check_controller_reachable(self):
        url = f"http://{self.ip_address}/"
        try:
            resp = requests.get(url, timeout=3)
            self.connected = resp.status_code == 200
            if not self.connected:
                self.init_error = f"ESP32 استجاب بكود {resp.status_code}"
            else:
                log.info(f"ESP32 ({self.board_type}) متاح كوحدة تحكم.")
        except Exception as exc:
            self.init_error = f"تعذر الوصول لـ ESP32 على {self.ip_address}: {exc}"
            log.warning(self.init_error)

    def read_frame(self):
        if not self.has_camera or not self.connected or self._stream is None:
            return None
        if not (HAS_CV2 and HAS_NUMPY):
            return None
        try:
            for chunk in self._stream.iter_content(chunk_size=2048):
                self._stream_bytes += chunk
                start = self._stream_bytes.find(b"\xff\xd8")
                end = self._stream_bytes.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    jpg = self._stream_bytes[start:end + 2]
                    self._stream_bytes = self._stream_bytes[end + 2:]
                    return cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                                        cv2.IMREAD_COLOR)
            return None
        except Exception as exc:
            log.error(f"خطأ في بث ESP32: {exc}")
            self.connected = False
            return None

    def send_command(self, command: str) -> bool:
        if not HAS_REQUESTS or not self.connected:
            return False
        endpoint_map = {
            "ALARM_ON": "/alarm_on",
            "ALARM_OFF": "/alarm_off",
            "PHONE_ALERT": "/phone_alert",
            "LOOK_FORWARD": "/look_forward",
        }
        path = endpoint_map.get(command)
        if not path:
            return False
        try:
            requests.get(f"http://{self.ip_address}{path}", timeout=2)
            return True
        except Exception as exc:
            log.error(f"فشل إرسال '{command}' إلى ESP32: {exc}")
            return False

    def close(self):
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass


                                                                                
                                                                   
                                                                                
class VisionResult:
    def __init__(self):
        self.face_found = False
        self.box: Optional[Tuple[int, int, int, int]] = None
        self.ear: Optional[float] = None
        self.mar: Optional[float] = None
        self.eyes_closed: Optional[bool] = None
        self.yawning = False
        self.yaw: Optional[float] = None
        self.pitch: Optional[float] = None
        self.roll: Optional[float] = None
        self.head_dir = "UNKNOWN"
        self.eye_boxes: List[Tuple[int, int, int, int]] = []
        self.landmarks_px: Optional[List[Tuple[int, int]]] = None
        self.gaze_dx = 0.0
        self.gaze_dy = 0.0
        self.backend = "NONE"
        self.error: Optional[str] = None


class FaceAnalyzer:
    """محلل الوجه باستخدام MediaPipe Tasks FaceLandmarker مثل النسخة العاملة."""

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT = 13, 14, 78, 308
    POSE_IDS = {"nose_tip": 1, "chin": 152, "left_eye_corner": 33,
                "right_eye_corner": 263, "left_mouth_corner": 61,
                "right_mouth_corner": 291}

    def __init__(self, prefer_mediapipe: bool = True):
        self.backend = "NONE"
        self.init_error = None
        self.tasks_landmarker = None
        self.face_mesh = None
        self._ts_counter = 0
        self._ts_start = time.time()
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype=np.float64)
        self.ear_history = deque(maxlen=Config.EAR_SMOOTHING_WINDOW)

        if not (HAS_CV2 and HAS_NUMPY):
            self.init_error = "opencv-python أو numpy غير مثبتة."
            return
        if prefer_mediapipe and HAS_MEDIAPIPE:
            if self._init_tasks():
                return
            if self._init_legacy():
                return
        self._init_opencv_fallback()

    def _init_tasks(self):
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            if not ensure_model_file(FACE_MODEL_PATH, FACE_MODEL_URL, "نموذج تحليل الوجه"):
                return False
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.50,
                min_face_presence_confidence=0.50,
                min_tracking_confidence=0.50,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.tasks_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._mp_module = mp
            self._ts_counter = 0
            self._ts_start = time.time()
            self.backend = "tasks"
            log.info("محرك الرؤية: MediaPipe Tasks FaceLandmarker")
            return True
        except Exception as exc:
            self.init_error = f"تعذر تشغيل FaceLandmarker: {exc}"
            log.warning(self.init_error)
            return False

    def _init_legacy(self):
        try:
            if not hasattr(mp, "solutions"):
                return False
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.backend = "mediapipe"
            log.info("محرك الرؤية: MediaPipe FaceMesh")
            return True
        except Exception as exc:
            log.warning("فشل MediaPipe القديم: %s", exc)
            return False

    def _init_opencv_fallback(self):
        try:
            face_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye_tree_eyeglasses.xml")
            self.face_cascade = cv2.CascadeClassifier(face_path)
            self.eye_cascade = cv2.CascadeClassifier(eye_path)
            if self.face_cascade.empty():
                raise RuntimeError("haarcascade_frontalface_default.xml غير متاح")
            self.backend = "opencv"
            self.init_error = None
            log.warning("استخدام OpenCV Haar كبديل احتياطي")
        except Exception as exc:
            self.backend = "NONE"
            self.init_error = f"فشل تشغيل محرك الرؤية: {exc}"

    @property
    def available(self):
        return self.backend in ("tasks", "mediapipe", "opencv")

    @staticmethod
    def _pt(landmarks, idx, w, h):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float64)

    def _ear(self, landmarks, idxs, w, h):
        p1, p2, p3, p4, p5, p6 = [self._pt(landmarks, i, w, h) for i in idxs]
        vertical = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
        horizontal = np.linalg.norm(p1-p4)
        return float(vertical / (2.0 * horizontal)) if horizontal > 1e-6 else 0.30

    def _mar(self, landmarks, w, h):
        top = self._pt(landmarks, self.MOUTH_TOP, w, h)
        bottom = self._pt(landmarks, self.MOUTH_BOTTOM, w, h)
        left = self._pt(landmarks, self.MOUTH_LEFT, w, h)
        right = self._pt(landmarks, self.MOUTH_RIGHT, w, h)
        horizontal = np.linalg.norm(left-right)
        return float(np.linalg.norm(top-bottom)/horizontal) if horizontal > 1e-6 else 0.0

    def _head_pose(self, landmarks, w, h):
        try:
            i = self.POSE_IDS
            image_points = np.array([
                self._pt(landmarks, i["nose_tip"], w, h),
                self._pt(landmarks, i["chin"], w, h),
                self._pt(landmarks, i["left_eye_corner"], w, h),
                self._pt(landmarks, i["right_eye_corner"], w, h),
                self._pt(landmarks, i["left_mouth_corner"], w, h),
                self._pt(landmarks, i["right_mouth_corner"], w, h),
            ], dtype=np.float64)
            f = float(w)
            cam = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)
            dist = np.zeros((4,1))
            ok, rvec, _ = cv2.solvePnP(self.model_points, image_points, cam, dist,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return 0.0, 0.0, 0.0
            rmat, _ = cv2.Rodrigues(rvec)
            sy = math.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
            if sy < 1e-6:
                pitch = math.atan2(-rmat[2,0], sy)
                yaw = 0.0
            else:
                pitch = math.atan2(-rmat[2,0], sy)
                yaw = math.atan2(rmat[1,0], rmat[0,0])
            roll = math.atan2(rmat[2,1], rmat[2,2])
            return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
        except Exception:
            return 0.0, 0.0, 0.0

    def _gaze(self, landmarks, w, h):
        try:
            def ratio(corners, iris):
                a = self._pt(landmarks, corners[0], w, h)
                b = self._pt(landmarks, corners[1], w, h)
                iris_pts = np.array([self._pt(landmarks, i, w, h) for i in iris])
                center = iris_pts.mean(axis=0)
                width = np.linalg.norm(b-a)
                if width < 1e-6:
                    return 0.0
                mid = (a+b)/2.0
                return float((center[0]-mid[0])/width)
            dx = (ratio((33,133), self.LEFT_IRIS) + ratio((362,263), self.RIGHT_IRIS)) / 2.0
            return dx, 0.0
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _dir(yaw, pitch, gaze_dx):
        if pitch < -Config.HEAD_PITCH_DOWN_THRESHOLD_DEG:
            return "DOWN"
        if gaze_dx > Config.GAZE_DEVIATION_THRESHOLD and yaw > Config.HEAD_YAW_THRESHOLD_DEG:
            return "RIGHT"
        if gaze_dx < -Config.GAZE_DEVIATION_THRESHOLD and yaw < -Config.HEAD_YAW_THRESHOLD_DEG:
            return "LEFT"
        if yaw > Config.HEAD_YAW_THRESHOLD_DEG:
            return "RIGHT"
        if yaw < -Config.HEAD_YAW_THRESHOLD_DEG:
            return "LEFT"
        return "CENTER"

    def _from_landmarks(self, landmarks, w, h, result):
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        result.face_found = True
        result.box = (max(0,min(xs)), max(0,min(ys)), min(w-1,max(xs)), min(h-1,max(ys)))
        result.landmarks_px = pts
        ear_l = self._ear(landmarks, self.LEFT_EYE, w, h)
        ear_r = self._ear(landmarks, self.RIGHT_EYE, w, h)
        raw_ear = (ear_l + ear_r) / 2.0
        self.ear_history.append(raw_ear)
        result.ear = float(np.mean(self.ear_history))
        result.eyes_closed = result.ear < Config.EAR_THRESHOLD
        result.mar = self._mar(landmarks, w, h)
        result.yawning = result.mar > Config.MAR_THRESHOLD
        result.yaw, result.pitch, result.roll = self._head_pose(landmarks, w, h)
        result.gaze_dx, result.gaze_dy = self._gaze(landmarks, w, h)
        result.head_dir = self._dir(result.yaw, result.pitch, result.gaze_dx)

        for idxs in (self.LEFT_EYE, self.RIGHT_EYE):
            p = [self._pt(landmarks, i, w, h) for i in idxs]
            x1,y1 = p[0].astype(int); x2,y2 = p[3].astype(int)
            result.eye_boxes.append((min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)))
        return result

    def process(self, frame):
        result = VisionResult(); result.backend = self.backend
        if not self.available:
            result.error = self.init_error or "محرك الرؤية غير متاح."
            return result
        try:
            h,w = frame.shape[:2]
            if self.backend == "tasks":
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                mp_image = self._mp_module.Image(image_format=self._mp_module.ImageFormat.SRGB, data=rgb)
                ts = int((time.time()-self._ts_start)*1000)
                if ts <= self._ts_counter: ts = self._ts_counter + 1
                self._ts_counter = ts
                out = self.tasks_landmarker.detect_for_video(mp_image, ts)
                if not out.face_landmarks: return result
                return self._from_landmarks(out.face_landmarks[0], w, h, result)
            if self.backend == "mediapipe":
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); rgb.flags.writeable = False
                out = self.face_mesh.process(rgb)
                if not out.multi_face_landmarks: return result
                return self._from_landmarks(out.multi_face_landmarks[0].landmark, w, h, result)
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(90,90))
            if len(faces)==0: return result
            fx,fy,fw,fh=max(faces,key=lambda x:x[2]*x[3])
            result.face_found=True; result.box=(fx,fy,fx+fw,fy+fh)
            roi=gray[fy:fy+int(fh*0.65), fx:fx+fw]
            eyes=self.eye_cascade.detectMultiScale(roi,1.08,6,minSize=(max(18,int(fw*.11)),max(12,int(fh*.07))))
            result.eyes_closed = len(eyes)==0
            result.head_dir = "CENTER" if len(eyes)>=1 else "UNKNOWN"
            for ex,ey,ew,eh in eyes[:2]: result.eye_boxes.append((fx+ex,fy+ey,fx+ex+ew,fy+ey+eh))
            return result
        except Exception as exc:
            result.error=f"خطأ أثناء تحليل الوجه: {exc}"
            log.error(result.error, exc_info=True)
            return result

    def close(self):
        for obj in (self.face_mesh, self.tasks_landmarker):
            if obj is not None:
                try: obj.close()
                except Exception: pass


                                                                                
                                                                     
                                                                                
class PhoneDetection:
    def __init__(self, box, confidence: Optional[float]):
        self.box = box
        self.confidence = confidence


class PhoneUseDetector:
    def __init__(self, use_yolo: bool = True):
        self.backend = "NONE"
        self.init_error: Optional[str] = None
        self.model = None
        self.hands = None
        self._target_class_id: Optional[int] = None
        self._frame_counter = 0
        self._cached: List[PhoneDetection] = []
        self._cached_at = 0.0
        self.hand_landmarks = []
        self.person_boxes = []
        self._person_class_id: Optional[int] = None

        if use_yolo and HAS_YOLO:
            self._init_yolo()
        if self.backend == "NONE":
            self._init_hands()

    def _init_yolo(self):
        try:
            self.model = YOLO(Config.YOLO_MODEL_PATH)
            for cid, cname in self.model.names.items():
                name = str(cname).lower()
                if name == Config.YOLO_CELL_PHONE_CLASS_NAME:
                    self._target_class_id = int(cid)
                if name == "person":
                    self._person_class_id = int(cid)
            if self._target_class_id is None:
                self.init_error = "النموذج لا يحتوي على صنف cell phone."
                log.warning(self.init_error)
                self.model = None
                return
            self.backend = "YOLO"
            log.info("كشف الهاتف: YOLO (cell phone)")
        except Exception as exc:
            self.init_error = f"فشل تحميل YOLO: {exc}"
            log.warning(self.init_error)
            self.model = None

    def _init_hands(self):
        if not HAS_MEDIAPIPE:
            self.init_error = (self.init_error or "") +\
                " | mediapipe غير مثبتة -- لا يوجد بديل لكشف انشغال الهاتف."
            log.warning("كشف انشغال الهاتف معطل (لا YOLO ولا mediapipe).")
            return
                                                                             
                             
        if not hasattr(mp, "solutions"):
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision
                hand_path = os.path.join(_THIS_DIR, "hand_landmarker.task")
                hand_url = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/1/hand_landmarker.task")
                if ensure_model_file(hand_path, hand_url, "نموذج تتبع اليدين"):
                    self.hands_tasks = mp_vision.HandLandmarker.create_from_options(
                        mp_vision.HandLandmarkerOptions(
                            base_options=mp_python.BaseOptions(model_asset_path=hand_path),
                            running_mode=mp_vision.RunningMode.VIDEO,
                            num_hands=2,
                            min_hand_detection_confidence=0.45,
                            min_hand_presence_confidence=0.45,
                            min_tracking_confidence=0.45))
                    self._hands_ts = 0
                    self._hands_start = time.time()
                    self.backend = "HAND+LOOK-DOWN"
                    log.info("كشف انشغال الهاتف: Tasks HandLandmarker + إطراق الرأس")
                    return
            except Exception as exc:
                log.warning(f"تعذر تشغيل تتبع اليدين (Tasks): {exc}")
            return

        try:
            self._mp_hands = mp.solutions.hands
            self.hands = self._mp_hands.Hands(
                static_image_mode=False, max_num_hands=2,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.backend = "HAND+LOOK-DOWN"
            log.info("كشف انشغال الهاتف: تقدير سلوكي (يد قرب الوجه + إطراق رأس)")
        except Exception as exc:
            self.init_error = f"فشل تهيئة MediaPipe Hands: {exc}"
            log.warning(self.init_error)

    def detect(self, frame, vision: VisionResult) -> List[PhoneDetection]:
        now = time.time()
        if self.backend == "YOLO":
            self._frame_counter += 1
            if self._frame_counter % Config.YOLO_EVERY_N_FRAMES != 0:
                if now - self._cached_at < Config.PHONE_MEMORY_SEC:
                    return self._cached
                return []
            dets = self._detect_yolo(frame)
            if dets:
                self._cached, self._cached_at = dets, now
                return dets
            if now - self._cached_at < Config.PHONE_MEMORY_SEC:
                return self._cached
            return []

        if self.backend == "HAND+LOOK-DOWN":
            return self._detect_hands(frame, vision)

        return []

    def _detect_yolo(self, frame) -> List[PhoneDetection]:
        try:
            results = self.model.predict(frame, verbose=False,
                                         conf=Config.PHONE_CONF_THRESHOLD,
                                         classes=[x for x in (self._target_class_id, self._person_class_id) if x is not None])
            dets = []
            self.person_boxes = []
            for r in results:
                if r.boxes is None:
                    continue
                for b in r.boxes:
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                    cls_id = int(b.cls[0]) if getattr(b, "cls", None) is not None else self._target_class_id
                    if cls_id == self._person_class_id:
                        self.person_boxes.append((x1, y1, x2, y2))
                    elif cls_id == self._target_class_id:
                        dets.append(PhoneDetection((x1, y1, x2, y2), float(b.conf[0])))
            return dets
        except Exception as exc:
            log.error(f"خطأ YOLO: {exc}")
            return []

    def _detect_hands(self, frame, vision: VisionResult) -> List[PhoneDetection]:
        try:
            self.hand_landmarks = []
            h, w = frame.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if getattr(self, "hands_tasks", None) is not None:
                ts = int((time.time() - self._hands_start) * 1000)
                if ts <= self._hands_ts:
                    ts = self._hands_ts + 1
                self._hands_ts = ts
                res = self.hands_tasks.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                hand_list = res.hand_landmarks
            else:
                rgb.flags.writeable = False
                out = self.hands.process(rgb)
                hand_list = out.multi_hand_landmarks
                hand_list = [h_.landmark for h_ in hand_list] if hand_list else None

            if not hand_list:
                return []

            looking_down = vision.head_dir == "DOWN"
            for hand in hand_list:
                self.hand_landmarks.append([(int(lm.x * w), int(lm.y * h)) for lm in hand])
                xs = [lm.x * w for lm in hand]
                ys = [lm.y * h for lm in hand]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                hand_high = (y1 + y2) / 2.0 < h * 0.75

                near_face = False
                if vision.box:
                    fx1, fy1, fx2, fy2 = vision.box
                    near_face = not (x2 < fx1 - (fx2 - fx1) or x1 > fx2 + (fx2 - fx1))

                if looking_down or (hand_high and near_face):
                    return [PhoneDetection((x1, y1, x2, y2), None)]
            return []
        except Exception as exc:
            log.error(f"خطأ Hands: {exc}")
            return []


                                                                                
                                       
                                                                                
class AdeebAIChat:
    def __init__(self, model: str = ADEEB_AI_MODEL):
        self.model = model
        self.enabled = False
        self.lock = threading.Lock()
        self.history = []
        self.last_error = ""

    def enable(self):
        self.enabled = True
        self.history.clear()
        ok, diag = self.check_ready()
        if not ok:
            return ADEEB_AI_GREETING + "\n\n⚠️ " + diag
        return ADEEB_AI_GREETING

    def check_ready(self) -> Tuple[bool, str]:
        """فحص سريع: هل مكتبة ollama مثبتة، وهل خدمة Ollama تعمل فعليًا،
        وهل النموذج المطلوب موجود محليًا؟ يُستخدم لإظهار سبب العطل فورًا
        بدل اكتشافه بعد أول سؤال فاشل."""
        if not HAS_OLLAMA:
            return False, "مكتبة ollama غير مثبتة. في الطرفية (CMD): pip install ollama"
        try:
            models_resp = ollama.list()
            try:
                names = [m.get("name") or m.get("model") for m in models_resp.get("models", [])]
            except Exception:
                names = [getattr(m, "model", getattr(m, "name", "")) for m in getattr(models_resp, "models", [])]
            names = [n for n in names if n]
            have_model = any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names)
            if not have_model:
                return False, (f"خدمة Ollama تعمل، لكن النموذج {self.model} غير موجود. "
                               f"في الطرفية (CMD): ollama pull {self.model}")
            return True, ""
        except Exception as exc:
            self.last_error = str(exc)
            return False, ("تعذر الاتصال بخدمة Ollama (البرنامج غير مُشغَّل). "
                            "افتح الطرفية (CMD) ونفّذ: ollama serve  -- أو شغّل تطبيق Ollama نفسه، "
                            f"ثم أعد تفعيل ADEEB AI. [{exc}]")

    def disable(self):
        self.enabled = False

    @staticmethod
    def _extract_answer(result) -> str:
        """يدعم كل إصدارات مكتبة ollama: قديمًا dict عادي، وحديثًا كائن
        ChatResponse (يدعم كلا أسلوبي الوصول dict وattribute)."""
        try:
            content = result["message"]["content"]
            if content:
                return str(content).strip()
        except Exception:
            pass
        try:
            msg = getattr(result, "message", None)
            content = getattr(msg, "content", None) if msg is not None else None
            if content:
                return str(content).strip()
        except Exception:
            pass
        return ""

    def _chat_once(self, messages):
        return ollama.chat(model=self.model, messages=messages)

    def ask(self, text: str) -> str:
        text = (text or "").strip()
        if not self.enabled:
            return "فعّل ADEEB AI أولاً."
        if not text:
            return "لم أسمع سؤالاً."
        if not HAS_OLLAMA:
            return "مكتبة ollama غير مثبتة على هذا الحاسوب. نفّذ: pip install ollama"
        with self.lock:
            messages = [{"role": "system", "content": ADEEB_AI_SYSTEM_PROMPT}] + self.history[-10:]
            messages.append({"role": "user", "content": text})
            try:
                result = self._chat_once(messages)
                answer = self._extract_answer(result) or "لم أحصل على إجابة من النموذج."
                self.history.extend([{"role": "user", "content": text},
                                     {"role": "assistant", "content": answer}])
                self.last_error = ""
                return answer
            except Exception as exc:
                self.last_error = str(exc)
                msg = str(exc).lower()
                log.error(f"خطأ Ollama: {exc}")
                                                                            
                if "not found" in msg or "404" in msg or "pull" in msg:
                    try:
                        log.info(f"النموذج {self.model} غير موجود -- جارٍ تنزيله تلقائيًا (قد يستغرق دقائق)...")
                        ollama.pull(self.model)
                        result = self._chat_once(messages)
                        answer = self._extract_answer(result) or "لم أحصل على إجابة من النموذج."
                        self.history.extend([{"role": "user", "content": text},
                                             {"role": "assistant", "content": answer}])
                        self.last_error = ""
                        return answer
                    except Exception as exc2:
                        self.last_error = str(exc2)
                        return (f"النموذج {self.model} غير موجود على Ollama، وفشل تنزيله تلقائيًا "
                                f"({exc2}). نفّذ يدويًا: ollama pull {self.model}")
                if "connection" in msg or "refused" in msg or "timed out" in msg or "timeout" in msg:
                    return ("تعذر الاتصال بخدمة Ollama. تأكد أن برنامج Ollama يعمل فعليًا "
                            "على هذا الحاسوب (شغّل الأمر: ollama serve) وأن المنفذ 11434 غير محجوب.")
                return f"تعذر الاتصال بـ Ollama: {exc}"

    def transcribe_wav(self, data: bytes) -> Tuple[str, str]:
        """يعيد (النص، رسالة خطأ). عند النجاح: (النص, "")."""
        if not HAS_SPEECH_RECOGNITION:
            return "", "مكتبة SpeechRecognition غير مثبتة. في الطرفية (CMD): pip install SpeechRecognition"
        if not data or len(data) < 200:
            return "", "لم يصل صوت فعلي (تسجيل فارغ) -- تحقق من صلاحية الميكروفون."
        try:
            r = sr.Recognizer()
            with sr.AudioFile(io.BytesIO(data)) as source:
                audio = r.record(source)
            text = r.recognize_google(audio, language="ar-SA")
            self.last_error = ""
            return text, ""
        except sr.UnknownValueError:
            return "", "لم يتم التعرف على كلام مفهوم في التسجيل. جرّب التحدث بوضوح وبصوت أعلى."
        except sr.RequestError as exc:
            self.last_error = str(exc)
            return "", ("تعذر الوصول لخدمة التعرّف على الصوت من Google (تحتاج اتصال إنترنت). "
                        f"[{exc}]")
        except Exception as exc:
            self.last_error = str(exc)
            return "", f"خطأ أثناء تحويل الصوت لنص: {exc}"


                                                                                
                                 
                                                                                
PHONE_PAGE_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ADEEB AI -- Phone</title>
<style>
  body{background:#05070a;color:#19ff8a;font-family:monospace;margin:0;padding:12px;text-align:center}
  h2{margin:4px 0 2px;font-size:20px;letter-spacing:2px}
  .ip{color:#ffd84d;font-size:13px;margin-bottom:8px;word-break:break-all}
  #shot{width:50%;max-width:260px;border:2px solid #19ff8a;border-radius:8px;background:#000}
  video{display:none}
  .panel{margin:10px auto;max-width:340px;background:#0b0f14;border:1px solid #14324a;
         border-radius:8px;padding:8px;font-size:13px;text-align:right}
  .row{display:flex;justify-content:space-between;padding:2px 6px;color:#6f8f8a}
  .row b{color:#19ff8a}
  #state{font-size:17px;font-weight:bold;margin:6px 0}
  #alertBox{color:#ff274d;min-height:20px;font-size:14px;margin-top:6px}
  #status{color:#6f8f8a;font-size:12px;margin-top:6px}
  button{background:#0b0f14;color:#19ff8a;border:1px solid #19ff8a;border-radius:8px;
         padding:10px 18px;font-family:monospace;font-size:14px;margin-top:10px}
  button:active{background:#19ff8a;color:#05070a}
  #aiPanel{max-width:340px;margin:10px auto;background:#0b0f14;border:1px solid #19ff8a;border-radius:10px;padding:10px;text-align:right;display:none}
  #aiLog{height:150px;overflow-y:auto;font-size:12px;line-height:1.55}
  .chatLine{margin:5px 0;padding:6px 8px;border-radius:7px}
  .driver{background:#14251d;color:#19ff8a}.adeeb{background:#171f18;color:#ffd84d}
  #aiBtn{width:100%;background:#19ff8a;color:#05070a;font-weight:bold}
  #aiBtn.active{background:#ff274d;color:white}
  #aiTalk{width:100%}
</style>
</head>
<body>
  <h2>ADEEB AI</h2>
  <div class="ip">اللابتوب: http://__SERVER_IP__:__SERVER_PORT__</div>

  <img id="shot" alt="processed">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="canvas" style="display:none"></canvas>

  <div id="state">--</div>
  <div class="panel">
    <div class="row"><span>العين</span><b id="eye">--</b></div>
    <div class="row"><span>اتجاه الرأس</span><b id="head">--</b></div>
    <div class="row"><span>الهاتف</span><b id="phone">--</b></div>
    <div class="row"><span>المحرك</span><b id="backend">--</b></div>
    <div class="row"><span>الإطارات/ث</span><b id="fps">--</b></div>
  </div>
  <div id="msgRow" style="max-width:340px;margin:6px auto 0;color:#ff274d;font-size:11px;word-break:break-word"></div>

  <div id="alertBox"></div>
  <button id="soundBtn">🔊 تفعيل الصوت</button>
  <button id="aiBtn">🟢 تفعيل ADEEB AI</button>
  <div id="aiPanel"><div id="aiLog"></div>
    <button id="aiTalk">🎙️ اضغط مع الاستمرار للتحدث مع ADEEB AI</button>
    <div id="aiToggleRow" style="display:flex;gap:6px;margin-top:6px">
      <button id="aiTalkToggle" style="flex:1">🎤 اضغط لبدء التسجيل</button>
      <button id="aiCancelBtn" style="display:none;background:#ff274d;color:#fff;border-color:#ff274d">إلغاء ✖</button>
    </div>
  </div>
  <button id="talkBtn">🎙️ رسالة صوتية إلى اللابتوب</button>
  <div id="status">جاري طلب صلاحية الكاميرا...</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
<script>
const statusEl=document.getElementById('status');
const alertBox=document.getElementById('alertBox');
const video=document.getElementById('video');
const canvas=document.getElementById('canvas');
const ctx=canvas.getContext('2d');
const shot=document.getElementById('shot');
const socket=io();


let audioCtx=null,osc=null,gain=null,alarmOn=false;
function ensureAudio(){ if(!audioCtx){ audioCtx=new (window.AudioContext||window.webkitAudioContext)(); } if(audioCtx.state==='suspended'){audioCtx.resume();} }
function startAlarm(){ if(alarmOn)return; ensureAudio(); alarmOn=true;
  osc=audioCtx.createOscillator(); gain=audioCtx.createGain();
  osc.type='square'; osc.frequency.value=2200; gain.gain.value=1.0;
  osc.connect(gain); gain.connect(audioCtx.destination); osc.start();
  let up=true; window._alarmTimer=setInterval(()=>{ if(!osc)return;
    osc.frequency.value = up?2600:1600; up=!up; },220); }
function stopAlarm(){ alarmOn=false; clearInterval(window._alarmTimer);
  if(osc){ try{osc.stop();}catch(e){} osc=null; } }

socket.on('connect',()=>{statusEl.textContent='متصل بالخادم.'; ensureAudio();});
socket.on('disconnect',()=>{statusEl.textContent='انقطع الاتصال!'; stopAlarm();});
socket.on('alarm',d=>{ d.on?startAlarm():stopAlarm(); });


let _voices=[];
function _loadVoices(){ if('speechSynthesis' in window) _voices=window.speechSynthesis.getVoices()||[]; }
_loadVoices();
if('speechSynthesis' in window) window.speechSynthesis.onvoiceschanged=_loadVoices;
function pickVoice(){
  const ar=_voices.find(v=>/^ar/i.test(v.lang||'')||/arabic/i.test(v.name||''));
  return ar||null;
}
function speakText(text){
  if(!('speechSynthesis' in window) || !text) return;
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(text);
  const v=pickVoice();
  if(v){ u.voice=v; u.lang=v.lang; } else { u.lang='ar-SA'; }
  u.rate=1.0; u.volume=1.0;
  window.speechSynthesis.speak(u);
}

socket.on('speak',d=>{
  const t=d.text||''; alertBox.textContent=t;
  speakText(t);
});

let recvCount=0,lastFpsTs=performance.now();
socket.on('result',d=>{
  if(d.image){ shot.src='data:image/jpeg;base64,'+d.image; }
  document.getElementById('state').textContent=d.state||'--';
  document.getElementById('state').style.color=d.danger?'#ff274d':'#19ff8a';
  document.getElementById('eye').textContent=d.eye||'--';
  document.getElementById('head').textContent=d.head||'--';
  document.getElementById('phone').textContent=d.phone||'--';
  document.getElementById('backend').textContent=d.backend||'--';
  document.getElementById('msgRow').textContent=d.message||'';
  recvCount++;
  const now=performance.now();
  if(now-lastFpsTs>1000){
    document.getElementById('fps').textContent=recvCount;
    recvCount=0; lastFpsTs=now;
  }
});


const aiBtn=document.getElementById('aiBtn'), aiPanel=document.getElementById('aiPanel'), aiLog=document.getElementById('aiLog'), aiTalk=document.getElementById('aiTalk');
const aiTalkToggle=document.getElementById('aiTalkToggle'), aiCancelBtn=document.getElementById('aiCancelBtn');
let aiEnabled=false, aiRecording=false, aiBuffers=[];
function addChat(role,text){const d=document.createElement('div');d.className='chatLine '+(role==='Driver'?'driver':'adeeb');d.textContent=role+': '+text;aiLog.appendChild(d);aiLog.scrollTop=aiLog.scrollHeight;}
function speakLocal(text){ speakText(text); }
aiBtn.addEventListener('click',()=>{ensureAudio();aiEnabled=!aiEnabled;aiBtn.classList.toggle('active',aiEnabled);aiBtn.textContent=aiEnabled?'🔴 إيقاف ADEEB AI':'🟢 تفعيل ADEEB AI';aiPanel.style.display=aiEnabled?'block':'none';socket.emit('ai_toggle',{enabled:aiEnabled});if(aiEnabled){const g='مرحباً، أنا أسترو إيه آي، نموذج ذكاء اصطناعي صممني أديب ابوكمييل. أنا هنا للمساعدة. وظيفتي هي مساعدة السائق.';addChat('ADEEB AI',g);speakLocal(g);}});


function aiStart(e){e.preventDefault();if(!aiEnabled||aiRecording||aiToggleRecording||!micStream)return;ensureAudio();aiRecording=true;aiBuffers=[];aiTalk.textContent='🔴 أستمع إليك...';recSource=audioCtx.createMediaStreamSource(micStream);recNode=audioCtx.createScriptProcessor(4096,1,1);recRate=audioCtx.sampleRate;recNode.onaudioprocess=ev=>{if(aiRecording)aiBuffers.push(new Float32Array(ev.inputBuffer.getChannelData(0)));};recSource.connect(recNode);recNode.connect(audioCtx.destination);}
function aiStop(e){e.preventDefault();if(!aiRecording)return;aiRecording=false;try{recSource.disconnect();recNode.disconnect();}catch(err){}aiTalk.textContent='🎙️ اضغط مع الاستمرار للتحدث مع ADEEB AI';let len=0;aiBuffers.forEach(b=>len+=b.length);if(len<1000)return;const all=new Float32Array(len);let off=0;aiBuffers.forEach(b=>{all.set(b,off);off+=b.length;});const wav=new Uint8Array(encodeWav(all,recRate));let bin='';const CH=0x8000;for(let i=0;i<wav.length;i+=CH)bin+=String.fromCharCode.apply(null,wav.subarray(i,i+CH));socket.emit('ai_audio',{data:btoa(bin),format:'wav'});}
aiTalk.addEventListener('touchstart',aiStart);aiTalk.addEventListener('touchend',aiStop);aiTalk.addEventListener('mousedown',aiStart);aiTalk.addEventListener('mouseup',aiStop);


let aiToggleRecording=false, aiToggleBuffers=[], aiToggleSource=null, aiToggleNode=null, aiToggleRate=44100;
function aiToggleStart(){
  if(!aiEnabled||aiRecording||aiToggleRecording||!micStream)return;
  ensureAudio();
  aiToggleRecording=true; aiToggleBuffers=[];
  aiTalkToggle.textContent='⏹️ إيقاف وإرسال';
  aiCancelBtn.style.display='inline-block';
  aiToggleSource=audioCtx.createMediaStreamSource(micStream);
  aiToggleNode=audioCtx.createScriptProcessor(4096,1,1);
  aiToggleRate=audioCtx.sampleRate;
  aiToggleNode.onaudioprocess=ev=>{ if(aiToggleRecording) aiToggleBuffers.push(new Float32Array(ev.inputBuffer.getChannelData(0))); };
  aiToggleSource.connect(aiToggleNode); aiToggleNode.connect(audioCtx.destination);
}
function aiToggleStop(send){
  if(!aiToggleRecording)return;
  aiToggleRecording=false;
  try{aiToggleSource.disconnect();aiToggleNode.disconnect();}catch(err){}
  aiTalkToggle.textContent='🎤 اضغط لبدء التسجيل';
  aiCancelBtn.style.display='none';
  if(!send){ aiToggleBuffers=[]; return; }
  let len=0; aiToggleBuffers.forEach(b=>len+=b.length);
  if(len<1000){ aiToggleBuffers=[]; return; }
  const all=new Float32Array(len); let off=0;
  aiToggleBuffers.forEach(b=>{all.set(b,off);off+=b.length;});
  const wav=new Uint8Array(encodeWav(all,aiToggleRate));
  let bin=''; const CH=0x8000;
  for(let i=0;i<wav.length;i+=CH) bin+=String.fromCharCode.apply(null,wav.subarray(i,i+CH));
  socket.emit('ai_audio',{data:btoa(bin),format:'wav'});
  aiToggleBuffers=[];
}
aiTalkToggle.addEventListener('click',()=>{ if(!aiToggleRecording) aiToggleStart(); else aiToggleStop(true); });
aiCancelBtn.addEventListener('click',()=> aiToggleStop(false));

socket.on('ai_message',d=>{const role=d.role||'ADEEB AI',text=d.text||'';addChat(role,text);if(role==='ADEEB AI')speakLocal(text);});


async function startCamera(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:{ideal:"environment"},width:{ideal:640}},audio:false});
    video.srcObject=stream;
    statusEl.textContent='الكاميرا تعمل -- يتم إرسال الفيديو للنظام.';
    requestAnimationFrame(sendFrames);
  }catch(err){ statusEl.textContent='فشل الوصول للكاميرا: '+err.message; }
}
let lastSent=0;
function sendFrames(ts){
  if(ts-lastSent>70 && video.videoWidth>0){
    lastSent=ts;
    const w=480, h=Math.round(video.videoHeight*480/video.videoWidth);
    canvas.width=w; canvas.height=h;
    ctx.drawImage(video,0,0,w,h);
    socket.emit('frame',{image:canvas.toDataURL('image/jpeg',0.55)});
  }
  requestAnimationFrame(sendFrames);
}
startCamera();


const soundBtn=document.getElementById('soundBtn');
soundBtn.addEventListener('click',()=>{
  ensureAudio();
  speakText('تم تفعيل الصوت');
  soundBtn.textContent='✅ الصوت مُفعّل';
});


const talkBtn=document.getElementById('talkBtn');
let micStream=null,recNode=null,recSource=null,recBuffers=[],recRate=44100,recording=false;

async function initMic(){
  try{
    micStream=await navigator.mediaDevices.getUserMedia({audio:true});
  }catch(e){ talkBtn.textContent='الميكروفون غير متاح'; talkBtn.disabled=true; }
}
initMic();

function encodeWav(samples,rate){
  const buf=new ArrayBuffer(44+samples.length*2);
  const v=new DataView(buf);
  const ws=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  ws(0,'RIFF'); v.setUint32(4,36+samples.length*2,true); ws(8,'WAVE'); ws(12,'fmt ');
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,1,true);
  v.setUint32(24,rate,true); v.setUint32(28,rate*2,true);
  v.setUint16(32,2,true); v.setUint16(34,16,true); ws(36,'data');
  v.setUint32(40,samples.length*2,true);
  let o=44;
  for(let i=0;i<samples.length;i++,o+=2){
    let s=Math.max(-1,Math.min(1,samples[i]));
    v.setInt16(o,s<0?s*0x8000:s*0x7FFF,true);
  }
  return buf;
}

function startRec(e){
  e.preventDefault();
  if(recording||!micStream) return;
  ensureAudio();
  recording=true; recBuffers=[]; recRate=audioCtx.sampleRate;
  recSource=audioCtx.createMediaStreamSource(micStream);
  recNode=audioCtx.createScriptProcessor(4096,1,1);
  recNode.onaudioprocess=ev=>{ if(recording) recBuffers.push(new Float32Array(ev.inputBuffer.getChannelData(0))); };
  recSource.connect(recNode); recNode.connect(audioCtx.destination);
  talkBtn.textContent='🔴 يتم التسجيل...';
}

function stopRec(e){
  e.preventDefault();
  if(!recording) return;
  recording=false;
  try{ recSource.disconnect(); recNode.disconnect(); }catch(err){}
  talkBtn.textContent='🎙️ اضغط مع الاستمرار للتحدث';
  let len=0; recBuffers.forEach(b=>len+=b.length);
  if(len<1000){ statusEl.textContent='التسجيل قصير جدًا.'; return; }
  const all=new Float32Array(len); let off=0;
  recBuffers.forEach(b=>{all.set(b,off); off+=b.length;});
  const wav=new Uint8Array(encodeWav(all,recRate));
  let bin=''; const CH=0x8000;
  for(let i=0;i<wav.length;i+=CH) bin+=String.fromCharCode.apply(null,wav.subarray(i,i+CH));
  socket.emit('audio',{data:btoa(bin),format:'wav'});
  statusEl.textContent='تم إرسال رسالتك الصوتية للابتوب.';
}

talkBtn.addEventListener('touchstart',startRec);
talkBtn.addEventListener('touchend',stopRec);
talkBtn.addEventListener('mousedown',startRec);
talkBtn.addEventListener('mouseup',stopRec);
document.body.addEventListener('click',ensureAudio,{once:true});
</script>
</body>
</html>
"""


class PhoneServer:
    def __init__(self, frame_queue: "queue.Queue", port: int = Config.PHONE_SERVER_PORT,
                 audio_player: Optional[AudioPlayer] = None, on_ai_toggle=None,
                 on_ai_audio=None, on_ai_text=None):
        self.available = HAS_FLASK
        self.port = port
        self.frame_queue = frame_queue
        self.audio_player = audio_player
        self.on_ai_toggle = on_ai_toggle
        self.on_ai_audio = on_ai_audio
        self.on_ai_text = on_ai_text
        self.init_error: Optional[str] = None
        self._app = None
        self._socketio = None
        self._thread = None
        self._connected_clients = 0
        self._alarm_state = False

        if not self.available:
            self.init_error = ("flask / flask-socketio غير مثبتة "
                               "(pip install flask flask-socketio) -- وضع الهاتف معطل.")
            log.warning(self.init_error)
            return

        self._app = Flask(__name__)
        self._socketio = SocketIO(self._app, cors_allowed_origins="*",
                                  async_mode="threading", logger=False,
                                  engineio_logger=False)
        self._register_routes()

    def _register_routes(self):
        app, sio = self._app, self._socketio
        ip = self.get_local_ip()
        page = (PHONE_PAGE_HTML
                .replace("__SERVER_IP__", ip)
                .replace("__SERVER_PORT__", str(self.port)))

        @app.route("/")
        def index():
            return render_template_string(page)

        @sio.on("connect")
        def on_connect():
            self._connected_clients += 1
            log.info(f"[Phone] اتصال جديد ({self._connected_clients})")
            try:
                sio.emit("speak", {"text": WELCOME_TEXT})
                if self._alarm_state:
                    sio.emit("alarm", {"on": True})
            except Exception:
                pass

        @sio.on("disconnect")
        def on_disconnect():
            self._connected_clients = max(0, self._connected_clients - 1)
            log.info(f"[Phone] انقطاع اتصال ({self._connected_clients})")

        @sio.on("frame")
        def on_frame(data):
            if not (HAS_CV2 and HAS_NUMPY):
                return
            try:
                _, encoded = data.get("image", "").split(",", 1)
                arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    return
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)
            except Exception as exc:
                log.error(f"[Phone] فشل فك ترميز الإطار: {exc}")

        @sio.on("ai_toggle")
        def on_ai_toggle(data):
            if self.on_ai_toggle:
                self.on_ai_toggle(bool(data.get("enabled")))

        @sio.on("ai_audio")
        def on_ai_audio(data):
            try:
                raw = base64.b64decode(data.get("data", ""))
            except Exception:
                return
            if self.on_ai_audio:
                threading.Thread(target=self.on_ai_audio, args=(raw,), daemon=True).start()

        @sio.on("ai_text")
        def on_ai_text(data):
            if self.on_ai_text:
                threading.Thread(target=self.on_ai_text, args=(str(data.get("text", "")),), daemon=True).start()

        @sio.on("audio")
        def on_audio(data):
            if not self.audio_player:
                return
            try:
                raw = base64.b64decode(data.get("data", ""))
                fmt = data.get("format", "wav")
                log.info(f"[Phone] وصلت رسالة صوتية ({len(raw)} بايت، {fmt})")
                if fmt == "wav":
                    self.audio_player.play_wav_bytes(raw)
                else:
                    self.audio_player.play_bytes(raw, ".webm")
            except Exception as exc:
                log.error(f"[Phone] فشل استقبال الصوت: {exc}")

                                                                             
    def _emit(self, event: str, payload: dict):
        if self._socketio and self.available:
            try:
                self._socketio.emit(event, payload)
            except Exception as exc:
                log.debug(f"[Phone] فشل البث {event}: {exc}")

    def broadcast_speak(self, text: str):
        self._emit("speak", {"text": text})

    def broadcast_alarm(self, on: bool):
        self._alarm_state = on
        self._emit("alarm", {"on": bool(on)})

    def broadcast_ai_message(self, role: str, text: str):
        self._emit("ai_message", {"role": role, "text": text})

    def broadcast_result(self, image_b64: str, info: dict):
        payload = dict(info)
        payload["image"] = image_b64
        self._emit("result", payload)

    @property
    def is_client_connected(self) -> bool:
        return self._connected_clients > 0

    @staticmethod
    def get_local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def start(self):
        if not self.available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info(f"[Phone] افتح على الهاتف: http://{self.get_local_ip()}:{self.port}")

    def _run(self):
        try:
            self._socketio.run(self._app, host="0.0.0.0", port=self.port,
                               allow_unsafe_werkzeug=True)
        except Exception as exc:
            log.error(f"[Phone] فشل تشغيل الخادم: {exc}")
            self.available = False


                                                                                
                
                                                                                
class InputMode:
    LAPTOP = "LAPTOP"
    PHONE = "PHONE"
    ESP32 = "ESP32"
    ARDUINO_ONLY = "ARDUINO_ONLY"


@dataclass
class RuntimeTelemetry:
    ear: Optional[float] = None
    mar: Optional[float] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    eye_text: str = "--"
    head_text: str = "--"
    phone_present: bool = False
    phone_conf: Optional[float] = None
    phone_source: str = "NONE"
    backend: str = "NONE"
    fps: float = 0.0
    state: str = SystemState.WAITING
    message: str = ""


class DriverMonitoringSystem:
    def __init__(self, mode: str, esp_ip: Optional[str] = None,
                 esp_board: Optional[str] = None,
                 arduino_port: Optional[str] = None,
                 use_yolo: bool = True,
                 prefer_mediapipe: bool = True,
                 language: str = LANGUAGE_AR):
        self.mode = mode
        self.language = language if language in (LANGUAGE_AR, LANGUAGE_EN) else LANGUAGE_AR
        self.running = False
        self.capture_error: Optional[str] = None

        self.frame_out_queue: "queue.Queue" = queue.Queue(maxsize=2)
        self._phone_in_queue: "queue.Queue" = queue.Queue(maxsize=2)

        self.vision = FaceAnalyzer(prefer_mediapipe=prefer_mediapipe)
        self.phone_detector = PhoneUseDetector(use_yolo=use_yolo)
        self.audio_player = AudioPlayer()
        self.ai = AdeebAIChat()
        self.ai_chat_queue: "queue.Queue" = queue.Queue(maxsize=50)
        self._ai_mic_stream = None
        self._ai_mic_buffers = []
        self._ai_mic_rate = 44100
        self._ai_mic_recording = False
        self.arduino = ArduinoController(port=arduino_port)
        self.voice = VoiceManager(on_speak_callback=self._broadcast_tts_to_phone)
        self.alarm = AlarmManager(arduino_controller=self.arduino)

        self.phone_server: Optional[PhoneServer] = None
        self.esp_camera: Optional[ESPCamera] = None
        self.cap = None

                      
        if mode == InputMode.PHONE:
            self.phone_server = PhoneServer(self._phone_in_queue,
                                            audio_player=self.audio_player,
                                            on_ai_toggle=self._ai_toggle,
                                            on_ai_audio=self._ai_audio,
                                            on_ai_text=self._ai_text)
            self.phone_server.start()
            self.alarm.set_phone_broadcast(self.phone_server.broadcast_alarm)
            if not self.phone_server.available:
                self.capture_error = self.phone_server.init_error
        elif mode == InputMode.ESP32:
            if not esp_ip or not esp_board:
                self.capture_error = "لم يتم توفير IP أو نوع لوحة ESP32."
            else:
                self.esp_camera = ESPCamera(esp_ip, esp_board)
                self.alarm.set_esp(self.esp_camera)
                if not self.esp_camera.has_camera:
                    self.capture_error = (
                        f"اللوحة {esp_board} بلا كاميرا -- تعمل كوحدة تحكم فقط. "
                        f"اختر Laptop أو Phone كمصدر صورة.")
        else:
            self._open_local_camera()

                                   
        if mode != InputMode.ESP32 and esp_ip:
            self.esp_camera = ESPCamera(esp_ip, esp_board or ESPBoardType.OTHER_UNKNOWN)
            self.alarm.set_esp(self.esp_camera)

        self._eye_closed_since: Optional[float] = None
        self._head_turn_since: Optional[float] = None
        self._phone_seen_since: Optional[float] = None
        self._phone_attention_since: Optional[float] = None
        self._danger_active = False
        self._last_phone_push = 0.0
        self._last_phone_alert = 0.0
        self._fps = 0.0
        self._fps_count = 0
        self._fps_ts = time.time()
        self._thread = None
        self._greeted = False

    def _open_local_camera(self):
        if not HAS_CV2:
            self.capture_error = "OpenCV غير مثبتة -- لا يمكن فتح الكاميرا."
            return
        backend_flags = []
        if platform.system() == "Windows":
            backend_flags = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
        else:
            backend_flags = [0]
        for flag in backend_flags:
            try:
                cap = cv2.VideoCapture(Config.CAMERA_INDEX, flag) if flag else\
                    cv2.VideoCapture(Config.CAMERA_INDEX)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap = cap
                    return
                cap.release()
            except Exception:
                continue
        self.capture_error = "تعذر فتح كاميرا الحاسوب (تحقق من الصلاحيات أو رقم الكاميرا)."

                                                                              
    def _broadcast_tts_to_phone(self, text: str):
        if self.phone_server and self.phone_server.available:
            self.phone_server.broadcast_speak(text)

    def _ai_broadcast(self, role: str, text: str):
        try:
            if self.ai_chat_queue.full(): self.ai_chat_queue.get_nowait()
            self.ai_chat_queue.put((role, text))
        except Exception:
            pass
        if self.phone_server and self.phone_server.available:
            self.phone_server.broadcast_ai_message(role, text)

    def _ai_toggle(self, enabled: bool):
        if enabled:
            greeting=self.ai.enable()
            self._ai_broadcast("ADEEB AI", greeting)
            self.voice.speak_now(greeting)
        else:
            self.ai.disable()

    def _handle_ai_text(self, text: str):
        if not self.ai.enabled: return
        text=(text or "").strip()
        if not text: return
        self._ai_broadcast("Driver", text)
        answer=self.ai.ask(text)
        self._ai_broadcast("ADEEB AI", answer)
        self.voice.speak_now(answer)

    def _ai_text(self, text: str):
        self._handle_ai_text(text)

    def _ai_audio(self, raw: bytes):
        if not self.ai.enabled: return
        text, err = self.ai.transcribe_wav(raw)
        if text:
            self._handle_ai_text(text)
        else:
            answer = "⚠️ " + (err or "لم أتمكن من فهم الصوت. حاول التحدث بوضوح مرة أخرى.")
            self._ai_broadcast("ADEEB AI", answer)
            self.voice.speak_now(answer)

    def ai_enable_from_laptop(self):
        if self.ai.enabled:
            self.ai.disable()
            return False
        greeting=self.ai.enable()
        self._ai_broadcast("ADEEB AI", greeting)
        self.voice.speak_now(greeting)
        return True

    def start_laptop_ai_recording(self) -> Tuple[bool, str]:
        """يبدأ تسجيل ميكروفون اللابتوب لإرساله لأولاما. يعيد (نجاح, رسالة)
        بدل فشل صامت، حتى تظهر للمستخدم سبب عدم عمل الزر بوضوح."""
        if not self.ai.enabled:
            return False, "فعّل ADEEB AI أولاً من الزر الأخضر."
        if self._ai_mic_recording:
            return False, "التسجيل يعمل بالفعل."
        if not HAS_SOUNDDEVICE:
            msg = "مكتبة sounddevice غير مثبتة. نفّذ: pip install sounddevice"
            log.warning(msg)
            return False, msg
        try:
            import sounddevice as sd
            self._ai_mic_buffers=[]
            self._ai_mic_rate=44100
            def callback(indata, frames, time_info, status):
                if self._ai_mic_recording:
                    self._ai_mic_buffers.append(indata[:,0].copy())
            self._ai_mic_recording=True
            self._ai_mic_stream=sd.InputStream(samplerate=self._ai_mic_rate, channels=1, dtype="float32", callback=callback)
            self._ai_mic_stream.start()
            return True, ""
        except Exception as exc:
            self._ai_mic_recording=False
            msg = f"تعذر فتح ميكروفون اللابتوب: {exc}"
            log.warning(msg)
            return False, msg

    def stop_laptop_ai_recording(self) -> Tuple[bool, str]:
        if not self._ai_mic_recording:
            return False, ""
        self._ai_mic_recording=False
        try:
            if self._ai_mic_stream:
                self._ai_mic_stream.stop(); self._ai_mic_stream.close()
        except Exception:
            pass
        self._ai_mic_stream=None
        if not self._ai_mic_buffers:
            return False, "لم يُسجَّل أي صوت (تحقق من صلاحية الميكروفون)."
        import numpy as _np
        samples=_np.concatenate(self._ai_mic_buffers).astype(_np.float32)
        pcm=_np.clip(samples,-1,1)
        raw=(pcm*32767).astype(_np.int16).tobytes()
        bio=io.BytesIO()
        with wave.open(bio,"wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self._ai_mic_rate); wf.writeframes(raw)
        threading.Thread(target=self._ai_audio,args=(bio.getvalue(),),daemon=True).start()
        return True, ""

    def send_intercom(self, text: str):
        """رسالة من اللابتوب إلى الهاتف تُنطق فورًا في المتصفح."""
        if not text.strip():
            return False
        if self.phone_server and self.phone_server.available:
            self.phone_server.broadcast_speak(text.strip())
            return True
        return False

    @staticmethod
    def _normalize_frame(frame):
        """بعض الكاميرات (خصوصًا عبر DirectShow) تُرجع BGRA أو رمادي بدل BGR.
        بدون هذا التطبيع، cv2.cvtColor داخل المعالجة يفشل بصمت ويظهر VISION ERROR
        رغم أن المحرك (tasks/mediapipe) نفسه سليم."""
        if frame is None:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    def _get_frame(self):
        if self.mode == InputMode.PHONE:
            try:
                return self._phone_in_queue.get(timeout=0.4), None
            except queue.Empty:
                if self.phone_server and not self.phone_server.is_client_connected:
                    ip = PhoneServer.get_local_ip()
                    return None, f"بانتظار الهاتف... افتح http://{ip}:{Config.PHONE_SERVER_PORT}"
                return None, None

        if self.mode == InputMode.ESP32:
            if not self.esp_camera or not self.esp_camera.has_camera:
                return None, self.capture_error or "لا توجد كاميرا على هذه اللوحة."
            if not self.esp_camera.connected:
                return None, self.esp_camera.init_error or "ESP32 غير متصل."
            frame = self.esp_camera.read_frame()
            return (frame, None) if frame is not None else (None, "انقطاع بث ESP32.")

        if self.cap is None:
            return None, self.capture_error or "الكاميرا غير متاحة."
        ok, frame = self.cap.read()
        if not ok:
            return None, "فشل قراءة إطار من الكاميرا."
        return frame, None

                                                                                
    def _evaluate(self, vision: VisionResult, phones: List[PhoneDetection], now: float) -> RuntimeTelemetry:
        t = RuntimeTelemetry()
        t.backend = vision.backend
        t.fps = self._fps
        t.phone_source = self.phone_detector.backend

        if vision.error:
            t.state = SystemState.VISION_ERROR
            t.message = vision.error
            self._reset_timers()
            self._release_danger()
            return t

                                             
        eye_hold = head_hold = 0.0
        if vision.face_found:
            t.ear, t.mar, t.yaw, t.pitch = vision.ear, vision.mar, vision.yaw, vision.pitch
            t.eye_text = "مغمضة" if vision.eyes_closed else "مفتوحة"
            t.head_text = {"CENTER":"للأمام","LEFT":"يسار","RIGHT":"يمين","DOWN":"للأسفل","UP":"للأعلى","UNKNOWN":"غير محدد"}.get(vision.head_dir, vision.head_dir)
            if vision.eyes_closed:
                if self._eye_closed_since is None: self._eye_closed_since = now
                eye_hold = now - self._eye_closed_since
            else:
                self._eye_closed_since = None
            if vision.head_dir in ("LEFT","RIGHT"):
                if self._head_turn_since is None: self._head_turn_since = now
                head_hold = now - self._head_turn_since
            else:
                self._head_turn_since = None
        else:
            t.eye_text = "--"; t.head_text = "--"
            self._eye_closed_since = None; self._head_turn_since = None

                                                    
        phone_present = bool(phones)
        t.phone_present = phone_present
        if phone_present:
            confs = [p.confidence for p in phones if p.confidence is not None]
            t.phone_conf = max(confs) if confs else None

        phone_attention = False
        if phone_present and vision.face_found and vision.box:
            fx1, fy1, fx2, fy2 = vision.box
            face_cx = (fx1 + fx2) / 2.0
            face_cy = (fy1 + fy2) / 2.0
            fw = max(1.0, fx2 - fx1)
                                             
            nearest = min(phones, key=lambda d: abs(((d.box[0]+d.box[2])/2.0)-face_cx) + abs(((d.box[1]+d.box[3])/2.0)-face_cy))
            px = (nearest.box[0] + nearest.box[2]) / 2.0
            py = (nearest.box[1] + nearest.box[3]) / 2.0
            rel_x = (px - face_cx) / fw
            phone_on_left = rel_x < -0.12
            phone_on_right = rel_x > 0.12
            gaze_right = vision.gaze_dx > Config.GAZE_DEVIATION_THRESHOLD
            gaze_left = vision.gaze_dx < -Config.GAZE_DEVIATION_THRESHOLD
            head_right = vision.head_dir == "RIGHT"
            head_left = vision.head_dir == "LEFT"
            same_direction = (phone_on_right and (gaze_right or head_right)) or (phone_on_left and (gaze_left or head_left))
            frontal_phone = abs(rel_x) <= 0.12 and abs(vision.gaze_dx) <= Config.GAZE_DEVIATION_THRESHOLD and vision.head_dir == "CENTER"
            phone_attention = same_direction or frontal_phone

        if phone_attention:
            if self._phone_attention_since is None: self._phone_attention_since = now
        else:
            self._phone_attention_since = None
        phone_hold = (now - self._phone_attention_since) if self._phone_attention_since else 0.0

                                         
        if phone_hold >= Config.PHONE_HOLD_SEC:
            t.state = SystemState.PHONE_DANGER
            t.message = "تحذير: السائق ينظر باتجاه الهاتف."
            self._trigger_danger()
            if now - self._last_phone_alert >= Config.PHONE_ALERT_COOLDOWN_SEC:
                self._last_phone_alert = now
                self.voice.speak("phone", localized(PHONE_ALERT_TEXT, PHONE_ALERT_TEXT_EN, self.language), force=True)
                self.arduino.send_command("PHONE_ALERT")
                if self.esp_camera and not self.esp_camera.has_camera:
                    self.esp_camera.send_command("PHONE_ALERT")
        elif vision.face_found and eye_hold >= Config.EYE_DANGER_SEC:
            t.state = SystemState.DANGER
            t.message = "خطر: العينان مغلقتان لأكثر من 4 ثوانٍ."
            self._trigger_danger()
            self.voice.speak("drowsy", localized(DROWSY_ALERT_TEXT, DROWSY_ALERT_TEXT_EN, self.language), force=True)
        elif vision.face_found and head_hold >= Config.HEAD_TURN_HOLD_SEC:
            t.state = SystemState.WARNING; t.message = HEAD_ALERT_TEXT
            self._release_danger(); self.voice.speak("head_turn", localized(HEAD_ALERT_TEXT, HEAD_ALERT_TEXT_EN, self.language))
            self.arduino.send_command("LOOK_FORWARD")
            if self.esp_camera and not self.esp_camera.has_camera: self.esp_camera.send_command("LOOK_FORWARD")
        elif vision.face_found and eye_hold >= Config.EYE_WARNING_SEC:
            t.state = SystemState.WARNING; t.message = "تحذير: العينان تُغلقان."
            self._release_danger()
        elif not vision.face_found:
            t.state = SystemState.NO_FACE; t.message = "لا يوجد وجه في الإطار."; self._release_danger()
        else:
            t.state = SystemState.NORMAL; t.message = "تثاؤب" if vision.yawning else ""; self._release_danger()
        return t

    def _trigger_danger(self):
        if not self._danger_active:
            self._danger_active = True
            self.alarm.start()

    def _release_danger(self):
        if self._danger_active:
            self._danger_active = False
            self.alarm.stop()

    def _reset_timers(self, keep_phone: bool = False):
        self._eye_closed_since = None
        self._head_turn_since = None
        self._phone_attention_since = None
        if not keep_phone:
            self._phone_seen_since = None

                                                                              
    def _draw(self, frame, vision: VisionResult, phones: List[PhoneDetection],
              t: RuntimeTelemetry):
        if not (HAS_CV2 and HAS_NUMPY):
            return frame
        try:
            green = (138, 255, 25)
            red = (77, 39, 255)
            yellow = (77, 216, 255)

            danger = t.state in (SystemState.DANGER, SystemState.PHONE_DANGER)
            box_color = red if danger else green

            person_boxes = list(getattr(self.phone_detector, "person_boxes", []))
            if not person_boxes and vision.box:
                person_boxes = [vision.box]
            for x1, y1, x2, y2 in person_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), green, 2)
                cv2.putText(frame, "PERSON", (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, green, 2)

            if vision.box:
                x1, y1, x2, y2 = vision.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
                cv2.putText(frame, "DRIVER", (x1, max(18, y1 - 26)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            for (ex1, ey1, ex2, ey2) in vision.eye_boxes:
                cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), yellow, 1)

            for hand in getattr(self.phone_detector, "hand_landmarks", []):
                for px, py in hand:
                    cv2.circle(frame, (px, py), 4, green, -1)

            for det in phones:
                x1, y1, x2, y2 = det.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), red, 2)
                label = "PHONE" if det.confidence is None else f"PHONE {det.confidence:.2f}"
                cv2.putText(frame, label, (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, red, 2)

                                                                     
            eye_en = "--"
            if vision.face_found and vision.eyes_closed is not None:
                eye_en = "CLOSED" if vision.eyes_closed else "OPEN"
            lines = [
                f"STATE : {t.state}",
                f"EYE T : {((now := time.time()) - self._eye_closed_since):.1f}s" if self._eye_closed_since else "EYE T : 0.0s",
                f"EYES  : {eye_en}" + (f"  EAR {vision.ear:.2f}" if vision.ear is not None else ""),
                f"HEAD  : {vision.head_dir}" + (f"  YAW {vision.yaw:.0f}" if vision.yaw is not None else ""),
                f"GAZE  : {vision.gaze_dx:+.2f}",
                f"PHONE : {'YES' if t.phone_present else 'NO'} [{t.phone_source}]",
                f"VISION: {vision.backend}   FPS {t.fps:.1f}",
            ]
            panel_w = 360
            if vision.error:
                                                                                      
                err_chunks = [vision.error[i:i + 46] for i in range(0, len(vision.error), 46)][:3]
                lines += [f"ERROR : {c}" for c in err_chunks]
                panel_w = 460
            overlay = frame.copy()
            cv2.rectangle(overlay, (8, 8), (panel_w, 8 + 24 * len(lines) + 8), (12, 15, 20), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            for i, line in enumerate(lines):
                is_err_line = line.startswith("ERROR") or (danger and i == 0)
                color = red if is_err_line else green
                cv2.putText(frame, line, (16, 30 + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            return frame
        except Exception as exc:
            log.error(f"خطأ الرسم: {exc}")
            return frame

                                                                            
    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if not self._greeted:
            self._greeted = True
            self.voice.speak("welcome", localized(WELCOME_TEXT, WELCOME_TEXT_EN, self.language), force=True)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.alarm.stop()
        self.stop_laptop_ai_recording()
        self.voice.stop()
        if self.cap:
            self.cap.release()
        if self.esp_camera:
            self.esp_camera.close()
        if self.arduino:
            self.arduino.close()
        if self.vision:
            self.vision.close()

    def _tick_fps(self):
        self._fps_count += 1
        now = time.time()
        if now - self._fps_ts >= 1.0:
            self._fps = self._fps_count / (now - self._fps_ts)
            self._fps_count = 0
            self._fps_ts = now

    def _loop(self):
        while self.running:
            frame, source_msg = self._get_frame()
            frame = self._normalize_frame(frame)

            if frame is None:
                self._push_output(None, RuntimeTelemetry(
                    state=SystemState.WAITING,
                    backend=self.vision.backend,
                    message=source_msg or "بانتظار الفيديو..."))
                time.sleep(0.05)
                continue

            vision = self.vision.process(frame)
            phones = self.phone_detector.detect(frame, vision)
            t = self._evaluate(vision, phones, time.time())
            frame = self._draw(frame, vision, phones, t)
            self._tick_fps()
            self._push_output(frame, t)
            self._maybe_push_to_phone(frame, vision, t)

    def _maybe_push_to_phone(self, frame, vision: VisionResult, t: RuntimeTelemetry):
        if self.mode != InputMode.PHONE or not self.phone_server:
            return
        if not self.phone_server.available or not self.phone_server.is_client_connected:
            return
        now = time.time()
        if now - self._last_phone_push < (1.0 / Config.PHONE_RESULT_FPS):
            return
        self._last_phone_push = now
        try:
            h, w = frame.shape[:2]
            nw = Config.PHONE_RESULT_WIDTH
            nh = max(1, int(h * nw / w))
            small = cv2.resize(frame, (nw, nh))
            ok, buf = cv2.imencode(".jpg", small,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ok:
                return
            phone_txt = "نعم" if t.phone_present else "لا"
            if t.phone_present and t.phone_conf is not None:
                phone_txt += f" ({t.phone_conf:.2f})"
            self.phone_server.broadcast_result(
                base64.b64encode(buf).decode("ascii"),
                {
                    "state": t.state,
                    "eye": t.eye_text,
                    "head": t.head_text,
                    "phone": phone_txt,
                    "backend": f"{t.backend}/{t.phone_source}",
                    "gaze": f"{vision.gaze_dx:+.2f}",
                    "message": t.message,
                    "danger": t.state in (SystemState.DANGER, SystemState.PHONE_DANGER),
                })
        except Exception as exc:
            log.debug(f"فشل إرسال النتيجة للهاتف: {exc}")

    def _push_output(self, frame, telemetry: RuntimeTelemetry):
        if self.frame_out_queue.full():
            try:
                self.frame_out_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_out_queue.put((frame, telemetry))


                                                                                
                                                         
                                                                                
class LauncherUI:
    """نافذة إعداد رسومية. تُرجع dict بالإعدادات أو None عند الإلغاء."""

    def __init__(self):
        self.result: Optional[dict] = None
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} -- Setup")
        self.root.configure(bg=Config.COLOR_BG)
        self.root.resizable(False, False)
        self._build()

    def _label(self, parent, text, color=None, size=10, bold=False):
        return tk.Label(parent, text=text, bg=Config.COLOR_BG,
                        fg=color or Config.COLOR_TEXT_DIM,
                        font=("Consolas", size, "bold" if bold else "normal"))

    def _build(self):
        pad = {"padx": 16, "pady": 4}

        self._label(self.root, APP_NAME, Config.COLOR_GREEN, 20, True).pack(padx=20, pady=(16, 0))
        self._label(self.root, "Smart Driver Monitoring -- أديب ابوكمييل",
                    Config.COLOR_TEXT_DIM, 9).pack(pady=(0, 10))

        frame = tk.Frame(self.root, bg=Config.COLOR_PANEL, bd=0)
        frame.pack(fill="x", padx=16, pady=6)

        self.mode_var = tk.StringVar(value=InputMode.LAPTOP)
        self.language_var = tk.StringVar(value=LANGUAGE_AR)
        self._label(frame, "اللغة / Language:", Config.COLOR_GREEN, 11, True).grid(row=0, column=1, sticky="w", **pad)
        tk.Radiobutton(frame, text="العربية", value=LANGUAGE_AR, variable=self.language_var, bg=Config.COLOR_PANEL, fg=Config.COLOR_GREEN, selectcolor=Config.COLOR_BG).grid(row=1, column=1, sticky="w", padx=28)
        tk.Radiobutton(frame, text="English", value=LANGUAGE_EN, variable=self.language_var, bg=Config.COLOR_PANEL, fg=Config.COLOR_GREEN, selectcolor=Config.COLOR_BG).grid(row=2, column=1, sticky="w", padx=28)
        self._label(frame, "مصدر الفيديو:", Config.COLOR_GREEN, 11, True).grid(
            row=0, column=0, sticky="w", **pad)
        modes = [("كاميرا اللابتوب", InputMode.LAPTOP),
                 ("كاميرا الهاتف (متصفح)", InputMode.PHONE),
                 ("ESP32", InputMode.ESP32),
                 ("Arduino + كاميرا اللابتوب", InputMode.ARDUINO_ONLY)]
        for i, (text, val) in enumerate(modes):
            tk.Radiobutton(frame, text=text, value=val, variable=self.mode_var,
                           bg=Config.COLOR_PANEL, fg=Config.COLOR_GREEN,
                           selectcolor=Config.COLOR_BG,
                           activebackground=Config.COLOR_PANEL,
                           activeforeground=Config.COLOR_GREEN,
                           font=("Consolas", 10), anchor="w",
                           command=self._on_mode_change).grid(
                row=1 + i, column=0, sticky="w", padx=28)

               
        esp = tk.Frame(self.root, bg=Config.COLOR_PANEL)
        esp.pack(fill="x", padx=16, pady=6)
        self._label(esp, "ESP32 (اختياري لغير وضع ESP32):",
                    Config.COLOR_GREEN, 10, True).grid(row=0, column=0, columnspan=2, sticky="w", **pad)
        self._label(esp, "اللوحة").grid(row=1, column=0, sticky="w", padx=16)
        self.board_var = tk.StringVar(value=ESPBoardType.ESP32_CAM_AI_THINKER)
        ttk.Combobox(esp, textvariable=self.board_var, values=ESPBoardType.ALL,
                     width=26, state="readonly").grid(row=1, column=1, sticky="w", pady=3)
        self._label(esp, "عنوان IP").grid(row=2, column=0, sticky="w", padx=16)
        self.ip_var = tk.StringVar()
        tk.Entry(esp, textvariable=self.ip_var, width=28,
                 bg=Config.COLOR_BG, fg=Config.COLOR_GREEN,
                 insertbackground=Config.COLOR_GREEN).grid(row=2, column=1, sticky="w", pady=3)

                 
        ard = tk.Frame(self.root, bg=Config.COLOR_PANEL)
        ard.pack(fill="x", padx=16, pady=6)
        self._label(ard, "Arduino (اختياري):", Config.COLOR_GREEN, 10, True).grid(
            row=0, column=0, columnspan=3, sticky="w", **pad)
        self._label(ard, "المنفذ").grid(row=1, column=0, sticky="w", padx=16)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(ard, textvariable=self.port_var,
                                       values=ArduinoController.list_available_ports(),
                                       width=20)
        self.port_combo.grid(row=1, column=1, sticky="w", pady=3)
        tk.Button(ard, text="تحديث", command=self._refresh_ports,
                  bg=Config.COLOR_BG, fg=Config.COLOR_GREEN,
                  font=("Consolas", 9), relief="flat").grid(row=1, column=2, padx=6)

                
        opt = tk.Frame(self.root, bg=Config.COLOR_PANEL)
        opt.pack(fill="x", padx=16, pady=6)
        self.mp_var = tk.BooleanVar(value=HAS_MEDIAPIPE)
        self.yolo_var = tk.BooleanVar(value=HAS_YOLO)
        tk.Checkbutton(opt, text="استخدام MediaPipe للوجه (وإلا OpenCV)",
                       variable=self.mp_var, bg=Config.COLOR_PANEL,
                       fg=Config.COLOR_GREEN, selectcolor=Config.COLOR_BG,
                       activebackground=Config.COLOR_PANEL,
                       font=("Consolas", 9), anchor="w").pack(anchor="w", padx=16, pady=2)
        tk.Checkbutton(opt, text="استخدام YOLO لكشف الهاتف (إن وُجد)",
                       variable=self.yolo_var, bg=Config.COLOR_PANEL,
                       fg=Config.COLOR_GREEN, selectcolor=Config.COLOR_BG,
                       activebackground=Config.COLOR_PANEL,
                       font=("Consolas", 9), anchor="w").pack(anchor="w", padx=16, pady=2)

                       
        self.status_text = tk.Text(self.root, height=6, width=54,
                                   bg=Config.COLOR_PANEL, fg=Config.COLOR_TEXT_DIM,
                                   font=("Consolas", 9), relief="flat")
        self.status_text.pack(padx=16, pady=(8, 4))
        self.status_text.insert("1.0", self._dependency_report())
        self.status_text.config(state="disabled")

        tk.Button(self.root, text="ابدأ التشغيل", command=self._start,
                  bg=Config.COLOR_GREEN, fg=Config.COLOR_BG,
                  font=("Consolas", 12, "bold"), relief="flat",
                  padx=20, pady=6).pack(pady=(6, 16))

        self._on_mode_change()

    @staticmethod
    def _dependency_report() -> str:
        def mark(ok):
            return "OK  " if ok else "NO  "
        lines = [
            f"{mark(HAS_CV2)}opencv-python      {mark(HAS_NUMPY)}numpy",
            f"{mark(HAS_MEDIAPIPE)}mediapipe          {mark(HAS_YOLO)}ultralytics (YOLO)",
            f"{mark(HAS_PYTTSX3)}pyttsx3 (نطق)      {mark(HAS_PYSERIAL)}pyserial",
            f"{mark(HAS_FLASK)}flask+socketio     {mark(HAS_REQUESTS)}requests",
            f"{mark(HAS_PIL)}pillow             {mark(HAS_WINSOUND)}winsound",
        ]
        if not HAS_MEDIAPIPE:
            lines.append("ملاحظة: بدون mediapipe يعمل محرك OpenCV تلقائيًا.")
        return "\n".join(lines)

    def _refresh_ports(self):
        self.port_combo["values"] = ArduinoController.list_available_ports()

    def _on_mode_change(self):
        pass

    def _start(self):
        self.result = {
            "mode": self.mode_var.get(),
            "esp_ip": self.ip_var.get().strip() or None,
            "esp_board": self.board_var.get(),
            "arduino_port": self.port_var.get().strip() or None,
            "language": self.language_var.get(),
            "use_yolo": bool(self.yolo_var.get()),
            "prefer_mediapipe": bool(self.mp_var.get()),
        }
        self.root.destroy()

    def run(self) -> Optional[dict]:
        self.root.mainloop()
        return self.result


                                                                                
             
                                                                                
class DashboardUI:
    POLL_MS = 20

    def __init__(self, system: DriverMonitoringSystem):
        if not HAS_TK:
            raise RuntimeError("tkinter غير متوفرة.")
        if not HAS_PIL:
            raise RuntimeError("Pillow غير متوفرة (pip install pillow).")

        self.system = system
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} -- SMART DRIVER MONITORING")
        self.root.configure(bg=Config.COLOR_BG)
        self.root.geometry(f"{Config.FRAME_WIDTH + 260}x{Config.FRAME_HEIGHT + 120}")
        self.root.minsize(900, 480)
        self._current_photo = None
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self):
        mono = ("Consolas", 10)

                                                                            
                                                                       
                                                                      
                                                                         
                                                                        
                                                                         
                                                              
                                                                            
        top = tk.Frame(self.root, bg=Config.COLOR_PANEL, height=36)
        top.pack(side="top", fill="x")

        self.dot_canvas = tk.Canvas(top, width=16, height=16, bg=Config.COLOR_PANEL,
                                    highlightthickness=0)
        self.dot_canvas.pack(side="left", padx=(10, 4), pady=9)
        self._dot = self.dot_canvas.create_oval(2, 2, 12, 12, fill=Config.COLOR_RED,
                                                outline="")

        tk.Label(top, text=APP_NAME, bg=Config.COLOR_PANEL, fg=Config.COLOR_GREEN,
                 font=("Consolas", 14, "bold")).pack(side="left", padx=6)

        self.status_label = tk.Label(top, text=SystemState.WAITING,
                                     bg=Config.COLOR_PANEL,
                                     fg=STATE_COLORS[SystemState.WAITING],
                                     font=("Consolas", 12, "bold"))
        self.status_label.pack(side="left", padx=14)

        self.clock_label = tk.Label(top, text="", bg=Config.COLOR_PANEL,
                                    fg=Config.COLOR_TEXT_DIM, font=mono)
        self.clock_label.pack(side="right", padx=10)

        self.ai_button = tk.Button(top, text="🟢 تفعيل ADEEB AI", command=self._toggle_ai,
                                   bg=Config.COLOR_GREEN, fg=Config.COLOR_BG,
                                   font=("Consolas", 9, "bold"), relief="flat")
        self.ai_button.pack(side="right", padx=8)

                                                                            
                                                                           
                                                             
        comm = tk.Frame(top, bg=Config.COLOR_PANEL)
        comm.pack(side="right", padx=8)
        tk.Label(comm, text="رسالة صوتية إلى الهاتف:", bg=Config.COLOR_PANEL,
                 fg=Config.COLOR_TEXT_DIM, font=("Consolas", 8)).pack(side="left", padx=(0, 4))
        self.msg_var = tk.StringVar()
        entry = tk.Entry(comm, textvariable=self.msg_var, bg=Config.COLOR_BG,
                         fg=Config.COLOR_GREEN, insertbackground=Config.COLOR_GREEN,
                         font=("Consolas", 9), width=22)
        entry.pack(side="left", padx=4, pady=6)
        entry.bind("<Return>", lambda e: self._send_msg())
        tk.Button(comm, text="إرسال", command=self._send_msg,
                  bg=Config.COLOR_GREEN, fg=Config.COLOR_BG, relief="flat",
                  font=("Consolas", 9, "bold")).pack(side="left", padx=4)
        self.comm_status = tk.Label(comm, text="", bg=Config.COLOR_PANEL,
                                    fg=Config.COLOR_YELLOW, font=("Consolas", 8))
        self.comm_status.pack(side="left", padx=6)

        self.ai_chat_frame = tk.Frame(self.root, bg=Config.COLOR_PANEL, width=300)
        self.ai_chat_frame.pack(side="right", fill="y", padx=(0,4), pady=4)
        tk.Label(self.ai_chat_frame, text="محادثة ADEEB AI", bg=Config.COLOR_PANEL,
                 fg=Config.COLOR_YELLOW, font=("Consolas", 11, "bold")).pack(pady=6)
        self.ai_chat = tk.Text(self.ai_chat_frame, width=32, height=18, bg=Config.COLOR_BG,
                               fg=Config.COLOR_GREEN, font=("Consolas", 9), relief="flat",
                               state="disabled", wrap="word")
        self.ai_chat.pack(fill="both", expand=True, padx=6, pady=4)
        self.ai_lap_btn = tk.Button(self.ai_chat_frame, text="🟢 تفعيل الذكاء الاصطناعي",
                                    command=self._toggle_ai, bg=Config.COLOR_GREEN, fg=Config.COLOR_BG,
                                    font=("Consolas", 10, "bold"), relief="flat")
        self.ai_lap_btn.pack(fill="x", padx=6, pady=5)
        self.ai_mic_btn = tk.Button(self.ai_chat_frame, text="🎙️ اضغط مع الاستمرار للتحدث",
                                    bg=Config.COLOR_PANEL, fg=Config.COLOR_GREEN,
                                    font=("Consolas", 9), relief="flat")
        self.ai_mic_btn.pack(fill="x", padx=6, pady=(0,6))
        self.ai_mic_btn.bind("<ButtonPress-1>", self._laptop_ai_mic_down)
        self.ai_mic_btn.bind("<ButtonRelease-1>", self._laptop_ai_mic_up)

        bottom = tk.Frame(self.root, bg=Config.COLOR_PANEL)
        bottom.pack(side="bottom", fill="x")
        self.telemetry_label = tk.Label(
            bottom, text="EYE: -- | HEAD: -- | PHONE: -- | FPS: --",
            bg=Config.COLOR_PANEL, fg=Config.COLOR_TEXT_DIM, font=mono)
        self.telemetry_label.pack(side="left", padx=10, pady=6)
        self.message_label = tk.Label(bottom, text="", bg=Config.COLOR_PANEL,
                                      fg=Config.COLOR_YELLOW, font=mono)
        self.message_label.pack(side="right", padx=10, pady=6)

                                                                          
                                       
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(side="top", fill="both", expand=True)

    def _laptop_ai_mic_down(self, event=None):
        ok, msg = self.system.start_laptop_ai_recording()
        if ok:
            self.ai_mic_btn.config(text="🔴 أستمع إليك...")
            self.comm_status.config(text="")
        elif msg:
            self.comm_status.config(text=msg)

    def _laptop_ai_mic_up(self, event=None):
        ok, msg = self.system.stop_laptop_ai_recording()
        self.ai_mic_btn.config(text="🎙️ اضغط مع الاستمرار للتحدث")
        if msg:
            self.comm_status.config(text=msg)
        elif ok:
            self.comm_status.config(text="تم الإرسال إلى ADEEB AI...")

    def _append_ai_chat(self, role: str, text: str):
        self.ai_chat.config(state="normal")
        self.ai_chat.insert("end", f"{role}: {text}\n\n")
        self.ai_chat.see("end")
        self.ai_chat.config(state="disabled")

    def _toggle_ai(self):
        enabled=self.system.ai_enable_from_laptop()
        label="🔴 إيقاف ADEEB AI" if enabled else "🟢 تفعيل ADEEB AI"
        color=Config.COLOR_RED if enabled else Config.COLOR_GREEN
        self.ai_button.config(text=label,bg=color,fg="white" if enabled else Config.COLOR_BG)
        self.ai_lap_btn.config(text=label,bg=color,fg="white" if enabled else Config.COLOR_BG)
        if enabled: self._append_ai_chat("ADEEB AI", ADEEB_AI_GREETING)

    def _send_msg(self):
        text = self.msg_var.get()
        ok = self.system.send_intercom(text)
        self.comm_status.config(text="تم الإرسال" if ok else "لا يوجد هاتف متصل")
        if ok:
            self.msg_var.set("")

    def _on_close(self):
        self.system.stop()
        self.root.destroy()

    def _poll(self):
        try:
            frame, telemetry = self.system.frame_out_queue.get_nowait()
        except queue.Empty:
            frame, telemetry = None, None

        self.clock_label.config(
            text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        try:
            while True:
                role,text=self.system.ai_chat_queue.get_nowait()
                self._append_ai_chat(role,text)
        except queue.Empty:
            pass

        if telemetry is not None:
            color = STATE_COLORS.get(telemetry.state, Config.COLOR_TEXT_DIM)
            self.status_label.config(text=telemetry.state, fg=color)
            self.dot_canvas.itemconfig(self._dot, fill=color)

            ear_txt = f" ({telemetry.ear:.2f})" if telemetry.ear is not None else ""
            yaw_txt = f" ({telemetry.yaw:.0f}°)" if telemetry.yaw is not None else ""
            phone_txt = "YES" if telemetry.phone_present else "NO"
            if telemetry.phone_present and telemetry.phone_conf is not None:
                phone_txt += f" {telemetry.phone_conf:.2f}"
            self.telemetry_label.config(
                text=(f"EYE: {telemetry.eye_text}{ear_txt} | "
                      f"HEAD: {telemetry.head_text}{yaw_txt} | "
                      f"PHONE: {phone_txt} [{telemetry.phone_source}] | "
                      f"VISION: {telemetry.backend} | FPS: {telemetry.fps:.1f}"))
            self.message_label.config(text=telemetry.message)

        if frame is not None and HAS_CV2:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                tw = max(self.video_label.winfo_width(), 320)
                th = max(self.video_label.winfo_height(), 240)
                ratio = min(tw / img.width, th / img.height)
                img = img.resize((max(1, int(img.width * ratio)),
                                  max(1, int(img.height * ratio))))
                photo = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=photo)
                self._current_photo = photo
            except Exception as exc:
                log.error(f"خطأ عرض الإطار: {exc}")
        elif frame is None and telemetry is not None:
            self.video_label.config(image="")
            self._current_photo = None

        self.root.after(self.POLL_MS, self._poll)

    def run(self):
        self.system.start()
        self.root.after(self.POLL_MS, self._poll)
        self.root.mainloop()


                                                                                
                                     
                                                                                
ARDUINO_UNO_SKETCH = r"""

const int BUZZER_PIN = 8;
const int LED_PIN = 13;

bool alarmActive = false;
unsigned long lastToggleMs = 0;
bool buzzerToggleState = false;
const unsigned long ALARM_TOGGLE_INTERVAL_MS = 200;

String inputBuffer = "";

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(9600);
  Serial.println("ADEEB AI -- Arduino Uno Ready");
}

void loop() {
  readSerialCommands();
  if (alarmActive) {
    unsigned long now = millis();
    if (now - lastToggleMs >= ALARM_TOGGLE_INTERVAL_MS) {
      lastToggleMs = now;
      buzzerToggleState = !buzzerToggleState;
      digitalWrite(BUZZER_PIN, buzzerToggleState ? HIGH : LOW);
      digitalWrite(LED_PIN, buzzerToggleState ? HIGH : LOW);
    }
  }
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      inputBuffer.trim();
      if (inputBuffer.length() > 0) handleCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}

void handleCommand(const String &cmd) {
  if (cmd == "ALARM_ON") {
    alarmActive = true;
    Serial.println("ACK ALARM_ON");
  } else if (cmd == "ALARM_OFF") {
    alarmActive = false;
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_PIN, LOW);
    Serial.println("ACK ALARM_OFF");
  } else if (cmd == "PHONE_ALERT") {
    shortAlertPattern(3, 120);
    Serial.println("ACK PHONE_ALERT");
  } else if (cmd == "LOOK_FORWARD") {
    shortAlertPattern(1, 400);
    Serial.println("ACK LOOK_FORWARD");
  } else {
    Serial.print("UNKNOWN_COMMAND: ");
    Serial.println(cmd);
  }
}

void shortAlertPattern(int pulses, int pulseDurationMs) {
  for (int i = 0; i < pulses; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    digitalWrite(LED_PIN, HIGH);
    delay(pulseDurationMs);
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_PIN, LOW);
    if (i < pulses - 1) delay(pulseDurationMs);
  }
}
"""

ESP32_AI_THINKER_SKETCH = r"""

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

httpd_handle_t streamServerHandle = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];
  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
    } else {
      if (fb->format != PIXFORMAT_JPEG) { esp_camera_fb_return(fb); continue; }
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
      if (res == ESP_OK) {
        size_t hlen = snprintf(part_buf, 64, STREAM_PART, fb->len);
        res = httpd_resp_send_chunk(req, part_buf, hlen);
      }
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      esp_camera_fb_return(fb);
    }
    if (res != ESP_OK) break;
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.ctrl_port = 32768;
  httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET,
                             .handler = stream_handler, .user_ctx = NULL };
  if (httpd_start(&streamServerHandle, &config) == ESP_OK) {
    httpd_register_uri_handler(streamServerHandle, &stream_uri);
  }
}

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  if (psramFound()) { config.frame_size = FRAMESIZE_VGA; config.jpeg_quality = 12; config.fb_count = 2; }
  else { config.frame_size = FRAMESIZE_QVGA; config.jpeg_quality = 15; config.fb_count = 1; }
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) Serial.printf("Camera init failed: 0x%x\n", err);
}

void setup() {
  Serial.begin(115200);
  setupCamera();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.println();
  Serial.print("IP: "); Serial.println(WiFi.localIP());
  startCameraServer();
  Serial.println("Stream on :81/stream");
}

void loop() { delay(10000); }
"""

ESP32_S3_CAM_SKETCH = r"""

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM      4
#define SIOC_GPIO_NUM      5
#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM        8
#define Y3_GPIO_NUM        9
#define Y2_GPIO_NUM       11
#define VSYNC_GPIO_NUM     6
#define HREF_GPIO_NUM      7
#define PCLK_GPIO_NUM     13

httpd_handle_t streamServerHandle = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[64];
  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) { res = ESP_FAIL; }
    else {
      if (fb->format != PIXFORMAT_JPEG) { esp_camera_fb_return(fb); continue; }
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
      if (res == ESP_OK) {
        size_t hlen = snprintf(part_buf, 64, STREAM_PART, fb->len);
        res = httpd_resp_send_chunk(req, part_buf, hlen);
      }
      if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      esp_camera_fb_return(fb);
    }
    if (res != ESP_OK) break;
  }
  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.ctrl_port = 32768;
  httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET,
                             .handler = stream_handler, .user_ctx = NULL };
  if (httpd_start(&streamServerHandle, &config) == ESP_OK) {
    httpd_register_uri_handler(streamServerHandle, &stream_uri);
  }
}

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  if (psramFound()) { config.frame_size = FRAMESIZE_VGA; config.jpeg_quality = 12; config.fb_count = 2; }
  else { config.frame_size = FRAMESIZE_QVGA; config.jpeg_quality = 15; config.fb_count = 1; }
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) Serial.printf("Camera init failed: 0x%x\n", err);
}

void setup() {
  Serial.begin(115200);
  setupCamera();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.println();
  Serial.print("IP: "); Serial.println(WiFi.localIP());
  startCameraServer();
  Serial.println("Stream on :81/stream");
}

void loop() { delay(10000); }
"""

ESP32_CONTROLLER_ONLY_SKETCH = r"""

#include <WiFi.h>
#include <WebServer.h>

const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

const int BUZZER_PIN = 4;
const int LED_PIN = 2;

WebServer server(80);

bool alarmActive = false;
unsigned long lastToggleMs = 0;
bool toggleState = false;
const unsigned long TOGGLE_INTERVAL_MS = 200;

void handleRoot() { server.send(200, "text/plain", "ADEEB AI -- ESP32 Controller Ready"); }
void handleAlarmOn() { alarmActive = true; server.send(200, "text/plain", "ACK ALARM_ON"); }
void handleAlarmOff() {
  alarmActive = false;
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  server.send(200, "text/plain", "ACK ALARM_OFF");
}

void shortAlertPattern(int pulses, int pulseDurationMs) {
  for (int i = 0; i < pulses; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    digitalWrite(LED_PIN, HIGH);
    delay(pulseDurationMs);
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_PIN, LOW);
    if (i < pulses - 1) delay(pulseDurationMs);
  }
}

void handlePhoneAlert() { shortAlertPattern(3, 120); server.send(200, "text/plain", "ACK PHONE_ALERT"); }
void handleLookForward() { shortAlertPattern(1, 400); server.send(200, "text/plain", "ACK LOOK_FORWARD"); }

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.println();
  Serial.print("IP: "); Serial.println(WiFi.localIP());
  server.on("/", handleRoot);
  server.on("/alarm_on", handleAlarmOn);
  server.on("/alarm_off", handleAlarmOff);
  server.on("/phone_alert", handlePhoneAlert);
  server.on("/look_forward", handleLookForward);
  server.begin();
}

void loop() {
  server.handleClient();
  if (alarmActive) {
    unsigned long now = millis();
    if (now - lastToggleMs >= TOGGLE_INTERVAL_MS) {
      lastToggleMs = now;
      toggleState = !toggleState;
      digitalWrite(BUZZER_PIN, toggleState ? HIGH : LOW);
      digitalWrite(LED_PIN, toggleState ? HIGH : LOW);
    }
  }
}
"""


                                                                                
                                      
                                                                                
def _select_from_menu(title: str, options: List[str]) -> int:
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        print(f"  [{i}] {opt}")
    while True:
        choice = input("اختر رقمًا: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice)
        print("اختيار غير صالح.")


def run_console_menu() -> dict:
    print("=" * 60)
    print(f"        {APP_NAME} -- SMART DRIVER MONITORING")
    print("=" * 60)
    print(LauncherUI._dependency_report())

    modes = ["Laptop Camera", "Phone Camera", "ESP32", "Arduino + Laptop Camera"]
    c = _select_from_menu("اختر مصدر الفيديو:", modes)
    mode = [InputMode.LAPTOP, InputMode.PHONE, InputMode.ESP32,
            InputMode.ARDUINO_ONLY][c - 1]

    esp_ip = esp_board = arduino_port = None
    if mode == InputMode.ESP32:
        b = _select_from_menu("اختر نوع اللوحة:", ESPBoardType.ALL)
        esp_board = ESPBoardType.ALL[b - 1]
        esp_ip = input("عنوان IP الخاص بـ ESP32: ").strip() or None

    ports = ArduinoController.list_available_ports()
    if ports:
        print("\nمنافذ Serial المتاحة: " + ", ".join(ports))
    arduino_port = input("منفذ Arduino (اتركه فارغًا للتخطي): ").strip() or None

    return {"mode": mode, "esp_ip": esp_ip, "esp_board": esp_board,
            "arduino_port": arduino_port, "language": LANGUAGE_AR if language == 1 else LANGUAGE_EN, "use_yolo": HAS_YOLO,
            "prefer_mediapipe": HAS_MEDIAPIPE}


def export_firmware_files(target_dir: str):
    mapping = {
        "adeeb_ai_uno.ino": ARDUINO_UNO_SKETCH,
        "adeeb_ai_esp32cam_ai_thinker.ino": ESP32_AI_THINKER_SKETCH,
        "adeeb_ai_esp32s3_cam.ino": ESP32_S3_CAM_SKETCH,
        "adeeb_ai_esp32_controller.ino": ESP32_CONTROLLER_ONLY_SKETCH,
    }
    os.makedirs(target_dir, exist_ok=True)
    for filename, content in mapping.items():
        path = os.path.join(target_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"تم إنشاء: {path}")
    print(f"\nالمجلد: {os.path.abspath(target_dir)}")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--export-firmware":
        export_firmware_files(sys.argv[2] if len(sys.argv) >= 3 else "./firmware_export")
        return

    cfg = None
    if HAS_TK:
        try:
            cfg = LauncherUI().run()
        except Exception as exc:
            log.error(f"فشل فتح واجهة الإعداد ({exc}) -- التحويل إلى التيرمنال.")
    if cfg is None:
        if HAS_TK:
                                         
            print("تم الإلغاء.")
            return
        cfg = run_console_menu()

    system = DriverMonitoringSystem(
        mode=cfg["mode"], esp_ip=cfg.get("esp_ip"), esp_board=cfg.get("esp_board"),
        arduino_port=cfg.get("arduino_port"), use_yolo=cfg.get("use_yolo", True),
        prefer_mediapipe=cfg.get("prefer_mediapipe", True), language=cfg.get("language", LANGUAGE_AR))

    if system.capture_error:
        print(f"\n[تنبيه مصدر الفيديو] {system.capture_error}")
    print(f"[محرك الوجه] {system.vision.backend}   "
          f"[كشف الهاتف] {system.phone_detector.backend}")
    if system.vision.backend == "NONE":
        print("\n" + "!" * 60)
        print("VISION ERROR -- محرك تحليل الوجه غير جاهز. السبب الدقيق:")
        print(f"  {system.vision.init_error}")
        print("\nالحلول بالترتيب:")
        print("  1) pip install mediapipe                (الأفضل: دقة EAR/زوايا الرأس)")
        print("  2) pip install --upgrade --force-reinstall opencv-python  (المسار الاحتياطي)")
        print("  3) إن كان Python لديك 3.13 فـ mediapipe قد لا تتوفر له -- "
              "استخدم Python 3.11 أو 3.12.")
        print("ملاحظة: YOLO لا علاقة له بهذه الرسالة -- هو لكشف الهاتف فقط.")
        print("!" * 60 + "\n")
    if cfg["mode"] == InputMode.PHONE:
        print(f"[الهاتف] افتح: http://{PhoneServer.get_local_ip()}:{Config.PHONE_SERVER_PORT}")

    if not HAS_TK or not HAS_PIL:
        print("\n[وضع نصي] tkinter/pillow غير متوفرة. Ctrl+C للإيقاف.")
        system.start()
        try:
            while True:
                try:
                    _, t = system.frame_out_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if t:
                    print(f"[{t.state}] EYE={t.eye_text} HEAD={t.head_text} "
                          f"PHONE={t.phone_present} FPS={t.fps:.1f} -- {t.message}")
        except KeyboardInterrupt:
            pass
        finally:
            system.stop()
        return

    ui = DashboardUI(system)
    try:
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()


if __name__ == "__main__":
    main()
