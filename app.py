"""
YOLOv10 Real-Time Object Detection GUI
Requires: pip install ultralytics opencv-python pillow
Model: YOLOv10n/s/m/b/l/x (auto-downloaded on first run)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
from PIL import Image, ImageTk
import sys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# WIDGETS

class ModernButton(tk.Button):
    """Flat button with hover-lighten effect."""
    def __init__(self, parent, **kwargs):
        defaults = {
            'relief': tk.FLAT, 'cursor': 'hand2',
            'bd': 0, 'highlightthickness': 0,
            'font': ('Courier New', 10, 'bold'),
            'padx': 22, 'pady': 11,
        }
        defaults.update(kwargs)
        super().__init__(parent, **defaults)
        self.original_bg = defaults.get('bg', '#404040')
        self.hover_bg    = self._lighten(self.original_bg)
        self.bind('<Enter>', lambda e: self.config(bg=self.hover_bg))
        self.bind('<Leave>', lambda e: self.config(bg=self.original_bg))

    @staticmethod
    def _lighten(color):
        if isinstance(color, str) and color.startswith('#') and len(color) == 7:
            r, g, b = (int(color[i:i+2], 16) for i in (1, 3, 5))
            return f"#{min(255,r+35):02x}{min(255,g+35):02x}{min(255,b+35):02x}"
        return color


class StatCard(tk.Frame):
    """Statistic display card with colored accent bar."""
    def __init__(self, parent, title, value="0", color="#4CAF50", **kwargs):
        super().__init__(parent, bg='#1e1e1e', relief=tk.FLAT, bd=0, **kwargs)
        tk.Label(self, text=title, bg='#1e1e1e', fg='#888888',
                 font=('Courier New', 9), anchor='w').pack(fill=tk.X, padx=14, pady=(14, 3))
        self.val = tk.Label(self, text=value, bg='#1e1e1e', fg=color,
                            font=('Courier New', 22, 'bold'), anchor='w')
        self.val.pack(fill=tk.X, padx=14, pady=(0, 14))
        tk.Frame(self, bg=color, height=3).pack(fill=tk.X, side=tk.BOTTOM)

    def set(self, value):
        self.val.config(text=str(value))


class LogPanel(tk.Frame):
    """Scrollable log output panel."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg='#141414', **kwargs)
        sb = tk.Scrollbar(self)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text = tk.Text(
            self, bg='#141414', fg='#00ff88',
            font=('Courier New', 9), relief=tk.FLAT,
            yscrollcommand=sb.set, state=tk.DISABLED,
            wrap=tk.WORD, insertbackground='#00ff88',
        )
        self.text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        sb.config(command=self.text.yview)

    def log(self, msg, tag='info'):
        colors = {'info': '#00ff88', 'warn': '#ffcc00', 'err': '#ff4444', 'det': '#44aaff'}
        self.text.config(state=tk.NORMAL)
        ts = time.strftime('%H:%M:%S')
        self.text.insert(tk.END, f"[{ts}] {msg}\n", tag)
        self.text.tag_config(tag, foreground=colors.get(tag, '#00ff88'))
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

# MAIN APPLICATION

class YOLOv10App:
    # ── Color palette ──────────────────────────────────────────────────────────
    C = {
        'bg':        '#0d0d0d',
        'bg2':       '#1a1a1a',
        'card':      '#1e1e1e',
        'border':    '#2a2a2a',
        'text':      '#e8e8e8',
        'sub':       '#888888',
        'green':     '#00ff88',
        'blue':      '#4488ff',
        'yellow':    '#ffcc00',
        'red':       '#ff4444',
        'btn_start': '#1a6b3a',
        'btn_stop':  '#6b1a1a',
        'btn_snap':  '#1a3a6b',
    }

    MODELS = ['yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt',
              'yolov10b.pt', 'yolov10l.pt', 'yolov10x.pt']

    def __init__(self, root: tk.Tk):
        self.root = root
        self._setup_window()

        # State
        self.model        = None
        self.cap          = None
        self.running      = False
        self._thread      = None
        self._photo       = None   # reference for tkinter GC
        self._frame_times = []

        # Build UI
        self._build_header()
        self._build_body()
        self._build_footer()

        self.log("YOLOv10 GUI ready. Select model & camera, then press Start.", 'info')
        if not YOLO_AVAILABLE:
            self.log("ultralytics not found! Run:  pip install ultralytics", 'err')

    # ── Window ─────────────────────────────────────────────────────────────────
    def _setup_window(self):
        self.root.title("YOLOv10 · Real-Time Detection")
        self.root.geometry("1280x780")
        self.root.minsize(900, 600)
        self.root.configure(bg=self.C['bg'])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Header ─────────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=self.C['bg2'], height=64)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⬡ YOLOv10", bg=self.C['bg2'], fg=self.C['green'],
                 font=('Courier New', 18, 'bold')).pack(side=tk.LEFT, padx=20, pady=14)
        tk.Label(hdr, text="Real-Time Object Detection", bg=self.C['bg2'], fg=self.C['sub'],
                 font=('Courier New', 10)).pack(side=tk.LEFT, padx=(0, 20), pady=14)

        # Status pill (right side)
        pill = tk.Frame(hdr, bg=self.C['bg2'])
        pill.pack(side=tk.RIGHT, padx=20)
        self._dot = tk.Label(pill, text="●", bg=self.C['bg2'], fg=self.C['red'],
                             font=('Courier New', 18))
        self._dot.pack(side=tk.LEFT)
        self._status_lbl = tk.Label(pill, text=" Idle", bg=self.C['bg2'], fg=self.C['text'],
                                    font=('Courier New', 11, 'bold'))
        self._status_lbl.pack(side=tk.LEFT)

    # ── Body (left panel + right panel) ────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=self.C['bg'])
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=(12, 6))

        self._build_left(body)
        self._build_right(body)

    # ── Left: video + stats ────────────────────────────────────────────────────
    def _build_left(self, parent):
        left = tk.Frame(parent, bg=self.C['bg'])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # ---- Video card ----
        vcard = tk.Frame(left, bg=self.C['card'])
        vcard.pack(fill=tk.BOTH, expand=True)

        tk.Label(vcard, text="📹  Live Feed", bg=self.C['card'], fg=self.C['text'],
                 font=('Courier New', 13, 'bold')).pack(anchor='w', padx=18, pady=(16, 8))

        border = tk.Frame(vcard, bg=self.C['border'])
        border.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 18))

        self.canvas = tk.Canvas(border, bg='#000', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._placeholder = self.canvas.create_text(
            400, 270,
            text="No camera feed\nPress  ▶  Start Detection  to begin",
            fill='#333', font=('Courier New', 12), justify=tk.CENTER
        )

        # ---- Stats row ----
        stats = tk.Frame(left, bg=self.C['bg'])
        stats.pack(fill=tk.X, pady=(10, 0))

        self._fps_card  = StatCard(stats, "⚡ FPS",             "0.0",  self.C['green'])
        self._obj_card  = StatCard(stats, "🔲 Objects",          "0",    self.C['blue'])
        self._cls_card  = StatCard(stats, "🏷  Unique Classes",   "0",    self.C['yellow'])
        self._conf_card = StatCard(stats, "🎯 Avg Confidence",   "0%",   self.C['red'])

        for card in (self._fps_card, self._obj_card, self._cls_card, self._conf_card):
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

    # ── Right: controls + log ──────────────────────────────────────────────────
    def _build_right(self, parent):
        right = tk.Frame(parent, bg=self.C['bg'], width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # ---- Control card ----
        ctrl = tk.Frame(right, bg=self.C['card'])
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text="⚙  Controls", bg=self.C['card'], fg=self.C['text'],
                 font=('Courier New', 12, 'bold')).pack(anchor='w', padx=16, pady=(16, 10))
        tk.Frame(ctrl, bg=self.C['border'], height=1).pack(fill=tk.X, padx=16)

        # Model selector
        self._section(ctrl, "MODEL")
        self._model_var = tk.StringVar(value=self.MODELS[0])
        model_dd = ttk.Combobox(ctrl, textvariable=self._model_var,
                                values=self.MODELS, state='readonly', width=22)
        model_dd.pack(padx=16, pady=(0, 10), fill=tk.X)
        self._style_combo(model_dd)

        # Camera index
        self._section(ctrl, "CAMERA INDEX")
        self._cam_var = tk.StringVar(value="0")
        cam_frame = tk.Frame(ctrl, bg=self.C['card'])
        cam_frame.pack(fill=tk.X, padx=16, pady=(0, 10))
        for i in range(4):
            tk.Radiobutton(cam_frame, text=str(i), variable=self._cam_var, value=str(i),
                           bg=self.C['card'], fg=self.C['text'], selectcolor=self.C['bg2'],
                           font=('Courier New', 10), activebackground=self.C['card']
                           ).pack(side=tk.LEFT, padx=(0, 8))

        # Confidence threshold
        self._section(ctrl, "CONFIDENCE THRESHOLD")
        self._conf_var = tk.DoubleVar(value=0.40)
        conf_row = tk.Frame(ctrl, bg=self.C['card'])
        conf_row.pack(fill=tk.X, padx=16, pady=(0, 10))
        self._conf_lbl = tk.Label(conf_row, text="0.40", bg=self.C['card'],
                                  fg=self.C['green'], font=('Courier New', 10, 'bold'), width=5)
        self._conf_lbl.pack(side=tk.RIGHT)
        tk.Scale(conf_row, variable=self._conf_var, from_=0.1, to=0.95,
                 resolution=0.05, orient=tk.HORIZONTAL,
                 bg=self.C['card'], fg=self.C['text'], troughcolor=self.C['border'],
                 highlightthickness=0, showvalue=False,
                 command=lambda v: self._conf_lbl.config(text=f"{float(v):.2f}")
                 ).pack(fill=tk.X, side=tk.LEFT, expand=True)

        # Show labels toggle
        self._show_labels = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Show detection labels", variable=self._show_labels,
                       bg=self.C['card'], fg=self.C['text'], selectcolor=self.C['bg2'],
                       font=('Courier New', 9), activebackground=self.C['card']
                       ).pack(anchor='w', padx=16, pady=(0, 6))

        # Show confidence toggle
        self._show_conf = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Show confidence scores", variable=self._show_conf,
                       bg=self.C['card'], fg=self.C['text'], selectcolor=self.C['bg2'],
                       font=('Courier New', 9), activebackground=self.C['card']
                       ).pack(anchor='w', padx=16, pady=(0, 14))

        tk.Frame(ctrl, bg=self.C['border'], height=1).pack(fill=tk.X, padx=16)

        # Buttons
        btn_frame = tk.Frame(ctrl, bg=self.C['card'])
        btn_frame.pack(fill=tk.X, padx=16, pady=14)

        self._btn_start = ModernButton(btn_frame, text="▶  Start",
                                       bg=self.C['btn_start'], fg='#ffffff',
                                       command=self._start)
        self._btn_start.pack(fill=tk.X, pady=(0, 6))

        self._btn_stop = ModernButton(btn_frame, text="■  Stop",
                                      bg=self.C['btn_stop'], fg='#ffffff',
                                      command=self._stop, state=tk.DISABLED)
        self._btn_stop.pack(fill=tk.X, pady=(0, 6))

        self._btn_snap = ModernButton(btn_frame, text="📷  Snapshot",
                                      bg=self.C['btn_snap'], fg='#ffffff',
                                      command=self._snapshot)
        self._btn_snap.pack(fill=tk.X)

        # ---- Log card ----
        log_card = tk.Frame(right, bg=self.C['card'])
        log_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        tk.Label(log_card, text="📋  Detection Log", bg=self.C['card'], fg=self.C['text'],
                 font=('Courier New', 11, 'bold')).pack(anchor='w', padx=16, pady=(14, 6))
        tk.Frame(log_card, bg=self.C['border'], height=1).pack(fill=tk.X, padx=16)

        self._log = LogPanel(log_card)
        self._log.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Clear log button
        ModernButton(log_card, text="Clear Log", bg='#2a2a2a', fg=self.C['sub'],
                     font=('Courier New', 8), padx=10, pady=5,
                     command=lambda: (self._log.text.config(state=tk.NORMAL),
                                      self._log.text.delete('1.0', tk.END),
                                      self._log.text.config(state=tk.DISABLED))
                     ).pack(anchor='e', padx=14, pady=8)

    # ── Footer ─────────────────────────────────────────────────────────────────
    def _build_footer(self):
        foot = tk.Frame(self.root, bg=self.C['bg2'], height=28)
        foot.pack(fill=tk.X, side=tk.BOTTOM)
        foot.pack_propagate(False)
        tk.Label(foot, text="YOLOv10 · ultralytics · Python",
                 bg=self.C['bg2'], fg=self.C['sub'],
                 font=('Courier New', 8)).pack(side=tk.LEFT, padx=14)
        self._frame_lbl = tk.Label(foot, text="Frames: 0",
                                   bg=self.C['bg2'], fg=self.C['sub'],
                                   font=('Courier New', 8))
        self._frame_lbl.pack(side=tk.RIGHT, padx=14)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _section(self, parent, text):
        tk.Label(parent, text=text, bg=self.C['card'], fg=self.C['sub'],
                 font=('Courier New', 8, 'bold')).pack(anchor='w', padx=16, pady=(10, 3))

    def _style_combo(self, combo):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox',
                         fieldbackground=self.C['bg2'],
                         background=self.C['bg2'],
                         foreground=self.C['text'],
                         selectbackground=self.C['border'],
                         selectforeground=self.C['text'])

    def log(self, msg, tag='info'):
        self._log.log(msg, tag)

    def _set_status(self, text, color):
        self._dot.config(fg=color)
        self._status_lbl.config(text=f" {text}")

    # ── Actions ────────────────────────────────────────────────────────────────
    def _start(self):
        if not YOLO_AVAILABLE:
            messagebox.showerror("Missing Dependency",
                                 "ultralytics not installed.\nRun: pip install ultralytics")
            return

        model_name = self._model_var.get()
        cam_idx    = int(self._cam_var.get())

        self._set_status("Loading model…", self.C['yellow'])
        self.log(f"Loading {model_name} …", 'warn')
        self.root.update()

        try:
            self.model = YOLO(f"models/{model_name}")  # auto-downloads if not cached
        except Exception as ex:
            self.log(f"Model load failed: {ex}", 'err')
            self._set_status("Error", self.C['red'])
            return

        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            self.log(f"Cannot open camera {cam_idx}", 'err')
            self._set_status("No Camera", self.C['red'])
            return

        self.log(f"Model loaded ✓  |  Camera {cam_idx} opened ✓", 'info')
        self.running = True
        self._frame_count = 0
        self._frame_times.clear()

        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._set_status("Detecting", self.C['green'])

        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()

    def _stop(self):
        self.running = False
        self._btn_start.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)
        self._set_status("Idle", self.C['red'])
        self.log("Detection stopped.", 'warn')
        if self.cap:
            self.cap.release()
            self.cap = None

    def _snapshot(self):
        """Save the current canvas frame as a PNG."""
        if not hasattr(self, '_last_frame') or self._last_frame is None:
            self.log("No frame to snapshot.", 'warn')
            return
        fname = f"snapshot_{int(time.time())}.png"
        cv2.imwrite(fname, self._last_frame)
        self.log(f"Snapshot saved → {fname}", 'info')

    # ── Detection loop (runs in thread) ────────────────────────────────────────
    def _detect_loop(self):
        self._last_frame = None
        frame_count = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.log("Frame read failed — stopping.", 'err')
                break

            t0 = time.time()
            conf_thresh = self._conf_var.get()

            # ── YOLOv10 inference ──────────────────────────────────────────────
            # YOLOv10 uses the same .predict() API as YOLOv8 via ultralytics
            results = self.model.predict(
                source=frame,
                conf=conf_thresh,
                verbose=False,
                imgsz=640,
            )
            result = results[0]

            # ── Parse detections ───────────────────────────────────────────────
            boxes      = result.boxes
            num_obj    = len(boxes)
            class_ids  = boxes.cls.tolist()  if num_obj else []
            confs      = boxes.conf.tolist() if num_obj else []
            names      = result.names         # dict: id → class name

            unique_cls = len(set(class_ids))
            avg_conf   = (sum(confs) / len(confs) * 100) if confs else 0.0

            # ── Draw on frame ──────────────────────────────────────────────────
            annotated = self._draw(frame, boxes, names, confs)

            # ── FPS ────────────────────────────────────────────────────────────
            elapsed = time.time() - t0
            self._frame_times.append(elapsed)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            fps = 1.0 / (sum(self._frame_times) / len(self._frame_times))

            self._last_frame = annotated
            frame_count += 1

            # Log detections every 30 frames
            if num_obj > 0 and frame_count % 30 == 0:
                det_str = ", ".join(
                    f"{names[int(c)]}({cf:.2f})" for c, cf in zip(class_ids, confs)
                )
                self.root.after(0, self.log, f"Detected: {det_str}", 'det')

            # ── Update UI on main thread ────────────────────────────────────────
            self.root.after(0, self._update_ui, annotated, fps, num_obj, unique_cls, avg_conf, frame_count)

        self.root.after(0, self._clear_canvas)

    def _draw(self, frame, boxes, names, confs):
        """Draw bounding boxes manually (gives full control over style)."""
        img = frame.copy()
        show_labels = self._show_labels.get()
        show_conf   = self._show_conf.get()

        # Color palette (BGR)
        palette = [
            (0, 255, 136), (68, 136, 255), (255, 204, 0),
            (255, 68,  68), (136, 68, 255), (0, 200, 255),
            (255, 136, 0),  (0, 255, 0),    (200, 0, 255),
        ]

        for box, cls_id, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), confs):
            x1, y1, x2, y2 = map(int, box)
            cid   = int(cls_id)
            color = palette[cid % len(palette)]
            label = names.get(cid, str(cid))

            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                text = label
                if show_conf:
                    text += f"  {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                # Label background
                cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
                cv2.putText(img, text, (x1 + 4, y1 - 5),
                            cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        return img

    def _update_ui(self, frame, fps, num_obj, unique_cls, avg_conf, frame_count):
        """Runs on main thread — update canvas + stat cards."""
        # Resize to fit canvas
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (nw, nh))
        pil = Image.fromarray(img)
        self._photo = ImageTk.PhotoImage(pil)

        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)

        # Stat cards
        self._fps_card.set(f"{fps:.1f}")
        self._obj_card.set(str(num_obj))
        self._cls_card.set(str(unique_cls))
        self._conf_card.set(f"{avg_conf:.0f}%")
        self._frame_lbl.config(text=f"Frames: {frame_count}")

    def _clear_canvas(self):
        self.canvas.delete('all')
        self.canvas.create_text(
            self.canvas.winfo_width() // 2 or 400,
            self.canvas.winfo_height() // 2 or 270,
            text="Camera stopped",
            fill='#333', font=('Courier New', 12), justify=tk.CENTER
        )

    # ── Cleanup ────────────────────────────────────────────────────────────────
    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    root = tk.Tk()
    app  = YOLOv10App(root)
    root.mainloop()