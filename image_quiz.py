"""Image Name Quiz.

Drop images into the local `img/` folder, then run this script.

Notes:
- Settings and scores are stored next to this script (JSON files).
"""

import json
import os
import random
import textwrap
import tkinter as tk
import traceback

import ttkbootstrap as ttk
from PIL import Image, ImageTk
from ttkbootstrap.constants import CENTER
from ttkbootstrap.dialogs import Messagebox

def _app_dir():
    return os.path.dirname(__file__)

IMAGE_DIR = os.path.join(_app_dir(), "img")
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}

SETTINGS_PATH = os.path.join(_app_dir(), "image_quiz_settings.json")
SCORES_PATH = os.path.join(_app_dir(), "image_quiz_scores.json")

class ImageQuizApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="flatly")
        self.title("Image Name Quiz")
        self._load_settings()

        self.report_callback_exception = self._report_callback_exception

        self.image_paths = self._load_images()
        if not self.image_paths:
            Messagebox.show_error("No images found", f"No images found in {IMAGE_DIR}")
            self.destroy()
            return

        self.scores = self._load_scores()
        self.recent_buffer_size = {"image_to_name": 6, "name_to_image": 6}
        self.recent_history = {"image_to_name": [], "name_to_image": []}
        self.mode3_sorted = []

        self.score = 0
        self.total = 0
        self.current_path = None
        self.current_photo = None
        self.mode = None
        self.current_choices = []
        self.history = {"image_to_name": [], "name_to_image": [], "reveal_grid": []}
        self.future = {"image_to_name": [], "name_to_image": [], "reveal_grid": []}
        self.used_correct = {"image_to_name": set(), "name_to_image": set(), "reveal_grid": set()}
        self.answer_count_var.trace_add("write", lambda *args: self._save_settings())
        self.min_question_height = 240
        self._reflow_job = None
        self._is_reflowing = False
        self._last_reflow_size = (0, 0)
        self._last_answer_count = {"image_to_name": None, "name_to_image": None, "reveal_grid": None}
        self.used_mode3 = set()
        self.mode3_state = None
        self._tooltip_window = None
        self._tooltip_photo = None
        self._answer_photo = None
        self._awaiting_continue = False
        self.current_choice_names = []
        self.current_choice_paths = []
        self.mode3_select_var = tk.StringVar(value="")
        self.app_style = ttk.Style()
        self.app_style.configure("outline-primary.TButton", font=("Segoe UI", 10))
        self.app_style.configure("secondary.TButton", font=("Segoe UI", 10))
        self.app_style.configure("danger.TButton", font=("Segoe UI", 10))
        self.app_style.configure("success.TButton", font=("Segoe UI", 10, "bold"))

        self._build_ui()
        self.bind("<Configure>", self._on_resize)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show_menu()

    def _report_callback_exception(self, exc, val, tb):
        traceback.print_exception(exc, val, tb)

    def _load_images(self):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        paths = []
        for name in os.listdir(IMAGE_DIR):
            path = os.path.join(IMAGE_DIR, name)
            if os.path.isfile(path):
                _, ext = os.path.splitext(name)
                if ext.lower() in SUPPORTED_EXTS:
                    paths.append(path)
        paths.sort()
        return paths

    def _build_ui(self):
        self.container = ttk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Top bar (shared)
        self.top_frame = ttk.Frame(self.container, padding=10)
        self.top_frame.pack(fill=tk.X)

        self.score_var = tk.StringVar(value="Score: 0/0")
        ttk.Label(self.top_frame, textvariable=self.score_var, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        self.mode3_selector = ttk.Combobox(
            self.top_frame,
            textvariable=self.mode3_select_var,
            values=[],
            width=40,
            state="readonly",
        )
        self.mode3_selector.pack(side=tk.RIGHT, padx=(0, 10))
        self.mode3_selector.bind("<<ComboboxSelected>>", self._on_mode3_select)
        self.mode3_selector.pack_forget()
        ttk.Button(self.top_frame, text="Back", command=self._go_back, bootstyle="secondary").pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(self.top_frame, text="Next", command=self._go_forward, bootstyle="secondary").pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(self.top_frame, text="Menu", command=self._show_menu, bootstyle="secondary").pack(side=tk.RIGHT)

        self.alert_var = tk.StringVar(value="")
        self.alert_label = ttk.Label(
            self.container,
            textvariable=self.alert_var,
            anchor=CENTER,
            font=("Segoe UI", 11, "bold"),
        )
        self.alert_label.pack(fill=tk.X, padx=10, pady=(0, 6))


        # Menu screen
        self.menu_frame = ttk.Frame(self.container, padding=30)
        ttk.Label(self.menu_frame, text="Choose a mode", font=("Segoe UI", 18, "bold")).pack(pady=10)
        options_frame = ttk.Frame(self.menu_frame)
        options_frame.pack(pady=10, fill=tk.X)
        ttk.Label(options_frame, text="Answer choices:").pack(side=tk.LEFT)
        ttk.Combobox(
            options_frame,
            textvariable=self.answer_count_var,
            values=["4", "6", "8", "10", "All"],
            width=8,
            state="readonly",
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            self.menu_frame,
            text="Mode 1: Image → Name",
            command=lambda: self._start_mode("image_to_name"),
            bootstyle="primary",
        )
        ttk.Button(
            self.menu_frame,
            text="Mode 2: Name → Image",
            command=lambda: self._start_mode("name_to_image"),
            bootstyle="primary",
        )
        ttk.Button(
            self.menu_frame,
            text="Mode 3: Reveal Grid",
            command=lambda: self._start_mode("reveal_grid"),
            bootstyle="primary",
        )
        ttk.Button(
            self.menu_frame,
            text="Scoreboard",
            command=self._show_scoreboard,
            bootstyle="info",
        )
        for child in self.menu_frame.winfo_children():
            if isinstance(child, ttk.Button):
                child.pack(pady=8, fill=tk.X)

        # Mode 1 screen
        self.mode1_frame = ttk.Frame(self.container, padding=10)
        self.mode1_image_container = ttk.Frame(self.mode1_frame)
        self.mode1_image_container.pack(padx=10, pady=10, expand=True)
        self.mode1_image_label = ttk.Label(self.mode1_image_container)
        self.mode1_image_label.pack(side=tk.LEFT, padx=(0, 8))
        self.mode1_magnifier = ttk.Label(self.mode1_image_container, text="🔍", bootstyle="secondary")
        self.mode1_magnifier.pack(side=tk.LEFT)
        self.mode1_magnifier.bind("<Enter>", lambda e: self._show_image_tooltip(self.current_path))
        self.mode1_magnifier.bind("<Leave>", lambda e: self._hide_image_tooltip())
        self.mode1_buttons_container, self.mode1_buttons_canvas, self.mode1_buttons_frame, _ = self._create_scrollable_area(
            self.mode1_frame,
            height=260,
        )
        self.mode1_buttons = []

        # Mode 2 screen
        self.mode2_frame = ttk.Frame(self.container, padding=10)
        self.mode2_label = ttk.Label(self.mode2_frame, font=("Segoe UI", 14, "bold"))
        self.mode2_label.pack(pady=10)
        self.mode2_buttons_container, self.mode2_buttons_canvas, self.mode2_buttons_frame, _ = self._create_scrollable_area(
            self.mode2_frame,
            height=360,
        )
        self.mode2_items = []
        self.mode2_photos = []

        # Mode 3 screen
        self.mode3_frame = ttk.Frame(self.container, padding=10)
        self.mode3_label = ttk.Label(self.mode3_frame, font=("Segoe UI", 14, "bold"))
        self.mode3_label.pack(pady=10)
        self.mode3_canvas_container = ttk.Frame(self.mode3_frame)
        self.mode3_canvas_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.mode3_canvas = tk.Canvas(self.mode3_canvas_container, highlightthickness=0)
        self.mode3_scroll_y = ttk.Scrollbar(self.mode3_canvas_container, orient="vertical", command=self.mode3_canvas.yview)
        self.mode3_scroll_x = ttk.Scrollbar(self.mode3_canvas_container, orient="horizontal", command=self.mode3_canvas.xview)
        self.mode3_canvas.configure(xscrollcommand=self.mode3_scroll_x.set, yscrollcommand=self.mode3_scroll_y.set)

        self.mode3_canvas.grid(row=0, column=0, sticky="nsew")
        self.mode3_scroll_y.grid(row=0, column=1, sticky="ns")
        self.mode3_scroll_x.grid(row=1, column=0, sticky="ew")
        self.mode3_canvas_container.grid_rowconfigure(0, weight=1)
        self.mode3_canvas_container.grid_columnconfigure(0, weight=1)

        # Scoreboard screen
        self.scoreboard_frame = ttk.Frame(self.container, padding=20)
        self.scoreboard_title = ttk.Label(self.scoreboard_frame, text="Scoreboard", font=("Segoe UI", 16, "bold"))
        self.scoreboard_title.pack(pady=(0, 10))
        self.scoreboard_text = tk.Text(self.scoreboard_frame, height=20, wrap="none")
        self.scoreboard_text.pack(fill=tk.BOTH, expand=True)

    def _show_menu(self):
        self._clear_frames()
        self._clear_alert()
        self._hide_mode3_selector()
        self.menu_frame.pack(fill=tk.BOTH, expand=True)

    def _show_scoreboard(self):
        self._clear_frames()
        self._clear_alert()
        self._hide_mode3_selector()
        self._render_scoreboard()
        self.scoreboard_frame.pack(fill=tk.BOTH, expand=True)

    def _start_mode(self, mode):
        self.mode = mode
        self._clear_frames()
        self._clear_alert()
        self._clear_answer()
        current_count = self.answer_count_var.get()
        if self._last_answer_count[mode] != current_count:
            self.history[mode].clear()
            self.future[mode].clear()
            self.used_correct[mode].clear()
            self._last_answer_count[mode] = current_count
        if mode == "image_to_name":
            self.recent_buffer_size[mode] = random.randint(5, 7)
            self._hide_mode3_selector()
            self.mode1_frame.pack(fill=tk.BOTH, expand=True)
            if self.history[mode]:
                self._render_image_to_name(self.history[mode][-1])
            else:
                self._new_question()
        elif mode == "name_to_image":
            self.recent_buffer_size[mode] = random.randint(5, 7)
            self._hide_mode3_selector()
            self.mode2_frame.pack(fill=tk.BOTH, expand=True)
            if self.history[mode]:
                self._render_name_to_image(self.history[mode][-1])
            else:
                self._new_question()
        else:
            self._show_mode3_selector()
            self.mode3_sorted = self._sorted_images_by_score()
            self.mode3_frame.pack(fill=tk.BOTH, expand=True)
            self._new_question()

    def _clear_frames(self):
        for frame in (self.menu_frame, self.mode1_frame, self.mode2_frame, self.mode3_frame, self.scoreboard_frame):
            frame.pack_forget()

    def _new_question(self):
        self._awaiting_continue = False
        self._clear_answer()
        self._set_buttons_state(True)
        self._reset_button_styles()
        if self.mode == "image_to_name":
            path = self._pick_next_image(self.mode)
            if not path:
                self._show_alert("No more images left in this mode. Return to menu to restart.", "warning")
                return
            correct_name = os.path.basename(path)
            choices = self._pick_name_choices(correct_name)
            question = {"path": path, "choices": choices}
            self.history[self.mode].append(question)
            self.future[self.mode].clear()
            self._render_image_to_name(question)
            self.recent_history[self.mode].append(path)
            self.recent_history[self.mode] = self.recent_history[self.mode][-
                self.recent_buffer_size[self.mode] :
            ]
        elif self.mode == "name_to_image":
            path = self._pick_next_image(self.mode)
            if not path:
                self._show_alert("No more images left in this mode. Return to menu to restart.", "warning")
                return
            choices = self._pick_image_choices(path)
            question = {"path": path, "choices": choices}
            self.history[self.mode].append(question)
            self.future[self.mode].clear()
            self._render_name_to_image(question)
            self.recent_history[self.mode].append(path)
            self.recent_history[self.mode] = self.recent_history[self.mode][-
                self.recent_buffer_size[self.mode] :
            ]
        elif self.mode == "reveal_grid":
            if not self.mode3_sorted:
                self.mode3_sorted = self._sorted_images_by_score()
            path = next((p for p in self.mode3_sorted if p not in self.used_correct[self.mode]), None)
            if not path:
                self._show_alert("No more images left in this mode. Return to menu to restart.", "warning")
                return
            question = {"path": path}
            self.history[self.mode].append(question)
            self.future[self.mode].clear()
            self._render_reveal_grid(question)

    def _render_image_to_name(self, question):
        self.current_path = question["path"]
        self._display_image(self.current_path)
        choices = question["choices"]
        self.current_choice_names = choices
        display_choices = [(self._display_name(name), name) for name in choices]

        self._ensure_mode1_buttons(len(display_choices))
        self.update_idletasks()
        available_w = max(600, self.mode1_buttons_canvas.winfo_width() or (self.winfo_width() - 60))
        col_count = max(2, min(4, available_w // 220))
        wrap_chars = 24
        wrapped = [textwrap.fill(d, width=wrap_chars) for d, _ in display_choices]

        for i, (btn, (display, full_name)) in enumerate(zip(self.mode1_buttons, display_choices)):
            btn.configure(text=wrapped[i], command=lambda n=full_name: self._check_name_answer(n))
            btn.configure(width=wrap_chars, padding=(10, 8))
            btn.grid(row=i // col_count, column=i % col_count, padx=10, pady=10, sticky="nsew")

        rows = (len(display_choices) + col_count - 1) // col_count
        for r in range(rows):
            self.mode1_buttons_frame.grid_rowconfigure(r, weight=1)
        for c in range(col_count):
            self.mode1_buttons_frame.grid_columnconfigure(c, weight=1)

    def _render_name_to_image(self, question):
        self.current_path = question["path"]
        correct_name = os.path.basename(self.current_path)
        self.mode2_label.configure(text=f"Pick the image for: {self._display_name(correct_name)}")
        choices = question["choices"]
        self.current_choice_paths = choices
        self.mode2_photos = []
        self.update_idletasks()
        self._ensure_mode2_buttons(len(choices))
        available_w = max(600, self.mode2_buttons_canvas.winfo_width() or (self.winfo_width() - 60))
        available_h = max(400, self.mode2_buttons_canvas.winfo_height() or (self.winfo_height() - 200))
        col_count = max(2, min(4, available_w // 260))
        max_w = max(200, available_w // col_count - 24)
        rows = (len(choices) + col_count - 1) // col_count
        max_h = max(180, available_h // max(1, rows) - 24)
        for i, (item, path) in enumerate(zip(self.mode2_items, choices)):
            btn = item["button"]
            magnifier = item["magnifier"]
            try:
                with Image.open(path) as img:
                    img.thumbnail((max_w, max_h))
                    photo = ImageTk.PhotoImage(img.copy())
            except Exception:
                continue
            self.mode2_photos.append(photo)
            btn.configure(image=photo, command=lambda p=path: self._check_image_answer(p))
            magnifier.unbind("<Enter>")
            magnifier.unbind("<Leave>")
            magnifier.bind("<Enter>", lambda e, p=path: self._show_image_tooltip(p))
            magnifier.bind("<Leave>", lambda e: self._hide_image_tooltip())
            item["frame"].grid(row=i // col_count, column=i % col_count, padx=10, pady=10, sticky="nsew")
        for r in range(rows):
            self.mode2_buttons_frame.grid_rowconfigure(r, weight=1)
        for c in range(col_count):
            self.mode2_buttons_frame.grid_columnconfigure(c, weight=1)

    def _display_image(self, path):
        try:
            with Image.open(path) as img:
                img = img.copy()
        except Exception:
            return
        self.update_idletasks()
        window_w = max(700, self.winfo_width() - 40)
        if self.mode == "image_to_name":
            buttons_h = self.mode1_buttons_frame.winfo_reqheight() or 220
            top_h = self.top_frame.winfo_reqheight() or 50
            available_h = max(self.min_question_height, self.winfo_height() - buttons_h - top_h - 80)
        else:
            available_h = 520
        max_w, max_h = window_w, available_h
        img.thumbnail((max_w, max_h))
        self.current_photo = ImageTk.PhotoImage(img)
        self.mode1_image_label.configure(image=self.current_photo)

    def _check_name_answer(self, guess):
        if self._awaiting_continue:
            return
        correct_name = os.path.basename(self.current_path)
        is_correct = guess == correct_name
        if is_correct:
            self.used_correct[self.mode].add(self.current_path)
        self._update_image_score(self.current_path, 1 if is_correct else -1)
        self._update_score(is_correct, correct_name)
        self._highlight_name_choices(guess, correct_name)
        self._awaiting_continue = True
        self._set_buttons_state(False)

    def _check_image_answer(self, guess_path):
        if self._awaiting_continue:
            return
        correct_name = os.path.basename(self.current_path)
        is_correct = guess_path == self.current_path
        if is_correct:
            self.used_correct[self.mode].add(self.current_path)
        self._update_image_score(self.current_path, 1 if is_correct else -1)
        self._update_score(is_correct, correct_name)
        self._highlight_image_choices(guess_path, self.current_path)
        self._awaiting_continue = True
        self._set_buttons_state(False)

    def _update_score(self, is_correct, correct_name):
        self.total += 1
        if is_correct:
            self.score += 1
            self._show_alert("Correct!", "success")
        else:
            self._show_alert(f"Incorrect. Correct answer: {correct_name}", "danger")
        self.score_var.set(f"Score: {self.score}/{self.total}")

    def _pick_name_choices(self, correct_name):
        all_names = [os.path.basename(p) for p in self.image_paths]
        count = self._get_answer_count()
        if count is None:
            choices = list(all_names)
        else:
            count = min(count, len(all_names))
            distractors = self._similar_names(correct_name, all_names)
            needed = max(0, count - 1 - len(distractors))
            remaining = [n for n in all_names if n not in distractors and n != correct_name]
            if needed > 0 and remaining:
                distractors.extend(random.sample(remaining, k=min(needed, len(remaining))))
            choices = [correct_name] + distractors[: max(0, count - 1)]
        random.shuffle(choices)
        return choices

    def _pick_image_choices(self, correct_path):
        count = self._get_answer_count()
        if count is None:
            choices = list(self.image_paths)
        else:
            count = min(count, len(self.image_paths))
            choices = [correct_path]
            others = [p for p in self.image_paths if p != correct_path]
            if count > 1:
                choices.extend(random.sample(others, k=min(count - 1, len(others))))
        random.shuffle(choices)
        return choices

    def _similar_names(self, target, all_names):
        target_tokens = set(self._name_tokens(target))
        scored = []
        for name in all_names:
            if name == target:
                continue
            tokens = set(self._name_tokens(name))
            score = len(target_tokens.intersection(tokens))
            scored.append((score, name))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [name for score, name in scored if score > 0][:3]
        if len(top) < 3:
            remaining = [name for _, name in scored if name not in top]
            top.extend(random.sample(remaining, k=min(3 - len(top), len(remaining))))
        return top

    def _name_tokens(self, name):
        base = os.path.splitext(name)[0]
        parts = [p.strip().lower() for p in base.replace("-", " ").replace("_", " ").split()]
        return [p for p in parts if p]

    def _display_name(self, name):
        return os.path.splitext(name)[0]

    def _clear_answer(self):
        self._answer_photo = None

    def _set_buttons_state(self, enabled):
        state = "normal" if enabled else "disabled"
        for btn in self.mode1_buttons:
            btn.configure(state=state)
        for item in self.mode2_items:
            item["button"].configure(state=state)

    def _reset_button_styles(self):
        for btn in self.mode1_buttons:
            btn.configure(bootstyle="outline-primary")
        for item in self.mode2_items:
            item["button"].configure(bootstyle="outline-primary")

    def _highlight_name_choices(self, guess, correct_name):
        for name, btn in zip(self.current_choice_names, self.mode1_buttons):
            if name == correct_name:
                btn.configure(bootstyle="success")
            elif name == guess:
                btn.configure(bootstyle="danger")
            else:
                btn.configure(bootstyle="secondary")

    def _highlight_image_choices(self, guess_path, correct_path):
        for path, item in zip(self.current_choice_paths, self.mode2_items):
            btn = item["button"]
            if path == correct_path:
                btn.configure(bootstyle="success")
            elif path == guess_path:
                btn.configure(bootstyle="danger")
            else:
                btn.configure(bootstyle="secondary")

    def _render_reveal_grid(self, question):
        path = question["path"]
        self.used_correct[self.mode].add(path)
        title = self._display_name(os.path.basename(path))
        self.mode3_label.configure(text=title)
        self._set_mode3_selector_value(path)

        self.mode3_canvas.delete("all")
        try:
            with Image.open(path) as img:
                img = img.copy()
        except Exception:
            return

        self.update_idletasks()
        avail_w = max(400, self.mode3_canvas.winfo_width())
        avail_h = max(300, self.mode3_canvas.winfo_height())
        scale = min(avail_w / img.width, avail_h / img.height, 1.0)
        display_w = int(img.width * scale)
        display_h = int(img.height * scale)
        if scale != 1.0:
            img = img.resize((display_w, display_h), Image.LANCZOS)

        self.mode3_state = {
            "path": path,
            "width": display_w,
            "height": display_h,
            "rects": [],
        }

        photo = ImageTk.PhotoImage(img)
        self.mode3_state["photo"] = photo
        self.mode3_canvas.create_image(0, 0, anchor="nw", image=photo)
        self.mode3_canvas.configure(scrollregion=(0, 0, display_w, display_h))

        count = self._get_answer_count()
        if count is None:
            count = 12
        cols = max(2, int(count ** 0.5))
        rows = max(2, (count + cols - 1) // cols)

        cell_w = display_w / cols
        cell_h = display_h / rows
        for r in range(rows):
            for c in range(cols):
                x1 = c * cell_w
                y1 = r * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                rect = self.mode3_canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill="#6c757d",
                    outline="#f8f9fa",
                    width=1,
                )
                self.mode3_state["rects"].append(rect)

        self.mode3_canvas.bind("<Button-1>", self._on_mode3_click)

    def _on_mode3_click(self, event):
        if not self.mode3_state:
            return
        items = self.mode3_canvas.find_overlapping(event.x, event.y, event.x, event.y)
        for item in items:
            if item in self.mode3_state["rects"]:
                self.mode3_canvas.delete(item)
                self.mode3_state["rects"].remove(item)
                break

    def _show_mode3_selector(self):
        self.mode3_selector.pack(side=tk.RIGHT, padx=(0, 10))
        self._update_mode3_selector_values()

    def _hide_mode3_selector(self):
        self.mode3_selector.pack_forget()

    def _update_mode3_selector_values(self):
        values = [os.path.basename(p) for p in self._sorted_images_by_score()]
        self.mode3_selector.configure(values=values)

    def _set_mode3_selector_value(self, path):
        if not path:
            return
        self.mode3_select_var.set(os.path.basename(path))

    def _on_mode3_select(self, event):
        name = self.mode3_select_var.get()
        if not name:
            return
        path = os.path.join(IMAGE_DIR, name)
        if not os.path.exists(path):
            return
        question = {"path": path}
        self.history[self.mode].append(question)
        self.future[self.mode].clear()
        self._render_reveal_grid(question)

    def _on_resize(self, event):
        if event.widget is not self:
            return
        size = (event.width, event.height)
        if size == self._last_reflow_size:
            return
        self._last_reflow_size = size
        if self._reflow_job is not None:
            try:
                self.after_cancel(self._reflow_job)
            except Exception:
                pass
        self._reflow_job = self.after(200, self._reflow_current)
        self._schedule_save_settings()

    def _reflow_current(self):
        if self._is_reflowing:
            return
        if not self.mode or not self.history[self.mode]:
            return
        try:
            self._is_reflowing = True
            if self.mode == "image_to_name":
                self._render_image_to_name(self.history[self.mode][-1])
            elif self.mode == "name_to_image":
                self._render_name_to_image(self.history[self.mode][-1])
            elif self.mode == "reveal_grid":
                self._render_reveal_grid(self.history[self.mode][-1])
        finally:
            self._is_reflowing = False

    def _get_answer_count(self):
        val = self.answer_count_var.get().strip().lower()
        if val == "all":
            return None
        try:
            return int(val)
        except ValueError:
            return 4

    def _sorted_images_by_score(self):
        paths = list(self.image_paths)
        paths.sort(key=lambda p: (self._score_for(p), os.path.basename(p).lower()))
        return paths

    def _pick_next_image(self, mode):
        remaining = [p for p in self.image_paths if p not in self.used_correct[mode]]
        if not remaining:
            return None
        recent = set(self.recent_history.get(mode, []))
        candidates = [p for p in remaining if p not in recent]
        if not candidates:
            candidates = remaining
        min_score = min(self._score_for(p) for p in candidates)
        lowest = [p for p in candidates if self._score_for(p) == min_score]
        return random.choice(lowest)

    def _ensure_mode1_buttons(self, count):
        while len(self.mode1_buttons) < count:
            btn = ttk.Button(self.mode1_buttons_frame, command=lambda: None, bootstyle="outline-primary")
            self.mode1_buttons.append(btn)
        for btn in self.mode1_buttons:
            btn.grid_forget()

    def _ensure_mode2_buttons(self, count):
        while len(self.mode2_items) < count:
            frame = ttk.Frame(self.mode2_buttons_frame)
            btn = ttk.Button(frame, command=lambda: None, bootstyle="outline-primary")
            btn.pack(side=tk.TOP, padx=4, pady=(0, 4))
            magnifier = ttk.Label(frame, text="🔍", bootstyle="secondary")
            magnifier.pack(side=tk.TOP)
            self.mode2_items.append({"frame": frame, "button": btn, "magnifier": magnifier})
        for item in self.mode2_items:
            item["frame"].grid_forget()

    def _create_scrollable_area(self, parent, height):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=False, pady=6)
        container.configure(height=height)
        container.pack_propagate(False)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(inner_id, width=event.width)

        inner.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        return container, canvas, inner, scrollbar

    def _load_settings(self):
        width, height = 900, 700
        answer_count = "4"
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                width = int(data.get("width", width))
                height = int(data.get("height", height))
                answer_count = str(data.get("answer_count", answer_count))
        except Exception:
            pass
        self.geometry(f"{width}x{height}")
        self.answer_count_var = tk.StringVar(value=answer_count)

    def _load_scores(self):
        scores = {}
        try:
            with open(SCORES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    scores = {k: int(v) for k, v in data.items() if isinstance(k, str)}
        except Exception:
            pass

        # Remove entries for missing images
        valid_names = {os.path.basename(p) for p in self.image_paths}
        scores = {k: v for k, v in scores.items() if k in valid_names}
        return scores

    def _save_scores(self):
        try:
            with open(SCORES_PATH, "w", encoding="utf-8") as f:
                json.dump(self.scores, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _render_scoreboard(self):
        rows = []
        for path in self._sorted_images_by_score():
            name = os.path.basename(path)
            rows.append((name, self.scores.get(name, 0)))
        rows.sort(key=lambda x: (-x[1], x[0].lower()))

        self.scoreboard_text.configure(state="normal")
        self.scoreboard_text.delete("1.0", tk.END)
        for name, score in rows:
            self.scoreboard_text.insert(tk.END, f"{score:>4}  {self._display_name(name)}\n")
        self.scoreboard_text.configure(state="disabled")

    def _score_for(self, path):
        return self.scores.get(os.path.basename(path), 0)

    def _update_image_score(self, path, delta):
        key = os.path.basename(path)
        self.scores[key] = self.scores.get(key, 0) + delta
        self._save_scores()

    def _schedule_save_settings(self):
        if hasattr(self, "_save_settings_job") and self._save_settings_job is not None:
            try:
                self.after_cancel(self._save_settings_job)
            except Exception:
                pass
        self._save_settings_job = self.after(300, self._save_settings)

    def _save_settings(self):
        try:
            width = self.winfo_width()
            height = self.winfo_height()
            data = {
                "width": width,
                "height": height,
                "answer_count": self.answer_count_var.get(),
            }
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _on_close(self):
        self._save_settings()
        self.destroy()

    def _show_image_tooltip(self, path):
        if not path:
            return
        self._hide_image_tooltip()
        try:
            with Image.open(path) as img:
                img = img.copy()
        except Exception:
            return

        self._tooltip_window = tk.Toplevel(self)
        self._tooltip_window.wm_overrideredirect(True)
        self._tooltip_window.attributes("-topmost", True)

        x = self.winfo_pointerx() + 12
        y = self.winfo_pointery() + 12

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        max_w = max(200, screen_w - 80)
        max_h = max(200, screen_h - 120)
        img.thumbnail((max_w, max_h))

        self._tooltip_photo = ImageTk.PhotoImage(img)
        label = ttk.Label(self._tooltip_window, image=self._tooltip_photo)
        label.pack()

        width = img.width
        height = img.height
        x = min(x, screen_w - width)
        y = min(y, screen_h - height)
        self._tooltip_window.geometry(f"{width}x{height}+{x}+{y}")

    def _hide_image_tooltip(self):
        if self._tooltip_window is not None:
            try:
                self._tooltip_window.destroy()
            except Exception:
                pass
        self._tooltip_window = None
        self._tooltip_photo = None

    def _go_back(self):
        if not self.mode:
            return
        if len(self.history[self.mode]) <= 1:
            return
        current = self.history[self.mode].pop()
        self.future[self.mode].append(current)
        prev = self.history[self.mode][-1]
        if self.mode == "image_to_name":
            self._render_image_to_name(prev)
        elif self.mode == "name_to_image":
            self._render_name_to_image(prev)
        else:
            self._render_reveal_grid(prev)

    def _go_forward(self):
        if not self.mode:
            return
        if self._awaiting_continue:
            self._new_question()
            return
        if self.mode == "reveal_grid":
            self._new_question()
            return
        if not self.future[self.mode]:
            return
        next_q = self.future[self.mode].pop()
        self.history[self.mode].append(next_q)
        if self.mode == "image_to_name":
            self._render_image_to_name(next_q)
        else:
            self._render_name_to_image(next_q)

    def _show_alert(self, message, style):
        self.alert_var.set(message)
        self.alert_label.configure(bootstyle=style)

    def _clear_alert(self):
        self.alert_var.set("")
        self.alert_label.configure(bootstyle="secondary")

if __name__ == "__main__":
    app = ImageQuizApp()
    app.mainloop()
