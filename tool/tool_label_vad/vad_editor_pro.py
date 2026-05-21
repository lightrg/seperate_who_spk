import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import os
import json
import threading
import tempfile
import whisper

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector

class VADEditorPro:
    def __init__(self, root):
        self.root = root
        self.root.title("VAD & ASR Annotator Pro (Merge, Auto-Save, Smart Export & Auto ASR)")
        self.root.geometry("1220x900")

        self.audio_data = None
        self.sample_rate = None
        self.max_duration = 0

        self.vad_spans = []
        self.current_segments = []

        self.current_txt_path = None

        self.is_playing = False
        self.playback_line = None
        self.playback_start_time = 0
        self.playback_start_sec = 0
        self.playback_end_sec = 0

        self.char_file = "vad_characters.json"
        self.char_list = self.load_characters()

        self.asr_model = None

        self.setup_ui()
        self.setup_hotkeys()

    def load_characters(self):
        if os.path.exists(self.char_file):
            try:
                with open(self.char_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def save_characters(self):
        try:
            with open(self.char_file, "w", encoding="utf-8") as f:
                json.dump(self.char_list, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Lỗi khi lưu danh sách nhân vật: {e}")

    def setup_ui(self):

        top_frame = tk.Frame(self.root, pady=5, padx=10)
        top_frame.pack(fill=tk.X)

        self.btn_load_audio = tk.Button(
            top_frame, text="🎵 1. Tải Audio (.wav)", command=self.load_audio,
            bg="#4CAF50", fg="white", font=("Arial", 10, "bold")
        )
        self.btn_load_audio.pack(side=tk.LEFT, padx=5)

        self.btn_load_txt = tk.Button(
            top_frame, text="📄 2. Tải Dataset (.txt)", command=self.load_txt,
            bg="#2196F3", fg="white", font=("Arial", 10, "bold")
        )
        self.btn_load_txt.pack(side=tk.LEFT, padx=5)

        self.btn_export = tk.Button(
            top_frame, text="✂️ 4. Xuất Dataset (Cắt Audio)", command=self.export_dataset,
            bg="#9C27B0", fg="white", font=("Arial", 10, "bold")
        )
        self.btn_export.pack(side=tk.RIGHT, padx=5)

        self.btn_save_txt = tk.Button(
            top_frame, text="💾 3. Lưu Dataset", command=self.save_txt,
            bg="#FF9800", fg="white", font=("Arial", 10, "bold")
        )
        self.btn_save_txt.pack(side=tk.RIGHT, padx=5)

        self.auto_save_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            top_frame, text="🔄 Tự động lưu", variable=self.auto_save_var,
            font=("Arial", 10, "bold"), fg="#FF9800"
        ).pack(side=tk.RIGHT, padx=15)

        tk.Button(
            top_frame, text="🔍 Tìm/Xóa (Ctrl+F)", command=self.open_find_replace,
            bg="#607D8B", fg="white", font=("Arial", 9, "bold")
        ).pack(side=tk.RIGHT, padx=5)

        self.lbl_status = tk.Label(
            top_frame, text="Vui lòng tải Audio và Text...", fg="gray",
            font=("Arial", 10, "italic")
        )
        self.lbl_status.pack(side=tk.LEFT, padx=20)

        wave_frame = tk.Frame(self.root, padx=10, pady=5)
        wave_frame.pack(fill=tk.X)

        wave_header = tk.Frame(wave_frame)
        wave_header.pack(fill=tk.X)

        tk.Label(
            wave_header,
            text="Biểu đồ Sóng âm (Lăn chuột: ZOOM | Phím ⬅️ ➡️: DI CHUYỂN | Bôi đỏ: CHỌN ĐOẠN):",
            font=("Arial", 10, "bold"), fg="#333"
        ).pack(side=tk.LEFT)

        self.show_vad_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            wave_header, text="👁 Hiển thị các đoạn VAD đã có",
            variable=self.show_vad_var, command=self.refresh_vad_overlays,
            font=("Arial", 10, "bold"), fg="#4CAF50"
        ).pack(side=tk.RIGHT)

        self.fig, self.ax = plt.subplots(figsize=(10, 2), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.set_facecolor('#ffffff')
        self.ax.set_yticks([])
        self.ax.set_xlabel("Thời gian (giây)")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=wave_frame)
        self.canvas.get_tk_widget().pack(fill=tk.X)

        toolbar_frame = tk.Frame(wave_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.span = SpanSelector(
            self.ax, self.on_select_waveform, 'horizontal', useblit=True,
            props=dict(alpha=0.5, facecolor="red"),
            onmove_callback=self.on_span_move
        )

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll_zoom)

        mid_frame = tk.Frame(self.root, padx=10, pady=5)
        mid_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("id", "start", "end", "text")
        self.tree = ttk.Treeview(mid_frame, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("id", text="STT")
        self.tree.heading("start", text="Start (s)")
        self.tree.heading("end", text="End (s)")
        self.tree.heading("text", text="Nội dung ASR")

        self.tree.column("id", width=40, anchor=tk.CENTER)
        self.tree.column("start", width=80, anchor=tk.CENTER)
        self.tree.column("end", width=80, anchor=tk.CENTER)
        self.tree.column("text", width=700, anchor=tk.W)

        scrollbar = ttk.Scrollbar(mid_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.bind("<<TreeviewSelect>>", self.on_select_row)

        bottom_frame = tk.LabelFrame(self.root, text="Tinh chỉnh / Thêm mới", padx=10, pady=10, font=("Arial", 10, "bold"))
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(bottom_frame, text="Bắt đầu:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.E, pady=5, padx=5)
        tk.Button(bottom_frame, text="-0.5s", command=lambda: self.tweak_time(self.entry_start, -0.5)).grid(row=0, column=1)
        tk.Button(bottom_frame, text="-0.1s", command=lambda: self.tweak_time(self.entry_start, -0.1)).grid(row=0, column=2)
        self.entry_start = tk.Entry(bottom_frame, width=10, font=("Arial", 12), justify="center")
        self.entry_start.grid(row=0, column=3, padx=5)
        tk.Button(bottom_frame, text="+0.1s", command=lambda: self.tweak_time(self.entry_start, 0.1)).grid(row=0, column=4)
        tk.Button(bottom_frame, text="+0.5s", command=lambda: self.tweak_time(self.entry_start, 0.5)).grid(row=0, column=5)

        tk.Label(bottom_frame, text="Kết thúc:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.E, pady=5, padx=5)
        tk.Button(bottom_frame, text="-0.5s", command=lambda: self.tweak_time(self.entry_end, -0.5)).grid(row=1, column=1)
        tk.Button(bottom_frame, text="-0.1s", command=lambda: self.tweak_time(self.entry_end, -0.1)).grid(row=1, column=2)
        self.entry_end = tk.Entry(bottom_frame, width=10, font=("Arial", 12), justify="center")
        self.entry_end.grid(row=1, column=3, padx=5)
        tk.Button(bottom_frame, text="+0.1s", command=lambda: self.tweak_time(self.entry_end, 0.1)).grid(row=1, column=4)
        tk.Button(bottom_frame, text="+0.5s", command=lambda: self.tweak_time(self.entry_end, 0.5)).grid(row=1, column=5)

        frame_label_asr = tk.Frame(bottom_frame)
        frame_label_asr.grid(row=2, column=0, sticky=tk.NE, pady=10, padx=5)

        tk.Label(frame_label_asr, text="Nội dung Text:", font=("Arial", 10, "bold")).pack(anchor=tk.E)

        self.btn_asr = tk.Button(
            frame_label_asr, text="🎙️ Auto ASR", command=self.run_asr_thread,
            bg="#FFC107", fg="black", font=("Arial", 9, "bold")
        )
        self.btn_asr.pack(anchor=tk.E, pady=(5, 0))

        tk.Label(frame_label_asr, text="Ngôn ngữ:", font=("Arial", 9)).pack(anchor=tk.E, pady=(6, 0))
        self.asr_language_var = tk.StringVar(value="auto")
        self.cmb_asr_language = ttk.Combobox(
            frame_label_asr,
            textvariable=self.asr_language_var,
            values=["auto", "vi", "en"],
            state="readonly",
            width=8
        )
        self.cmb_asr_language.pack(anchor=tk.E, pady=(2, 0))

        self.text_editor = tk.Text(bottom_frame, width=85, height=2, font=("Arial", 11))
        self.text_editor.grid(row=2, column=1, columnspan=6, pady=10)

        char_control_frame = tk.Frame(bottom_frame)
        char_control_frame.grid(row=3, column=1, columnspan=6, sticky=tk.W, pady=(0, 10))

        tk.Label(char_control_frame, text="Tên nhân vật:", font=("Arial", 10, "bold"), fg="#FF5722").pack(side=tk.LEFT, padx=(0, 5))

        self.char_buttons_frame = tk.Frame(char_control_frame)
        self.char_buttons_frame.pack(side=tk.LEFT)

        tk.Label(char_control_frame, text="|  Thêm mới:", font=("Arial", 9, "italic"), fg="gray").pack(side=tk.LEFT, padx=(10, 2))
        self.entry_new_char = tk.Entry(char_control_frame, width=12, font=("Arial", 10))
        self.entry_new_char.pack(side=tk.LEFT, padx=2)
        self.entry_new_char.bind('<Return>', lambda event: self.add_custom_char())

        tk.Button(
            char_control_frame, text="➕", command=self.add_custom_char,
            bg="#8BC34A", fg="white", font=("Arial", 8, "bold")
        ).pack(side=tk.LEFT)

        self.render_char_buttons()

        action_frame = tk.Frame(bottom_frame)
        action_frame.grid(row=4, column=1, columnspan=6, sticky=tk.W)

        tk.Button(action_frame, text="▶ NGHE ĐOẠN", command=self.play_segment, bg="#E91E63", fg="white", font=("Arial", 10, "bold"), width=12).pack(side=tk.LEFT, padx=3)
        tk.Button(action_frame, text="⏹ Dừng", command=self.stop_audio, width=6).pack(side=tk.LEFT, padx=3)

        self.play_to_end_var = tk.BooleanVar(value=False)
        tk.Checkbutton(action_frame, text="Phát tới cuối", variable=self.play_to_end_var, font=("Arial", 9, "bold"), fg="#2196F3").pack(side=tk.LEFT, padx=2)

        self.lbl_stop_time = tk.Label(action_frame, text="", fg="red", font=("Arial", 10, "bold"))
        self.lbl_stop_time.pack(side=tk.LEFT, padx=2)

        tk.Label(action_frame, text=" | ").pack(side=tk.LEFT)
        tk.Button(action_frame, text="➕ THÊM", command=self.add_new_row, bg="#9C27B0", fg="white", font=("Arial", 10, "bold"), width=8).pack(side=tk.LEFT, padx=3)
        tk.Button(action_frame, text="️ CẬP NHẬT", command=self.update_row, bg="#009688", fg="white", font=("Arial", 10, "bold"), width=10).pack(side=tk.LEFT, padx=3)
        tk.Button(action_frame, text="🔗 GỘP ĐOẠN", command=self.merge_segments, bg="#3F51B5", fg="white", font=("Arial", 10, "bold"), width=10).pack(side=tk.LEFT, padx=3)
        tk.Button(action_frame, text="🗑 XÓA", command=self.delete_row, bg="#F44336", fg="white", font=("Arial", 10, "bold"), width=8).pack(side=tk.LEFT, padx=3)

        vad_adjust_frame = tk.Frame(bottom_frame)
        vad_adjust_frame.grid(row=5, column=1, columnspan=6, sticky=tk.W, pady=(8, 0))

        tk.Label(vad_adjust_frame, text="Điều chỉnh toàn bộ VAD:", font=("Arial", 10, "bold"), fg="#795548").pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(vad_adjust_frame, text="Cắt đầu (s):", font=("Arial", 9)).pack(side=tk.LEFT)
        self.entry_trim_start_all = tk.Entry(vad_adjust_frame, width=6, font=("Arial", 10), justify="center")
        self.entry_trim_start_all.pack(side=tk.LEFT, padx=3)
        self.entry_trim_start_all.insert(0, "0.10")

        tk.Label(vad_adjust_frame, text="Thêm cuối (s):", font=("Arial", 9)).pack(side=tk.LEFT, padx=(8, 0))
        self.entry_extend_end_all = tk.Entry(vad_adjust_frame, width=6, font=("Arial", 10), justify="center")
        self.entry_extend_end_all.pack(side=tk.LEFT, padx=3)
        self.entry_extend_end_all.insert(0, "0.10")

        tk.Button(
            vad_adjust_frame,
            text="Áp dụng toàn bộ",
            command=self.apply_vad_adjust_all,
            bg="#795548", fg="white", font=("Arial", 9, "bold")
        ).pack(side=tk.LEFT, padx=8)

    def load_whisper_model(self):
        if self.asr_model is None:
            self.lbl_status.config(text="Đang tải model Whisper lần đầu (sẽ mất vài giây)...", fg="blue")
            self.root.update()
            try:

                self.asr_model = whisper.load_model("large")
                self.lbl_status.config(text="Đã tải xong model Whisper!", fg="green")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải Whisper model: {e}")
                self.lbl_status.config(text="Lỗi tải model ASR!", fg="red")
                return False
        return True

    def run_asr_thread(self):
        if self.audio_data is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải audio trước!")
            return

        try:
            start_sec = float(self.entry_start.get())
            end_sec = float(self.entry_end.get())
        except ValueError:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn đoạn âm thanh trên biểu đồ trước!")
            return

        if start_sec >= end_sec:
            messagebox.showwarning("Cảnh báo", "Thời gian không hợp lệ!")
            return

        self.btn_asr.config(text="⏳ Đang nghe...", state=tk.DISABLED)

        language = self.asr_language_var.get().strip().lower()

        threading.Thread(target=self.perform_asr, args=(start_sec, end_sec, language), daemon=True).start()

    def perform_asr(self, start_sec, end_sec, language="auto"):
        if not self.load_whisper_model():
            self.root.after(0, self.reset_asr_button)
            return

        tmp_path = None
        try:

            start_frame = int(start_sec * self.sample_rate)
            end_frame = int(end_sec * self.sample_rate)
            chunk = self.audio_data[start_frame:end_frame]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            sf.write(tmp_path, chunk, self.sample_rate, subtype='PCM_16')

            if language == "auto":
                result = self.asr_model.transcribe(tmp_path)
            else:
                result = self.asr_model.transcribe(tmp_path, language=language)

            recognized_text = result["text"].strip()

            self.root.after(0, self.update_text_from_asr, recognized_text)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Lỗi ASR", str(e)))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            self.root.after(0, self.reset_asr_button)

    def update_text_from_asr(self, recognized_text):
        current_text = self.text_editor.get("1.0", tk.END).strip()

        pattern = r"^(?:\|?\s*([^|]+)\s*\|\s*|([^:]+):\s*)"
        match = re.match(pattern, current_text)

        self.text_editor.delete("1.0", tk.END)

        if match:
            prefix = match.group(0)
            self.text_editor.insert(tk.END, f"{prefix}{recognized_text}")
        else:
            self.text_editor.insert(tk.END, recognized_text)

    def reset_asr_button(self):
        self.btn_asr.config(text="🎙️ Auto ASR", state=tk.NORMAL)
        self.lbl_status.config(text="Sẵn sàng", fg="black")

    def render_char_buttons(self):
        for widget in self.char_buttons_frame.winfo_children():
            widget.destroy()

        for char_name in self.char_list:
            btn = tk.Button(
                self.char_buttons_frame, text=char_name, bg="#E0E0E0",
                font=("Arial", 9, "bold"),
                command=lambda n=char_name: self.replace_character_name(n)
            )
            btn.bind("<Button-3>", lambda e, n=char_name: self.remove_custom_char(n))
            btn.pack(side=tk.LEFT, padx=2)

    def add_custom_char(self):
        new_name = self.entry_new_char.get().strip()
        if new_name and new_name not in self.char_list:
            self.char_list.append(new_name)
            self.entry_new_char.delete(0, tk.END)
            self.save_characters()
            self.render_char_buttons()

    def remove_custom_char(self, name_to_remove):
        if messagebox.askyesno("Xóa nhân vật", f"Bạn có muốn xóa '{name_to_remove}' khỏi danh sách gắn nhanh không?"):
            if name_to_remove in self.char_list:
                self.char_list.remove(name_to_remove)
                self.save_characters()
                self.render_char_buttons()

    def replace_character_name(self, new_name):
        current_text = self.text_editor.get("1.0", tk.END).strip()

        if not current_text:
            new_text = f"| {new_name} | "
        else:
            pattern = r"^(?:\|?\s*([^|]+)\s*\|\s*|([^:]+):\s*)(.*)"
            match = re.match(pattern, current_text)

            if match:
                rest_of_text = match.group(3).strip()
                new_text = f"| {new_name} | {rest_of_text}"
            else:
                new_text = f"| {new_name} | {current_text}"

        self.text_editor.delete("1.0", tk.END)
        self.text_editor.insert(tk.END, new_text)

        selected = self.tree.selection()
        if len(selected) == 1:
            item = selected[0]
            vals = list(self.tree.item(item)['values'])
            vals[3] = new_text
            self.tree.item(item, values=vals)
            self.perform_auto_save()

    def export_dataset(self):
        if self.audio_data is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải file Audio trước khi xuất!")
            return

        children = self.tree.get_children()
        if not children:
            messagebox.showwarning("Cảnh báo", "Bảng dữ liệu đang trống. Không có gì để xuất!")
            return

        output_dir = filedialog.askdirectory(title="Chọn thư mục lưu Dataset cắt ra")
        if not output_dir:
            return

        real_padding_sec = 0.05
        silence_tail_sec = 0.25

        speaker_pattern = re.compile(r"^\|?\s*([^|:]+)[\s|:]+(.*)", re.IGNORECASE)
        speaker_counts = {}

        self.lbl_status.config(text="Đang xử lý cắt, chuẩn hóa và thêm khoảng lặng...", fg="blue")
        self.root.update()

        try:
            for item in children:
                vals = self.tree.item(item)['values']
                start_sec = float(vals[1])
                end_sec = float(vals[2])
                raw_text = str(vals[3]).strip()

                speaker = "Unknown"
                clean_text = raw_text

                match = speaker_pattern.match(raw_text)
                if match:
                    speaker = match.group(1).strip()
                    clean_text = match.group(2).strip()

                clean_text = clean_text.replace('"', '').replace('\\', '')
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                speaker_folder_name = speaker.strip().title().replace(" ", "_").replace("/", "_")

                if speaker_folder_name not in speaker_counts:
                    speaker_counts[speaker_folder_name] = 1
                else:
                    speaker_counts[speaker_folder_name] += 1

                local_idx = speaker_counts[speaker_folder_name]

                padded_start = max(0.0, start_sec - real_padding_sec)
                padded_end = min(self.max_duration, end_sec + real_padding_sec)

                start_frame = int(padded_start * self.sample_rate)
                end_frame = int(padded_end * self.sample_rate)

                chunk = self.audio_data[start_frame:end_frame].copy()

                max_amp = np.max(np.abs(chunk))
                if max_amp > 0:
                    chunk = chunk * (0.95 / max_amp)

                silence_frames = int(silence_tail_sec * self.sample_rate)
                if len(chunk.shape) > 1:
                    silence = np.zeros((silence_frames, chunk.shape[1]), dtype=chunk.dtype)
                else:
                    silence = np.zeros(silence_frames, dtype=chunk.dtype)

                final_chunk = np.concatenate((chunk, silence))

                speaker_dir = os.path.join(output_dir, speaker_folder_name)
                os.makedirs(speaker_dir, exist_ok=True)

                base_filename = f"{speaker_folder_name}_{local_idx:04d}"
                wav_out_path = os.path.join(speaker_dir, f"{base_filename}.wav")
                script_out_path = os.path.join(speaker_dir, f"{speaker_folder_name}_script.txt")

                sf.write(wav_out_path, final_chunk, self.sample_rate, subtype='PCM_16')

                if local_idx == 1:
                    open(script_out_path, "w", encoding="utf-8").close()

                with open(script_out_path, "a", encoding="utf-8") as f_out:
                    f_out.write(f"[{base_filename}.wav] {clean_text}\n")

            self.lbl_status.config(text="Xuất Dataset hoàn tất!", fg="green")
            messagebox.showinfo(
                "Thành công",
                f"Đã xuất xong {len(children)} đoạn!\n\n- Đã Chuẩn hóa âm lượng to rõ đồng đều.\n- Tên file được đánh số lại rất gọn gàng."
            )

        except Exception as e:
            messagebox.showerror("Lỗi khi xuất", f"Đã xảy ra lỗi:\n{e}")
            self.lbl_status.config(text="Lỗi xuất file!", fg="red")

    def open_find_replace(self, event=None):
        if hasattr(self, 'fr_window') and self.fr_window.winfo_exists():
            self.fr_window.focus()
            return

        self.fr_window = tk.Toplevel(self.root)
        self.fr_window.title("Tìm kiếm & Xóa Text (Ctrl + F)")
        self.fr_window.geometry("480x210")
        self.fr_window.attributes('-topmost', True)

        tk.Label(self.fr_window, text="Tìm chữ:", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.entry_find = tk.Entry(self.fr_window, width=35, font=("Arial", 11))
        self.entry_find.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(self.fr_window, text="Thay bằng:\n(Để trống để XÓA)", font=("Arial", 9, "italic"), fg="gray").grid(row=1, column=0, padx=10, sticky="e")
        self.entry_replace = tk.Entry(self.fr_window, width=35, font=("Arial", 11))
        self.entry_replace.grid(row=1, column=1, padx=10)

        self.search_scope = tk.StringVar(value="box")
        scope_frame = tk.Frame(self.fr_window)
        scope_frame.grid(row=2, column=0, columnspan=2, pady=5)
        tk.Radiobutton(scope_frame, text="Chỉ ô Text", variable=self.search_scope, value="box", font=("Arial", 9, "bold"), fg="#2196F3").pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(scope_frame, text="Toàn bộ Bảng", variable=self.search_scope, value="table", font=("Arial", 9)).pack(side=tk.LEFT, padx=10)

        btn_frame = tk.Frame(self.fr_window)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=10)

        tk.Button(btn_frame, text="Tìm Tiếp", command=self.find_next, width=10, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Thay / Xóa", command=self.replace_current, width=12, bg="#FF9800", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Thay thế / Xóa TẤT CẢ", command=self.replace_all, width=20, bg="#F44336", fg="white", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)

        self.entry_find.focus()

    def find_next(self):
        query = self.entry_find.get()
        if not query:
            return

        if self.search_scope.get() == "box":
            start_pos = self.text_editor.index(tk.INSERT)
            if self.text_editor.tag_ranges(tk.SEL):
                start_pos = self.text_editor.index(tk.SEL_LAST)

            pos = self.text_editor.search(query, start_pos, stopindex=tk.END, nocase=True)
            if not pos:
                pos = self.text_editor.search(query, "1.0", stopindex=start_pos, nocase=True)

            if pos:
                end_pos = f"{pos}+{len(query)}c"
                self.text_editor.tag_remove(tk.SEL, "1.0", tk.END)
                self.text_editor.tag_add(tk.SEL, pos, end_pos)
                self.text_editor.mark_set(tk.INSERT, end_pos)
                self.text_editor.see(pos)
            else:
                messagebox.showinfo("Tìm kiếm", "Không tìm thấy từ khóa này trong ô Text!")

        else:
            children = self.tree.get_children()
            if not children:
                return
            start_idx = 0
            selected = self.tree.selection()
            if selected:
                start_idx = children.index(selected[-1]) + 1

            for i in range(start_idx, len(children)):
                item = children[i]
                text = self.tree.item(item)['values'][3]
                if query.lower() in str(text).lower():
                    self.tree.selection_set(item)
                    self.tree.focus(item)
                    self.tree.see(item)
                    self.on_select_row(None)
                    return

            for i in range(0, start_idx):
                item = children[i]
                text = self.tree.item(item)['values'][3]
                if query.lower() in str(text).lower():
                    self.tree.selection_set(item)
                    self.tree.focus(item)
                    self.tree.see(item)
                    self.on_select_row(None)
                    return
            messagebox.showinfo("Tìm kiếm", "Không tìm thấy kết quả nào trong bảng!")

    def replace_current(self):
        query = self.entry_find.get()
        replacement = self.entry_replace.get()
        if not query:
            return

        if self.search_scope.get() == "box":
            if self.text_editor.tag_ranges(tk.SEL):
                sel_start = self.text_editor.index(tk.SEL_FIRST)
                sel_end = self.text_editor.index(tk.SEL_LAST)
                selected_text = self.text_editor.get(sel_start, sel_end)
                if selected_text.lower() == query.lower():
                    self.text_editor.delete(sel_start, sel_end)
                    self.text_editor.insert(sel_start, replacement)
            self.find_next()

        else:
            selected = self.tree.selection()
            if not selected:
                self.find_next()
                return

            item = selected[-1]
            vals = list(self.tree.item(item)['values'])
            text = str(vals[3])

            new_text = re.sub(re.escape(query), replacement, text, flags=re.IGNORECASE)
            new_text = re.sub(' +', ' ', new_text).strip()

            if new_text != text:
                vals[3] = new_text
                self.tree.item(item, values=vals)
                self.text_editor.delete(1.0, tk.END)
                self.text_editor.insert(tk.END, new_text)
                self.perform_auto_save()
                self.find_next()
            else:
                self.find_next()

    def replace_all(self):
        query = self.entry_find.get()
        replacement = self.entry_replace.get()
        if not query:
            return

        if self.search_scope.get() == "box":
            text = self.text_editor.get("1.0", tk.END)
            new_text = re.sub(re.escape(query), replacement, text, flags=re.IGNORECASE)
            new_text = re.sub(' +', ' ', new_text).strip()
            self.text_editor.delete("1.0", tk.END)
            self.text_editor.insert(tk.END, new_text)
            messagebox.showinfo("Thành công", "Đã thay thế/xóa toàn bộ trong ô Text!\n\n(Nhớ bấm nút '️ CẬP NHẬT' để chốt lên bảng nhé)")

        else:
            count = 0
            for item in self.tree.get_children():
                vals = list(self.tree.item(item)['values'])
                text = str(vals[3])

                new_text = re.sub(re.escape(query), replacement, text, flags=re.IGNORECASE)
                new_text = re.sub(' +', ' ', new_text).strip()

                if new_text != text:
                    vals[3] = new_text
                    self.tree.item(item, values=vals)
                    count += 1

            if count > 0:
                self.perform_auto_save()
                self.on_select_row(None)
                messagebox.showinfo("Thành công", f"Đã Xóa/Thay thế ở {count} vị trí trong Bảng!")
            else:
                messagebox.showinfo("Thông báo", "Không tìm thấy từ khóa nào để xóa!")

    def merge_segments(self):
        selected = self.tree.selection()
        if len(selected) < 2:
            messagebox.showwarning("Cảnh báo", "Vui lòng giữ Chuột Trái (hoặc Shift) kéo chọn ít nhất 2 dòng trên bảng để có thể gộp lại!")
            return

        items_data = []
        for item in selected:
            vals = self.tree.item(item)['values']
            items_data.append((float(vals[1]), float(vals[2]), str(vals[3]), item))

        items_data.sort(key=lambda x: x[0])

        new_start = items_data[0][0]
        new_end = items_data[-1][1]

        texts = [x[2].strip() for x in items_data if x[2].strip()]
        new_text = " ".join(texts)

        for _, _, _, item in items_data:
            self.tree.delete(item)

        new_item = self.tree.insert("", tk.END, values=("", f"{new_start:.2f}", f"{new_end:.2f}", new_text))

        self.sort_and_reindex_tree()
        self.refresh_vad_overlays()
        self.perform_auto_save()

        self.tree.selection_set(new_item)
        self.tree.focus(new_item)
        self.tree.see(new_item)
        self.on_select_row(None)

        self.lbl_status.config(text=f"Đã gộp thành công {len(selected)} đoạn!", fg="green")

    def perform_auto_save(self):
        if self.auto_save_var.get():
            if self.current_txt_path:
                self.save_txt_silent()
            else:
                self.save_txt()

    def save_txt_silent(self):
        try:
            all_items = []
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
                all_items.append((float(values[1]), float(values[2]), values[3]))

            all_items.sort(key=lambda x: x[0])

            with open(self.current_txt_path, "w", encoding="utf-8") as f:
                for start, end, text in all_items:
                    f.write(f"[{start:05.2f}s - {end:05.2f}s] Text: {text}\n")

            current_time = time.strftime('%H:%M:%S')
            self.lbl_status.config(text=f"Đã tự động lưu ({current_time})", fg="blue")
        except Exception as e:
            self.lbl_status.config(text=f"Lỗi tự động lưu: {e}", fg="red")

    def on_span_move(self, vmin, vmax):
        overlap = False
        for st, en in self.current_segments:
            if max(vmin, st) < min(vmax, en):
                overlap = True
                break

        color = '#FF9800' if overlap else 'red'
        try:
            if hasattr(self.span, '_rect'):
                self.span._rect.set_facecolor(color)
            elif hasattr(self.span, 'rect'):
                self.span.rect.set_facecolor(color)
            elif hasattr(self.span, 'poly'):
                self.span.poly.set_facecolor(color)
        except Exception:
            pass

    def refresh_vad_overlays(self):
        if self.audio_data is None:
            return

        for span in self.vad_spans:
            try:
                span.remove()
            except Exception:
                pass
        self.vad_spans.clear()
        self.current_segments.clear()

        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            start_val = float(values[1])
            end_val = float(values[2])

            self.current_segments.append((start_val, end_val))

            if self.show_vad_var.get():
                span = self.ax.axvspan(start_val, end_val, color='#4CAF50', alpha=0.25)
                self.vad_spans.append(span)

        self.canvas.draw_idle()

    def setup_hotkeys(self):
        self.root.bind("<Left>", self.pan_left)
        self.root.bind("<Right>", self.pan_right)
        self.root.bind("<Control-f>", self.open_find_replace)
        self.root.bind("<Control-F>", self.open_find_replace)

    def pan_left(self, event):
        self.pan_waveform(direction=-1)

    def pan_right(self, event):
        self.pan_waveform(direction=1)

    def pan_waveform(self, direction):
        if self.audio_data is None:
            return
        focused_widget = self.root.focus_get()
        if isinstance(focused_widget, (tk.Entry, tk.Text)):
            return

        cur_xlim = self.ax.get_xlim()
        window_size = cur_xlim[1] - cur_xlim[0]
        step = window_size * 0.2

        new_xmin = cur_xlim[0] + direction * step
        new_xmax = cur_xlim[1] + direction * step

        if new_xmin < 0:
            new_xmin = 0
            new_xmax = window_size
        if new_xmax > self.max_duration:
            new_xmax = self.max_duration
            new_xmin = self.max_duration - window_size

        self.ax.set_xlim([new_xmin, new_xmax])
        self.canvas.draw_idle()

    def on_scroll_zoom(self, event):
        if self.audio_data is None:
            return
        if event.inaxes != self.ax:
            return

        base_scale = 1.3
        cur_xlim = self.ax.get_xlim()
        xdata = event.xdata
        if xdata is None:
            return

        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])

        new_xmin = xdata - new_width * (1 - relx)
        new_xmax = xdata + new_width * relx

        if new_xmin < 0:
            new_xmin = 0
        if new_xmax > self.max_duration:
            new_xmax = self.max_duration

        self.ax.set_xlim([new_xmin, new_xmax])
        self.canvas.draw_idle()

    def play_segment(self):
        if self.audio_data is None:
            return
        try:
            start_sec = float(self.entry_start.get())

            if self.play_to_end_var.get():
                end_sec = self.max_duration
            else:
                end_sec = float(self.entry_end.get())

            start_frame = int(start_sec * self.sample_rate)
            end_frame = int(end_sec * self.sample_rate)
            chunk = self.audio_data[start_frame:end_frame]

            self.lbl_stop_time.config(text="")
            self.remove_playback_cursor()

            self.is_playing = True
            self.playback_start_sec = start_sec
            self.playback_end_sec = end_sec

            self.playback_line = self.ax.axvline(x=start_sec, color='#FFEB3B', linewidth=2.5, zorder=10)
            self.canvas.draw_idle()

            sd.stop()
            self.playback_start_time = time.time()
            sd.play(chunk, self.sample_rate)

            self.update_playback_cursor()
        except ValueError:
            pass

    def update_playback_cursor(self):
        if not self.is_playing:
            return
        elapsed_time = time.time() - self.playback_start_time
        current_sec = self.playback_start_sec + elapsed_time

        if current_sec >= self.playback_end_sec:
            self.is_playing = False
            self.remove_playback_cursor()
        else:
            if self.playback_line:
                self.playback_line.set_xdata([current_sec, current_sec])
                self.canvas.draw_idle()
            self.root.after(40, self.update_playback_cursor)

    def stop_audio(self):
        if self.is_playing:
            sd.stop()
            elapsed_time = time.time() - self.playback_start_time
            stopped_sec = self.playback_start_sec + elapsed_time
            if stopped_sec > self.playback_end_sec:
                stopped_sec = self.playback_end_sec
            self.lbl_stop_time.config(text=f"[Dừng ở: {stopped_sec:.2f}s]")
            self.is_playing = False
            self.remove_playback_cursor()
        else:
            sd.stop()

    def remove_playback_cursor(self):
        if self.playback_line is not None:
            try:
                self.playback_line.remove()
                self.playback_line = None
                self.canvas.draw_idle()
            except Exception:
                pass

    def sort_and_reindex_tree(self):
        children = list(self.tree.get_children(""))
        children.sort(key=lambda x: float(self.tree.item(x)['values'][1]))

        for index, child in enumerate(children):
            self.tree.move(child, "", index)
            vals = list(self.tree.item(child)['values'])
            vals[0] = index + 1
            self.tree.item(child, values=vals)

    def on_select_waveform(self, xmin, xmax):
        try:
            if hasattr(self.span, '_rect'):
                self.span._rect.set_facecolor('red')
            elif hasattr(self.span, 'rect'):
                self.span.rect.set_facecolor('red')
            elif hasattr(self.span, 'poly'):
                self.span.poly.set_facecolor('red')
        except Exception:
            pass

        self.entry_start.delete(0, tk.END)
        self.entry_start.insert(0, f"{xmin:.2f}")
        self.entry_end.delete(0, tk.END)
        self.entry_end.insert(0, f"{xmax:.2f}")
        self.text_editor.delete(1.0, tk.END)
        if self.tree.selection():
            self.tree.selection_remove(self.tree.selection())

    def draw_waveform(self):
        if self.audio_data is None:
            return
        self.ax.clear()
        self.vad_spans.clear()
        self.remove_playback_cursor()

        if len(self.audio_data.shape) > 1:
            mono_data = self.audio_data.mean(axis=1)
        else:
            mono_data = self.audio_data

        self.max_duration = len(mono_data) / self.sample_rate

        max_points = 100000
        step = max(1, len(mono_data) // max_points)
        plot_data = mono_data[::step]
        time_axis = np.arange(len(plot_data)) * step / self.sample_rate

        self.ax.plot(time_axis, plot_data, color="#2196F3", linewidth=0.5)
        self.ax.set_xlim(0, self.max_duration)

        self.refresh_vad_overlays()
        self.canvas.draw()

    def apply_vad_adjust_all(self):
        children = self.tree.get_children()
        if not children:
            messagebox.showwarning("Cảnh báo", "Bảng dữ liệu đang trống!")
            return

        try:
            trim_start = float(self.entry_trim_start_all.get().strip())
            extend_end = float(self.entry_extend_end_all.get().strip())
        except ValueError:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập số hợp lệ cho thời gian điều chỉnh!")
            return

        if trim_start < 0 or extend_end < 0:
            messagebox.showwarning("Cảnh báo", "Thời gian điều chỉnh phải lớn hơn hoặc bằng 0!")
            return

        for item in children:
            vals = list(self.tree.item(item)['values'])
            start_val = float(vals[1])
            end_val = float(vals[2])

            new_start = max(0.0, start_val + trim_start)
            new_end = end_val + extend_end

            if new_end < new_start:
                new_end = new_start

            vals[1] = f"{new_start:.2f}"
            vals[2] = f"{new_end:.2f}"
            self.tree.item(item, values=vals)

        self.sort_and_reindex_tree()
        self.refresh_vad_overlays()
        self.perform_auto_save()
        self.on_select_row(None)
        self.lbl_status.config(text="Đã áp dụng điều chỉnh toàn bộ VAD", fg="green")

    def tweak_time(self, entry_widget, delta):
        try:
            current_val = float(entry_widget.get())
            new_val = max(0.0, current_val + delta)
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, f"{new_val:.2f}")

            if self.audio_data is not None:
                start_val = float(self.entry_start.get())
                end_val = float(self.entry_end.get())
                self.span.extents = (start_val, end_val)
        except ValueError:
            pass

    def load_audio(self):
        filepath = filedialog.askopenfilename(title="Chọn file Audio (.wav)", filetypes=[("WAV files", "*.wav")])
        if filepath:
            try:
                self.audio_data, self.sample_rate = sf.read(filepath)
                self.lbl_status.config(text=f"Đã tải Audio: {os.path.basename(filepath)}", fg="green")
                self.draw_waveform()
            except Exception as e:
                messagebox.showerror("Lỗi Audio", f"Không thể đọc file:\n{e}")

    def load_txt(self):
        filepath = filedialog.askopenfilename(title="Chọn Dataset (.txt)", filetypes=[("Text files", "*.txt")])
        if not filepath:
            return

        self.current_txt_path = filepath
        self.tree.delete(*self.tree.get_children())
        pattern = r"\[\s*([\d\.]+)\s*s\s*-\s*([\d\.]+)\s*s\s*\]\s*(?:Text:\s*)?(.*)"
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    start_val, end_val, text_val = match.groups()
                    self.tree.insert("", tk.END, values=("", start_val, end_val, text_val.strip()))

            self.sort_and_reindex_tree()
            self.refresh_vad_overlays()
            self.lbl_status.config(text=f"Đang xử lý: {os.path.basename(filepath)}", fg="black")
        except Exception as e:
            messagebox.showerror("Lỗi Text", f"Lỗi đọc file:\n{e}")

    def on_select_row(self, event):
        selected = self.tree.selection()
        if not selected:
            return

        if len(selected) == 1:
            values = self.tree.item(selected[0])['values']
            start_val = float(values[1])
            end_val = float(values[2])
            text_val = str(values[3])
        else:
            items_data = []
            for item in selected:
                vals = self.tree.item(item)['values']
                items_data.append((float(vals[1]), float(vals[2]), str(vals[3])))

            items_data.sort(key=lambda x: x[0])
            start_val = items_data[0][0]
            end_val = items_data[-1][1]
            texts = [x[2].strip() for x in items_data if x[2].strip()]
            text_val = " ".join(texts)

        self.entry_start.delete(0, tk.END)
        self.entry_start.insert(0, f"{start_val:.2f}")
        self.entry_end.delete(0, tk.END)
        self.entry_end.insert(0, f"{end_val:.2f}")

        self.text_editor.delete(1.0, tk.END)
        self.text_editor.insert(tk.END, text_val)

        if self.audio_data is not None:
            try:
                if hasattr(self.span, '_rect'):
                    self.span._rect.set_facecolor('red')
                elif hasattr(self.span, 'rect'):
                    self.span.rect.set_facecolor('red')
                elif hasattr(self.span, 'poly'):
                    self.span.poly.set_facecolor('red')
            except Exception:
                pass

            self.span.extents = (start_val, end_val)
            padding = 2.0
            new_xmin = max(0, start_val - padding)
            new_xmax = min(self.max_duration, end_val + padding)
            self.ax.set_xlim([new_xmin, new_xmax])
            self.canvas.draw_idle()

    def add_new_row(self):
        new_start = self.entry_start.get().strip()
        new_end = self.entry_end.get().strip()
        new_text = self.text_editor.get(1.0, tk.END).strip()

        if not new_start or not new_end:
            messagebox.showwarning("Cảnh báo", "Vui lòng kéo chọn thời gian trên sóng âm!")
            return
        if not new_text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập Nội dung Text!")
            return

        item = self.tree.insert("", tk.END, values=("", new_start, new_end, new_text))

        self.sort_and_reindex_tree()
        self.refresh_vad_overlays()
        self.perform_auto_save()

        self.tree.selection_set(item)
        self.tree.focus(item)
        self.tree.see(item)

    def update_row(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một dòng trên bảng để cập nhật!")
            return

        if len(selected) > 1:
            messagebox.showwarning("Cảnh báo", "Bạn đang bôi đen nhiều đoạn cùng lúc!\n\nHãy bấm nút '🔗 GỘP ĐOẠN' nếu muốn nối chúng lại, hoặc chỉ chọn 1 dòng để cập nhật.")
            return

        new_start = self.entry_start.get().strip()
        new_end = self.entry_end.get().strip()
        new_text = self.text_editor.get(1.0, tk.END).strip()

        stt = self.tree.item(selected[0])['values'][0]
        self.tree.item(selected[0], values=(stt, new_start, new_end, new_text))

        self.sort_and_reindex_tree()
        self.refresh_vad_overlays()
        self.perform_auto_save()

        next_item = self.tree.next(selected[0])
        if next_item:
            self.tree.selection_set(next_item)
            self.tree.focus(next_item)
            self.on_select_row(None)

    def delete_row(self):
        selected = self.tree.selection()
        if selected:
            next_item = self.tree.next(selected[-1])

            for item in selected:
                self.tree.delete(item)

            self.sort_and_reindex_tree()
            self.refresh_vad_overlays()
            self.perform_auto_save()

            if next_item and self.tree.exists(next_item):
                self.tree.selection_set(next_item)
                self.tree.focus(next_item)
                self.on_select_row(None)

    def save_txt(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile=self.current_txt_path
        )
        if not filepath:
            return

        self.current_txt_path = filepath

        try:
            all_items = []
            for item in self.tree.get_children():
                values = self.tree.item(item)['values']
                all_items.append((float(values[1]), float(values[2]), values[3]))

            all_items.sort(key=lambda x: x[0])

            with open(filepath, "w", encoding="utf-8") as f:
                for start, end, text in all_items:
                    f.write(f"[{start:05.2f}s - {end:05.2f}s] Text: {text}\n")

            self.lbl_status.config(text=f"Đang xử lý: {os.path.basename(filepath)}", fg="black")
            messagebox.showinfo("Xong!", "Đã lưu bộ Dataset hoàn chỉnh!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không lưu được: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VADEditorPro(root)
    root.mainloop()