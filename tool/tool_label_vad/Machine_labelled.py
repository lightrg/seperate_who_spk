import os
import re
import glob
import sys
import subprocess
import numpy as np
import soundfile as sf
import yt_dlp
import torch
import torchaudio
import warnings
import time
import requests
import whisper as openai_whisper

if not hasattr(np, 'NAN'): np.NAN = np.nan
if not hasattr(np, 'NaN'): np.NaN = np.nan
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
warnings.filterwarnings("ignore")

from pyannote.audio import Pipeline
from faster_whisper import WhisperModel

OUTPUT_DIR = "output_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load HF_TOKEN from environment variable (set via: source set_hf_token.sh)
# Never hardcode your token here — see set_hf_token.sh in the project root.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise EnvironmentError(
        "[ERROR] HF_TOKEN is not set.\n"
        "  Please fill in your token and run: source set_hf_token.sh\n"
        "  (See README.md — 'Hugging Face Token Setup' section)"
    )
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "conan_transcript_log.txt")

MAX_DURATION = 10.0
MIN_DURATION = 0.5

JUNK_KEYWORDS = [
    "đăng ký", "đăng kí", "subscribe", "subcribe",
    "nhấn chuông", "bấm chuông", "like và share",
    "chia sẻ video", "ủng hộ kênh", "theo dõi kênh",
    "ghiền mì gõ", "cảm ơn các bạn đã theo dõi"
]

def download_data_and_subs(youtube_url, target_lang):
    print(f"\n[+] Đang tải Audio và Phụ đề YouTube vào thư mục '{OUTPUT_DIR}'...")
    session_id = int(time.time())

    base_name = os.path.join(OUTPUT_DIR, f"audio_raw_{session_id}")

    if target_lang:
        sub_langs = [target_lang]
    else:
        sub_langs = ['vi']

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{base_name}.%(ext)s',
        'quiet': True,
        'noplaylist': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': sub_langs,
        'subtitlesformat': 'vtt',
        'ignoreerrors': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    final_audio = f"{base_name}.wav"

    vtt_path = None
    if target_lang:
        vtt_files = glob.glob(f"{base_name}*{target_lang}*.vtt")
        if vtt_files: vtt_path = vtt_files[0]

    if not vtt_path:
        vtt_files = glob.glob(f"{base_name}*.vtt")
        if vtt_files: vtt_path = vtt_files[0]

    if vtt_path: print(f"[+] Đã tìm thấy file phụ đề: {vtt_path}")
    else: print("[-] Không tìm thấy phụ đề. AI Whisper sẽ tự động chép lời.")

    return final_audio, vtt_path

def convert_to_16k_mono(audio_path):
    out_path = audio_path.replace(".wav", "_16k.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
    return out_path

def get_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def apply_sponsorblock(audio_path, video_url):
    video_id = get_video_id(video_url)
    if not video_id: return audio_path

    print("\n[+] Đang truy vấn dữ liệu từ SponsorBlock API...")
    categories = '["sponsor","intro","outro","interaction","preview"]'
    api_url = f"https://sponsor.ajay.app/api/skipSegments?videoID={video_id}&categories={categories}"

    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code != 200: return audio_path

        segments = response.json()
        print(f"    - Tìm thấy {len(segments)} đoạn intro/outro/sponsor. Đang làm câm...")

        data, samplerate = sf.read(audio_path)
        for seg in segments:
            start_sec, end_sec = seg['segment'][0], seg['segment'][1]
            start_sample = max(0, int(start_sec * samplerate))
            end_sample = min(len(data), int(end_sec * samplerate))
            data[start_sample:end_sample] = 0

        silenced_path = audio_path.replace(".wav", "_silenced.wav")
        sf.write(silenced_path, data, samplerate)
        return silenced_path
    except Exception as e:
        return audio_path

def parse_vtt(filepath):
    subs = []
    if not filepath or not os.path.exists(filepath): return subs
    with open(filepath, 'r', encoding='utf-8') as f: lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            parts = line.split('-->')
            def time_to_sec(t_str):
                t_str = t_str.strip().replace(',', '.')
                p = t_str.split(':')
                if len(p) == 3: return int(p[0])*3600 + int(p[1])*60 + float(p[2])
                elif len(p) == 2: return int(p[0])*60 + float(p[1])
                return 0.0

            start_sec, end_sec = time_to_sec(parts[0]), time_to_sec(parts[1])
            text = ""
            i += 1
            while i < len(lines) and lines[i].strip() != "":
                clean_line = re.sub(r'<[^>]+>', '', lines[i].strip())
                if clean_line and not (clean_line.startswith('[') and clean_line.endswith(']')):
                    text += clean_line + " "
                i += 1
            if text.strip(): subs.append({'start': start_sec, 'end': end_sec, 'text': text.strip()})
        else: i += 1
    return subs

def get_aligned_text(t_start, t_end, subs, fallback_text):
    if not subs: return fallback_text
    matched_texts = []
    for sub in subs:
        overlap = max(0, min(t_end, sub['end']) - max(t_start, sub['start']))
        sub_duration = sub['end'] - sub['start']
        if overlap > 0.5 or (sub_duration > 0 and (overlap / sub_duration) > 0.4):
            matched_texts.append(sub['text'])
    return " ".join(matched_texts) if matched_texts else fallback_text

def extract_vocals(audio_path):
    print("\n[+] Đang chạy Demucs tách BGM (Nhạc nền)...")
    cmd = [sys.executable, "-m", "demucs.separate", "--two-stems=vocals", "-n", "htdemucs", "-o", OUTPUT_DIR, audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = os.path.join(OUTPUT_DIR, "htdemucs", base_name, "vocals.wav")

    return vocals_path if os.path.exists(vocals_path) else audio_path

def run_diarization(audio_file):
    print("\n[+] Chạy Pyannote phân tích nhân vật...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    diarization = pipeline(audio_file)
    del pipeline
    torch.cuda.empty_cache()
    return [{"start": t.start, "end": t.end, "speaker": s} for t, _, s in diarization.itertracks(yield_label=True)]

def assign_speaker(t_start, t_end, speaker_segments):
    max_overlap, assigned_speaker = 0, "Unknown"
    for seg in speaker_segments:
        overlap = max(0, min(t_end, seg['end']) - max(t_start, seg['start']))
        if overlap > max_overlap:
            max_overlap, assigned_speaker = overlap, seg['speaker']
    return assigned_speaker

if __name__ == "__main__":
    url = input("Insert link YouTube: ")
    if not url.strip(): exit()

    lang_input = input("Enter language code (e.g., vi, en, ja). Leave blank for AI to detect: ").strip()
    target_lang = lang_input if lang_input else None

    raw_audio, vtt_path = download_data_and_subs(url, target_lang)
    raw_audio = convert_to_16k_mono(raw_audio)
    parsed_subs = parse_vtt(vtt_path)

    silenced_audio = apply_sponsorblock(raw_audio, url)
    clean_audio_path = extract_vocals(silenced_audio)
    clean_audio_path = convert_to_16k_mono(clean_audio_path)

    speaker_segments = run_diarization(clean_audio_path)

    print("\n[+] Bật Faster-Whisper trích xuất Timestamps (Lượt 1)...")
    whisper_model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

    segments_gen, info = whisper_model.transcribe(
        clean_audio_path,
        language=target_lang,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(threshold=0.2, min_silence_duration_ms=250, speech_pad_ms=400),
        word_timestamps=True
    )

    if target_lang is None:
        print(f"[!] AI nhận diện ngôn ngữ audio: '{info.language}'")

    initial_segments = list(segments_gen)
    all_segments = []

    class GapSegment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

    print("\n[+] ĐANG KIỂM TRA VÀ QUÉT LẠI CÁC KHOẢNG TRỐNG (RESCAN GAPS)...")
    GAP_THRESHOLD = 4.0
    fallback_whisper_model = None

    for i in range(len(initial_segments)):
        all_segments.append(initial_segments[i])

        if i < len(initial_segments) - 1:
            current_end = initial_segments[i].end
            next_start = initial_segments[i+1].start
            gap_duration = next_start - current_end

            if gap_duration > GAP_THRESHOLD:
                print(f"    [!] Phát hiện khoảng trống {gap_duration:.2f}s (từ {current_end:.2f}s đến {next_start:.2f}s). Đang xử lý...")

                data, samplerate = sf.read(clean_audio_path)
                start_sample = int(current_end * samplerate)
                end_sample = int(next_start * samplerate)
                gap_audio = data[start_sample:end_sample]

                if np.max(np.abs(gap_audio)) < 0.0001:
                    print("        -> Âm thanh trống hoàn toàn (có thể do SponsorBlock). Bỏ qua.")
                    continue

                if fallback_whisper_model is None:
                    print("    [+] Đang tải mô hình OpenAI Whisper Large vào VRAM để 'vét' chữ...")
                    fallback_whisper_model = openai_whisper.load_model("large")

                temp_gap_path = os.path.join(OUTPUT_DIR, "temp_gap.wav")
                sf.write(temp_gap_path, gap_audio, samplerate)

                result = fallback_whisper_model.transcribe(
                    temp_gap_path,
                    language=target_lang or info.language,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.4,
                    logprob_threshold=-1.0
                )

                valid_gap_found = False

                for g_seg in result["segments"]:
                    text_clean = g_seg["text"].strip()
                    text_lower = text_clean.lower()

                    if text_clean:
                        is_junk = any(kw in text_lower for kw in JUNK_KEYWORDS)

                        if is_junk:
                            print(f"        -> [Đã chặn ảo giác]: {text_clean}")
                            continue

                        all_segments.append(GapSegment(
                            start=g_seg["start"] + current_end,
                            end=g_seg["end"] + current_end,
                            text=text_clean
                        ))
                        print(f"        -> Vớt được bằng AI: {text_clean}")
                        valid_gap_found = True

                if not valid_gap_found:
                    vtt_recovered_count = 0
                    for sub in parsed_subs:

                        overlap = max(0, min(next_start, sub['end']) - max(current_end, sub['start']))
                        sub_duration = sub['end'] - sub['start']

                        if overlap > 0.5 or (sub_duration > 0 and (overlap / sub_duration) > 0.5):
                            clean_sub_text = sub['text'].strip()

                            if clean_sub_text and not any(kw in clean_sub_text.lower() for kw in JUNK_KEYWORDS):
                                all_segments.append(GapSegment(
                                    start=max(current_end, sub['start']),
                                    end=min(next_start, sub['end']),
                                    text=clean_sub_text
                                ))
                                print(f"        -> [Vớt thành công bằng Phụ đề YT]: {clean_sub_text}")
                                vtt_recovered_count += 1

                    if vtt_recovered_count == 0:
                        print("        -> [Thất bại]: Không có phụ đề YT hợp lệ trong khoảng này.")

                if os.path.exists(temp_gap_path):
                    os.remove(temp_gap_path)

    if fallback_whisper_model is not None:
        del fallback_whisper_model
        torch.cuda.empty_cache()

    all_segments = sorted(all_segments, key=lambda x: x.start)

    final_log = []
    current_spk, current_start, current_end, current_text = None, 0.0, 0.0, ""

    def save_segment_log():
        global current_start, current_end, current_spk, current_text
        if not current_text: return

        duration = current_end - current_start
        final_text = get_aligned_text(current_start, current_end, parsed_subs, fallback_text=current_text).strip()

        if any(kw in final_text.lower() for kw in JUNK_KEYWORDS): return

        if duration >= MIN_DURATION and final_text:
            final_text = final_text[0].upper() + final_text[1:]
            if not final_text.endswith(('.', '?', '!')): final_text += '.'

            final_text = re.sub(r'\s+', ' ', final_text).replace('\n', ' ')
            final_text = re.sub(r'\s+([.,?!])', r'\1', final_text)

            log_line = f"[{current_start:.2f}s - {current_end:.2f}s] Text: | {current_spk} | {final_text}"

            final_log.append(log_line)
            print(log_line)

    print("\n[+] ĐANG TRÍCH XUẤT TIMESTAMPS VÀ GẮN NHÃN...\n")
    for seg in all_segments:
        t_start, t_end, text = seg.start, seg.end, seg.text.strip()
        if not text: continue

        speaker = assign_speaker(t_start, t_end, speaker_segments)

        if speaker.startswith("SPEAKER_"):
            char_num = speaker.split("_")[1]
            clean_speaker = f"Char {char_num}"
        else:
            clean_speaker = "Char ??"

        if current_spk is None:
            current_spk, current_start, current_end, current_text = clean_speaker, t_start, t_end, text
        else:
            projected_duration = t_end - current_start
            is_same_speaker = (clean_speaker == current_spk)
            gap = t_start - current_end

            should_cut = False
            if not is_same_speaker: should_cut = True
            elif gap > 1.2: should_cut = True
            elif projected_duration > MAX_DURATION: should_cut = True
            elif current_text.endswith(('.', '?', '!')) and (current_end - current_start) > 4.0: should_cut = True

            if not should_cut:
                if current_text and not current_text.strip().endswith(('.', '?', '!', ',', ':', ';')):
                    if text[0].isupper():
                        current_text += ". " + text
                    else:
                        current_text += " " + text
                else:
                    current_text += " " + text

                current_end = t_end
            else:
                save_segment_log()
                current_spk, current_start, current_end, current_text = clean_speaker, t_start, t_end, text

    save_segment_log()

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(final_log))

    print(f"\n=============================================")
    print(f"[+] HOÀN TẤT! Đã xuất dữ liệu Dataset ra file: '{OUTPUT_TXT}'")