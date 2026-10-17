import os
import json
import subprocess
import torch
import whisper
import psycopg2
import cv2
import numpy as np
from dotenv import load_dotenv
from transformers import pipeline
from ultralytics import YOLO

load_dotenv()

# Official Category Weights (Allotted ONLY ONCE per movie if present)
CATEGORY_WEIGHTS = {
    "Nudity": 5,
    "Gore": 5,
    "Strong abuse": 4,
    "Violence": 4,
    "Smoking or Drugs": 3,
    "Mild Language": 1
}

class ModerationPipeline:
    def __init__(self):
        print("Loading AI models into memory...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Whisper Audio Model (Generates English Subtitles & Transcripts)
        self.audio_model = whisper.load_model("small", device=self.device)

        # 2. XLM-RoBERTa Text Toxicity Model
        self.text_model = pipeline(
            "text-classification",
            model="./text_model",
            tokenizer="./text_model",
            device=0 if self.device == "cuda" else -1,
            top_k=None
        )

        # 3. YOLOv8 Custom Model (Weapons / Blood / Smoking / Violence)
        self.weapon_model = YOLO('./best.pt')

        # 4. AWS RDS PostgreSQL Connection
        self.db_conn = None
        self.db_cursor = None
        try:
            self.db_conn = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                port=os.getenv('DB_PORT', '5432')
            )
            self.db_cursor = self.db_conn.cursor()
            print("Connected to AWS RDS successfully!")
        except Exception as e:
            print(f"Warning: RDS connection failed ({e}). Proceeding without DB logging.")

    def log_to_rds(self, video_name, timestamp_start, timestamp_end, category, score, action):
        if not self.db_cursor:
            return
        try:
            query = """
                INSERT INTO audit_logs (video_name, start_time, end_time, category, severity_score, action)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            self.db_cursor.execute(query, (video_name, timestamp_start, timestamp_end, category, score, action))
            self.db_conn.commit()
        except Exception as e:
            print(f"Failed to log to RDS: {e}")

    @staticmethod
    def _format_vtt_time(seconds):
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"

    @staticmethod
    def _format_mmss(seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _map_visual_label(self, raw_label):
        """Maps YOLO class names to the 6 official scoring categories."""
        lbl = raw_label.lower()
        if any(k in lbl for k in ["blood", "gore", "cut", "wound", "dead", "corpse"]):
            return "Gore"
        elif any(k in lbl for k in ["smoke", "smoking", "cigarette", "drug", "alcohol", "weed"]):
            return "Smoking or Drugs"
        elif any(k in lbl for k in ["nude", "nudity", "explicit", "nsfw"]):
            return "Nudity"
        else:
            # Weapons (sword, gun, knife, axe, pistol, rifle) and fighting map to Violence
            return "Violence"

    def process_audio_and_subtitles(self, video_path):
        """Translates speech to English subtitles (.vtt) and flags Strong Abuse vs Mild Language."""
        # task='translate' ensures English subtitles even if audio is Tamil/Hindi
        result = self.audio_model.transcribe(video_path, task="translate", word_timestamps=True)
        
        audio_flags = []
        vtt_lines = ["WEBVTT\n"]
        transcript_snippets = []

        for idx, segment in enumerate(result.get("segments", [])):
            text = segment["text"].strip()
            if not text:
                continue
            transcript_snippets.append(text)
            start_s = round(segment["start"], 2)
            end_s = round(segment["end"], 2)

            # Build English Subtitle track
            vtt_lines.append(f"{idx + 1}")
            vtt_lines.append(f"{self._format_vtt_time(start_s)} --> {self._format_vtt_time(end_s)}")
            vtt_lines.append(f"{text}\n")

            # Check toxicity with XLM-RoBERTa
            predictions = self.text_model(text[:512])[0]
            max_score = max([p['score'] for p in predictions])

            if max_score >= 0.80:
                audio_flags.append({
                    "start": start_s,
                    "end": end_s,
                    "category": "Strong abuse",
                    "detail": f'Strong abusive dialogue detected ("{text}")',
                    "score": round(max_score, 2),
                    "action": "mute"
                })
            elif 0.50 <= max_score < 0.80:
                audio_flags.append({
                    "start": start_s,
                    "end": end_s,
                    "category": "Mild Language",
                    "detail": f'Mild offensive language detected ("{text}")',
                    "score": round(max_score, 2),
                    "action": "mute"
                })

        vtt_content = "\n".join(vtt_lines)
        full_summary_text = " ".join(transcript_snippets[:8])
        return audio_flags, vtt_content, full_summary_text

    def process_video_frames(self, video_path):
        """Draws bounding boxes on weapons/blood/smoking across the movie and extracts metadata."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = round(total_frames / float(fps), 2) if fps > 0 else 0

        raw_annotated_path = video_path.replace(".mp4", "_raw_boxed.mp4")
        final_annotated_path = video_path.replace(".mp4", "_annotated.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(raw_annotated_path, fourcc, fps, (width, height))

        raw_flags = []
        frame_idx = 0
        last_boxes = [] # Persist boxes across 3 frames for fast 10-min video processing

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO every 3rd frame to keep 10-min video inference fast & responsive
            if frame_idx % 3 == 0:
                results = self.weapon_model.predict(frame, conf=0.25, verbose=False)
                r = results[0]
                last_boxes = []
                timestamp = round(frame_idx / float(fps), 2)

                if len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        raw_label = r.names[cls_id]
                        conf = round(float(box.conf[0]), 2)
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        category = self._map_visual_label(raw_label)

                        last_boxes.append((xyxy, f"{category}: {raw_label} ({conf})", category))
                        raw_flags.append({
                            "start": timestamp,
                            "end": round(timestamp + 1.5, 2),
                            "category": category,
                            "detail": f"{raw_label.title()} detected on screen",
                            "score": conf,
                            "action": "skip"
                        })

            # Draw bounding boxes on current frame
            for (x1, y1, x2, y2), label_text, cat in last_boxes:
                color = (0, 0, 255) if cat == "Gore" else ((0, 140, 255) if cat == "Violence" else (0, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label_text, (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # Merge original audio & convert to H.264 so HTML5 browser player streams smoothly
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", raw_annotated_path,
                "-i", video_path,
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                final_annotated_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(raw_annotated_path):
                os.remove(raw_annotated_path)
        except Exception:
            # Fallback if ffmpeg remux fails
            final_annotated_path = raw_annotated_path

        # Merge overlapping visual intervals of the same category
        merged_flags = []
        for flag in sorted(raw_flags, key=lambda x: (x['start'], x['category'])):
            if merged_flags and flag['category'] == merged_flags[-1]['category'] and flag['start'] <= merged_flags[-1]['end'] + 1.0:
                merged_flags[-1]['end'] = max(merged_flags[-1]['end'], flag['end'])
            else:
                merged_flags.append(flag)

        metadata = {
            "duration_sec": duration_sec,
            "duration_str": self._format_mmss(duration_sec),
            "pixel_quality": f"{width}x{height} ({height}p)"
        }
        return merged_flags, final_annotated_path, metadata

    def evaluate_scoring_and_certificate(self, all_timeline_events):
        """
        Applies ONE-TIME category point allotment and strict U / UA / A rules:
        - Nudity: 5, Gore: 5, Strong abuse: 4, Violence: 4, Smoking or Drugs: 3, Mild Language: 1
        - U: Score <= 4 AND NO Nudity, Strong abuse, or Smoking or Drugs
        - UA: Score 5 to 11 (or <=4 with Smoking/Drugs) AND NO Nudity or Strong abuse
        - A: Score >= 12 OR contains Nudity or Strong abuse
        """
        detected_categories = set(ev["category"] for ev in all_timeline_events)
        
        category_breakdown = {}
        total_score = 0
        for cat, weight in CATEGORY_WEIGHTS.items():
            if cat in detected_categories:
                category_breakdown[cat] = weight # Allotted ONLY ONCE regardless of frequency
                total_score += weight
            else:
                category_breakdown[cat] = 0

        has_nudity = "Nudity" in detected_categories
        has_strong_abuse = "Strong abuse" in detected_categories
        has_smoking_drugs = "Smoking or Drugs" in detected_categories

        if total_score >= 12 or has_nudity or has_strong_abuse:
            certificate = "A"
        elif total_score <= 4 and not has_smoking_drugs:
            certificate = "U"
        else:
            certificate = "UA"

        return certificate, total_score, category_breakdown

    def build_report_and_llm_review(self, filename, metadata, all_events, certificate, total_score, category_breakdown):
        """Builds the timeline-based Report, auto Title/Description, and LLM Review + Suggestions."""
        clean_title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()

        # 1. Chronological Report based on Timeline
        report_lines = []
        for ev in all_events:
            t_start = self._format_mmss(ev['start'])
            t_end = self._format_mmss(ev['end'])
            report_lines.append(
                f"At timeline {t_start} - {t_end}: [{ev['category']}] — {ev['detail']}."
            )
        if not report_lines:
            report_lines.append("00:00 - End: No restricted visual or audio scenes detected across the timeline.")

        # 2. Auto-generated Movie Description
        active_cats = [c for c, pts in category_breakdown.items() if pts > 0]
        cat_summary = ", ".join(active_cats) if active_cats else "clean family-friendly content"
        auto_description = (
            f"{clean_title} ({metadata['pixel_quality']}, {metadata['duration_str']}) is an AI-screened feature "
            f"certified '{certificate}' with a one-time severity score of {total_score}. "
            f"Multi-agent inspection identified {cat_summary}."
        )

        # 3. LLM Review ("Why", "How", "For What")
        if active_cats:
            pts_str = ", ".join([f"{c} (+{category_breakdown[c]} pts)" for c in active_cats])
            why_text = f"WHY: Issued '{certificate}' because the timeline contains {pts_str}, totaling {total_score} severity points."
        else:
            why_text = "WHY: Issued 'U' because zero restricted categories occurred across the timeline (0 points)."

        how_text = (
            "HOW: Evaluated via YOLOv8 visual bounding-box tracking, Whisper English translation, "
            "and XLM-RoBERTa toxicity scoring using strict one-time category point allotment."
        )
        for_what_text = (
            f"FOR WHAT: {len(all_events)} total flagged timeline interval(s) across {len(active_cats)} unique violation category(s)."
        )
        llm_review = f"{why_text}\n\n{how_text}\n\n{for_what_text}"

        # 4. Director / Production Team Suggestions (When & Where to cut for UA and U)
        a_only_events = [
            f"{self._format_mmss(e['start'])}-{self._format_mmss(e['end'])} ({e['category']})"
            for e in all_events if e['category'] in ["Nudity", "Strong abuse", "Gore"]
        ]
        all_viol_events = [
            f"{self._format_mmss(e['start'])}-{self._format_mmss(e['end'])} ({e['category']})"
            for e in all_events if e['category'] != "Mild Language"
        ]

        option_ua = (
            f"To be eligible for a 'UA' Certificate: The production team must remove/avoid the severe scenes at "
            f"{', '.join(a_only_events) if a_only_events else ', '.join(all_viol_events)} "
            f"so that Nudity and Strong Abuse are eliminated and total points drop to 5–11."
        )
        option_u = (
            f"To be eligible for a 'U' Certificate: The director/production team must avoid/cut the flagged scenes at "
            f"{', '.join(all_viol_events) if all_viol_events else 'all flagged timestamps'} "
            f"(eliminating all Nudity, Gore, Strong Abuse, Violence, and Smoking/Drugs so total score is <= 4)."
        )

        return clean_title, auto_description, report_lines, llm_review, option_ua, option_u

    def generate_censor_manifest(self, video_path):
        video_filename = os.path.basename(video_path)

        # 1. Run Audio + Subtitle Agent & Vision Bounding-Box Agent
        audio_flags, vtt_subtitles, _ = self.process_audio_and_subtitles(video_path)
        video_flags, annotated_video_path, metadata = self.process_video_frames(video_path)

        # Combine and sort all events chronologically for Timeline & Report
        all_events = sorted(audio_flags + video_flags, key=lambda x: x["start"])

        # Log to RDS
        for ev in all_events:
            self.log_to_rds(video_filename, ev['start'], ev['end'], ev['category'], ev['score'], ev['action'])

        # 2. One-Time Category Scoring & Certificate Assignment
        certificate, total_score, category_breakdown = self.evaluate_scoring_and_certificate(all_events)

        # 3. Generate LLM Report, Review, Metadata, and Cut Suggestions
        title, description, report_lines, llm_review, option_ua, option_u = self.build_report_and_llm_review(
            video_filename, metadata, all_events, certificate, total_score, category_breakdown
        )

        manifest = {
            "file": video_filename,
            "title": title,
            "description": description,
            "duration_str": metadata["duration_str"],
            "duration_sec": metadata["duration_sec"],
            "pixel_quality": metadata["pixel_quality"],
            "censor_rating": certificate,
            "total_severity_score": total_score,
            "category_breakdown": category_breakdown,
            "timeline_events": all_events,
            "mute_intervals": audio_flags,
            "skip_intervals": video_flags,
            "report_lines": report_lines,
            "llm_review": llm_review,
            "suggestion_for_ua": option_ua,
            "suggestion_for_u": option_u,
            "vtt_subtitles": vtt_subtitles
        }

        return manifest, annotated_video_path