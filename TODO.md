# 📋 TODO.md — HighlightClipper

---

## ✅ COMPLETED

- [x] `mkv_converter.py` – Converts `.mkv` to `.mp4` format
- [x] `chunker.py` – Chunks videos into smaller pieces for analysis
- [x] `game_classifier.py` – Detects whether a chunk is Valorant or another game
- [x] `killfeed_detector.py` – Detects Valorant killfeed using template matching
- [x] `headshot_audio.py` – Detects headshot audio cues using trained classifier
- [x] `audio_spike_detector.py` – Detects general audio spikes as proxy for action
- [x] `highlight_filter_and_trimmer.py` – Trims and saves clips around convergent events
- [x] `clip_scorer.py` – Scores clips based on audio/visual convergence
- [x] `temporal_convergence_scorer.py` – Aligns events to determine highlight intensity
- [x] `highlight_logger.py` – Logs detected events with timestamps
- [x] `run_all.py` – Orchestrates full pipeline from raw videos to highlight outputs
- [x] `self_updating_trainer.py` – Triggers pipeline improvements after every 3 videos
- [x] `.env` and OpenAI/Whisper API integration (optional)

---

## 🧠 IN PROGRESS

- [ ] `visual_headshot_flash_detector.py` – Detect flash animation for confirmed headshots  
- [ ] `score_fusion.py` – Normalize and combine headshot flash + audio + killfeed signal
- [ ] `highlight_feedback_logger.py` – Log which clips were good/bad to improve AI trainer
- [ ] `auto_config_tuner.py` – Adjust internal settings based on clip outcome feedback

---

## 🗺️ UPCOMING / PLANNED

- [ ] `storyboard_generator.py`  
  - Uses transcripts + timestamps to outline structure of highlights  
  - Export as `.json` or `.md` for review/edit before final assembly

- [ ] `uploader.py`  
  - Auto-upload clips to TikTok, YouTube Shorts, and Instagram Reels  
  - Optional: queue for manual review before post

- [ ] `hud_verifier.py`  
  - Detect “spectating” HUD indicator to avoid clipping non-player footage

- [ ] `highlight_ranker.py`  
  - Rank top clips per video session based on total score + novelty  
  - Tag with metadata: kills, headshots, streaks, timestamps, etc.

- [ ] `game_recognition_trainer.py`  
  - Improves per-chunk classification to detect more games beyond Valorant  
  - Build training dataset from existing labeled chunks

---

## 💡 IDEAS

- Auto-caption generator w/ Whisper + visual alignment  
- Support for Apex, LoL, Overwatch kill detection  
- Web dashboard for reviewing clips, scores, and metrics  
- Self-hosted transcription + tagging to reduce OpenAI API dependence