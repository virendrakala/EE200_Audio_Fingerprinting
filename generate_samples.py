"""
generate_samples.py — Create 5 sample query clips from the song library.
Run once:  python generate_samples.py
"""

import os
import soundfile as sf
from fingerprint import load_audio, SAMPLE_RATE

SONG_DIR = "Q3_data"
OUT_DIR  = "samples"
CLIP_SEC = 10                   # each sample is 10 seconds

PICKS = [
    "Never Gonna Give You Up",
    "Hey Jude",
    "Two Of Us",
    "We Will Rock You",
    "While My Guitar Gently Weeps",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for i, name in enumerate(PICKS, 1):
        path = os.path.join(SONG_DIR, f"{name}.mp3")
        if not os.path.exists(path):
            print(f"  ⚠  {path} not found — skipping")
            continue

        y = load_audio(path)
        clip_len = CLIP_SEC * SAMPLE_RATE
        start = max(0, len(y) // 2 - clip_len // 2)      # take from the middle
        clip = y[start : start + clip_len]

        out_path = os.path.join(OUT_DIR, f"sample{i}.wav")
        sf.write(out_path, clip, SAMPLE_RATE)
        print(f"  sample{i}.wav  <-  {name}  ({len(clip)/SAMPLE_RATE:.1f} s)")

    print("Done.")


if __name__ == "__main__":
    main()
