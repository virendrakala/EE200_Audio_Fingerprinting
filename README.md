# EE200 Q3A/B — Audio Fingerprinting Identifier (v2)

## Files
- `app.py` — Streamlit app: Library / Identify / Batch tabs.
- `fingerprint.py` — core engine: spectrogram, constellation peaks, paired-hash
  fingerprinting, database building, matching, and `best_offset()` (used to
  position the highlighted window in the full-song reconstruction view).
- `database.pkl` — pre-built database for all 50 songs (67 MB): per-hash lookup
  table **plus** each song's full constellation and metadata (duration, peak
  count, hash count) — the richer structure needed for the Library tab and the
  full-song fingerprint reconstruction view.
- `samples/sample1.mp3` … `sample5.mp3` — real 30-second clips cut from 5
  different songs in the database, for the "Try a sample" feature.
- `requirements.txt` — Python (pip) dependencies.
- `packages.txt` — system-level (apt) dependencies — `ffmpeg` + `libsndfile1`,
  needed for MP3 decoding on Streamlit Cloud.

## A note on a real bug found and fixed during testing

The first "We Will Rock You" sample clip was cut from the song's famous
stomp-stomp-clap intro. That section is so percussive (broadband noise-like
content) that the peak-picker found **more than double** the normal number of
constellation peaks (2,999 vs. ~1,300 typical), which produced a combinatorial
explosion of spurious hash collisions — every song in the database scored
500,000+ "matches," and a wrong song occasionally narrowly out-scored the
correct one. The fix: re-cut the sample from a less percussive section
(starting at 60s instead of 5s) — peak count returned to normal (~1,250) and
the match became correct and confident (74x margin over the runner-up).

This is a genuine limitation of the constellation/peak-picking approach worth
knowing: **very dense, broadband, percussive audio (drum solos, claps, white
noise) degrades match confidence**, because the fixed `N_PEAKS`-per-frame cap
combined with extremely peaky/noisy content produces far more candidate hashes
than melodic content does, diluting the signal-to-collision ratio.


## Running locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Re-building the database from scratch
```python
import fingerprint as fp, pickle
db = fp.build_database()   # expects songs_db/ folder with the 50 .mp3 files
with open('database.pkl', 'wb') as f:
    pickle.dump(db, f)
```
The app auto-builds it on first run if `database.pkl` is missing and
`fp.SONG_DIR` ("songs_db") is present (see `get_database()` in `app.py`).

## Fingerprinting parameters (in `fingerprint.py`)
| Parameter | Value | Meaning |
|---|---|---|
| `SR` | 22050 Hz | audio sample rate used throughout |
| `NPERSEG` | 1024 | spectrogram FFT window length |
| `NOVERLAP` | 512 | spectrogram window overlap |
| `N_PEAKS` | 30 | max constellation peaks kept per time frame |
| `FAN_OUT` | 10 | max pairs formed per anchor peak |
| `DT_MIN`, `DT_MAX` | 1, 40 | allowed time-frame gap between paired peaks |
| `DF_MAX` | 200 | allowed frequency-bin gap between paired peaks |
