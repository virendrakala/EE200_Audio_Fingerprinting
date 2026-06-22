"""
Shazam-style audio fingerprinting — core engine (v2)

Database structure (database.pkl):
{
    'hashes': { (f1, f2, dt): [(song_name, t_anchor), ...], ... },
    'songs': {
        song_name: {
            'peaks': [(t_idx, f_idx), ...],   # FULL constellation for the song
            'n_frames': int,                   # total spectrogram time frames
            'n_hashes': int,
            'duration_sec': float,
        },
        ...
    }
}
"""
import numpy as np
import librosa
from scipy import signal as sp_signal
from scipy.ndimage import maximum_filter
import os
from collections import defaultdict

SONG_DIR = "songs_db"
SR = 22050
NPERSEG = 1024
NOVERLAP = 512
N_PEAKS = 30
FAN_OUT = 10
DT_MIN = 1
DT_MAX = 40
DF_MAX = 200


def load_audio(path, sr=SR, duration=None):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return y


def make_spectrogram(y, sr=SR):
    f, t, Sxx = sp_signal.spectrogram(y, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    Sdb = (10 * np.log10(Sxx + 1e-10)).astype(np.float32)
    return f, t, Sdb


def find_peaks(Sdb, n_peaks=N_PEAKS):
    local_max = maximum_filter(Sdb, size=(20, 20))
    peaks_mask = (Sdb == local_max)
    peaks = []
    n_time = Sdb.shape[1]
    for t_idx in range(n_time):
        col_peaks = np.where(peaks_mask[:, t_idx])[0]
        if len(col_peaks) == 0:
            continue
        col = Sdb[:, t_idx]
        ranked = col_peaks[np.argsort(col[col_peaks])[::-1]][:n_peaks]
        for f_idx in ranked:
            peaks.append((t_idx, int(f_idx)))
    peaks.sort()
    return peaks


def hash_peaks(peaks):
    hashes = []
    for i, (t1, f1) in enumerate(peaks):
        count = 0
        for j in range(i + 1, len(peaks)):
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < DT_MIN:
                continue
            if dt > DT_MAX:
                break
            if abs(f2 - f1) > DF_MAX:
                continue
            hashes.append(((f1, f2, dt), t1))
            count += 1
            if count >= FAN_OUT:
                break
    return hashes


def build_database(song_dir=SONG_DIR, progress_callback=None):
    """
    Build the full database: hashes for matching + full per-song constellations
    for the "full song fingerprint reconstruction" visualization.
    """
    hash_db = defaultdict(list)
    song_meta = {}
    songs = sorted([f for f in os.listdir(song_dir) if f.endswith('.mp3')])

    for i, fname in enumerate(songs):
        name = os.path.splitext(fname)[0]
        path = os.path.join(song_dir, fname)
        y = load_audio(path)
        duration_sec = len(y) / SR
        _, t_axis, Sdb = make_spectrogram(y)
        peaks = find_peaks(Sdb)
        hashes = hash_peaks(peaks)

        for h, t_anchor in hashes:
            hash_db[h].append((name, t_anchor))

        song_meta[name] = {
            'peaks': peaks,
            'n_frames': Sdb.shape[1],
            'n_hashes': len(hashes),
            'duration_sec': duration_sec,
            'filename': fname,
        }

        if progress_callback:
            progress_callback(i + 1, len(songs), name)
        else:
            print(f"  [{i+1}/{len(songs)}] {name}  ({len(peaks)} peaks, {len(hashes)} hashes)")

    return {'hashes': dict(hash_db), 'songs': song_meta}


def match(query_audio, db, top_k=5):
    """
    Returns: ranked (list of (song_name, score)), histograms (dict), peaks, hashes,
             song_offsets (dict song -> list of offsets, for the alignment-spike plot)
    """
    _, _, Sdb = make_spectrogram(query_audio)
    peaks = find_peaks(Sdb)
    query_hashes = hash_peaks(peaks)

    hash_db = db['hashes']
    # Use a dictionary to count occurrences of offsets directly instead of appending to a massive list.
    # This prevents allocating millions of integers in memory which leads to OOM crashes.
    song_offsets = defaultdict(lambda: defaultdict(int))
    for (h, t_q) in query_hashes:
        if h in hash_db:
            for (song_name, t_db) in hash_db[h]:
                song_offsets[song_name][t_db - t_q] += 1

    scores = {}
    histograms = {}
    for song_name, hist in song_offsets.items():
        if not hist:
            continue
        # Find the peak count directly from the dictionary values
        peak_count = max(hist.values())
        if peak_count < 3:
            continue
        scores[song_name] = peak_count
        # To maintain compatibility with app.py which expects (counts, bins), we mock it.
        # However app.py's plot_offset_histogram only needs histograms[song_name] to be plottable.
        histograms[song_name] = hist

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_k], histograms, peaks, query_hashes, song_offsets


def best_offset(histograms, song_name):
    """Return the offset (frame difference) where the histogram peak occurs."""
    if song_name not in histograms:
        return None
    hist = histograms[song_name]
    if not hist:
        return None
    # Find the offset with the maximum count
    best_off = max(hist, key=hist.get)
    return best_off
