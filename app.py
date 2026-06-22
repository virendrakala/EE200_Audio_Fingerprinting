"""
EE200 Q3B — 'Zapp-tain America'
Full-featured Shazam-style audio identifier:
  - Library tab: browse the 50 indexed songs
  - Identify tab: single-clip ID with "try a sample", pipeline timing,
    spectrogram/constellation, full-song fingerprint reconstruction,
    and the offset-histogram "alignment spike" proof
  - Batch tab: identify many clips at once -> results.csv
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, os, io, csv, tempfile, time
from collections import defaultdict

import fingerprint as fp

DB_PATH = "database.pkl"
SAMPLES_DIR = "samples"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")

# ── Dark terminal theme (CSS) ────────────────────────────────────────────────
DARK_CSS = """
<style>
:root {
    --bg: #0a0e14;
    --bg-panel: #0f1420;
    --accent: #5eead4;
    --accent-dim: #2dd4bf;
    --warn: #f59e0b;
    --text: #e2e8f0;
    --text-dim: #64748b;
    --border: #1e293b;
}
.stApp {
    background-color: var(--bg);
    color: var(--text);
}
section[data-testid="stSidebar"] { background-color: var(--bg-panel); }

h1, h2, h3, h4 { color: var(--text) !important; font-family: 'Courier New', monospace; }

.eyebrow {
    color: var(--accent);
    font-family: 'Courier New', monospace;
    letter-spacing: 0.15em;
    font-size: 0.75rem;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.step-block {
    border-left: 3px solid var(--accent);
    padding-left: 1rem;
    margin: 1.5rem 0 1rem 0;
}
.match-hero {
    background: linear-gradient(135deg, #0d1f1a 0%, #0a1614 100%);
    border: 1px solid var(--accent-dim);
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
}
.match-hero h1 {
    color: var(--accent) !important;
    font-size: 2.4rem;
    margin: 0.2rem 0;
}
.cluster-score {
    color: var(--warn);
    font-weight: bold;
}
.pipeline-stage {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.8rem;
    text-align: center;
}
.pipeline-stage .label {
    color: var(--accent);
    font-family: 'Courier New', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.pipeline-stage .ms {
    font-size: 1.4rem;
    font-weight: bold;
    color: var(--text);
}
.pipeline-stage .detail {
    color: var(--text-dim);
    font-size: 0.7rem;
}
.candidate-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.4rem 0;
    font-family: 'Courier New', monospace;
}
.candidate-bar-bg {
    flex: 1;
    background: var(--border);
    border-radius: 3px;
    height: 10px;
    overflow: hidden;
}
.candidate-bar-fill {
    background: var(--accent);
    height: 100%;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ── Database loading (cached) ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fingerprint database…")
def get_database():
    import pickle
    # Load the friend's split database format to prevent memory fragmentation during load
    with open('song_database.pkl', 'rb') as f:
        hashes = pickle.load(f)
    with open('song_metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
        
    # Map friend's metadata keys to our UI's expected keys so our UI stays exactly as it is
    for song_name, m in meta.items():
        if 'duration_sec' not in m:
            m['duration_sec'] = m.get('duration', 0)
        if 'n_frames' not in m:
            m['n_frames'] = int(m.get('duration', 0) * fp.SR / fp.HOP_LENGTH)
        if 'n_hashes' not in m:
            m['n_hashes'] = m.get('num_hashes', 0)
            
    return {'hashes': hashes, 'songs': meta}


def load_audio_file(file_obj, sr=fp.SR):
    import io
    import librosa
    
    if isinstance(file_obj, bytes):
        mem_stream = io.BytesIO(file_obj)
    else:
        file_obj.seek(0)
        # Detach from Tornado's internal buffer to prevent memory leaks
        mem_stream = io.BytesIO(file_obj.read())
        
    try:
        y, _ = librosa.load(mem_stream, sr=sr, mono=True)
    finally:
        mem_stream.close()
        
    return y


# ── Timed pipeline: runs each stage separately so we can report ms per stage ─
def timed_identify(audio_bytes, db):
    timings = {}

    t0 = time.perf_counter()
    y = load_audio_file(audio_bytes)
    t1 = time.perf_counter()
    timings['load'] = (t1 - t0) * 1000

    f_ax, t_ax, Sdb = fp.make_spectrogram(y)
    t2 = time.perf_counter()
    timings['spectrogram'] = (t2 - t1) * 1000
    timings['spectrogram_shape'] = Sdb.shape

    peaks = fp.find_peaks(Sdb)
    t3 = time.perf_counter()
    timings['constellation'] = (t3 - t2) * 1000
    timings['n_peaks'] = len(peaks)

    hashes = fp.hash_peaks(peaks)
    t4 = time.perf_counter()
    timings['hashing'] = (t4 - t3) * 1000
    timings['n_hashes'] = len(hashes)

    hash_db = db['hashes']
    song_offsets = defaultdict(list)
    for (h, t_q) in hashes:
        if h in hash_db:
            for (song_name, t_db) in hash_db[h]:
                song_offsets[song_name].append(t_db - t_q)
    t5 = time.perf_counter()
    timings['db_lookup'] = (t5 - t4) * 1000
    timings['n_tracks_hit'] = len(song_offsets)

    scores, histograms = {}, {}
    for song_name, offsets in song_offsets.items():
        if len(offsets) < 3:
            continue
        counts, bins = np.histogram(offsets, bins=300)
        scores[song_name] = int(counts.max())
        histograms[song_name] = (counts, bins)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    t6 = time.perf_counter()
    timings['scoring'] = (t6 - t5) * 1000
    timings['total'] = (t6 - t0) * 1000

    return {
        'y': y, 'f_ax': f_ax, 't_ax': t_ax, 'Sdb': Sdb,
        'peaks': peaks, 'hashes': hashes,
        'song_offsets': song_offsets, 'scores': scores,
        'histograms': histograms, 'ranked': ranked,
        'timings': timings,
    }


# ── Plotting helpers (dark theme to match the terminal aesthetic) ───────────
PLOT_BG = "#0a0e14"
PLOT_FG = "#5eead4"
PLOT_FG2 = "#94a3b8"

def _dark_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=PLOT_FG2, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#1e293b')
    ax.xaxis.label.set_color(PLOT_FG2)
    ax.yaxis.label.set_color(PLOT_FG2)
    ax.title.set_color('#e2e8f0')
    return fig, ax


def plot_spectrogram(t_ax, f_ax, Sdb, title="Spectrogram"):
    fig, ax = _dark_fig((6, 3.2))
    t_max = t_ax[-1] if len(t_ax) else 1
    f_max = f_ax[-1] if len(f_ax) else 1
    ax.imshow(Sdb, aspect='auto', origin='lower', cmap='magma', extent=[0, t_max, 0, f_max])
    ax.set_ylim(0, 5000)
    ax.set_xlabel('time (s)'); ax.set_ylabel('frequency (Hz)')
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig


def plot_constellation(t_ax, Sdb, peaks, title=None):
    fig, ax = _dark_fig((6, 3.2))
    peak_t = [t_ax[ti] for ti, fi in peaks]
    peak_f = [fi * fp.SR / 2 / Sdb.shape[0] for ti, fi in peaks]
    ax.scatter(peak_t, peak_f, s=4, c=PLOT_FG, alpha=0.8)
    ax.set_xlim(0, t_ax[-1] if len(t_ax) else 1)
    ax.set_ylim(0, 5000)
    ax.set_xlabel('time (s)'); ax.set_ylabel('frequency (Hz)')
    ax.set_title(title or f"{len(peaks)} peaks", fontsize=10)
    plt.tight_layout()
    return fig


def plot_full_song_fingerprint(song_meta, song_name, query_n_frames, best_offset_frames):
    """
    Reconstruct the FULL song's constellation from its stored peaks, and
    highlight the window where the query clip sits (using the matched offset).
    """
    info = song_meta[song_name]
    peaks = info['peaks']
    n_frames = info['n_frames']

    fig, ax = _dark_fig((9, 3.6))
    xs = [p[0] for p in peaks]
    ys = [p[1] for p in peaks]
    ax.scatter(xs, ys, s=2, c=PLOT_FG, alpha=0.5)

    # Highlight the matched window: [best_offset_frames, best_offset_frames + query_n_frames]
    if best_offset_frames is not None:
        x0 = max(0, best_offset_frames)
        x1 = min(n_frames, best_offset_frames + query_n_frames)
        ax.axvspan(x0, x1, color='#f59e0b', alpha=0.25)
        ax.axvline(x0, color='#f59e0b', linewidth=1)
        ax.axvline(x1, color='#f59e0b', linewidth=1)

    ax.set_xlim(0, n_frames)
    ax.set_xlabel('time (frames)'); ax.set_ylabel('freq bin')
    ax.set_title(f'Full fingerprint reconstruction — "{song_name}"  ({len(peaks)} stored peaks)', fontsize=10)
    plt.tight_layout()
    return fig


def plot_offset_histogram(histograms, song_name):
    hist = histograms[song_name]
    fig, ax = _dark_fig((9, 3.2))
    
    # Extract keys and values from the dictionary
    offsets = list(hist.keys())
    counts = list(hist.values())
    max_count = max(counts)
    
    colors = ['#f59e0b' if c == max_count else PLOT_FG for c in counts]
    ax.bar(offsets, counts, width=1.0, color=colors, edgecolor='none')
    
    peak_offset = max(hist, key=hist.get)
    # Safely compute arrow position
    offset_range = max(offsets) - min(offsets) if offsets else 1
    
    ax.annotate(f'{max_count:,} hashes\nalign here',
                xy=(peak_offset, max_count),
                xytext=(peak_offset + offset_range * 0.15, max_count * 0.8),
                color='#f59e0b', fontsize=9,
                arrowprops=dict(color='#f59e0b', arrowstyle='->'))
    ax.set_xlabel('time offset (database frame − query frame)')
    ax.set_ylabel('# hashes')
    ax.set_title('The alignment spike — chance matches scatter, a true match converges', fontsize=10)
    plt.tight_layout()
    return fig


def render_candidate_bars(ranked, max_show=5):
    if not ranked:
        st.info("No candidates cleared the minimum hash threshold.")
        return
    max_score = ranked[0][1]
    for name, score in ranked[:max_show]:
        pct = 100 * score / max_score if max_score else 0
        st.markdown(f"""
        <div class="candidate-row">
            <div style="width:220px; color:#e2e8f0;">{name}</div>
            <div class="candidate-bar-bg"><div class="candidate-bar-fill" style="width:{pct}%;"></div></div>
            <div style="width:60px; text-align:right; color:#94a3b8;">{score:,}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="eyebrow">SIGNALS, SYSTEMS &amp; NETWORKS · PROJECT DEMO</div>', unsafe_allow_html=True)
st.markdown("# EE200: Audio Fingerprinting")

db = get_database()
n_songs = len(db['songs'])
n_hashes = len(db['hashes'])
st.caption(f"Database ready — **{n_songs} songs** indexed into **{n_hashes:,} fingerprint hashes**")

tab_library, tab_identify, tab_batch = st.tabs(["LIBRARY", "IDENTIFY", "BATCH"])

# ══════════════════════════════════════════════════════════════════════════════
# LIBRARY TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_library:
    st.markdown("### Indexed song library")
    st.caption("Every track below has been fingerprinted and is searchable from the Identify and Batch tabs.")

    song_items = sorted(db['songs'].items(), key=lambda kv: kv[0])
    search = st.text_input("Filter by name", "")
    if search:
        song_items = [(n, m) for n, m in song_items if search.lower() in n.lower()]

    for name, meta in song_items:
        cols = st.columns([3, 1, 1, 1, 3])
        cols[0].markdown(f"**{name}**")
        cols[1].markdown(f"<span style='color:#94a3b8'>{meta['duration_sec']:.0f}s</span>", unsafe_allow_html=True)
        cols[2].markdown(f"<span style='color:#94a3b8'>{len(meta['peaks']):,} peaks</span>", unsafe_allow_html=True)
        cols[3].markdown(f"<span style='color:#94a3b8'>{meta['n_hashes']:,} hashes</span>", unsafe_allow_html=True)
        song_path = os.path.join(fp.SONG_DIR, meta['filename'])
        if os.path.exists(song_path):
            cols[4].audio(song_path)
        else:
            cols[4].caption("(audio file not bundled with this deployment)")


# ══════════════════════════════════════════════════════════════════════════════
# IDENTIFY TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_identify:
    st.markdown("### Upload a query clip")
    uploaded = st.file_uploader("MP3 or WAV", type=["mp3", "wav"], key="identify_uploader")

    st.markdown("##### Or try a sample")
    sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith('.mp3')]) if os.path.isdir(SAMPLES_DIR) else []

    chosen_sample = None
    for sf_name in sample_files:
        c1, c2, c3 = st.columns([1.2, 5, 1])
        c1.markdown(f"`{sf_name}`")
        c2.audio(os.path.join(SAMPLES_DIR, sf_name))
        if c3.button("Try", key=f"try_{sf_name}"):
            chosen_sample = sf_name

    audio_bytes = None
    source_label = None
    if uploaded is not None:
        audio_bytes = uploaded.read()
        source_label = uploaded.name
    elif chosen_sample is not None:
        with open(os.path.join(SAMPLES_DIR, chosen_sample), 'rb') as f:
            audio_bytes = f.read()
        source_label = chosen_sample

    if audio_bytes is not None:
        st.markdown("---")
        st.caption(f"Identifying: **{source_label}**")

        try:
            with st.spinner("Running fingerprint pipeline…"):
                result = timed_identify(audio_bytes, db)

            timings = result['timings']
            ranked = result['ranked']

            # ---- Pipeline timing breakdown ----
            stage_cols = st.columns(5)
            stage_defs = [
                ("SPECTROGRAM", timings['spectrogram'], f"{timings['spectrogram_shape'][0]}×{timings['spectrogram_shape'][1]}"),
                ("CONSTELLATION", timings['constellation'], f"{timings['n_peaks']} peaks"),
                ("HASHING", timings['hashing'], f"{timings['n_hashes']:,} hashes"),
                ("DB LOOKUP", timings['db_lookup'], f"{timings['n_tracks_hit']} tracks"),
                ("SCORING", timings['scoring'], f"offset {0}"),
            ]
            for col, (label, ms, detail) in zip(stage_cols, stage_defs):
                col.markdown(f"""
                <div class="pipeline-stage">
                    <div class="label">{label}</div>
                    <div class="ms">{ms:.0f} ms</div>
                    <div class="detail">{detail}</div>
                </div>
                """, unsafe_allow_html=True)
            st.caption(f"total {timings['total']:.0f} ms")

            st.markdown("---")

            if not ranked or ranked[0][1] < 10:
                st.error("No match found — try a longer or cleaner clip. (Confidence threshold not met)")
                if ranked:
                    st.caption(f"Top candidate '{ranked[0][0]}' only scored {ranked[0][1]} aligned hashes. A true match requires at least 10.")
            else:
                top_name, top_score = ranked[0]
                runner_up = ranked[1][1] if len(ranked) > 1 else 1
                margin = top_score / max(runner_up, 1)

                st.markdown(f"""
                <div class="match-hero">
                    <div class="eyebrow" style="color:#5eead4;">MATCH FOUND</div>
                    <h1>{top_name}</h1>
                    <div>cluster score <span class="cluster-score">{top_score:,}</span>
                         &nbsp;·&nbsp; <span class="cluster-score">{margin:.0f}×</span> the runner-up</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("##### Candidate scores")
                render_candidate_bars(ranked, max_show=5)

                # ---- Step 1: spectrogram + constellation ----
                st.markdown("""
                <div class="step-block">
                    <div class="eyebrow">STEP 1 · FEATURE EXTRACTION</div>
                    <h4>From spectrogram to constellation</h4>
                    <p style="color:#94a3b8;">The clip was converted into a time-frequency map (left); brighter means
                    louder at that frequency and moment. From that rich image, only the
                    <b style="color:#e2e8f0;">strongest peaks</b> were kept (right) — discarding amplitude and phase
                    makes the fingerprint robust to EQ, volume changes, and noise.</p>
                </div>
                """, unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                fig_spec = plot_spectrogram(result['t_ax'], result['f_ax'], result['Sdb'])
                c1.pyplot(fig_spec)
                import matplotlib.pyplot as plt
                plt.close(fig_spec)
                
                fig_const = plot_constellation(result['t_ax'], result['Sdb'], result['peaks'], title=f"{len(result['peaks'])} peaks")
                c2.pyplot(fig_const)
                plt.close(fig_const)

                # ---- Step 2: full-song fingerprint reconstruction ----
                st.markdown(f"""
                <div class="step-block">
                    <div class="eyebrow">STEP 2 · DATABASE SEARCH</div>
                    <h4>Where in the song?</h4>
                    <p style="color:#94a3b8;">The <b style="color:#e2e8f0;">{timings['n_hashes']:,} fingerprint hashes</b>
                    were looked up against every indexed track. Below is the full fingerprint of
                    <i>{top_name}</i> reconstructed from the database — each dot is a stored hash anchor.
                    The highlighted window is exactly where the query clip sits inside the full song.</p>
                </div>
                """, unsafe_allow_html=True)

                offset = fp.best_offset(result['histograms'], top_name)
                fig_full = plot_full_song_fingerprint(
                    db['songs'], top_name, result['Sdb'].shape[1],
                    int(offset) if offset is not None else None
                )
                st.pyplot(fig_full)
                plt.close(fig_full)

                # ---- Step 3: offset histogram (the proof) ----
                st.markdown("""
                <div class="step-block">
                    <div class="eyebrow">STEP 3 · THE PROOF</div>
                    <h4>The alignment spike</h4>
                    <p style="color:#94a3b8;">Every matched hash votes for a time offset (database frame minus query
                    frame). Chance matches scatter votes randomly, forming a flat noise floor. A genuine match
                    makes them converge into a single sharp spike — that spike cannot be a coincidence.</p>
                </div>
                """, unsafe_allow_html=True)
                fig_hist = plot_offset_histogram(result['histograms'], top_name)
                st.pyplot(fig_hist)
                plt.close(fig_hist)
        except Exception as e:
            st.error(f"Error analyzing audio: The file may be corrupted, too short, or in an unsupported format. Details: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# BATCH TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Identify many clips at once")
    st.caption(
        "Upload a set of query clips. Each is identified against the currently indexed "
        "library, and the results are written to a standardised `results.csv` with columns "
        "`filename, prediction`. The prediction is the matched track's filename without its "
        "extension, or `none` when no candidate clears the confidence threshold."
    )

    CONFIDENCE_THRESHOLD = 10  # minimum top-bin score to accept a match

    batch_files = st.file_uploader(
        "MP3 or WAV files", type=["mp3", "wav"], accept_multiple_files=True, key="batch_uploader"
    )

    if batch_files and st.button("Run batch", type="primary"):
        try:
            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, f in enumerate(batch_files):
                status.text(f"Identifying {f.name} …")
                
                # Safety Limit: Skip excessively large files (e.g. > 20MB) to prevent Streamlit OOM
                if f.size > 20 * 1024 * 1024:
                    prediction = "File too large (skipped)"
                else:
                    try:
                        y = load_audio_file(f)
                        match_result = fp.match(y, db, top_k=1)
                        ranked = match_result[0]
                        if ranked and ranked[0][1] >= CONFIDENCE_THRESHOLD:
                            prediction = ranked[0][0]
                        else:
                            prediction = "none"
                        
                        del y
                        del match_result
                        del ranked
                    except Exception as e:
                        prediction = f"Error: {str(e)}"
                
                results.append((os.path.splitext(f.name)[0], prediction))
                
                # Use integer progress (0 to 100) to avoid any float type-mismatch errors
                pct = int((i + 1) * 100 / len(batch_files))
                progress.progress(min(100, pct))
                
                # Force garbage collection to prevent memory fragmentation across multiple batch runs
                import gc
                gc.collect()
                
                # Yield the thread briefly so Streamlit Cloud can run health checks
                import time
                time.sleep(0.1)

            status.text("Done.")
            st.session_state.batch_results = results
            st.session_state.batch_file_names = [f.name for f in batch_files]
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    if "batch_results" in st.session_state and batch_files and [f.name for f in batch_files] == st.session_state.get("batch_file_names", []):
        try:
            results = st.session_state.batch_results
            st.markdown("##### Results")
            n_matched = sum(1 for _, p in results if p != "none")
            for fname, pred in results:
                color = "#5eead4" if pred != "none" else "#64748b"
                st.markdown(
                    f"<div style='font-family:monospace;'>{fname} &nbsp;→&nbsp; "
                    f"<span style='color:{color};'>{pred}</span></div>",
                    unsafe_allow_html=True
                )
            st.caption(f"{n_matched} / {len(results)} clips matched to a track "
                       f"({len(results)-n_matched} returned `none`).")

            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["filename", "prediction"])
            for fname, pred in results:
                writer.writerow([fname, pred])
            
            st.download_button(
                "Download results.csv",
                data=csv_buf.getvalue().encode('utf-8'),
                file_name="results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error rendering results: {str(e)}")
# Defensive EOF check
