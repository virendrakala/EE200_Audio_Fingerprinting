"""
build_db.py — Pre-index the song library into a fingerprint database.
Run once before starting the app:
    python build_db.py
"""

import pickle
import os
import matplotlib.pyplot as plt
import random
from fingerprint import index_songs

SONG_DIR  = "Q3_data"
DB_FILE   = "song_database.pkl"
META_FILE = "song_metadata.pkl"

def generate_thumbnails(metadata):
    os.makedirs("thumbnails", exist_ok=True)
    colors = ["#00d4ff", "#ffb84d", "#ff7373", "#ccff00", "#b266ff"]
    songs = sorted(metadata.keys())
    
    print(f"\nGenerating {len(songs)} thumbnails ...")
    for i, name in enumerate(songs):
        color = colors[i % len(colors)]
        peaks = metadata[name]["peaks"]
        plot_peaks = random.sample(peaks, min(len(peaks), 1000))
        
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.scatter([p[0] for p in plot_peaks], [p[1] for p in plot_peaks], s=0.5, c=color, alpha=0.8)
        ax.set_facecolor("#0e1117")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.patch.set_facecolor("#0e1117")
        plt.tight_layout(pad=0)
        
        fig.savefig(os.path.join("thumbnails", f"{name}.png"), format="png", bbox_inches='tight', pad_inches=0, facecolor="#0e1117")
        plt.close(fig)
    print("Thumbnails generated successfully.")


def main():
    def progress(i, total, name):
        if name == "done":
            print(f"\nDone — indexed {total} songs.")
        else:
            print(f"  [{i+1:2d}/{total}] {name}")

    print("Building fingerprint database …")
    hash_db, metadata = index_songs(SONG_DIR, progress_cb=progress)

    total_hashes = sum(len(v) for v in hash_db.values())
    print(f"  Total hashes in DB : {total_hashes:,}")
    print(f"  Unique hash keys   : {len(hash_db):,}")

    with open(DB_FILE, "wb") as f:
        pickle.dump(hash_db, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    generate_thumbnails(metadata)

    import os
    db_mb = os.path.getsize(DB_FILE) / 1e6
    print(f"  Saved {DB_FILE} ({db_mb:.1f} MB) and {META_FILE}")


if __name__ == "__main__":
    main()
