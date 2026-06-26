import app
import os
import io

db = app.get_database()
songs = os.listdir('songs_db')
if songs:
    test_song = songs[0]
    with open(os.path.join('songs_db', test_song), 'rb') as f:
        audio_bytes = f.read()
    print(f"Testing with {test_song}")
    try:
        res = app.timed_identify(audio_bytes, db)
        top_name = res['ranked'][0][0]
        fig = app.plot_offset_histogram(res['histograms'], top_name)
        print("Success!")
    except Exception as e:
        import traceback
        traceback.print_exc()
else:
    print("No songs found")
