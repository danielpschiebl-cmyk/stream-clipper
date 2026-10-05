import streamlit as st
import librosa
import numpy as np
import whisper
import os
import yt_dlp

try:
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.video.VideoClip import TextClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

# --- DASHBOARD KONFIGURATION ---
st.set_page_config(page_title="KI Stream Clipper", layout="wide")
st.title("🎬 KI Stream Clipper & Shorts-Generator")
st.write("Lade dein Stream-Video hoch oder gib einen Twitch-Link ein (empfohlen: Twitch-Clips oder kurze VODs)!")

# --- SESSION STATE INITIALISIERUNG ---
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# --- SIDEBAR EINSTELLUNGEN ---
st.sidebar.header("⚙️ Einstellungen")
twitch_username = st.sidebar.text_input("Twitch-Kanalname für Wasserzeichen:", value="twitch.tv/DanCliffX")
clip_dauer = st.sidebar.slider("Dauer je Short (Sekunden):", 5, 30, 20)
schwellenwert = st.sidebar.slider("Lautstärke-Empfindlichkeit (Spike):", 0.1, 1.0, 0.5)

# --- INPUTCENTER: UPLOAD ODER TWITCH LINK ---
st.subheader("1. Video bereitstellen")
tab1, tab2 = st.tabs(["🔗 Twitch-Link eingeben", "📁 Datei hochladen"])

with tab1:
    twitch_url = st.text_input("Twitch VOD oder Clip Link:")
    if st.button("Video von Twitch laden") and twitch_url:
        with st.spinner("Lade Video von Twitch herunter (speicherschonend)..."):
            # VODs/Clips in komprimierter Qualität herunterladen, um RAM-Abstürze zu verhindern
            ydl_opts = {
                'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]/best',
                'outtmpl': 'downloaded_stream.mp4',
                'overwrites': True,
                'max_filesize': 300 * 1024 * 1024  # Max 300MB Schutz
            }
            try:
                if os.path.exists("downloaded_stream.mp4"):
                    os.remove("downloaded_stream.mp4")
                    
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([twitch_url])
                st.session_state.video_path = "downloaded_stream.mp4"
                st.success("Download erfolgreich!")
            except Exception as e:
                st.error(f"Fehler beim Download (Datei eventuell zu groß): {e}")

with tab2:
    uploaded_file = st.file_uploader("Stream-Video hochladen (.mp4)", type=["mp4", "mov"])
    if uploaded_file is not None:
        path = f"temp_{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.video_path = path
        st.success("Datei hochgeladen!")

# Anzeige, wenn ein Video bereitsteht
if st.session_state.video_path and os.path.exists(st.session_state.video_path):
    st.info(f"Bereit zur Verarbeitung: `{st.session_state.video_path}`")
    
    st.subheader("2. Analyse & Shorts-Erstellung")
    if st.button("🚀 Shorts jetzt generieren"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. Audio-Analyse
        status_text.text("🔊 Analysiere Audiospur auf laute Highlights...")
        progress_bar.progress(20)
        
        try:
            y, sr = librosa.load(st.session_state.video_path, sr=None, duration=180) # Max 3 Min scannen
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.times_like(rms, sr=sr)
            
            spike_threshold = np.max(rms) * schwellenwert
            spikes = np.where(rms > spike_threshold)[0]
            
            if len(spikes) > 0:
                best_spike_idx = spikes[np.argmax(rms[spikes])]
                start_time = max(0, times[best_spike_idx] - (clip_dauer / 2))
            else:
                start_time = 0
            end_time = start_time + clip_dauer
        except Exception as e:
            start_time, end_time = 0, clip_dauer

        # 2. Transkription mit Whisper
        status_text.text("🎙️ KI transkribiert Audio (Whisper tiny)...")
        progress_bar.progress(50)
        try:
            model = whisper.load_model("tiny")
            result = model.transcribe(st.session_state.video_path)
            transcript_text = result.get("text", "Stream Highlight")
        except Exception as e:
            transcript_text = "Highlight"

        # 3. Videoschnitt im 9:16 Format
        status_text.text("🎬 Schneide Short im 9:16 Format...")
        progress_bar.progress(75)
        
        try:
            clip = VideoFileClip(st.session_state.video_path).subclip(start_time, end_time)
            
            w, h = clip.size
            target_w = int(h * (9 / 16))
            if target_w < w:
                crop_x1 = (w - target_w) // 2
                clip_cropped = clip.crop(x1=crop_x1, width=target_w)
            else:
                clip_cropped = clip

            output_path = "generated_short.mp4"
            clip_cropped.write_videofile(output_path, codec="libx264", audio_codec="aac", preset="ultrafast")
            
            progress_bar.progress(100)
            status_text.text("✅ Fertig!")
            
            st.success("Dein Short wurde erfolgreich erstellt!")
            st.video(output_path)
            
            with open(output_path, "rb") as file:
                st.download_button(
                    label="⬇ Short herunterladen",
                    data=file,
                    file_name="stream_short.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"Fehler bei der Videoerstellung: {e}")
