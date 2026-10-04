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
st.write("Lade dein Stream-Video hoch oder gib einen Twitch-Link ein: Die KI findet laute Momente, transkribiert das Audio und schneidet Shorts!")

# --- SIDEBAR EINSTELLUNGEN ---
st.sidebar.header("⚙️ Einstellungen")
twitch_username = st.sidebar.text_input("Twitch-Kanalname für Wasserzeichen:", value="twitch.tv/DeinKanal")
clip_dauer = st.sidebar.slider("Dauer je Short (Sekunden):", 5, 30, 12)
schwellenwert = st.sidebar.slider("Lautstärke-Empfindlichkeit (Spike):", 0.1, 1.0, 0.5)

# --- INPUTCENTER: UPLOAD ODER TWITCH LINK ---
st.subheader("1. Video bereitstellen")
tab1, tab2 = st.tabs(["🔗 Twitch-Link eingeben", "📁 Datei hochladen"])

video_path = None

with tab1:
    twitch_url = st.text_input("Twitch VOD oder Clip Link:")
    if st.button("Video von Twitch laden") and twitch_url:
        with st.spinner("Lade Video von Twitch herunter..."):
            ydl_opts = {
                'format': 'best',
                'outtmpl': 'downloaded_stream.mp4',
                'overwrites': True
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([twitch_url])
                video_path = "downloaded_stream.mp4"
                st.success("Download erfolgreich!")
            except Exception as e:
                st.error(f"Fehler beim Download: {e}")

with tab2:
    uploaded_file = st.file_uploader("Stream-Video hochladen (.mp4)", type=["mp4", "mov"])
    if uploaded_file is not None:
        video_path = f"temp_{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Datei hochgeladen!")

# --- VERARBEITUNG ---
if video_path and os.path.exists(video_path):
    st.subheader("2. Analyse & Shorts-Erstellung")
    if st.button("🚀 Shorts jetzt generieren"):
        with st.spinner("KI analysiert das Video... Bitte warten..."):
            st.info("Verarbeitung läuft...")
