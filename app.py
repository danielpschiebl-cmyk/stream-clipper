import streamlit as st
import librosa
import numpy as np
import whisper
import os

try:
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.video.VideoClip import TextClip
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip


# --- DASHBOARD KONFIGURATION ---
st.set_page_config(page_title="KI Stream Clipper", layout="wide")
st.title("🎬 KI Stream Clipper & Shorts-Generator")
st.write("Lade deinen Stream hoch: Die KI findet laute Momente, transkribiert das Audio, schneidet 9:16 Shorts mit Wasserzeichen & Hook und erstellt Upload-Texte.")

# --- SIDEBAR EINSTELLUNGEN ---
st.sidebar.header("⚙️ Einstellungen")
twitch_username = st.sidebar.text_input("Twitch-Kanalname für Wasserzeichen:", value="twitch.tv/DeinKanal")
clip_dauer = st.sidebar.slider("Dauer je Short (Sekunden)", 5, 30, 12)
schwellenwert = st.sidebar.slider("Lautstärke-Empfindlichkeit (Spike)", 0.1, 1.0, 0.5)

# --- HOOK & METADATEN GENERATOR ---
def generiere_hook_und_titel(transkript_text):
    text = transkript_text.strip()
    if not text or len(text) < 5:
        text = "Unglaublicher Stream Moment!"
    
    hooks = [
        "DAS ging komplett schief! 😱",
        "WIE HAT ER DAS GEMACHT?! 🔥",
        "ABSOLUTER CHAOS MOMENT 💥",
        "Das müsst ihr sehen! 👀"
    ]
    selected_hook = np.random.choice(hooks)

    titel_vorschlaege = [
        f"{selected_hook} | {text[:35]}...",
        f"Kranker Clip auf Twitch! 🔥 | {text[:35]}..."
    ]
    
    hashtags = "#gaming #twitchgermany #streamer #highlight #shorts #fyp #viral"
    return selected_hook, titel_vorschlaege, hashtags

# --- MAIN UPLOAD & VERARBEITUNG ---
video_datei = st.file_uploader("1. Stream-Video hochladen (.mp4)", type=["mp4", "mov"])

if video_datei:
    with open("stream_input.mp4", "wb") as f:
        f.write(video_datei.getbuffer())
    st.success("Stream-Video erfolgreich geladen!")

    if st.button("🚀 Stream analysieren & Shorts generieren"):
        with st.spinner("1/3 Analysiere Audio & Lautstärke-Peaks..."):
            y, sr = librosa.load("stream_input.mp4", sr=None)
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.times_like(rms, sr=sr)
            target_rms = np.max(rms) * schwellenwert
            loud_peaks = times[np.where(rms > target_rms)[0]]

        with st.spinner("2/3 Transkribiere Audio mit OpenAI Whisper..."):
            model = whisper.load_model("base")
            audio_path = "temp_audio.mp3"
            video_temp = VideoFileClip("stream_input.mp4")
            video_temp.audio.write_audiofile(audio_path, logger=None)
            transkript_data = model.transcribe(audio_path, language="de")

            ausgewaehlte_zeiten = []
            last_t = -999
            for t in loud_peaks:
                if t - last_t > clip_dauer:
                    ausgewaehlte_zeiten.append(t)
                    last_t = t

        st.spinner("3/3 Schneide 9:16 Shorts & erstelle Overlays...")
        st.subheader("📹 Fertige Shorts zur Vorschau & Download")
        video = VideoFileClip("stream_input.mp4")
        
        for idx, start_t in enumerate(ausgewaehlte_zeiten[:4]):
            start = max(0, start_t - 2)
            end = min(video.duration, start + clip_dauer)
            
            clip_text = ""
            for seg in transkript_data["segments"]:
                if seg["start"] >= start and seg["end"] <= end:
                    clip_text += " " + seg["text"]

            clip = video.subclip(start, end)
            w, h = clip.size
            target_width = int(h * (9 / 16))
            crop_x1 = max(0, (w - target_width) // 2)
            clip_916 = clip.crop(x1=crop_x1, width=target_width, height=h)

            hook_text, titel_liste, hashtags = generiere_hook_und_titel(clip_text)

            elemente = [clip_916]

            # Wasserzeichen oben links
            if twitch_username:
                txt_watermark = TextClip(
                    twitch_username, fontsize=22, color='white', bg_color='rgba(0,0,0,0.6)'
                ).set_position(('left', 'top')).set_duration(clip_916.duration)
                elemente.append(txt_watermark)

            # Hook-Banner in den ersten 3 Sekunden
            txt_hook = TextClip(
                hook_text, fontsize=28, color='yellow', bg_color='black', method='caption', size=(target_width - 40, None)
            ).set_position(('center', 100)).set_duration(3)
            elemente.append(txt_hook)

            final_clip = CompositeVideoClip(elemente)
            clip_pfad = f"short_clip_{idx+1}.mp4"
            final_clip.write_videofile(clip_pfad, codec="libx264", audio_codec="aac", logger=None)

            # --- UI DARSTELLUNG ---
            st.markdown("---")
            col_vid, col_info = st.columns([1, 2])
            
            with col_vid:
                st.video(clip_pfad)
                with open(clip_pfad, "rb") as file:
                    st.download_button(
                        label=f"⬇️ Short #{idx+1} herunterladen",
                        data=file,
                        file_name=f"stream_short_{idx+1}.mp4",
                        mime="video/mp4"
                    )

            with col_info:
                st.markdown(f"### Short #{idx+1} - Upload-Inhalte")
                st.text_input("Titel-Vorschlag 1:", value=titel_liste[0], key=f"t1_{idx}")
                st.text_input("Titel-Vorschlag 2:", value=titel_liste[1], key=f"t2_{idx}")
                st.text_area("Beschreibung & Hashtags:", value=f"{clip_text.strip()}\n\n{hashtags}", key=f"hash_{idx}")

        video.close()
