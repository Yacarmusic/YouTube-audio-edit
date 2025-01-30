import streamlit as st
import yt_dlp
import os
import io
import tempfile
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from typing import Optional

# Rango de sliders
PITCH_RANGE = (-12, 12)
SPEED_RANGE = (0.2, 2.0)

# Formatos de audio disponibles
AUDIO_FORMATS = {"MP3": "mp3", "WAV": "wav"}

def check_ffmpeg() -> bool:
    """Verifica si FFmpeg está instalado."""
    return os.system("ffmpeg -version > /dev/null 2>&1") == 0

def aplicar_estilos():
    """Estilos corregidos con ticks legibles (fondo oscuro, texto blanco)"""
    st.markdown("""
    <style>
    body, .stApp {
        background: #0E1117 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stTickBar"] {
        color: #FFFFFF !important;
    }
    [data-testid="stTickBar"] > div > span {
        color: #FFFFFF !important;
        background: transparent !important;
    }
    [data-testid="stSliderValue"] {
        color: #FFFFFF !important;
        background: #161B22 !important;
        padding: 2px 8px !important;
        border-radius: 4px !important;
    }
    [data-baseweb="input"] {
        background: #161B22 !important;
        border-color: #2D3436 !important;
    }
    [data-baseweb="input"] input {
        color: #FFFFFF !important;
    }
    button:hover {
        opacity: 0.9 !important;
        transform: scale(1.02) !important;
    }
    button:active {
        transform: scale(0.98) !important;
    }
    </style>
    """, unsafe_allow_html=True)

def convertir_a_segundos(mmss: str) -> float:
    """Convierte 'MM:SS' a segundos (float)."""
    try:
        mm, ss = mmss.split(':')
        mm = int(mm)
        ss = float(ss)
        return mm * 60 + ss
    except:
        return 0.0

def convertir_a_mmss(segundos: float) -> str:
    """Convierte un número de segundos a 'MM:SS'."""
    mm = int(segundos // 60)
    ss = int(segundos % 60)
    return f"{mm:02d}:{ss:02d}"

def obtener_duracion(audio_bytes: bytes) -> float:
    """Calcula duración del audio en segundos usando pydub."""
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(audio_bytes)
        audio = AudioSegment.from_file(tmp.name)
        return len(audio) / 1000.0

def descargar_contenido(url: str, formato: str) -> Optional[bytes]:
    """Descarga video/MP3/WAV usando yt_dlp."""
    ydl_opts = {'outtmpl': 'tempfile.%(ext)s'}
    try:
        if formato == "Video":
            ydl_opts['format'] = ('bestvideo[ext=mp4]+bestaudio[ext=m4a]'
                                  '/best[ext=mp4]/best')
        else:
            ydl_opts.update({
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': AUDIO_FORMATS[formato],
                }]
            })
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if formato != "Video":
                filename = filename.rsplit('.', 1)[0] + f'.{AUDIO_FORMATS[formato]}'
            with open(filename, 'rb') as f:
                content = f.read()
            os.remove(filename)
            return content
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def procesar_audio(audio_bytes: bytes, params: dict) -> Optional[bytes]:
    """
    Aplica recorte, pitch shift (librosa) sin alterar la velocidad,
    y time stretch (librosa) sin alterar el pitch.
    """
    try:
        # Guardamos audio en un archivo temporal
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            input_path = tmp_in.name

        # 1) Cargar con librosa (sr=None para no forzar sample rate)
        y, sr = librosa.load(input_path, sr=None)

        # 2) Recortar
        start_sample = int(params['start_seg'] * sr)
        end_sample = int(params['end_seg'] * sr)
        end_sample = min(end_sample, len(y))  # evitar salir del rango
        y = y[start_sample:end_sample]

        # 3) Pitch shift (NO altera duración)
        if params['semitones'] != 0:
            y = librosa.effects.pitch_shift(y, sr, n_steps=params['semitones'])

        # 4) Time stretch (NO altera pitch)
        if params['speed'] != 1.0:
            y = librosa.effects.time_stretch(y, rate=params['speed'])

        # 5) Exportar a WAV en memoria (para no perder calidad)
        # Luego, si es MP3, convertimos con pydub.
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, y, sr, format="WAV")
        wav_bytes.seek(0)

        if params['output_format'] == "wav":
            final_bytes = wav_bytes.getvalue()
        else:
            # Convertimos a MP3 con pydub
            # Reabrimos el 'wav_bytes' con pydub para exportar a mp3
            temp_wav = AudioSegment.from_file(wav_bytes, format="wav")
            mp3_io = io.BytesIO()
            temp_wav.export(mp3_io, format="mp3")
            mp3_io.seek(0)
            final_bytes = mp3_io.getvalue()

        # Limpieza
        os.remove(input_path)
        return final_bytes
    except Exception as e:
        st.error(f"Error procesando audio: {e}")
        return None

def main():
    if not check_ffmpeg():
        st.error("Por favor, instala FFmpeg (ej: brew install ffmpeg en Mac).")
        return

    # Inyectar estilos
    aplicar_estilos()

    st.title("Descarga vídeos y edita audios")
    st.subheader("Aplicación creada por @yacarmusic 100% con IA")

    # Variables en session_state
    if 'audio_data' not in st.session_state:
        st.session_state['audio_data'] = None
        st.session_state['audio_duration'] = 0.0
        st.session_state['show_editor'] = False

    # --- Sección: Descargar ---
    st.header("1. Descargar contenido")
    url = st.text_input("URL de YouTube:")
    formato = st.radio("Formato:", ["Video", "MP3", "WAV"], horizontal=True)

    if st.button("Descargar"):
        if url.strip():
            with st.spinner("Descargando..."):
                result = descargar_contenido(url, formato)
                if result:
                    if formato == "Video":
                        st.download_button("Descargar Video", data=result,
                                          file_name="video.mp4", mime="video/mp4")
                    else:
                        st.session_state['audio_data'] = result
                        st.session_state['audio_duration'] = obtener_duracion(result)
                        st.session_state['show_editor'] = True
                        st.success("¡Audio descargado! Editor activado ↓")
        else:
            st.warning("Ingresa una URL válida")

    # --- Sección: Editor ---
    if st.session_state['show_editor'] and st.session_state['audio_data']:
        with st.expander("✂️ Editor de Audio", expanded=True):
            st.header("2. Editar Audio")

            # Previsualizar el audio original
            st.subheader("Previsualizar audio original")
            st.audio(st.session_state['audio_data'], format=f"audio/{formato.lower()}")

            # Duración total en mm:ss
            total_mmss = convertir_a_mmss(st.session_state['audio_duration'])
            st.write(f"**Duración total:** {total_mmss} (MM:SS)")

            # Campos en formato "MM:SS" para inicio y fin
            col1, col2 = st.columns(2)
            with col1:
                start_str = st.text_input("Inicio (MM:SS)", value="00:00")
            with col2:
                end_str = st.text_input("Fin (MM:SS)", value=total_mmss)

            # Ajustes de pitch y velocidad
            semitones = st.slider("Semitonos", PITCH_RANGE[0], PITCH_RANGE[1], 0)
            speed = st.slider("Velocidad", SPEED_RANGE[0], SPEED_RANGE[1], 1.0, 0.01)

            if st.button("Procesar Audio"):
                with st.spinner("Procesando..."):
                    # Convertir a segundos
                    start_seg = convertir_a_segundos(start_str)
                    end_seg = convertir_a_segundos(end_str)
                    dur = st.session_state['audio_duration']
                    if end_seg <= 0 or end_seg > dur:
                        end_seg = dur
                    if start_seg < 0:
                        start_seg = 0
                    if start_seg >= end_seg:
                        st.warning("El inicio debe ser menor que el fin.")
                    else:
                        params = {
                            'start_seg': start_seg,
                            'end_seg': end_seg,
                            'semitones': semitones,
                            'speed': speed,
                            'output_format': formato.lower()
                        }
                        procesado = procesar_audio(st.session_state['audio_data'], params)
                        if procesado:
                            # Reproducir
                            st.audio(procesado, format=f"audio/{formato.lower()}")
                            # Botón descargar
                            st.download_button(
                                "Descargar Audio Editado",
                                data=procesado,
                                file_name=f"audio_editado.{formato.lower()}",
                                mime=f"audio/{formato.lower()}",
                                key="descargar_audio_editado",
                                help="Descarga tu audio procesado"
                            )

if __name__ == "__main__":
    main()