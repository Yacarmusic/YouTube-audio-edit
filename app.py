import streamlit as st
import yt_dlp
import os
import io
import tempfile
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import shutil
import re
from typing import Optional

# Requerimientos:
# pip install streamlit yt_dlp librosa numpy soundfile pydub
# Asegúrate de tener instalado FFmpeg.
# En macOS: brew install ffmpeg
# En Windows: descárgalo desde https://ffmpeg.org/download.html

# Rango de sliders
PITCH_RANGE = (-12, 12)
SPEED_RANGE = (0.2, 2.0)

# Formatos de audio disponibles
AUDIO_FORMATS = {"MP3": "mp3", "WAV": "wav"}

def check_ffmpeg() -> bool:
    """Verifica si FFmpeg está instalado de manera multiplataforma."""
    return shutil.which("ffmpeg") is not None

def aplicar_estilos():
    """Inyecta estilos personalizados para un tema oscuro con buen contraste."""
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
    """Convierte un tiempo en formato 'MM:SS' a segundos (float)."""
    try:
        mm, ss = mmss.split(':')
        mm = int(mm)
        ss = float(ss)
        return mm * 60 + ss
    except Exception:
        return 0.0

def convertir_a_mmss(segundos: float) -> str:
    """Convierte un número de segundos a formato 'MM:SS'."""
    mm = int(segundos // 60)
    ss = int(segundos % 60)
    return f"{mm:02d}:{ss:02d}"

def obtener_duracion(audio_bytes: bytes) -> float:
    """Calcula la duración del audio en segundos usando pydub."""
    # Se usa un archivo temporal que se elimina automáticamente al salir del bloque
    with tempfile.NamedTemporaryFile(suffix=".tmp") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        audio = AudioSegment.from_file(tmp.name)
        return len(audio) / 1000.0

def validar_url(url: str) -> bool:
    """Valida que la URL tenga un formato correcto."""
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// o https://
        r'\S+$', re.IGNORECASE)
    return re.match(regex, url) is not None

def descargar_contenido(url: str, formato: str) -> Optional[bytes]:
    """Descarga video o audio usando yt_dlp."""
    # Se guarda en el directorio temporal del sistema para mayor compatibilidad
    ydl_opts = {'outtmpl': os.path.join(tempfile.gettempdir(), 'tempfile.%(ext)s')}
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
            try:
                with open(filename, 'rb') as f:
                    content = f.read()
            finally:
                if os.path.exists(filename):
                    os.remove(filename)
            return content
    except Exception as e:
        st.error(f"Error al descargar contenido: {e}")
        return None

def procesar_audio(audio_bytes: bytes, params: dict) -> Optional[bytes]:
    """
    Aplica recorte, cambio de pitch y ajuste de velocidad al audio.
    Utiliza librosa para el procesamiento.
    """
    input_path = None
    try:
        # Crear un archivo temporal para el audio de entrada
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            input_path = tmp_in.name

        # Indicador de progreso (simulado)
        progress_bar = st.progress(0)
        progress_bar.progress(20)

        # 1) Cargar el audio sin modificar la tasa de muestreo
        y, sr = librosa.load(input_path, sr=None)
        progress_bar.progress(40)

        # 2) Recortar el audio
        start_sample = int(params['start_seg'] * sr)
        end_sample = int(params['end_seg'] * sr)
        end_sample = min(end_sample, len(y))
        y = y[start_sample:end_sample]
        progress_bar.progress(60)

        # 3) Aplicar pitch shift si es necesario
        if params['semitones'] != 0:
            y = librosa.effects.pitch_shift(y, sr, n_steps=params['semitones'])
        progress_bar.progress(75)

        # 4) Aplicar time stretch si es necesario
        if params['speed'] != 1.0:
            y = librosa.effects.time_stretch(y, rate=params['speed'])
        progress_bar.progress(90)

        # 5) Exportar a WAV en memoria
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, y, sr, format="WAV")
        wav_bytes.seek(0)

        # Convertir a MP3 si se requiere
        if params['output_format'] == "wav":
            final_bytes = wav_bytes.getvalue()
        else:
            temp_wav = AudioSegment.from_file(wav_bytes, format="wav")
            mp3_io = io.BytesIO()
            temp_wav.export(mp3_io, format="mp3")
            mp3_io.seek(0)
            final_bytes = mp3_io.getvalue()
        progress_bar.progress(100)
        return final_bytes
    except Exception as e:
        st.error(f"Error procesando audio: {e}")
        return None
    finally:
        # Asegurarse de eliminar el archivo temporal
        if input_path is not None and os.path.exists(input_path):
            os.remove(input_path)

def main():
    # Verificar que FFmpeg esté instalado
    if not check_ffmpeg():
        st.error("FFmpeg no está instalado. Por favor, instálalo.\n"
                 "En macOS: brew install ffmpeg\n"
                 "En Windows: descarga desde https://ffmpeg.org/download.html")
        return

    # Aplicar estilos personalizados
    aplicar_estilos()

    st.title("Descarga de vídeos y edición de audio")
    st.subheader("Aplicación creada por @yacarmusic 100% con IA")

    # Inicializar variables en session_state
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
            if not validar_url(url):
                st.warning("La URL ingresada no es válida. Asegúrate de incluir 'http://' o 'https://'.")
            else:
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
            st.warning("Ingresa una URL válida.")

    # --- Sección: Editor ---
    if st.session_state.get('show_editor') and st.session_state.get('audio_data'):
        with st.expander("✂️ Editor de Audio", expanded=True):
            st.header("2. Editar Audio")

            # Previsualización del audio original
            st.subheader("Previsualizar audio original")
            st.audio(st.session_state['audio_data'])

            total_mmss = convertir_a_mmss(st.session_state['audio_duration'])
            st.write(f"**Duración total:** {total_mmss} (MM:SS)")

            col1, col2 = st.columns(2)
            with col1:
                start_str = st.text_input("Inicio (MM:SS)", value="00:00")
            with col2:
                end_str = st.text_input("Fin (MM:SS)", value=total_mmss)

            semitones = st.slider("Semitonos", PITCH_RANGE[0], PITCH_RANGE[1], 0)
            speed = st.slider("Velocidad", SPEED_RANGE[0], SPEED_RANGE[1], 1.0, 0.01)

            if st.button("Procesar Audio"):
                with st.spinner("Procesando audio..."):
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
                            st.audio(procesado, format=f"audio/{formato.lower()}")
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
