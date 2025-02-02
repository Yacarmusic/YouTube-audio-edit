# Usamos una imagen oficial de Python optimizada (versión slim)
FROM python:3.10-slim

# Actualizamos e instalamos FFmpeg, que es necesario para tu app
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Establecemos el directorio de trabajo en la imagen
WORKDIR /app

# Copiamos el archivo de dependencias y lo instalamos
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiamos el resto de la aplicación a la imagen
COPY . .

# Exponemos el puerto que utilizará Streamlit (8501 por defecto)
EXPOSE 8501

# Configuramos variables de entorno para que Streamlit se ejecute en modo headless (sin la barra superior)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "app.py"]