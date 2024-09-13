# Usa l'immagine base ufficiale di Python
FROM python:3.11-slim

# Imposta la cartella di lavoro all'interno del container
WORKDIR /app

# Copia il requirements.txt e installa le dipendenze
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libhdf5-dev \
    python3-dev \
    pkg-config
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutti i file di codice nell'immagine
COPY . /app

# Espone la porta 8501 per Streamlit
EXPOSE 8501

# Comando per avviare l'applicazione Streamlit
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]