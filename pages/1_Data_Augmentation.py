import streamlit as st
import os
from datetime import datetime

import yaml
from core.augmentation import DataAugmentation
from core.processor import DataProcessor
from yaml import SafeLoader

st.set_page_config(
    page_title="Generazione Dati Sintetici",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open('settings.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

MODELS_DIR = config['save_dirs']['models']
CTGAN_MODEL_DIR = config['save_dirs']['ctgan']
DATA_DIR = config['save_dirs']['data']

st.title("Generazione Dati Sintetici")

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    try:
        st.header("Genera dataset sintetico")

        # Leggi i nomi dei file dei modelli dalla directory 'models'
        model_dir = "models/ctgan"
        data_path = DATA_DIR
        model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

        # Verifica che la directory 'models' esista e contenga file
        if os.path.exists(model_dir) and model_files:
            model_name = st.selectbox("Seleziona un modello GenAI addestrato", model_files)
        else:
            st.warning(
                "Nessun modello trovato nella directory 'models'. Assicurati che la directory esista e contenga file di modello.")
            st.stop()  # Ferma l'esecuzione del resto della pagina se non ci sono modelli

        num_data_points = st.number_input("Numero di dati da generare", min_value=100, max_value=10000, step=100)

        if st.button("Genera Dati"):
            augmentation = DataAugmentation("", [], [])  # Assuming data is not needed

            model_path = os.path.join(model_dir, model_name)
            augmentation.load_model(model_path)

            synthetic_data = augmentation.generate_data(num_data_points)

            proc = DataProcessor(dataframe=synthetic_data)
            data = proc.process_data()

            st.dataframe(data)

            # Save synthetic data
            filename = f'synthetic_{datetime.now().strftime("%d-%m-%y_%H:%M:%S")}_{num_data_points}.csv'
            csv_filename = os.path.join(data_path, filename)
            synthetic_data.to_csv(csv_filename,index=False)
            st.success(f"Dati sintetici salvati in {csv_filename}")
    except Exception as e:
        st.error(f"{e}")
else:
    st.error('Effettua il login per accedere a questa funzionalità')
    st.switch_page("Home.py")
