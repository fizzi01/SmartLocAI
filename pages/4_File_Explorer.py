import streamlit as st
import pandas as pd
import yaml
from core.utils import *
from yaml import SafeLoader

# Directory dove sono salvati i modelli e i dati sintetici
MODELS_DIR = "models"
DATA_DIR = "data"

with open('settings.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

MODELS_DIR = config['save_dirs']['models']
DATA_DIR = config['save_dirs']['data']

# Funzione per visualizzare i file in una tabella e permettere la selezione per il download
def display_files(files, key):
    if files:
        st.write("Seleziona i file da scaricare")
        df = pd.DataFrame(files)
        df['Scarica'] = False  # Aggiungi una colonna per la selezione del download
        selected_files = st.data_editor(df, use_container_width=True,hide_index=True, key=key, disabled=["Directory","File","Percorso"])

        # Filtra solo i file selezionati per il download
        files_to_download = selected_files[selected_files['Scarica']]
        return files_to_download['Percorso'].tolist()
    else:
        st.write("Nessun file trovato.")
        return []


if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    try:
        # Ottieni tutti i file nelle directory
        model_files = get_all_files(MODELS_DIR)
        synthetic_files = get_all_files(DATA_DIR)

        st.title("File Explorer")

        # Mostra i file dei modelli
        selected_model_files = []
        if model_files:
            st.subheader("Modelli Disponibili")
            selected_model_files = display_files(model_files, key='model_files')

        # Mostra i file dei dati sintetici
        selected_synthetic_files = []
        if synthetic_files:
            st.subheader("Dati Disponibili")
            selected_synthetic_files = display_files(synthetic_files, key='synthetic_files')
        else:
            st.write("Nessun dato sintetico trovato.")

        # Combina le selezioni e crea un archivio ZIP
        all_selected_files = selected_model_files + selected_synthetic_files
        if all_selected_files:
            zip_file = create_zip(all_selected_files)
            st.download_button(
                label="Scarica tutti i file selezionati",
                data=zip_file,
                file_name="files.zip",
                mime="application/zip"
            )
        else:
            st.write("Seleziona almeno un file per abilitare il download.")

    except Exception as e:
        st.error(f"{e}")
else:
    st.error('Effettua il login per accedere a questa funzionalità')
    st.switch_page("Home.py")
