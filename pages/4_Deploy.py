# Page to select models and to deploy them by sending them to the server with api

import os
from datetime import datetime

import streamlit as st
import pandas as pd
import yaml
from core.net import APIDataProcessor
from core.processor import PreClustering, IndoorPositioning
from core.utils import *
from yaml import SafeLoader

st.set_page_config(
    page_title="Model Deployment",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open('settings.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

MODELS_DIR = config['save_dirs']['models']
KM_MODEL_DIR = config['save_dirs']['kmeans']
KNN_MODEL_DIR = config['save_dirs']['knn']

MODEL_URI = config['api']['models_upload']
LOGIN_URI = config['api']['login']


def display_files(files, key):
    if files:
        st.write("Select a file")
        df = pd.DataFrame(files)
        df.sort_values(by=['File'], inplace=True)
        df['Select'] = False  # Aggiungi una colonna per la selezione del download
        selected_files = st.data_editor(df, use_container_width=True, hide_index=True, key=key,
                                        disabled=["Directory", "File", "Percorso"])

        # Filtra solo i file selezionati per il download
        files_to_download = selected_files[selected_files['Select']]
        return files_to_download['Percorso'].tolist()
    else:
        st.write("Nessun file trovato.")
        return []


st.title("Model Deployment")

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    try:
        model_files = get_all_files(MODELS_DIR)

        if not 'selected_model_files' in st.session_state:
            st.session_state['selected_model_files'] = []

        selected_model_files = st.session_state['selected_model_files']
        if model_files:
            st.subheader("Available Models")
            selected_model_files = display_files(model_files, key='model_files')
            # Save in session state
            st.session_state['selected_model_files'] = selected_model_files

        # Clean selected files by removing from session state
        if 'deployed_models' in st.session_state:
            deployed_models = st.session_state['deployed_models']
            selected_model_files = [file for file in selected_model_files if file not in deployed_models]
        else:
            deployed_models = []
            st.session_state['deployed_models'] = deployed_models

        st.header("Choose a Model to Deploy")

        net = APIDataProcessor("")
        if 'api_access' not in st.session_state:
            with st.form(key="login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                form_button = st.form_submit_button("Get API Access")

            if len(username) > 0 and len(password) > 0:
                try:
                    res = net.login(LOGIN_URI, username, password)
                    st.session_state['api_access'] = res
                    st.session_state['api_processor'] = net
                    st.success("API Access granted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop()
        else:
            col = st.columns(6)
            col[0].success("**API Access granted**")
            if col[0].button("Clear access"):
                st.session_state.pop('api_access')
                st.session_state.pop('deployed_models')
                st.rerun()

            if selected_model_files:
                for model_path in selected_model_files:
                    model_name = os.path.basename(model_path).split(".")[0]
                    model_description = None
                    timestamp = None
                    model_type = None
                    form_button = None
                    with st.form(key=f"deploy_model_{model_path}"):
                        # Carica il modello KMeans
                        premodel = PreClustering(None)
                        premodel.load_model(model_path)

                        # Visualizza i nomi delle colonne RSSI
                        rssi_columns = premodel.columns
                        rssi_columns = [col for col in rssi_columns if col.startswith('RSSI')]

                        st.subheader("Model Information")
                        st.write(f"Model Name: **{model_name}**")
                        st.write(f"Model Path: {model_path}")

                        api_url = st.text_input("API URL", value=MODEL_URI)
                        model_name = st.text_input("Model Name", value=model_name)
                        timestamp = datetime.now().isoformat()
                        model_description = st.selectbox("Model Type", ["KMeans", "KNN"])
                        form_button = st.form_submit_button("Deploy Model")

                    if form_button:
                        with st.spinner("Deploying model..."):
                            net = st.session_state['api_processor']
                            net.set_url(api_url)
                            res = net.upload_file(model_path, model_name, model_description, timestamp)

                            st.success(f"{model_name}: {res['message']}")

                        # Remove file from selected files
                        if 'deployed_models' in st.session_state:
                            st.session_state['deployed_models'] = st.session_state.get('deployed_models', [])
                            st.session_state['deployed_models'].append(model_path)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()
else:
    st.warning("Please authenticate to access this page.")
    st.switch_page("Home.py")
