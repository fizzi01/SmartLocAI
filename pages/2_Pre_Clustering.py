import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import yaml
from core.processor import *
from core.plot import *

import pickle
import os

from yaml import SafeLoader

st.set_page_config(
    page_title="Clustering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Data Clustering")

with open('settings.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

DATA_DIR = config['save_dirs']['data']
MODELS_DIR = config['save_dirs']['models']
KM_MODEL_DIR = config['save_dirs']['kmeans']
KNN_MODEL_DIR = config['save_dirs']['knn']

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    try:
        st.header("KMeans Clustering")

        model_dir = os.path.join(MODELS_DIR, KM_MODEL_DIR)
        # Leggi i nomi dei file di modelli KMeans dalla directory 'models/kmeans'
        kmeans_model_files = [f for f in os.listdir(model_dir) if
                              os.path.isfile(os.path.join(model_dir, f)) and f.endswith('.pkl')]

        # Sort the files by date
        kmeans_model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)

        data_dir = DATA_DIR
        data_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

        if not os.path.exists(data_dir) and data_files:
            st.warning(
                "No dataset available. Please upload a dataset to perform clustering.")
            st.stop()  # Ferma l'esecuzione del resto della pagina se non ci sono dataset

        clustering_option = st.selectbox("Select an option", ["KMeans model fit", "Model fine tuning"])
        data = None

        if clustering_option == "KMeans model fit":

            dataset_option = st.selectbox("Choose a dataset",
                                          ["Existing dataset", "Upload a new dataset"])
            st.subheader("Dataset")
            if dataset_option == "Existing dataset":
                dataset_name = st.selectbox("Data", data_files)
                dataset_path = os.path.join(data_dir, dataset_name)
                data = pd.read_csv(dataset_path)
                st.dataframe(data.head(50))
            elif dataset_option == "Upload a new dataset":
                uploaded_file = st.file_uploader("Upload a file", type=["csv"])

                if uploaded_file is not None:
                    data = pd.read_csv(uploaded_file)
                    st.dataframe(data.head(50))
                else:
                    st.warning("Please upload a dataset to perform clustering.")
                    st.stop()

            st.subheader("Clustering Parameters")
            num_clusters = st.number_input("Clusters", min_value=2, max_value=10, value=4)
            numerical_cols = [col for col in data.columns if col.startswith('RSSI')]
            st.write("Nuemerical columns found:", numerical_cols)

            if st.button("Execute Clustering"):
                kmeans = PreClustering(input_data=data, columns=numerical_cols, clusters=num_clusters, batch_size=60)
                results = kmeans.fit()

                st.success("Clustering Completed!")
                st.write("Centers:", kmeans.get_centers())

                st.write("Scores (silhoutte, inertia):",
                         kmeans.evaluate(kmeans.get_normalized_data(), results['cluster']))
                st.text("Lower inertia and higher silhoutte score are better.")

                # Save model
                model_filename = f'k_{num_clusters}means_model_{datetime.now().strftime("%d-%m-%y_%H:%M:%S")}.pkl'
                model_path = os.path.join(model_dir, model_filename)
                kmeans.save_model(model_path)
                kmeans.light_save_model(model_path+".light")
                st.success(f"Model saved as {model_filename}")

                # Plotting
                plot_kmeans_clusters(data, kmeans.get_centers(), columns=numerical_cols)
        elif clustering_option == "Model fine tuning":
            model_name = st.selectbox("Choose a model", kmeans_model_files)
            dataset_name = st.selectbox("Choose a dataset for Partial Fit", data_files)

            if model_name and dataset_name:
                model_path = os.path.join(model_dir, model_name)
                dataset_path = os.path.join(data_dir, dataset_name)

                data = pd.read_csv(dataset_path)
                st.dataframe(data.head())

                if st.button("Execute Partial Fit"):
                    kmeans = PreClustering(input_data=data)
                    kmeans.load_model(model_path)
                    # Old centers
                    st.write("Old Centers:", kmeans.get_centers())

                    results = kmeans.fit()
                    centers = kmeans.get_centers()

                    st.success("Partial Fit completed!")
                    st.write("Centers:", centers)

                    st.write("Scores (silhoutte, inertia):",
                             kmeans.evaluate(kmeans.get_normalized_data(), results['cluster']))
                    st.text("Lower inertia and higher silhoutte score are better.")

                    # Salva il modello KMeans aggiornato
                    updated_model_filename = f'kmeans_model_partial_{datetime.now().strftime("%d-%m-%y_%H:%M:%S")}.pkl'
                    updated_model_path = os.path.join(model_dir, updated_model_filename)
                    kmeans.save_model(updated_model_path)
                    st.success(f"Model saved as {updated_model_filename}")

                    # Plotting
                    plot_kmeans_clusters(kmeans.data, centers, columns=kmeans.columns)
    except Exception as e:
        st.error(f"{e}")
else:
    st.error('Login required to access this page.')
    st.switch_page("Home.py")
