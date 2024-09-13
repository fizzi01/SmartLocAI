import os
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from core.processor import PreClustering, IndoorPositioning
from yaml import SafeLoader

st.set_page_config(
    page_title="Localization",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open('settings.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

DATA_DIR = config['save_dirs']['data']
MODELS_DIR = config['save_dirs']['models']
KM_MODEL_DIR = config['save_dirs']['kmeans']
KNN_MODEL_DIR = config['save_dirs']['knn']

st.title("KNN Positioning Test")

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    try:
        # Directory per i modelli KMeans
        kmeans_model_dir = os.path.join(MODELS_DIR, KM_MODEL_DIR)

        # Leggi i nomi dei file di modelli KMeans dalla directory 'models/kmeans'
        kmeans_model_files = [f for f in os.listdir(kmeans_model_dir) if
                              os.path.isfile(os.path.join(kmeans_model_dir, f)) and f.endswith('.pkl')]

        # RIordina i file in ordine alfabetico
        kmeans_model_files.sort()

        st.header("Choose a KMeans Model to Test KNN Positioning")

        # Selezione del modello KMeans
        model_name = st.selectbox("Choose a model", kmeans_model_files)

        input_method = st.radio(
            "Select input method",
            ('Test on a Dataset', 'Manual Input')
        )

        # Caricamento del dataset di test
        test_file = None
        current_rss = None

        if model_name:
            # Percorso del modello selezionato
            model_path = os.path.join(kmeans_model_dir, model_name)

            # Carica il modello KMeans
            premodel = PreClustering(None)
            premodel.load_model(model_path)

            # Visualizza i nomi delle colonne RSSI
            rssi_columns = premodel.data.columns
            rssi_columns = [col for col in rssi_columns if col.startswith('RSSI')]

            if input_method == 'Test on a Dataset':
                test_file = st.file_uploader("Upload dataset", type=['csv'])

                if test_file:
                    test_file = pd.read_csv(test_file)
                    st.dataframe(test_file)
                else:
                    st.warning("Please upload a dataset to test the KNN positioning")
                    st.stop()
            else:
                # Su un unico input inserire valori di rssi separati da virgola o spazio o punto e virgola o tab
                st.subheader("Manual Input")
                rssi_values = st.text_input("Enter RSSI values separated by comma")

                if not rssi_values:
                    st.warning("Please enter RSSI values")
                    st.stop()

                try:
                    # Se piu spazi consecutivi vengono considerati come un solo spazio
                    rssi_values = rssi_values.replace(',', ' ').replace(';', ' ').replace('\t', ' ').replace(',', ' ')
                    rssi_values = rssi_values.split()

                    rssi_values = [int(rssi) for rssi in rssi_values if rssi.isdigit()]

                    if len(rssi_values) != len(rssi_columns):
                        st.error("Please enter the correct number of RSSI values")
                        st.stop()

                    st.write(rssi_values)

                except ValueError:
                    st.error("Please enter valid RSSI values")
                    st.stop()

                # Converti i valori RSSI in un array numpy
                current_rss = np.array(rssi_values)

            col = st.columns(2)

            # Input per il valore di k per KNN
            col[0].subheader("KNN Parameters")
            k = col[0].number_input("**k** value", min_value=1, max_value=100, value=28)
            leaf_size = col[0].number_input("**Leaf size**", min_value=1, max_value=100, value=30)
            p = col[0].number_input("**p** value", min_value=1, max_value=100, value=2)
            use_kmeans = col[0].checkbox("Use KMeans", value=True)
            model_data = premodel.data

            optimal_button = None
            optmized_k = None
            knn_positioning = None

            if input_method == 'Test on a Dataset':
                col[1].subheader("Actions")
                col[1].write("Choose an action to perform")
                #optimal_button = col[1].button("Find optimal params")
                optmized_k = col[1].button("Find optimal K")

            if col[0].button("Run KNN Positioning"):

                if not use_kmeans:
                    premodel = None

                if test_file is not None:
                    # Testa il posizionamento KNN
                    knn_positioning = IndoorPositioning(clustered_data=model_data, test_data=test_file,
                                                        columns=rssi_columns,
                                                        category='RP')

                    rows = st.columns(2)
                    rows[0].write("**KNN Evaluation**")
                    performance, results = knn_positioning.evaluate(kmeans=premodel, k=k, leaf_size=leaf_size, p=p)

                    for key, value in performance.items():
                        rows[0].success(f"**{key}**: {value}")

                    rows[1].write("Positioning results:")
                    rows[1].dataframe(results)
                else:

                    # Esegui la predizione usando il modello KMeans caricato
                    cluster_data = premodel.predict(current_rss)

                    # Testa il posizionamento KNN
                    knn_positioning = IndoorPositioning(clustered_data=cluster_data, columns=rssi_columns,
                                                        category='RP')

                    estimated_rp = knn_positioning.fit_predict(current_rss, k=k, leaf_size=leaf_size, p=p)
                    st.success(f"Estimated RP Position: **{estimated_rp}**")
            elif optimal_button:

                if not use_kmeans:
                    premodel = None

                if test_file is not None:
                    with st.spinner('Processing...'):
                        # Testa il posizionamento KNN
                        knn_positioning = IndoorPositioning(clustered_data=model_data, test_data=test_file,
                                                            columns=rssi_columns,
                                                            category='RP')

                        results, params = knn_positioning.optimal_params(kmeans=premodel)

                    rows = st.columns(2)

                    rows[0].write("**Optimal Parameters**")
                    for key, value in params.items():
                        rows[0].success(f"**{key}**: {value}")

                    rows[1].write("Results:")
                    rows[1].dataframe(results)
            elif optmized_k:
                if not use_kmeans:
                    premodel = None

                if test_file is not None:
                    with st.spinner('Processing...'):
                        # Testa il posizionamento KNN
                        knn_positioning = IndoorPositioning(clustered_data=model_data, test_data=test_file,
                                                            columns=rssi_columns,
                                                            category='RP')

                        best_k, result_params = knn_positioning.optimal_k(kmeans=premodel, leaf_size=leaf_size, p=p)

                    st.write("**KNN Evaluation**")
                    rows = st.columns(3)
                    for key, value in best_k.items():
                        rows[0].success(f"Cluster **{key}** - k: {value[0]}")
                        rows[1].success(f"Accuracy: **{value[1] * 100}%**")
                        rows[2].success(f"Mean Distance Error: **{value[2]}**")

                    st.session_state['optimal_params'] = result_params
                    st.warning("Optimal parameters saved in session")

            # Fit and save model
            knn_positioning = IndoorPositioning(clustered_data=model_data, columns=rssi_columns,
                                                category='RP')
            save_dir = os.path.join(MODELS_DIR, KNN_MODEL_DIR)
            params = None
            choise = None
            start_button = None
            st.header("Fit and Save Model")
            with st.form(key='fit_save_form'):
                # Give choise to enter hyperparameters as dict or use the one from previous inputs
                choise = st.radio("Choose a method to enter hyperparameters", ['Enter optimal', 'Use fixed params'])
                params = st.text_input("Enter optimal hyperparameters as a dictionary",
                                       value=str(st.session_state.get('optimal_params', {})))
                start_button = st.form_submit_button("Start")

            if start_button:
                with st.spinner('Processing ...'):
                    if len(params) <= 0:
                        params = None
                    else:
                        # Parse string into dict
                        try:
                            params = eval(params)
                            st.success("Hyperparameters parsed successfully!")
                            st.write(params)
                        except Exception as e:
                            st.error(f"{e}")
                            st.stop()

                    knn_positioning.fit_clusters(k=k, leaf_size=leaf_size, p=p, kmeans=premodel,
                                                 save_dir_path=save_dir, params=params)

                st.success("Model fitted and saved successfully!")


    except Exception as e:
        st.error(f"{e}")
else:
    st.error('Login required to access this page')
    st.switch_page("Home.py")
