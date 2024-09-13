import pandas as pd
import streamlit as st
import yaml
from core.net import APIDataProcessor
from core.processor import DataProcessor
from core.plot import plot_reference_and_additional_data
import os

from yaml import SafeLoader

st.set_page_config(page_title="Data Extraction and Transformation", layout="wide", page_icon="📉")

# Configurazione della pagina Streamlit
st.title("Data Extraction and Transformation")

with open('settings.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Impostazioni della connessione MongoDB
main_data_url = config['api']['data']
test_data_url = config['api']['test_data']
data_path = config['save_dirs']['data']

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    # Seleziona il tipo di dati da estrarre

    data_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    rows = st.columns(2)

    data_type = rows[0].radio(
        "Select Data Type",
        ('Training Data', 'Test Data')
    )

    with rows[1].form(key='upload_form'):
        st.write("Upload a file:")
        uploaded_file = st.file_uploader("Choose a file", type=['csv'])
        st.write("Or")
        # Choose from data folder
        selected_file = st.selectbox("Choose a file", data_files)
        submit = st.form_submit_button("Extract and Process Data")

    if selected_file:
        uploaded_file = pd.read_csv(os.path.join(data_path, selected_file))

    if uploaded_file is not None and submit:
        processor = APIDataProcessor("")
        uploaded_file = pd.DataFrame(uploaded_file)
        df, rows_affected = processor.data_cleaning(uploaded_file)

        if df is None or len(df) == 0:
            st.warning("No data found.")
            st.stop()

        st.success(f"Data extracted successfully.")
        st.warning(f"Rows affected: {rows_affected}")

        # Select columns that start with RSSI
        rssi_columns = [col for col in df.columns if col.lower().startswith('rssi')]
        # Apply absolute value to RSSI columns
        df[rssi_columns] = df[rssi_columns].abs()

        st.subheader("Processed Data")
        st.dataframe(df, use_container_width=True)

        # Salvo i dati in un file CSV
        df.to_csv(os.path.join(data_path, f"extracted_data_uploaded.csv"), index=False)

        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="extracted_data.csv",
            mime="text/csv"
        )

        # Mostra statistiche descrittive sui dati
        st.subheader("Statistical Summary")

        rows = st.columns(3)

        # Statistiche descrittive per le coordinate x e y
        if 'x' in df.columns and 'y' in df.columns:
            rows[0].write("**Coordinates stats `x` & `y`:**")
            rows[0].write(df[['x', 'y']].describe())

            # Valori unici di x e y
            rows[2].write("**Unique `x` & `y` values:**")
            unique_x = df['x'].nunique()
            unique_y = df['y'].nunique()
            rows[2].write(f"**`x`**: {unique_x}")
            rows[2].write(f"**`y`**: {unique_y}")

        # Statistiche per i valori RSSI
        rows[1].write("**RSSI Stats (RSSIA, RSSIB, RSSIC):**")
        rssi_stats = df[['RSSIA', 'RSSIB', 'RSSIC']].describe()
        rows[1].write(rssi_stats)

        # Conteggio totale dei record
        rows[0].write("")
        rows[0].write(f"**Records counting:** {len(df)}")

        # Plot reference and additional data
        if data_type == 'Training Data':
            plot_reference_and_additional_data(df)

        st.stop()

    # Determina il nome della collezione in base alla selezione dell'utente
    api_url = main_data_url if data_type == 'Training Data' else test_data_url

    if rows[0].button("Extract Data"):
        try:
            # Inizializza e connette il processore dei dati Mongo
            processor = APIDataProcessor(api_url)

            # Recupera i dati dalla collezione selezionata
            data = processor.fetch_data()

            if data is None or len(data) == 0:
                st.warning("No data found.")
                st.stop()

            # Trasforma i dati in CSV e DataFrame
            csv_data, df, cleaned = processor.transform_data(data)
            st.success(f"Data extracted successfully.")
            st.warning(f"Data cleaned: {cleaned}")

            if not df.empty and data_type == 'Training Data':
                labeler = DataProcessor(df)
                df = labeler.process_data(abs_rssi=True)
                csv_data = df.to_csv(index=False)
            elif not df.empty and data_type == 'Test Data':
                # Select columns that start with RSSI
                rssi_columns = [col for col in df.columns if col.lower().startswith('rssi')]
                # Apply absolute value to RSSI columns
                df[rssi_columns] = df[rssi_columns].abs()
                csv_data = df.to_csv(index=False)
            else:
                st.warning("No data to process.")
                st.stop()

            # Mostra i dati trasformati su Streamlit
            st.subheader("Processed Data")
            st.dataframe(df, use_container_width=True)

            # Pulsante per il download del CSV
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="extracted_data.csv",
                mime="text/csv"
            )

            if df.empty:
                st.warning("No data to save.")
                st.stop()

            # Salva i dati in un file CSV
            df.to_csv(os.path.join(data_path, f"extracted_data_{'test' if data_type == 'Test Data' else 'training'}.csv"), index=False)

            st.success("Data saved successfully.")

            # Mostra statistiche descrittive sui dati
            st.subheader("Statistical Summary")

            rows = st.columns(3)

            # Statistiche descrittive per le coordinate x e y
            if 'x' in df.columns and 'y' in df.columns:
                rows[0].write("**Coordinates stats `x` & `y`:**")
                rows[0].write(df[['x', 'y']].describe())

                # Valori unici di x e y
                rows[2].write("**Unique `x` & `y` values:**")
                unique_x = df['x'].nunique()
                unique_y = df['y'].nunique()
                rows[2].write(f"**`x`**: {unique_x}")
                rows[2].write(f"**`y`**: {unique_y}")

            # Statistiche per i valori RSSI
            rows[1].write("**RSSI Stats (RSSIA, RSSIB, RSSIC):**")
            rssi_stats = df[['RSSIA', 'RSSIB', 'RSSIC']].describe()
            rows[1].write(rssi_stats)

            # Conteggio totale dei record
            rows[0].write("")
            rows[0].write(f"**Records counting:** {len(df)}")

            # Plot reference and additional data
            if data_type == 'Training Data':
                plot_reference_and_additional_data(df)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.error('Login to access this functionality')
    st.switch_page("Home.py")