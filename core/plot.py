import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go


def plot_rssi_values(csv_file):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Extract the columns for x, y, and RSSI values
    x = data['x']
    y = data['y']
    rssi_a = data['RSSI A']
    rssi_b = data['RSSI B']
    rssi_c = data['RSSI C']

    # Create a scatter plot for RSSI A
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, c=rssi_a, cmap='Reds', alpha=0.6, edgecolors='w', s=100)
    plt.colorbar(label='RSSI A')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('RSSI A Values at Different Points in Space')
    plt.grid(True)
    plt.show()

    # Create a scatter plot for RSSI B
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, c=rssi_b, cmap='Greens', alpha=0.6, edgecolors='w', s=100)
    plt.colorbar(label='RSSI B')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('RSSI B Values at Different Points in Space')
    plt.grid(True)
    plt.show()

    # Create a scatter plot for RSSI C
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, c=rssi_c, cmap='Blues', alpha=0.6, edgecolors='w', s=100)
    plt.colorbar(label='RSSI C')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('RSSI C Values at Different Points in Space')
    plt.grid(True)
    plt.show()


def plot_kmeans_clusters(data, centroids, columns):
    # 1. Grafico 3D di Dispersione dei Cluster Basati su RSSI

    # Creazione del grafico 3D con Plotly
    fig = go.Figure()

    # Aggiunta dei punti dei dati
    fig.add_trace(go.Scatter3d(
        x=data[columns[0]],
        y=data[columns[1]],
        z=data[columns[2]],
        mode='markers',
        marker=dict(
            size=5,
            color=data['cluster'],  # Coloriamo i punti in base al cluster
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Dati RSSI'
    ))

    # Aggiunta dei centroidi
    fig.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            symbol='x',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        name='Centroidi'
    ))

    # Aggiornamento del layout del grafico
    fig.update_layout(
        title='K-means Clustering of RSSI Data (3D Scatter Plot)',
        scene=dict(
            xaxis_title='RSSI A',
            yaxis_title='RSSI B',
            zaxis_title='RSSI C'
        )
    )

    # Visualizzazione del grafico interattivo su Streamlit
    st.plotly_chart(fig)

    # 2. Proiezione su Piano 2D (RSSI A vs RSSI B)
    fig2 = plt.figure(figsize=(8, 6))
    scatter_2d = plt.scatter(data[columns[0]], data[columns[1]], c=data['cluster'], cmap='viridis')
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title(f'K-means Clustering: {columns[0]} vs {columns[1]}')
    plt.colorbar(scatter_2d)
    st.pyplot(fig2)

    # 2.1. Proiezione su Piano 2D (RSSI A vs RSSI C)
    fig2_1 = plt.figure(figsize=(8, 6))
    scatter_2d_1 = plt.scatter(data[columns[0]], data[columns[2]], c=data['cluster'], cmap='viridis')
    plt.xlabel(columns[0])
    plt.ylabel(columns[2])
    plt.title(f'K-means Clustering: {columns[0]} vs {columns[2]}')
    plt.colorbar(scatter_2d_1)
    st.pyplot(fig2_1)

    # 2.2. Proiezione su Piano 2D (RSSI B vs RSSI C)
    fig2_2 = plt.figure(figsize=(8, 6))
    scatter_2d_2 = plt.scatter(data[columns[1]], data[columns[2]], c=data['cluster'], cmap='viridis')
    plt.xlabel(columns[1])
    plt.ylabel(columns[2])
    plt.title(f'K-means Clustering: {columns[1]} vs {columns[2]}')
    plt.colorbar(scatter_2d_2)
    st.pyplot(fig2_2)

    # 3. Visualizzazione nello Spazio Fisico (x, y)
    fig3 = plt.figure(figsize=(8, 6))
    scatter_space = plt.scatter(data['x'], data['y'], c=data['cluster'], cmap='viridis')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('K-means Clustering in Physical Space (x, y)')
    plt.colorbar(scatter_space)
    #st.pyplot(fig3)


def check_parameter_type(param):
    if isinstance(param, str):
        return "String"
    elif isinstance(param, pd.DataFrame):
        return "DataFrame"
    else:
        return "Unknown"


def plot_reference_and_additional_data(reference_file, additional_file_path=None):
    """
    Crea un plot delle zone di riferimento e opzionalmente dei dati aggiuntivi nello spazio.

    :param reference_file_path: Percorso del file CSV contenente le zone di riferimento e le coordinate.
    :param additional_file_path: Percorso opzionale del file CSV contenente i dati aggiuntivi da visualizzare.
    """
    # Carica il dataset delle zone di riferimento
    if check_parameter_type(reference_file) == "String":
        reference_data = pd.read_csv(reference_file)
    elif check_parameter_type(reference_file) == "DataFrame":
        reference_data = reference_file
    else:
        raise ValueError("Il parametro deve essere una stringa o un DataFrame.")

    # Assumendo che il dataset abbia le colonne 'RP', 'x', e 'y'
    if not {'RP', 'x', 'y'}.issubset(reference_data.columns):
        raise ValueError("Il dataset delle zone di riferimento deve contenere le colonne 'RP', 'x', e 'y'.")

    # Creare una lista di colori con 16 colori differenti
    colors = plt.cm.get_cmap('tab20', 16)

    # Crea il plot delle zone di riferimento
    fig = plt.figure(figsize=(10, 8))
    for idx, zone in enumerate(reference_data['RP'].unique()):
        zone_data = reference_data[reference_data['RP'] == zone]
        plt.scatter(zone_data['x'], zone_data['y'], label=zone, color=colors(idx))
        for _, row in zone_data.iterrows():
            plt.text(row['x'], row['y'], row['RP'], fontsize=12, ha='right', va='bottom')

    # Se è fornito un file di dati aggiuntivi, aggiungi questi dati al plot
    if additional_file_path:
        additional_data = pd.read_csv(additional_file_path)
        # Assumendo che il dataset aggiuntivo abbia le colonne 'x' e 'y'
        if not {'x', 'y'}.issubset(additional_data.columns):
            raise ValueError("Il dataset dei dati aggiuntivi deve contenere le colonne 'x' e 'y'.")
        for i, (x, y) in enumerate(zip(additional_data['x'], additional_data['y']), start=1):
            plt.scatter(x, y, color='red', marker='x')
            plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom', color='red')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Reference Points in Space')
    #plt.legend()
    plt.grid(True)
    st.pyplot(fig)
