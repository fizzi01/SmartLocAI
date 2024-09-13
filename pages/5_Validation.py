import os.path

import streamlit as st
import pandas as pd
import yaml
from core.trilateration import TrilaterationEstimator
from matplotlib import pyplot as plt
from yaml import SafeLoader

with open('settings.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

data_path = config['save_dirs']['data']

st.set_page_config(page_title="Validation", layout="wide", page_icon="📊")

# Function to plot AP and RP positions
def plot_ap_rp_positions(ap_positions_df, training_df):
    """Plot the positions of Access Points and Reference Points."""
    plt.figure(figsize=(8, 8))

    # Plot the access points from the ap_positions dataset
    for _, row in ap_positions_df.iterrows():
        plt.scatter(row['x'], row['y'], color='red', s=100, label=f"AP_{row['AP']}")
        plt.text(row['x'], row['y'], f"{row['AP']}",
                 fontsize=12, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.3, edgecolor='black'))

    # Plot the reference points (RP) from the training data
    for _, row in training_df.iterrows():
        plt.scatter(row['x'], row['y'], color='blue', s=50)
        plt.text(row['x'], row['y'], f"{row['RP']}", fontsize=10, ha='right')

    # Set plot limits and labels
    plt.xlim(0, 4.45)
    plt.ylim(0, 3.80)
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Positions of Access Points and Reference Points (RPs)")
    plt.grid(True)

    # Show the plot in Streamlit
    st.pyplot(plt)

if 'authentication_status' in st.session_state and st.session_state['authentication_status']:
    st.title("Validation")

    st.markdown("""
    # Processo per la Stima dei Parametri del Path-Loss e della Distanza

    Il processo di stima dei parametri del *path loss* e della distanza si basa su modelli di propagazione del segnale radio. Di seguito viene descritto l'approccio utilizzato per stimare i parametri $n$ (esponente del *path loss*) e $RSSI(d_0)$ (RSSI alla distanza di riferimento $d_0$, tipicamente 1 metro), e come questi vengono sfruttati per calcolare le distanze dagli access points (AP).
    """)

    # Equazione del path-loss model
    st.latex(r'''
    RSSI(d) = RSSI(d_0) - 10n \log_{10}\left(\frac{d}{d_0}\right)
    ''')

    st.markdown("""
    Dove:
    - $ RSSI(d) $ è la potenza del segnale ricevuto alla distanza $ d $,
    - $ RSSI(d_0) $ è il valore RSSI a una distanza di riferimento $ d_0 $ (tipicamente 1 metro),
    - $ n $ è l'esponente del *path loss*, che varia a seconda dell'ambiente (tipicamente tra 2 e 4 in ambienti indoor),
    - $ d $ è la distanza tra l'AP e il dispositivo di misurazione.

    ## Processo di Stima dei Parametri del Path Loss

    Per ogni *access point (AP)*, stimiamo i parametri del modello di *path loss* $n$ e $RSSI(d_0)$ utilizzando i dati di *fingerprinting* raccolti. Il processo è il seguente:

    ### 1. Raccolta delle Misurazioni

    In ogni punto di riferimento (RP) viene effettuata una serie di misurazioni dell'RSSI per ogni AP. Questi dati vengono raggruppati per ciascun RP. Per ciascun AP, viene calcolata la **media** delle misurazioni di RSSI in ogni RP.

    ### 2. Calcolo delle Distanze Reali

    Utilizzando le coordinate note degli access points e dei punti di riferimento (RP), viene calcolata la **distanza reale** tra l'AP e ciascun RP utilizzando la formula della distanza euclidea:
    """)

    # Equazione della distanza euclidea
    st.latex(r'''
    d = \sqrt{(x_{AP} - x_{RP})^2 + (y_{AP} - y_{RP})^2}
    ''')

    st.markdown("""
    ### 3. Fit del Modello di Path Loss

    Per stimare i parametri $n$ e $RSSI(d_0)$, applichiamo una **regressione lineare** tra il logaritmo della distanza e i valori di RSSI misurati:
    """)

    # Equazione del fit del path-loss
    st.latex(r'''
    RSSI(d) = RSSI(d_0) - 10n \log_{10}(d)
    ''')

    st.markdown("""
    La regressione lineare si esegue tra $ \log_{10}(d) $ e i valori $ RSSI(d) $. La **pendenza** della retta risultante fornisce l'esponente del *path loss* $ n $, mentre l'**intercetta** della retta fornisce $ RSSI(d_0) $.

    ## Stima della Distanza

    Una volta stimati i parametri del modello di *path loss* $n$ e $RSSI(d_0)$ per ciascun AP, possiamo stimare la distanza tra il dispositivo e l'AP usando le misurazioni di RSSI.

    ### Formula per la Stima della Distanza
    """)

    # Equazione per la stima della distanza
    st.latex(r'''
    d = d_0 \cdot 10^{\frac{RSSI(d_0) - RSSI(d)}{10n}}
    ''')

    st.markdown("""
    Dove:
    - $ RSSI(d) $ è la potenza del segnale misurata in un punto,
    - $ RSSI(d_0) $ e $ n $ sono i parametri stimati dal modello di *path loss*.

    ### Esempio di Stima della Distanza

    Supponiamo di avere un access point con i seguenti parametri stimati:
    - $ RSSI(d_0) = -40 , dBm$ a 1 metro,
    - $ n = 3.0 $ (esponente del *path loss*).

    Se in un punto misuriamo un valore $ RSSI(d) = -60 \, dBm $, possiamo calcolare la distanza $ d $ come:
    """)

    # Equazione del calcolo della distanza con esempio numerico
    st.latex(r'''
    d = 1 \cdot 10^{\frac{-40 - (-60)}{10 \cdot 3}} = 10^{\frac{20}{30}} \approx 2.15 \, metri
    ''')

    st.markdown("""
    ## Considerazioni sull'Accuratezza

    L'accuratezza della stima della distanza dipende fortemente dalla **stabilità delle misurazioni di RSSI** e dalle **condizioni ambientali**. Ostacoli, riflessioni e rumori interferenti possono causare variazioni significative nei valori di RSSI, portando a una stima errata della distanza. Il valore dell'esponente del *path loss* $n$ varia a seconda dell'ambiente (es. spazi aperti o indoor con molte pareti), e dovrebbe essere adattato per scenari specifici.

    ## Utilizzo nella Trilaterazione

    Una volta stimate le distanze da tre o più AP, possiamo utilizzare queste informazioni per stimare la posizione del dispositivo tramite **trilaterazione**, risolvendo un sistema di equazioni non lineari basato sulle distanze stimate da ciascun AP.
    """)

    st.subheader("Validation Process")
    ap_pos_path = os.path.join(data_path, 'ap_positions.csv')
    st.dataframe(pd.read_csv(ap_pos_path), use_container_width=True, hide_index=True)

    ap_positions = pd.read_csv(ap_pos_path).set_index('AP').to_dict('index')
    estimator = TrilaterationEstimator(ap_positions)

    df = pd.read_csv(os.path.join(data_path, 'extracted_data_training_real.csv'))

    # Assuming 'df' contains RSSI measurements and positions
    df_with_distances = estimator.calculate_distances(df)
    estimator.estimate_path_loss_and_rssi_d0(df_with_distances)

    #Show dataset
    st.write("**Processed Data**")
    st.dataframe(df, use_container_width=True)

    plot_ap_rp_positions(pd.read_csv(ap_pos_path), df)

    # Show path loss exponents
    cols = st.columns(2)
    cols[0].write("Path loss exponents (n):")
    cols[0].write(estimator.path_loss_exponents)

    cols[1].write("RSSI(d0) [dB]:")
    cols[1].write(estimator.rssi_d0)

    # Mapping reference points to true positions
    rp_to_position = df[['RP', 'x', 'y']].drop_duplicates().set_index('RP').to_dict('index')
    test_df = pd.read_csv(os.path.join(data_path, 'extracted_data_test.csv'))

    mean_error, positions = estimator.run_trilateration(df, rp_to_position)
    if mean_error > 1.2:
        st.error(f"Mean error: {mean_error:.2f} meters")
    else:
        st.warning(f"Mean error: {mean_error:.2f} meters")
    if st.button("Plot positions"):
        estimator.plot_positions_rp(rp_to_position, positions)

else:
    st.error("You need to login to access this page.")
    st.switch_page("Home.py")
