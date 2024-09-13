
# Processing Dashboard

## Overview

La **Processing Dashboard**, sviluppata in **Streamlit**, è uno strumento avanzato per la localizzazione indoor basata su **Wi-Fi** e **Bluetooth Low Energy (BLE)**. Questa applicazione facilita l'elaborazione dei dati e il miglioramento della precisione di localizzazione sfruttando Generative AI e ML.

![workflow](https://drive.google.com/uc?export=view&id=1X_hAKv6Gfx64jVT-aF3GQcP7hRWn_DOr)



### Caratteristiche Principali

1. **Generazione di Dati Sintetici con CTGAN**:
   - Utilizzo di **CTGAN** per la generazione di dati sintetici, in modo da aumentare artificialmente la quantità e la qualità dei dati raccolti nel mondo reale. Questa componente è fondamentale per migliorare l'efficacia dei modelli di localizzazione in presenza di dataset limitati o sbilanciati.

2. **Processing dei Dati**:
   - Raccolta dei dati RSSI da **access points** Wi-Fi e **beacon BLE** per identificare la posizione dei dispositivi mobili. Il sistema include il **preprocessing dei dati**, come pulizia, normalizzazione e etichettatura dei **Reference Points (RP)** per una corretta elaborazione successiva.

3. **Pre-Clustering tramite K-Means**:
   - Implementazione dell'algoritmo **K-Means** per raggruppare i dati RSSI in cluster omogenei, riducendo la variabilità e migliorando l'efficacia del modello KNN.

4. **Addestramento Ottimale di KNN e Ottimizzazione Parametri**:
   - Utilizzo dell'algoritmo **K-Nearest Neighbors (KNN)** per la localizzazione indoor. La dashboard include una **ricerca di parametri ottimali** per KNN. Vengono esplorati diversi valori di **k** attraverso procedure di ottimizzazione automatica, selezionando quello che garantisce le migliori performance in termini di accuratezza predittiva.

5. **Trilaterazione per il Calcolo della Posizione**:
   - Integrazione della **trilaterazione** per stimare con precisione la posizione del dispositivo mobile, utilizzando i segnali ricevuti da diversi punti di accesso Wi-Fi e beacon BLE. Questa componente viene utilizzata per puri scopi comparativi.

6. **Visualizzazione dei Risultati**:
   - La dashboard offre visualizzazioni interattive e dinamiche dei risultati. Include rappresentazioni 3D dei cluster generati da K-Means, tabelle dei risultati per monitorare la precisione delle stime di posizione e le performance dei modelli.

7. **Deployment dei Modelli tramite LocalizationService**:
   - La dashboard permette il **deployment automatico** dei modelli K-Means e KNN utilizzando un'API dedicata ([**LocalizationService**](https://github.com/UniSalento-IDALab-IoTCourse-2023-2024/wot-project-2023-2024-LocalizationService-IzziBarone)). Questo servizio facilita il caricamento, versionamento e utilizzo dei modelli per la localizzazione in tempo reale. I modelli vengono caricati nel sistema per essere richiamati durante la fase di predizione, garantendo che l'applicazione utilizzi sempre le versioni più aggiornate e ottimizzate.

## Come Iniziare

1. Clona il repository:
   ```bash
   git clone https://github.com/UniSalento-IDALab-IoTCourse-2023-2024/wot-project-2023-2024-Dashboard-IzziBarone.git
   ```

2. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```

3. Avvia l'applicazione:
   ```bash
   streamlit run Home.py
   ```
____

Questa versione della dashboard include funzionalità avanzate come il **deployment automatico** dei modelli tramite API, l'integrazione con **CTGAN** per la generazione di dati sintetici e strumenti per l'ottimizzazione dei modelli di machine learning.
