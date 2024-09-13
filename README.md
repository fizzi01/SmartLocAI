# SmartLocAI - Processing Dashboard

## Overview

![workflow](https://drive.google.com/uc?export=view&id=1X_hAKv6Gfx64jVT-aF3GQcP7hRWn_DOr)

La **Processing Dashboard** è una piattaforma interattiva sviluppata con **Streamlit** che gestisce l'intero flusso di lavoro per la localizzazione indoor, dall'acquisizione e preprocessamento dei dati RSSI, all'addestramento di modelli di machine learning fino alla predizione della posizione dei dispositivi mobili. Questa soluzione sfrutta tecniche avanzate come la **Generative AI**, il **clustering K-Means**, il **K-Nearest Neighbors (KNN)** e la **trilaterazione** per ottenere stime accurate della posizione basate su segnali Wi-Fi e BLE.

## a) Architettura del Sistema

![architettura](https://drive.google.com/uc?export=view&id=13Cb9Iq9cTK-zhTe3yJ_fPGRLT2n5gV35)

L'architettura del sistema è modulare e comprende i seguenti componenti chiave:

- **App Mobile SmartLocAI**: Raccoglie dati RSSI da access points Wi-Fi e beacon BLE tramite dispositivi mobili. Invia i dati al **DataService API** per l'elaborazione.
- **DataService API**: Gestisce la raccolta, l'archiviazione e il preprocessamento dei dati.
- **LocalizationService API**: Gestisce il deployment e l'esecuzione dei modelli addestrati, restituendo la posizione stimata in base ai dati RSSI in ingresso.
- **Preprocessing Dashboard (Streamlit)**: Interfaccia che consente di eseguire operazioni di visualizzazione e preprocessamento dei dati, addestramento dei modelli e gestione del loro deployment.
- **CTGAN**: Basato su architettura **Generative Adversarial Network**, viene utilizzato per generare dati sintetici da dataset reali, ampliando i dati disponibili per migliorare le performance dei modelli di localizzazione.
- **K-Means e KNN**: Algoritmi utilizzati per il clustering e la classificazione dei dati. K-Means organizza i dati in cluster, mentre KNN viene utilizzato per stimare la posizione dei dispositivi.

## b) Repositori dei Componenti

- **SmartLocAI Mobile App**: [Link al repository]()
- **DataService API**: [Link al repository](https://github.com/UniSalento-IDALab-IoTCourse-2023-2024/wot-project-2023-2024-DataService-IzziBarone.git)
- **LocalizationService API**: [Link al repository](https://github.com/UniSalento-IDALab-IoTCourse-2023-2024/wot-project-2023-2024-LocalizationService-IzziBarone)
- **Preprocessing Dashboard**: Questo repository attuale descrive l'interfaccia principale di gestione dell'intero flusso di preprocessing, addestramento e deployment.

## c) Descrizione della Dashboard

La **Processing Dashboard** è il cuore operativo della soluzione, integrando e orchestrando tutte le componenti del sistema. La Dashboard permette di:

1. **Visualizzare e Preprocessare i Dati**: I dati raccolti dai dispositivi mobili vengono mostrati in tempo reale con statistiche e grafici interattivi. Puoi eseguire operazioni di pulizia, aggregazione e normalizzazione dei dati.
   
2. **Generare Dati Sintetici con CTGAN**: Se il dataset è limitato, puoi utilizzare il modulo di **Data Augmentation** basato su **CTGAN** per ampliare i dati disponibili e migliorare l'addestramento dei modelli.

3. **Addestrare i Modelli**:
   - **K-Means**: Suddivide i dati RSSI in cluster omogenei.
   - **KNN**: Viene addestrato su ciascun cluster per garantire previsioni accurate. La dashboard include un modulo di ottimizzazione automatica del parametro **k** per migliorare la precisione del modello.

4. **Deployment dei Modelli**: Una volta completato l'addestramento, i modelli K-Means e KNN possono essere distribuiti tramite l'API **LocalizationService**, garantendo che i modelli più recenti siano utilizzati per le predizioni in tempo reale.

5. **Monitoraggio e Visualizzazione dei Risultati**: La dashboard offre strumenti di visualizzazione per monitorare le performance dei modelli, come grafici 3D per i cluster K-Means e metriche di accuratezza per i modelli KNN.

---

### Come Iniziare

1. Clona il repository:
   ```bash
   git clone https://github.com/UniSalento-IDALab-IoTCourse-2023-2024/wot-project-2023-2024-Dashboard-IzziBarone.git
   ```

2. Installa le dipendenze:
   ```bash
   docker compose up -d
   ```
