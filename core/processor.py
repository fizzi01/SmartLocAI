import os.path
from datetime import datetime

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np


class DataProcessor:
    """
    Classe per il processamento dei dati

    :param dataframe: DataFrame contenente i dati RSSI
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def process_data(self, abs_rssi=False):
        # Identifica i punti distinti
        distinct_points = self.dataframe[['x', 'y']].drop_duplicates().sort_values(['x', 'y'])

        # Crea etichette per ciascun punto in modo ordinato (ad es. A11, A12, ..., B21, B22, ...)
        rows = len(distinct_points['y'].unique())
        cols = len(distinct_points['x'].unique())

        # Genera etichette seguendo la logica di matrice
        labels = []
        for i, (index, row) in enumerate(distinct_points.iterrows()):
            label = f"{chr(65 + i // cols)}{i % cols + 1}"
            labels.append((row['x'], row['y'], label))

        # Crea un dizionario per mappare le coordinate alle etichette
        label_mapping = {(x, y): label for x, y, label in labels}

        # Aggiungi una colonna con le etichette corrispondenti nel dataset originale
        self.dataframe['RP'] = self.dataframe.apply(lambda row: label_mapping[(row['x'], row['y'])], axis=1)

        # Se presente la colonna "Point" viene droppata
        if 'Point' in self.dataframe.columns:
            self.dataframe.drop('Point', axis=1, inplace=True)

        # Applica valore assoluto ai dati RSSI
        if abs_rssi:
            self.dataframe['RSSIA'] = self.dataframe['RSSIA'].abs()
            self.dataframe['RSSIB'] = self.dataframe['RSSIB'].abs()
            self.dataframe['RSSIC'] = self.dataframe['RSSIC'].abs()

        return self.dataframe

    def save_to_csv(self, filename):
        self.dataframe.to_csv(filename, index=False)


class PreClustering:
    """
    Classe per il clustering preliminare dei dati RSSI

    Caricando il modello preaddestrato è possibile effettuare il clustering senza dover ricalcolare i centri (fine-tuning).

    Usage
    ----------

    >>> import pandas as pd
    >>> from processor import *
    >>> from plot import plot_kmeans_clusters
    >>> data = pd.read_csv('data.csv')
    >>> kmeans = PreClustering(input_data=data)
    >>> results = kmeans.fit()
    >>> normalized_data = kmeans.get_normalized_data()
    >>> print("Score: ", kmeans.evaluate(normalized_data, results['cluster']))
    >>> print("Centers: ", kmeans.get_centers())
    >>> plot_kmeans_clusters(results, kmeans.get_centers())

    Methods
    ----------
    transform(data)
        Normalizza i nuovi dati

    inverse_transform(data)
        Effettua la trasformazione inversa dei dati

    clustering(n_clusters, batch_size)
        Esegue il clustering

    save_model(filename)
        Salva il modello e lo scaler

    load_model(filename)
        Carica il modello e lo scaler

    get_centers()
        Restituisce i centri dei cluster

    get_normalized_data()
        Restituisce i dati normalizzati

    get_optimal(data)
        Calcola il numero di cluster ottimale

    evaluate(data, labels)
        Valuta la qualità del clustering

    fit()
        Esegue il fitting del modello

    predict(new_data)
        Determina il cluster al quale il RSS corrente appartiene

    mle_predict(current_rssi)
        Determina il cluster al quale il RSS corrente appartiene usando MLE con istogrammi

    Parameters
    ----------
    input_data : DataFrame
        DataFrame contenente i dati RSSI

    columns : list
        Lista di colonne da utilizzare per il clustering

    clusters : int
        Numero di cluster

    batch_size : int
        Dimensione del batch per il MiniBatchKMeans


    Attributes
    ----------
    data : DataFrame
        DataFrame contenente i dati RSSI

    columns : list
        Lista di colonne da utilizzare per il clustering

    input : DataFrame
        DataFrame contenente i dati normalizzati

    model : MiniBatchKMeans
        Modello di clustering

    scaler : MinMaxScaler
        Scaler per la normalizzazione dei dati
    """

    def __init__(self, input_data, columns=None, clusters=3, batch_size=10):

        if columns is None:
            columns = ['RSSI A', 'RSSI B', 'RSSI C']
        self.data = input_data
        self.columns = columns

        self.input = None
        self.model = None
        self.scaler = None
        self.old_data = None

        self._batch_size = batch_size
        self._clusters = clusters
        self._histograms = None

    def _data_preprocessing(self):
        """
        Normalizza i dati input del modello
        :return:
        """
        self.input = self.data.copy()

        # Normalizza i dati
        if self.scaler is None:
            self.scaler = MinMaxScaler()  # old: StandardScaler() << slight better results with minmax

        if isinstance(self.input, pd.DataFrame):
            values = self.input[self.columns].values
        else:
            values = self.input[self.columns]

        self.input[self.columns] = self.scaler.fit_transform(values)

    def transform(self, data):
        """
        Normalizza i nuovi dati
        :param data:
        :return:
        """
        if self.scaler is None:
            print("Scaler not found!")
            return
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """
        Effettua la trasformazione inversa dei dati
        :param data: Dati da trasformare
        :return: Dati trasformati
        """
        if self.scaler is None:
            print("Scaler not found!")
            return
        return self.scaler.inverse_transform(data)

    def clustering(self, n_clusters=3, batch_size=10):
        if self.model is None:
            self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size)
            train_data = self.input[self.columns].values
            self.model.fit(train_data)

            start_time = datetime.timestamp(datetime.now())
            self.data['cluster'] = self.model.predict(train_data)
            end_time = datetime.timestamp(datetime.now())

            print(f"Clustering completed in {end_time - start_time} seconds.")
        else:
            # Partial fit
            start_time = datetime.timestamp(datetime.now())
            self.model.partial_fit(self.input[self.columns])
            end_time = datetime.timestamp(datetime.now())

            print(f"Partial fit completed in {end_time - start_time} seconds.")

            self.data['cluster'] = self.model.predict(self.input[self.columns])

            # Aggiungo i nuovi dati al dataset
            if self.old_data is not None:
                self.data = pd.concat([self.data, self.old_data], ignore_index=True)
                self._data_preprocessing()  # Aggiorno i dati normalizzati

    def save_model(self, filename):
        import joblib

        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'columns': self.columns,
                'clusters': self._clusters,
                'batch_size': self._batch_size,
                'data': self.data,
                'hist': self._histograms
            }, filename)
        except Exception as e:
            print(e)

    def light_save_model(self, filename):
        import joblib

        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'columns': self.columns,
                'clusters': self._clusters,
                'batch_size': self._batch_size,
                'data': None,
                'hist': self._histograms
            }, filename)
        except Exception as e:
            print(e)

    def load_model(self, filename):
        import joblib
        import os

        # Check file
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")

        try:
            file = joblib.load(filename)

            # Check if data is present
            if ('model' or 'scaler' or 'hist' or 'columns') not in file.keys():
                raise ValueError("Model file is missing data.")

            self.model = file['model']
            self.scaler = file['scaler']
            self._histograms = file['hist']
            self.columns = file['columns']

            if ('clusters' or 'batch_size') not in file.keys():
                raise ValueError("Model file is missing clustering parameters.")

            self._clusters = file['clusters']
            self._batch_size = file['batch_size']

            if self.data is None:
                self.data = file['data']
            else:
                self.old_data = file['data']

        except Exception as e:
            print(e)

    def get_centers(self):
        return self.inverse_transform(self.model.cluster_centers_)

    def get_clusters(self):
        """
        Restituisce i dati di ciascun cluster
        :return: Dizionario contenente i dati di ciascun cluster
        """
        clusters = {}
        for cluster_label in self.data['cluster'].unique():
            cluster_data = self.data[self.data['cluster'] == cluster_label]
            clusters[cluster_label] = cluster_data

        return clusters

    def get_normalized_data(self):
        if self.input is None:
            self._data_preprocessing()

        return self.input[self.columns]

    @staticmethod
    def get_optimal(data, columns, batch_size=10):
        """
        Calcola il numero di cluster ottimale
        :param data: DataFrame contenente i dati
        :return: Array contenente i punteggi Silhouette per ogni numero di cluster (da 2 a 10)
        """
        from sklearn.metrics import silhouette_score

        scores = []
        for i in range(2, 11):
            model = MiniBatchKMeans(n_clusters=i, random_state=42, batch_size=batch_size)
            res = model.fit_predict(data[columns])
            score = silhouette_score(data, res)
            scores.append((score, model.inertia_))

        return scores

    def evaluate(self, data, labels):
        """
        Valuta la qualità del clustering
        :param data: DataFrame contenente i dati RSSI normalizzati
        :param labels: Cluster di appartenenza
        :return: Silhouette score
        """
        from sklearn.metrics import silhouette_score

        return silhouette_score(data, labels), self.model.inertia_

    def fit(self):
        self._data_preprocessing()
        self.clustering(n_clusters=self._clusters, batch_size=self._batch_size)
        self._frequency_histogram()

        return self.data

    def predict(self, new_data, results=False):
        """
        Effettua il clustering sui nuovi dati
        :param new_data: dataframe contenente i nuovi dati
        :return: Il cluster di appartenenza, i dati del cluster
        """
        # Controlla se i nuovi dati sono in un formato corretto
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)

        new = self.transform(new_data)
        cluster = self.model.predict(new)

        print(f"Best cluster found: {cluster}")

        if results:  # Return cluster and data
            return cluster, self.data[self.data['cluster'] == cluster[0]]

        # Return only data
        return self.data[self.data['cluster'] == cluster[0]]

    def _frequency_histogram(self):
        """
        Calcola l'istogramma di frequenza per ciascun cluster
        :return: Dizionario contenente gli istogrammi di frequenza
        """
        import numpy as np

        cluster_histograms = {}
        for cluster_label in self.data['cluster'].unique():
            cluster_data = self.data[self.data['cluster'] == cluster_label]

            histograms = {}
            for col in self.columns:
                hist, bin_edges = np.histogram(cluster_data[col], bins=10,
                                               range=(self.data[col].min(), self.data[col].max()))
                histograms[col] = (hist, bin_edges)

            cluster_histograms[cluster_label] = histograms

        self._histograms = cluster_histograms

        return cluster_histograms

    def get_histograms(self):
        return self._histograms

    def mle_predict(self, current_rssi, results=False):
        """
        Determina il cluster al quale il RSS corrente appartiene usando MLE con istogrammi.
        Restituisce i dati appartenenti al cluster con la probabilità più alta.

        :param current_rssi: Valori RSSI correnti
        :param results: Se True restituisce il cluster e i dati del cluster
        :return: Il cluster di appartenenza, i dati del cluster (se results=True) altrimenti solo i dati
        """
        import numpy as np

        # Assicurati che il clustering sia stato eseguito e i dati siano presenti
        if ('cluster' not in self.data.columns) and (self._histograms is None):
            raise ValueError("Clustering not performed. Please run the fit method first.")

        # Calcola la frequenza di ciascun valore RSS in ogni cluster
        if self._histograms is None:
            self._frequency_histogram()

        cluster_histograms = self._histograms

        # Calcola la probabilità del RSS corrente in ciascun cluster
        cluster_probabilities = []
        epsilon = 1e-6  # Evita che la probabilità sia zero
        for cluster_label, histograms in cluster_histograms.items():
            probability = 1
            for i, col in enumerate(self.columns):
                hist, bin_edges = histograms[col]

                # Prende un singolo valore scalare di current_rssi
                current_value = current_rssi[i]

                # Usa np.digitize con un singolo valore
                bin_index = np.digitize(current_value, bin_edges) - 1

                # Assicura che bin_index sia un singolo valore scalare e sia valido
                bin_index = max(0, min(bin_index, len(hist) - 1))

                frequency = hist[bin_index]
                probability *= max(frequency, epsilon)

            cluster_probabilities.append((cluster_label, probability))

        # Trova il cluster con la massima probabilità
        best_cluster = max(cluster_probabilities, key=lambda x: x[1])[0]

        # Restituisce tutti i dati del cluster
        best_cluster_data = None
        if self.data is not None:
            best_cluster_data = self.data[self.data['cluster'] == best_cluster]

        print(f"Cluster probabilities: {cluster_probabilities}")
        print(f"Best cluster found: {best_cluster}")

        if results:  # Return cluster and data
            return best_cluster, best_cluster_data

        return best_cluster_data


class IndoorPositioning:
    """
    Classe per la stima della posizione basata sui dati clusterizzati

    Usage
    ----------
    >>> current_rss = np.array([31, 47, 43])  # x=0,02	y=1,205
    >>> cluster_data = premodel.mle_predict(current_rss)

    >>> test_data = pd.read_csv('test_data.csv')
    >>> knn_positioning = IndoorPositioning(clustered_data=premodel.data,test_data=test_data ,columns=['RSSI A', 'RSSI B', 'RSSI C'], category='RP')
    >>> estimated_rp = knn_positioning.fit_predict(current_rss, k=28)

    >>> print(knn_positioning.evaluate())

    Parameters
    ----------
    clustered_data : DataFrame
        DataFrame contenente i dati clusterizzati con colonne per RSSI e 'cluster'

    test_data : DataFrame
        DataFrame contenente i dati di test

    columns : list
        Lista delle colonne RSSI da utilizzare

    category : str
        Colonna contenente le informazioni etichettate (es. RP)
    """

    def __init__(self, clustered_data, test_data=None, columns=None, category='RP'):
        """
        Inizializza la classe KNNPositioning con i dati clusterizzati.

        :param clustered_data: DataFrame contenente i dati clusterizzati con colonne per RSSI e 'cluster'
        :param columns: Lista delle colonne RSSI da utilizzare
        """
        if columns is None:
            columns = ['RSSI A', 'RSSI B', 'RSSI C']

        self.data = clustered_data
        self.columns = columns

        self._category = category
        self._test_data = test_data
        self._model = None
        self._scaler = None

        self._best_params = None
        self._position_map = None

    def save_model(self, filename, light=False):
        import joblib

        save_data = {
                'model': self._model,
                'scaler': self._scaler,
                'columns': self.columns,
                'category': self._category,
                'best_params': self._best_params,
                'position_map': self._position_map
            }

        if not light:
            save_data['data'] = self.data
            save_data['test_data'] = self._test_data

        try:
            joblib.dump(save_data, filename)
        except Exception as e:
            print(e)

    def load_model(self, filename):
        import joblib
        import os

        # Check file
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")

        try:
            file = joblib.load(filename)

            # Check if data is present
            if ('model' or 'scaler' or 'columns' or 'category' or 'position_map') not in file.keys():
                raise ValueError("Model file is missing data.")

            self._model = file['model']
            self._scaler = file['scaler']
            self.columns = file['columns']
            self._category = file['category']
            self._position_map = file['position_map']

            if ('data' or 'test_data' or 'best_params') not in file.keys():
                raise ValueError("Model file is missing clustering parameters.")

            self.data = file['data']
            self._test_data = file['test_data']
            self._best_params = file['best_params']

        except Exception as e:
            print(e)

    def _set_mapposition(self, force=False):
        """
        Crea una mappa per la posizione di ciascun RP
        """
        if self.data is None or 'RP' not in self.data.columns:
            return

        if self._position_map is None or force:
            self._position_map = self.data[['RP', 'x', 'y']].drop_duplicates().set_index('RP')

    def get_mapposition(self):
        """
        Restituisce la mappa delle posizioni
        """
        self._set_mapposition()
        return self._position_map

    def get_position(self, rp):
        """
        Restituisce la posizione di un RP
        """
        self._set_mapposition()
        return self._position_map.loc[rp]

    def transform(self, data, force=False):
        """
        Normalizza i nuovi dati
        :param data:
        :return:
        """
        # Check if data is DataFrame
        if isinstance(data, pd.DataFrame):
            data = data.values

        if self._scaler is None or force:
            self._scaler = MinMaxScaler()
            self._scaler.fit(data)

        return self._scaler.transform(data)

    def inverse_transform(self, data):
        """
        Effettua la trasformazione inversa dei dati
        :param data: Dati da trasformare
        :return: Dati trasformati
        """
        return self._scaler.inverse_transform(data)

    def _fit(self, k=3, p=2, leaf_size=30):
        """
        Addestra un modello KNN per la stima della posizione basata sui dati clusterizzati.

        :param k: Numero di vicini più prossimi da considerare
        :param p: Valore della distanza (1 per Manhattan, 2 per Euclidean)
        """
        knn = KNeighborsClassifier(n_neighbors=k, p=p, leaf_size=leaf_size)
        transformed_data = self.transform(self.data[self.columns])
        categories = self.data[self._category]
        start_time = datetime.timestamp(datetime.now())
        knn.fit(transformed_data, categories)
        end_time = datetime.timestamp(datetime.now())
        print(f"Training completed in {end_time - start_time} seconds.")

        self._model = knn

    def _predict(self, current_rssi):
        import pandas as pd
        """
        Usa KNN per stimare l'RP di appartenenza basandosi sui valori RSSI attuali.

        :param current_rssi: Lista contenente i valori RSS correnti per ciascun AP (es: [RSSI_A, RSSI_B, RSSI_C])
        :param k: Numero di vicini più prossimi da considerare
        :param p: Valore della distanza (1 per Manhattan, 2 per Euclidean)
        :return: L'RP stimato
        """

        if self._category not in self.data.columns:
            raise ValueError("RP information not available in data. Ensure data contains 'RP' column.")

        # Crea un DataFrame con current_rssi e nomi delle colonne
        current_rssi_df = pd.DataFrame([current_rssi], columns=self.columns)

        # Normalizza i dati
        current_rssi_df = self.transform(current_rssi_df)

        # if scaler minmax is used, drop rows with negative values and corresponding categories
        if isinstance(self._scaler, MinMaxScaler):
            current_rssi_df = current_rssi_df[(current_rssi_df >= 0).all(1)]

        if len(current_rssi_df) == 0:
            return None

        estimated_rp = self._model.predict(current_rssi_df)
        return estimated_rp[0]

    def optimal_params(self, kmeans=None):
        from sklearn.metrics import accuracy_score

        if kmeans is not None:
            for index, row in self._test_data.iterrows():
                # Trova il cluster al quale il RSS corrente appartiene usando kmeans
                cluster, _ = kmeans.mle_predict(row[self.columns].values, results=True)
                self._test_data.at[index, 'cluster'] = cluster

            clusters = kmeans.get_clusters()
        else:
            clusters = {0: self.data}
            self._test_data['cluster'] = 0

        # Per ogni cluster, addestra il KNN con un valore di k e testa le performance con i dati di test appartenenti al cluster e migliora il valore di k
        best_params = {}
        results = {}
        old_data = self.data.copy()

        for cluster_label, cluster_data in clusters.items():
            self.data = cluster_data

            leaf_size = list(range(1, 50))
            n_neighbors = list(range(1, 30))
            p = [1, 2]
            hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
            knn = KNeighborsClassifier()

            # Prendi tutti i dati di test che appartengono al cluster corrente e valuta le performance
            test_data = self._test_data[self._test_data['cluster'] == cluster_label].copy()

            transformed_data = self.transform(self.data[self.columns], force=True)
            categories = self.data[self._category]
            clf = GridSearchCV(estimator=knn, param_grid=hyperparameters, cv=10)
            best_model = clf.fit(transformed_data, categories)

            best_params[cluster_label] = clf.best_params_

            test_categories = test_data[self._category]
            test_transformed_data = self.transform(test_data[self.columns])
            test_predicted = clf.predict(test_transformed_data)

            accuracy = accuracy_score(test_categories, test_predicted)
            print(f"Cluster {cluster_label} - Accuracy: {accuracy}")
            results[cluster_label] = accuracy

        self.data = old_data
        self._best_params = best_params
        print(f"Best parameters found: {best_params}")
        return results, best_params

    def optimal_k(self, kmeans=None, p=2, leaf_size=30):
        from sklearn.metrics import accuracy_score
        # Assegno ciascun dato di test al cluster più vicino

        if kmeans is not None:
            for index, row in self._test_data.iterrows():
                # Trova il cluster al quale il RSS corrente appartiene usando kmeans
                cluster, _ = kmeans.mle_predict(row[self.columns].values, results=True)
                self._test_data.at[index, 'cluster'] = cluster

            clusters = kmeans.get_clusters()
        else:
            self._test_data['cluster'] = 0
            clusters = {0: self.data}

        # Per ogni cluster, addestra il KNN con un valore di k e testa le performance con i dati di test appartenenti al cluster e migliora il valore di k
        # best_k = 0
        best_score = {}
        old_data = self.data.copy()
        rp_positions = old_data[['RP', 'x', 'y']].drop_duplicates().set_index('RP').to_dict('index')
        # Ciclo per trovare il miglior valore di k per ogni cluster

        for cluster_label, cluster_data in clusters.items():
            best_score[cluster_label] = {1: 0}
            self.data = cluster_data.copy()
            self.transform(self.data[self.columns], force=True)  # Re-normalize data with new cluster

            for k in range(1, 100):
                try:
                    self.fit(k=k, p=p, leaf_size=leaf_size)
                except Exception as e:
                    print(e)
                    break

                test_data = self._test_data[self._test_data['cluster'] == cluster_label].copy()
                predicted_rps = []
                for i, row in test_data.iterrows():
                    predicted_rp = self._predict(row[self.columns].values)
                    if predicted_rp is not None:
                        predicted_rps.append(predicted_rp)
                    else:
                        # drop row if negative values are present
                        test_data.drop(index=i, inplace=True)

                test_data['predicted_rp'] = predicted_rps

                accuracy = accuracy_score(test_data['RP'], test_data['predicted_rp'])
                best_score[cluster_label][k] = accuracy

                print(f"Cluster {cluster_label} - k={k}, Accuracy: {accuracy}")

        self.data = old_data

        # Trova per ogni cluster trova il k con la migliore accuracy e restituisci il valore di k e l'accuracy
        best_k = {}
        for cluster_label, scores in best_score.items():
            best_k[cluster_label] = (max(scores, key=scores.get), scores[max(scores, key=scores.get)])
            print(
                f"Cluster {cluster_label} - Best k found: {best_k[cluster_label][0]}, Best Accuracy: {best_k[cluster_label][1]}")

        # Per ogni cluster, con il miglior k trovato, calcola l'errore medio di distanza tra le posizioni previste delle RPs e quelle reali
        for cluster_label, (k, _) in best_k.items():
            self.data = clusters[cluster_label].copy(deep=True)
            self.transform(self.data[self.columns], force=True)

            self.fit(k=k, p=p, leaf_size=leaf_size)
            test_data = self._test_data[self._test_data['cluster'] == cluster_label].copy()
            predicted_rps = []

            for i, row in test_data.iterrows():
                predicted_rp = self._predict(row[self.columns].values)
                if predicted_rp is not None:
                    predicted_rps.append(predicted_rp)
                else:
                    # drop row if negative values are present
                    test_data.drop(index=i, inplace=True)

            test_data['predicted_rp'] = predicted_rps

            # Calcola l'errore medio di distanza tra le posizioni previste delle RPs e quelle reali

            errors = []

            for index, row in test_data.iterrows():
                if row['RP'] in rp_positions.keys():
                    real_position = np.array([rp_positions[row['RP']]['x'], rp_positions[row['RP']]['y']])
                    predicted_rp = row['predicted_rp']
                    if predicted_rp in rp_positions.keys():
                        predicted_position = np.array(
                            [rp_positions[predicted_rp]['x'], rp_positions[predicted_rp]['y']])
                        error_distance = np.linalg.norm(real_position - predicted_position)
                        errors.append(error_distance)
                    else:
                        print(f"Warning: RP {predicted_rp} not found in cluster data.")
                else:
                    print(f"Warning: RP {row['RP']} not found in cluster data.")

            mean_error_distance = np.mean(errors)

            print(f"Cluster {cluster_label} - Best k found: {k}, Mean Error Distance: {mean_error_distance}")

            best_k[cluster_label] = (k, best_k[cluster_label][1], mean_error_distance)

        self.data = old_data

        # Restituisce anche la percentuale dei dati di training appartenenti al cluster
        cluster_test_percentage = {}
        for cluster_label, cluster_data in clusters.items():
            tot_data = len(self.data)
            cluster_data = len(cluster_data)
            cluster_test_percentage[cluster_label] = cluster_data / tot_data

        # Restituisce un dizionario del tipo {cluster_label: {'leaf_size': leaf_size, 'n_neighbors': best_cluster_k, 'p': 2}}
        res_params = {}
        for cluster_label, (best_cluster_k, accuracy, _) in best_k.items():
            res_params[str(cluster_label)] = {'leaf_size': leaf_size, 'n_neighbors': best_cluster_k, 'p': p}

        return best_k, res_params

    def evaluate(self, kmeans=None, k=3, p=2, leaf_size=30):
        """
        Valuta le performance del modello KNN utilizzando il dataset di test.

        :return: Dizionario con metriche di valutazione
        """
        if kmeans is None:
            print("Warning: KMeans model not found. Evaluating on whole dataset.")
            self.fit(k=k, p=p, leaf_size=leaf_size)

        # Prevedi le RPs usando il modello KNN
        predicted_rps = []
        old_data = self.data.copy()

        for index, row in self._test_data.iterrows():
            current_rssi = row[self.columns].values

            # Trova il cluster al quale il RSS corrente appartiene
            if kmeans is not None:
                cluster_data = kmeans.predict(current_rssi)
                self.data = cluster_data
                self.transform(self.data[self.columns], force=True)  # Re-normalize data with new cluster
                self.fit(k=k, p=p, leaf_size=leaf_size)

            predicted_rp = self._predict(current_rssi)
            if predicted_rp is not None:
                predicted_rps.append(predicted_rp)
            else:
                # drop row if negative values are present
                self._test_data.drop(index=index, inplace=True)

        self.data = old_data

        print(predicted_rps)
        self._test_data['predicted_rp'] = predicted_rps

        # Estrarre le posizioni delle RPs dai dati del cluster
        rp_positions = self.data[['RP', 'x', 'y']].drop_duplicates().set_index('RP').to_dict('index')

        # Calcola l'errore medio di distanza tra le posizioni previste delle RPs e quelle reali
        errors = []
        for index, row in self._test_data.iterrows():
            if row['RP'] in rp_positions.keys():
                real_position = np.array([rp_positions[row['RP']]['x'], rp_positions[row['RP']]['y']])
                predicted_rp = row['predicted_rp']
                if predicted_rp in rp_positions.keys():
                    predicted_position = np.array([rp_positions[predicted_rp]['x'], rp_positions[predicted_rp]['y']])
                    error_distance = np.linalg.norm(real_position - predicted_position)
                    errors.append(error_distance)
                else:
                    print(f"Warning: RP {predicted_rp} not found in cluster data.")
            else:
                print(f"Warning: RP {row['RP']} not found in cluster data.")

        mean_error_distance = np.mean(errors)
        mse_x = mean_squared_error([rp_positions[row['RP']]['x'] for _, row in self._test_data.iterrows()],
                                   [rp_positions[row['predicted_rp']]['x'] for _, row in self._test_data.iterrows()
                                    if row['predicted_rp'] in rp_positions.keys()])

        mse_y = mean_squared_error([rp_positions[row['RP']]['y'] for _, row in self._test_data.iterrows()],
                                   [rp_positions[row['predicted_rp']]['y'] for _, row in self._test_data.iterrows()
                                    if row['predicted_rp'] in rp_positions.keys()])

        mse = (mse_x + mse_y) / 2

        accuracy = len(self._test_data[self._test_data['RP'] == self._test_data['predicted_rp']]) / len(self._test_data)

        results = {
            'mean_error_distance': mean_error_distance,
            'mse': mse,
            'mse_x': mse_x,
            'mse_y': mse_y,
            'accuracy': f"{round(accuracy * 100, 3)}%"
        }

        return results, predicted_rps

    def fit_clusters(self, kmeans=None, k=3, p=2, leaf_size=30, params=None, save_dir_path=None):
        """
        Effettua il fitting del modello KNN
        :param k: Numero di vicini più prossimi da considerare
        :param p: Valore della distanza (1 per Manhattan, 2 per Euclidean)
        """
        if params:
            self._best_params = params

        if kmeans is None:
            print("Warning: KMeans model not found. Evaluating on whole dataset.")
            self.fit(k=k, p=p, leaf_size=leaf_size)
        else:
            clusters = kmeans.get_clusters()

            for cluster_label, cluster_data in clusters.items():
                self.data = cluster_data.copy()

                # Position map
                self._set_mapposition()

                self.transform(self.data[self.columns], force=True)

                if self._best_params is not None:
                    params = self._best_params[str(cluster_label)]
                    k = params['n_neighbors']
                    p = params['p']
                    leaf_size = params['leaf_size']

                self.fit(k=k, p=p, leaf_size=leaf_size)

                if save_dir_path is not None:
                    save_path = os.path.join(save_dir_path, f"model_cluster_{cluster_label}.joblib")
                    # Save model for each cluster without data
                    self.save_model(save_path, light=True)

    def fit_predict(self, data, k=3, p=2, leaf_size=30):
        """
        Effettua il fitting e la predizione del modello KNN
        :param data: Lista contenente i valori RSS correnti per ciascun AP
        :param k: Numero di vicini più prossimi da considerare
        :param p: Valore della distanza (1 per Manhattan, 2 per Euclidean)
        :return: L'RP stimato
        """
        self._fit(
            k=k,
            p=p,
            leaf_size=leaf_size
        )

        return self._predict(data)

    def predict(self, data):
        """
        Effettua la predizione del modello KNN
        :param data: Lista contenente i valori RSS correnti per ciascun AP
        :return: L'RP stimato
        """
        return self._predict(data)

    def fit(self, k=3, p=2, leaf_size=30):
        """
        Effettua il fitting del modello KNN
        :param k: Numero di vicini più prossimi da considerare
        :param p: Valore della distanza (1 per Manhattan, 2 per Euclidean)
        """
        self._fit(
            k=k,
            p=p,
            leaf_size=leaf_size
        )

