from pymongo import MongoClient
import pandas as pd

import requests


class MongoDataProcessor:
    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None

    def connect(self):
        """ Connette al database MongoDB. """
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]

    def fetch_data(self, collection_name):
        """ Recupera i dati dalla collezione MongoDB specificata. """
        if self.db is None:
            raise Exception("Non connesso a nessun database.")
        collection = self.db[collection_name]
        data = list(collection.find())
        return data

    @staticmethod
    def transform_data(data):
        """ Trasforma i dati JSON in un DataFrame pandas e CSV. """
        rows = []
        for item in data:
            x = item['x']
            y = item['y']

            # Escludi i dati con x=0, y=0 o x=999, y=999
            if (x == 0 and y == 0) or (x == 999 and y == 999):
                continue

            for record in item['data']:
                # Escludi i record che non hanno tutti i tre valori RSSI
                if 'rssiA' not in record or 'rssiB' not in record or 'rssiC' not in record:
                    continue

                row = {
                    'x': x,
                    'y': y,
                    'RSSIA': record['rssiA'],
                    'RSSIB': record['rssiB'],
                    'RSSIC': record['rssiC']
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False)
        return csv_data, df

    def close_connection(self):
        """ Chiude la connessione al database. """
        if self.client:
            self.client.close()


class APIDataProcessor:
    def __init__(self, api_url):
        self.api_url = api_url
        self.token = ""

    def set_url(self, api_url):
        self.api_url = api_url

    def login(self, login_url, username, password):
        """ Effettua il login all'API e salva il token di accesso. """
        response = requests.post(login_url, json={'username': username, 'password': password})
        response.raise_for_status()
        self.token = response.json()['access_token']
        return self.token

    def fetch_data(self):
        """ Effettua una richiesta GET all'API e restituisce i dati. """
        response = requests.get(self.api_url)
        response.raise_for_status()  # Solleva un'eccezione se la richiesta fallisce
        data = response.json()
        return data

    def upload_data(self, data):
        """ Effettua una richiesta POST all'API per caricare i dati. """
        response = requests.post(self.api_url, json=data)
        response.raise_for_status()
        return response.json()

    def upload_file(self, file_path, name, description, timestamp):
        """
        Effettua una richiesta POST all'API per caricare un file con informazioni aggiuntive.

        :param file_path: Path del file da caricare.
        :param name: Nome del file o altra informazione.
        :param description: Descrizione del file.
        """
        filename = file_path.split('/')[-1]  # Extract the filename from the path

        # Open the file in binary mode
        with open(file_path, 'rb') as f:
            files = {'file': (filename, f)}  # Set up the file for the form data

            # Include additional metadata like name and description
            data = {
                'name': name,
                'description': description,
                'timestamp': timestamp
            }

            headers = {'Authorization': f'Bearer {self.token}'}

            # Make the POST request to the API
            response = requests.post(self.api_url, files=files, data=data, headers=headers)

            # Raise an error if the request failed
            response.raise_for_status()

        # Return the API's response in JSON format
        return response.json()

    @staticmethod
    def transform_data(data):
        """ Trasforma i dati JSON in un DataFrame pandas e CSV. """
        rows = []
        for item in data:
            # Controlla se presente x,y e data oppure solo RP e data
            if 'x' in item and 'y' in item:
                x = item['x']
                y = item['y']
                rp = None
            else:
                x = None
                y = None
                rp = item['RP']

            # Escludi i dati con x=0, y=0 o x=999, y=999
            if (x == 0 and y == 0) or (x == 999 and y == 999) or (x == 1 and y == 1):
                continue

            for record in item['data']:
                # Escludi i record che non hanno tutti i tre valori RSSI (non case sensitive)
                # Mette tutto in minuscolo per evitare errori di battitura
                keys = [key.lower() for key in record.keys()]

                # Controlla che ci siano almeno tre chiavi che contengono 'rssi'
                if len(keys) < 3:
                    continue

                # Estrae i valori RSSI in maniera dinamica
                rssis = {key.upper(): record[key] for key in record if 'rssi' in key.lower()}

                # Se presenti aggiunge x e y e i valori RSSI, se x e y non presenti il dataset ha come colonne solo RSSI e RP
                if x and y:
                    row = {'x': x, 'y': y, **rssis}
                else:
                    row = {**rssis, 'RP': rp}

                rows.append(row)

        # Clean outliers in rssis columns using the IQR method
        df = pd.DataFrame(rows)

        df, rows_removed = APIDataProcessor.data_cleaning(df)

        csv_data = df.to_csv(index=False)
        return csv_data, df, rows_removed

    @staticmethod
    def data_cleaning(df):
        """
        Clean outliers in RSSI columns using the IQR method.
        """
        rows_removed = len(df)

        rssis_columns = [col for col in df.columns if col.lower().startswith('rssi')]
        q1 = df[rssis_columns].quantile(0.25)
        q3 = df[rssis_columns].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[~((df[rssis_columns] < lower_bound) | (df[rssis_columns] > upper_bound)).any(axis=1)]

        df = df[(df[rssis_columns] > -70).all(axis=1)]

        rows_removed -= len(df)

        return df, rows_removed
