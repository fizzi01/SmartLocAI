import streamlit as st
import yaml
from streamlit_authenticator import Authenticate
from yaml.loader import SafeLoader

st.set_page_config(
    page_title="Clustering e Posizionamento Indoor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


with open('settings.yml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

try:
    name, authentication_status, username = authenticator.login("main",captcha=True)

    if authentication_status:

        st.sidebar.markdown("Use the sidebar to navigate between different sections of the application.")
        authenticator.logout('Logout', 'sidebar')

        st.title("SmartLocAI: Clustering and Indoor Positioning")

        st.markdown("""
This application is designed to analyze and visualize RSSI data from WiFi and BLE signals, utilizing clustering algorithms to estimate positions within an indoor environment. The goal is to create a workflow that leverages RSSI data to accurately estimate spatial positions, which is crucial for developing robust indoor positioning systems.

## Key Features

1. **Synthetic Data Generation**:
    - Generate synthetic datasets using pre-trained AI models. Users can select a model and specify the number of data points to generate. The generated data is displayed within the app and can be downloaded for further analysis.

2. **Data Clustering**:
    - Apply K-means clustering to RSSI data to identify signal clusters. Users can train a new clustering model or load an existing model to perform partial fitting. The clustering results are visualized through interactive 3D plots, allowing users to explore the data's spatial distribution.

3. **KNN Positioning Test**:
    - Allows users to upload a test dataset and perform KNN (K-Nearest Neighbors) positioning. Users can also manually specify RSSI values. The estimated positioning result is displayed and evaluated, providing insights into the accuracy of the positioning process.

4. **Data Extraction and Analysis from API**:
    - Extract RSSI data from an API that provides positioning information. Data can be filtered and transformed into a CSV format, displayed within the app, and downloaded for further analysis. Descriptive statistics are included to better understand the extracted data.

5. **Interactive Visualizations**:
    - Utilize interactive 3D plots to visualize clustering results, enabling users to explore the data interactively and intuitively. These visualizations help in understanding the relationships between different RSSI values and their corresponding spatial positions.

## How to Use This Application

1. **Navigation**: Use the sidebar to navigate between different sections of the application. Each section provides specific functionalities for generating, clustering, and analyzing data.
2. **Uploading Files**: Upload your test data or pre-trained models through the file upload interfaces available in the respective sections.
3. **Interacting with Graphs**: Explore the 3D interactive plots to visualize clusters and understand the spatial distribution of RSSI signals.
4. **Downloading Data**: Download the processed data, synthetic datasets, or clustering results for further analysis or integration with other tools.

## Objective

The primary objective is to provide a comprehensive workflow that facilitates the use of RSSI data from WiFi and BLE signals for accurate spatial position estimation. By enabling users to explore clustering patterns and improve positioning accuracy, this application serves as a valuable tool for researchers, developers, and professionals working in the field of wireless networks and localization systems.

---

**Note**: For optimal results provide suitable test datasets. The accuracy of the clustering and positioning results depends on the quality and quantity of the input data.
        """)
    elif authentication_status is None:
        st.warning('Please enter your username and password')
    elif not authentication_status:
        st.error('Username/password is incorrect')

except Exception as e:
    st.error(f"{e}")

