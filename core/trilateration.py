# Define a Python class to encapsulate all the steps for trilateration and error calculation
import streamlit
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import numpy as np
import pandas as pd


class TrilaterationEstimator:
    def __init__(self, ap_positions, use_effective_distance=False):
        """
        Initialize the TrilaterationEstimator with the positions of the access points.

        :param ap_positions: Dictionary of AP positions {AP: (x, y)}
        """
        self.ap_positions = ap_positions
        self.rssi_d0 = {}
        self.path_loss_exponents = {}
        self.use_effective_distance = use_effective_distance

    def estimate_distance_from_rssi(self, rssi, rssi_d0, path_loss_exponent, d0=1):
        """Estimate the distance based on RSSI values."""
        return d0 * 10 ** ((rssi_d0 - rssi) / (10 * path_loss_exponent))

    def estimate_distances(self, rssi_values, rp_position=None):
        """
        Estimate the distances to each AP from the RSSI values.

        :param rssi_values: Dictionary of RSSI values for each AP.
        :param rp_position: Position of the reference point (RP) being evaluated (only used if using effective distance).
        """
        distances = {}
        for ap, rssi in rssi_values.items():
            if self.use_effective_distance:
                # Calculate the effective distance between the AP and the RP (real position)
                d_eff = self.calculate_real_distance(self.ap_positions[ap], rp_position)
            else:
                d_eff = 1  # Fixed d0 = 1 meter if the traditional method is used

            distances[ap] = self.estimate_distance_from_rssi(rssi, self.rssi_d0[ap], self.path_loss_exponents[ap],
                                                             d_eff)
        return distances

    @staticmethod
    def calculate_real_distance(ap_position, rp_position):
        """Calculate the Euclidean distance between an AP and an RP."""
        return np.sqrt((ap_position['x'] - rp_position['x']) ** 2 + (ap_position['y'] - rp_position['y']) ** 2)

    def trilateration(self, positions, distances):
        """Perform trilateration to estimate position."""

        class TrilaterationFunction:
            def __init__(self, positions, distances):
                self.positions = np.array(positions)
                self.distances = np.array(distances)

            def residuals(self, point):
                return np.sqrt(np.sum((self.positions - point) ** 2, axis=1)) - self.distances

        # Convert the list of dictionaries to a NumPy array of coordinates
        positions_array = np.array([[pos['x'], pos['y']] for pos in positions])

        initial_guess = np.mean(positions_array, axis=0)  # Now this works correctly
        result = least_squares(TrilaterationFunction(positions_array, distances).residuals, initial_guess)
        return result.x

    def estimate_position(self, rssi_values):
        """Estimate the position given RSSI values from each AP."""
        ap_positions_array = np.array([self.ap_positions['A'], self.ap_positions['B'], self.ap_positions['C']])
        estimated_distances = self.estimate_distances(rssi_values)
        distances = [estimated_distances['A'], estimated_distances['B'], estimated_distances['C']]
        return self.trilateration(ap_positions_array, distances)

    def calculate_mean_error(self, estimated_positions, true_positions):
        """Calculate the mean distance error between estimated and true positions."""
        errors = []
        for est_pos, true_pos in zip(estimated_positions, true_positions):
            error = np.sqrt((est_pos[0] - true_pos[0]) ** 2 + (est_pos[1] - true_pos[1]) ** 2)
            errors.append(error)
        return np.mean(errors)

    def estimate_path_loss_and_rssi_d0(self, df):
        """Estimate the path loss exponent and RSSI(d0) for each access point."""

        def estimate_params(d, rssi):
            log_d = np.log10(d)
            A = np.vstack([log_d, np.ones(len(log_d))]).T
            model = np.linalg.lstsq(A, rssi, rcond=None)
            path_loss_exponent, rssi_d0 = model[0]
            if path_loss_exponent < 10e-6:
                path_loss_exponent = 5.6
            return path_loss_exponent, rssi_d0

        def estimate_params_with_constraints(d, rssi):
            from scipy.optimize import minimize
            log_d = np.log10(d)
            A = np.vstack([log_d, np.ones(len(log_d))]).T

            # Define the objective function (squared error) to minimize
            def objective_function(params):
                path_loss_exponent, rssi_d0 = params
                predicted_rssi = path_loss_exponent * log_d + rssi_d0
                error = np.sum((predicted_rssi - rssi) ** 2)
                return error

            # Initial guess for path loss exponent and RSSI(d0)
            initial_guess = [-2, np.mean(rssi)]  # Negative initial guess to simulate realistic cases

            # Define the constraint: path_loss_exponent >= 0
            constraints = [{'type': 'ineq', 'fun': lambda x: x[0]}]

            # Minimize the objective function with the constraint on path_loss_exponent
            result = minimize(objective_function, initial_guess, constraints=constraints)

            # Extract the optimized path loss exponent and RSSI(d0)
            path_loss_exponent, rssi_d0 = result.x
            if path_loss_exponent < 10e-6:
                path_loss_exponent = 0.1  # Set a small value if the optimization fails
            return path_loss_exponent, rssi_d0

        # Raggruppa per RP e calcola la media dei valori RSSI per ogni AP
        #grouped = df.groupby('RP').mean()
        grouped = df

        # Calcola per ciascun AP
        for ap in ['A', 'B', 'C']:
            distances = grouped[f'distance_to_AP_{ap}']
            rssi_values = grouped[f'RSSI{ap}']
            path_loss_exponent, rssi_d0 = estimate_params(distances, rssi_values)
            self.path_loss_exponents[ap] = path_loss_exponent
            self.rssi_d0[ap] = rssi_d0



    def calculate_distances(self, df):
        """Calculate the distances to the APs based on positions."""

        def calculate_distance(df, ap_position):
            return np.sqrt((df['x'] - ap_position['x']) ** 2 + (df['y'] - ap_position['y']) ** 2)

        df[f'distance_to_AP_A'] = calculate_distance(df, self.ap_positions['A'])
        df[f'distance_to_AP_B'] = calculate_distance(df, self.ap_positions['B'])
        df[f'distance_to_AP_C'] = calculate_distance(df, self.ap_positions['C'])
        return df

    def run_trilateration(self, df, rp_to_position):
        """
        Run the full trilateration process on a dataframe with RSSI values.

        :param df: DataFrame containing RSSI values and reference points (RP).
        :param rp_to_position: Dictionary mapping reference points (RP) to true positions.
        :return: Mean error of the estimated positions.
        """
        # Estimate positions and calculate the error
        estimated_positions = []
        true_positions = []

        for _, row in df.iterrows():
            rssi_values = {'A': row['RSSIA'], 'B': row['RSSIB'], 'C': row['RSSIC']}
            estimated_position = self.estimate_position(rssi_values)
            estimated_positions.append(estimated_position)

            true_position = (rp_to_position[row['RP']]['x'], rp_to_position[row['RP']]['y'])
            true_positions.append(true_position)

        mean_error = self.calculate_mean_error(estimated_positions, true_positions)
        return mean_error, estimated_positions

    @staticmethod
    def plot_positions_rp(rp_positions, estimated_positions):
        """
        Plot the true positions (RP) and the estimated positions obtained through trilateration.

        :param rp_positions: Dictionary of true positions {RP: {'x': x_value, 'y': y_value}}.
        :param estimated_positions: List of estimated positions (x, y).
        """
        # Extract true positions from rp_positions
        true_x = [pos['x'] for pos in rp_positions.values()]
        true_y = [pos['y'] for pos in rp_positions.values()]

        # Extract estimated positions
        est_x = [pos[0] for pos in estimated_positions]
        est_y = [pos[1] for pos in estimated_positions]

        fig = plt.figure(figsize=(8, 8))

        # Plot true positions (RP)
        plt.scatter(true_x, true_y, color='blue', label='True Positions (RP)', marker='o')

        # Plot estimated positions
        plt.scatter(est_x, est_y, color='red', label='Estimated Positions', marker='x')

        # Draw lines between true and estimated positions to visualize the error
        for tx, ty, ex, ey in zip(true_x, true_y, est_x, est_y):
            plt.plot([tx, ex], [ty, ey], color='gray', linestyle='--')

        plt.title('True Positions (RP) vs Estimated Positions')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True)
        streamlit.pyplot(fig)


# Example usage:
# 1. Initialize the class with access point positions, rssi_d0, and path loss exponents.
# 2. Run trilateration on a given dataset and compute the mean error.
# ap_pos_path = '../data/ap_positions.csv'
# ap_positions = pd.read_csv(ap_pos_path).set_index('AP').to_dict('index')
# estimator = TrilaterationEstimator(ap_positions)
#
# df = pd.read_csv('../data/extracted_data_training_real.csv')
#
# # Assuming 'df' contains RSSI measurements and positions
# df_with_distances = estimator.calculate_distances(df)
# estimator.estimate_path_loss_and_rssi_d0(df_with_distances)
# print(estimator.path_loss_exponents)
#
# # Mapping reference points to true positions
# rp_to_position = df[['RP', 'x', 'y']].drop_duplicates().set_index('RP').to_dict('index')
# test_df = pd.read_csv('../data/extracted_data_test.csv')
#
# mean_error, positions = estimator.run_trilateration(df, rp_to_position)
#
# print(f"Mean error: {mean_error}")
# estimator.plot_positions_rp(rp_to_position, positions)