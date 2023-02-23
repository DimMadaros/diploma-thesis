"""
Created on Fri Dec 30 20:17:47 2022

@author: Dimitris Madaros
"""
import numpy as np
import pandas as pd
from numpy.random import default_rng

DEFAULT_DATASET_NO = 3  # In case no dataset number is given
MAX_COSTUMER_DEMAND = 10
DEPOT_LOCATION = 0
WITH_SENSOR_DATA = False
TWL = 0.70  # Threshold Waste Level


# noinspection PyUnusedLocal,PyMethodMayBeStatic
class Data:
    def __init__(self, dataset_no, with_sensor_data=WITH_SENSOR_DATA):
        self.dataset = self.choose_dataset(dataset_no)  # DataFrame
        self.distance_matrix = self.calc_distance_matrix(self.dataset)  # DataFrame
        self.with_sensor_data = with_sensor_data  # If sensor data will be used
        self.x = self.dataset["X"].values  # A NumPy array of the x coordinates of the nodes
        self.y = self.dataset["Y"].values  # A NumPy array of the y coordinates of the nodes
        self._all_nodes = list(self.distance_matrix.columns.values)  # A list of the nodes
        self.all_size = len(self._all_nodes)  # The size of the tour
        self._nodes_without_depot = np.delete(self._all_nodes,
                                              DEPOT_LOCATION)  # A NumPy array of the nodes without the depot
        self.size_without_depot = len(self._nodes_without_depot)  # The size of the tour without the depot
        self.default_dataset_number = DEFAULT_DATASET_NO
        self._random_demand = self.calc_random_demand(self.dataset, MAX_COSTUMER_DEMAND)  # A NumPy array of demands
        self._dataset_demand = self.dataset['Demand']  # The Dataset's demands
        self.sensor_dataset = self.calc_sensor_dataset()  # The dataset with demands above TWL
        self.sensor_distance_matrix = self.calc_distance_matrix(self.sensor_dataset)
        self.sensor_nodes = list(self.sensor_distance_matrix.columns.values)  # A list of the sensor nodes
        self.sensor_nodes_without_depot = np.delete(self.sensor_nodes,
                                                    DEPOT_LOCATION)  # A NumPy array of the nodes without the depot

    def choose_dataset(self, dataset_no=None):
        """
        Chooses one of the five datasets based on the input number and returns it.
        :param int dataset_no: A Pandas DataFrame with coordinates.
        :return: The specified Dataset.
        """
        # Check if an input has been given
        if dataset_no is None:
            dataset_no = self.default_dataset_number

        # Choose between different coordinates-datasets
        match dataset_no:
            case 1:
                dataset = pd.read_excel("dataset1.xlsx")  # Pandas Dataframe
            case 2:
                dataset = pd.read_excel("dataset2.xlsx")  # Pandas Dataframe
            case 3:
                dataset = pd.read_excel("dataset3.xlsx")  # Pandas Dataframe
            case 4:
                dataset = pd.read_excel("dataset4.xlsx")  # Pandas Dataframe
            case 5:
                dataset = pd.read_excel("dataset5.xlsx")  # Pandas Dataframe
            case other:
                dataset = pd.read_excel("dataset3.xlsx")  # Pandas Dataframe

        return dataset

    def calc_distance_matrix(self, dataset=None):
        """
        Derive the Distance Matrix from the given Dataset.
        :param DataFrame dataset: A Pandas DataFrame with coordinates.
        :return: A Pandas Dataframe with the euclidean distances between the nodes.
        """
        # Check if an input has been given
        if dataset is None:
            dataset = self.dataset

        # Initialize a (num_of_nodes x num_of_nodes) NumPy array with zeros
        distance_matrix = np.zeros([len(dataset), len(dataset)])
        x = dataset['X'].values
        y = dataset['Y'].values

        for i in range(len(dataset)):
            # Get the x and y coordinates of every node
            x1 = x[i]
            y1 = y[i]
            for j in range(len(dataset)):
                # Get the x and y coordinates of every other node again
                x2 = x[j]
                y2 = y[j]
                # Calculate every distance between every node and populate the distances_matrix
                if i != j:
                    distance_matrix[i, j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    # We are rounding up in order to avoid violations of the triangle inequality
                else:
                    distance_matrix[i, j] = np.inf

        # Create a list of the indexes for the Pandas DataFrame
        indexes = list(np.arange(0, len(dataset)))  # Because we start from 1 instead of 0
        # Create the Pandas DataFrame of the distances
        distance_dataframe = pd.DataFrame(distance_matrix, columns=indexes, index=indexes)

        return distance_dataframe

    def create_random_tour(self, nodes=None):
        """
        Generates a random tour from a list of nodes.
        :param list nodes: A list of nodes.
        :return: A random permutation of the input list.
        """
        # Check if an input has been given
        if nodes is None:
            nodes = self._all_nodes

        rng = default_rng()  # Create a new instance of numpy Generator for random methods

        # Make a random permutation of the nodes to create a random tour
        random_tour = rng.permutation(nodes)

        # To make random tour an actual tour, we have to add the first node to the end of the list
        random_tour = np.append(random_tour, random_tour[0])

        return random_tour

    def calc_tour_cost(self, tour, distance_matrix):
        """
        Calculates the cost of a tour given the distances.
        :param list tour: A list of nodes in the order visited.
        :param DataFrame distance_matrix: A matrix of the distances between each node.
        :return: The sum of the distances of the tour.
        """
        # Create a list of the distances of each edge
        distances = []

        for i in range(len(tour) - 1):
            distance = distance_matrix.loc[tour[i], tour[i + 1]]  # Each city and the city after it
            distances.append(distance)  # Append the distances

        cost = sum(distances)  # Add up the distances

        return cost

    def calc_random_demand(self, dataset=None, max_bin_capacity=MAX_COSTUMER_DEMAND):
        """
        Calculates a random demand for every costumer.
        :param DataFrame dataset: A Pandas DataFrame with coordinates.
        :param int max_bin_capacity: The maximum demand a costumer can have.
        :return: A random list of demands.
        """
        # Check if an input has been given
        if dataset is None:
            dataset = self.dataset

        rng = default_rng()  # Create a new instance of numpy Generator for random methods

        demand = rng.integers(1, max_bin_capacity + 1, size=len(dataset))
        demand[0] = 0  # The depot has zero demand

        # self.dataset['Demand'] = demand

        return demand

    def calc_sensor_dataset(self):
        """
        Creates a new Dataset based on the sensor data. Nodes with demands lower than the TWL are being discarded.
        :return: The new Dataset based on the sensor data.
        """
        df = self.dataset.copy()  # Copy the original Dataset
        df.loc[0, ['Demand']] = MAX_COSTUMER_DEMAND  # Make the depot demand equal to the max bin capacity
        df = df[df['Demand'] >= TWL*MAX_COSTUMER_DEMAND]  # Create a new dataset based on the TWL
        df.loc[0, ['Demand']] = 0  # Make the depot demand equal to 0 again

        return df
