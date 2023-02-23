"""
Created on Fri Dec 30 21:55:32 2022

@author: Dimitris Madaros
"""
import data as dt

MAX_BIN_CAPACITY = 10
MAX_TRUCK_CAPACITY = 70
TWL = 0.7  # Threshold Waste Level
WITH_SENSOR_DATA = False
WITH_DEPOT = True
WITH_RANDOM_DEMAND = True


class VRP(dt.Data):
    def __init__(self, dataset_no, with_depot=WITH_DEPOT, with_random_demand=WITH_RANDOM_DEMAND,
                 with_sensor_data=WITH_SENSOR_DATA):
        super().__init__(dataset_no)
        self.with_sensor_data = with_sensor_data  # If sensor data will be used
        self.dataset_copy = self.choose_correct_dataset(self.with_sensor_data)  # A copy to avoid references
        self.distance_matrix_copy = self.choose_correct_distance_matrix(
            self.with_sensor_data)  # A copy to avoid references
        self.fitness_distance_matrix = \
            self.choose_fitness_distance_matrix()  # The distance matrix from which the fitness will be calculated
        self.max_bin_capacity = MAX_BIN_CAPACITY  # Max capacity of the bins
        self.max_truck_capacity = MAX_TRUCK_CAPACITY  # Max capacity of the trucks
        self.twl = TWL  # Threshold Waste Level - TWL
        self.with_depot = with_depot  # Choose between nodes and node_without_depot
        self.with_random_demand = with_random_demand  # Use random demand or Dataset Demand
        self.demand = self.choose_demand(self.with_random_demand)
        self.total_demand = sum(list(self.demand))  # The total random demand of the bins
        self.truck_demands = []  # A list of every truck's collected demand
        self.nodes = self.choose_correct_nodes(self.with_depot, self.with_sensor_data)
        self.size = len(self.nodes)
        self.x0 = self.create_random_tour(self.nodes)  # Calculate an Initial Random Tour (Numpy Array)
        self.x0_fitness = self.calc_tour_cost(self.x0, self.distance_matrix)  # The Fitness of the Initial Tour
        self.init_vrp = self.create_vrp(self.x0, self.demand,
                                        self.max_truck_capacity)  # Repurpose the tour to truck tours
        self.init_vrp_fitness = self.fitness_function(self.init_vrp,
                                                      self.distance_matrix)  # Calculate the fitness of the initial VRP
        self.best_vrp = self.init_vrp  # The best Tour
        self.best_vrp_fitness = self.init_vrp_fitness  # The Fitness of the best Tour
        self.all_fitness = [self.init_vrp_fitness]  # Set up a list of every fitness for Iteration v Fitness plot

    def choose_correct_nodes(self, with_depot=WITH_DEPOT, with_sensor_data=WITH_SENSOR_DATA):
        """
        Chooses the nodes that will be used based on the specifications of the problem. Specifically, if the
        depot is needed and if sensor data will be used.
        :param bool with_depot: Check if the depot is needed for the calculations.
        :param bool with_sensor_data: Check if sensor data will be used.
        :return: The list of desired nodes.
        """
        if with_depot and with_sensor_data:  # Keep the set of sensor nodes
            nodes = self.sensor_nodes
        elif with_depot and not with_sensor_data:
            nodes = self._all_nodes  # Keep the complete set of nodes
        elif not with_depot and not with_sensor_data:  # If the depot is not needed, remove it
            nodes = self._nodes_without_depot
        else:
            nodes = self.sensor_nodes_without_depot

        return nodes

    def choose_correct_dataset(self, with_sensor_data=WITH_SENSOR_DATA):
        """
        Chooses between the original dataset and the sensor dataset.
        :param bool with_sensor_data: Checks if sensor data will be used.
        :return:
        """
        if with_sensor_data:
            dataset = self.sensor_dataset.copy()
        else:
            dataset = self.dataset.copy()

        return dataset

    def choose_correct_distance_matrix(self, with_sensor_data=WITH_SENSOR_DATA):
        """
        Chooses between the original distance matrix and the sensor distance matrix.
        :param bool with_sensor_data: Checks if sensor data will be used.
        :return:
        """
        if with_sensor_data:
            distance_matrix = self.sensor_distance_matrix.copy()
        else:
            distance_matrix = self.distance_matrix.copy()

        return distance_matrix

    def choose_fitness_distance_matrix(self):
        """
        Chooses which distance matrix will be used to calculate the fitness function. May be used with ACO.
        :return: The distance matrix.
        """
        if self.with_sensor_data:
            distance_matrix = self.sensor_distance_matrix
        else:
            distance_matrix = self.distance_matrix

        return distance_matrix

    def choose_demand(self, with_random_demand=WITH_RANDOM_DEMAND):
        """
        Checks if a new set of demands needs to be calculated.
        :param bool with_random_demand: Check if we want a random set of demands,
        :return: A list of demands.
        """
        if with_random_demand:
            demand = self._random_demand  # Choose the random demand
        else:
            demand = self._dataset_demand  # Choose the original demand

        return demand

    # noinspection PyMethodMayBeStatic
    def create_vrp(self, tour, demand, max_truck_capacity):
        """
        Generates a random list of truck tours, without violating the maximum truck capacities. If sensor data are
        considered, then the tour is formed based on the TWL. The nodes bellow TWL are discarded.
        :param list tour: A list of nodes in the order visited.
        :param demand: A list of the demands of the tour nodes in the order visited.
        :param int max_truck_capacity: The maximum capacity of the trucks.
        :return: A list of lists of nodes visited by each truck.
        """

        vrp = []  # A list of all the truck tours
        current_truck = []  # The tour of a truck
        current_truck_capacity = 0  # The current capacity of the truck

        tour = list(tour)  # Tour argument is a NumPy array. Make it a list
        tour.pop()  # Remove the last node of the tour, because it is the same as the first

        for i in range(len(tour)):
            current_truck_capacity += demand[tour[i]]  # Add to the truck the demand of the ith node of the tour

            # Check if the max truck capacity has been violated
            if current_truck_capacity <= max_truck_capacity:
                current_truck.append(tour[i])  # Append this node to the truck tour
            else:
                # The truck capacity has been violated
                current_truck_capacity -= demand[tour[i]]  # Remove the capacity of the last node

                # Add the depot at the beginning and at the end of the current truck to complete its tour
                truck = [0] + current_truck + [0]
                # Add this truck to the vrp list
                vrp.append(truck)

                # Create the new truck
                current_truck_capacity = demand[tour[i]]  # Append to the new truck the demand of this node
                current_truck = [tour[i]]  # Create the tour of the new truck starting with the current node

        # Don't forget the last truck!
        truck = [0] + current_truck + [0]
        vrp.append(truck)

        return vrp

    def fitness_function(self, vrp, distance_matrix):
        """
        The Fitness Function of the VRP.
        :param list vrp: A list of truck tours.
        :param DataFrame distance_matrix: A matrix of the distances between each node.
        :return: The sum of the cost of each truck tour.
        """
        fitness = []
        for i in range(len(vrp)):
            cost = self.calc_tour_cost(vrp[i], distance_matrix)
            fitness.append(cost)

        return sum(fitness)
