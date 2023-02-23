"""
Created on Mon Jan 9 13:33:19 2023

@author: Dimitris Madaros
"""
import time
import numpy as np
from numpy.random import default_rng


class ACO:
    def __init__(self, vrp, max_iter=100, num_of_ants=10, sa=None):
        self.vrp = vrp  # An instance of a VRP problem
        self.sa = sa  # An SA algorithm for refining
        self.distance_matrix = vrp.distance_matrix_copy  # A copy of the distance matrix to avoid references
        self.max_iter = max_iter  # Max number of iterations
        self.num_of_ants = num_of_ants  # Number of ants in the colony
        self.c = 10  # Parameter for the initial pheromone calculation
        self.rho = 0.05  # Evaporation rate. 5% of the pheromone evaporates with each iteration
        self.a = 1  # Pheromone exponential parameter Alpha
        self.b = 1  # Desirability exponential parameter Beta
        self.tau0 = self.calc_init_pheromone()  # τ0 = Initial Pheromone level (float)
        self.tau = self.calc_pheromone_matrix()  # τ  = Pheromone Matrix (NumPy ndarray)
        self.eta = self.calc_desirability_matrix()  # η  = Desirability of each edge (NumPy ndarray)
        self.colony = self.create_colony()  # Create a colony of ants (a list of tours)
        self.colony_vrp = self.create_colony_vrp(self.colony)  # The VRPs formed from every ant
        self.colony_fitness = self.calc_colony_fitness(self.colony_vrp)  # Calculate colony's fitness
        self.refined_vrp = self.vrp.best_vrp  # The VRP after the refining procedure
        self.refined_vrp_fitness = self.vrp.best_vrp_fitness  # The fitness value of the refined VRP
        self.aco_time = 0  # The performance time of the ACO
        self.refining_time = 0  # The performance time of the refining
        self.aco_time_with_refining = self.aco_time +\
                                      self.refining_time  # The total performance time of ACO with refining

    def calc_init_pheromone(self):
        """
        Calculates the initial pheromone level based on the expression:
         c * [1 / (number_of_nodes * average_of_all_distances)]
        :return: The initial pheromone level τ0 (tau 0)
        """
        # Convert the distance matrix to NumPy ndarray
        distance_matrix = self.distance_matrix.to_numpy()

        # Make the main diagonal equal to zero, because it has inf values
        np.fill_diagonal(distance_matrix, 0)

        # Collapse the matrix to a NumPy array
        distance_matrix = distance_matrix.flatten()

        # Calculate the mean of the array
        distance_matrix_mean = np.mean(distance_matrix)

        # Calculate the initial pheromone value: c * [1 / (number_of_nodes * average_of_all_costs)]
        tau0 = self.c * 1 / (self.vrp.size * distance_matrix_mean)

        return tau0

    def calc_pheromone_matrix(self):
        """
        Initializes the Pheromone Matrix τ (tau) with the initial pheromone value τ0.
        :return: The Pheromone Matrix τ
        """
        # Create the matrix of the pheromone levels of each edge in the graph
        tau = self.tau0 * np.ones((self.vrp.size, self.vrp.size))

        # Make the pheromone levels of the main diagonal !!!and the depot (node 0)!!! equal to zero
        np.fill_diagonal(tau, 0)
        tau[0] = 0  # !!! For VRP only
        tau[:, 0] = 0  # !!! For VRP only

        return tau

    def calc_desirability_matrix(self):
        """
        Calculates the Desirability matrix η (eta). Desirability of an edge is the tendency of the ants towards
        the shorter-distance edges.
        :return: The Desirability Matrix η
        """
        # Convert the distance matrix to NumPy ndarray
        distance = self.distance_matrix.to_numpy()

        # Consider division by zero. We want the diagonal !!!and the depot!!! to have zero desirability.
        np.fill_diagonal(distance, np.inf)
        distance[0] = np.inf  # !!! For VRP only
        distance[:, 0] = np.inf  # !!! For VRP only

        # Calculate the Desirability matrix of each edge in the graph
        eta = 1 / distance

        return eta

    def create_colony(self):
        """
        Creates the Ant Colony. Every ant is a tour.
        :return: A list with the ants of the colony.
        """
        # Create a colony to store the ants
        colony = []

        # Create every ant's tour
        for i in range(self.num_of_ants):
            # Choose a random node to start, except the depot
            nodes = np.delete(self.vrp.nodes, 0)
            ant = np.array(np.random.choice(nodes, 1))

            # Choose the rest of the nodes
            # Because we have made the depot pheromone and desirability 0, the depot will never be chosen from
            # the roulette wheel. The depot will remain the last node and the np.nonzero() will return an empty
            # list in the final round of the loop and will throw an error. That's why we loop until self.vrp.size-1
            for j in range(1, self.vrp.size - 1):
                current_node = ant[-1]  # Each time we want the last node that has been appended to the tour

                # Calculate the probability of each connected node to the current to be chosen
                prob_all = self.tau[current_node, :] ** self.a * self.eta[current_node, :] ** self.b  # τij^α * ηij^β
                prob_all[ant] = 0  # Assign 0 probability to every node visited so far. We used NumPy fancy indexing
                p = prob_all / sum(
                    prob_all)  # (τij^α * ηij^β) / Σ(τij^α * ηij^β) -> Probability of the ant to choose a node

                next_node = self.roulette_wheel(p)  # Choose the next node the ant will go
                ant = np.append(ant, next_node)

            # Complete the Tour
            ant = list(np.append(ant, ant[0]))
            colony.append(ant)

        return colony

    # noinspection PyMethodMayBeStatic
    def roulette_wheel(self, p):
        """
        Decides what will be the next node the ant will follow.
        :param list p: A list of probabilities.
        :return: The next node the ant will follow.
        """
        rng = default_rng()  # Create a new instance of numpy Generator for random methods

        # Roulette Wheel chooses the next node an ant will follow, based on the probability (p) values
        csp = np.cumsum(p)  # Calculate the Cumulative Sum of the p array

        r = rng.random()  # Random number

        next_node = np.nonzero(r < csp)  # Find the indices of the values that are less than r (Returns a tuple)
        next_node = next_node[0]  # Grab the first element of the tuple that has the row indices
        next_node = next_node[0]  # Grab the first node

        return next_node

    def create_colony_vrp(self, colony):
        """
        Generates a list of truck tours from every ant in the colony.
        :param list colony: A list of ants/tours
        :return: A list of lists with truck tours
        """
        # Create a vrp from every ant
        colony_vrp = []

        for i in range(len(colony)):
            tour = colony[i]
            vrp = self.vrp.create_vrp(tour, self.vrp.demand, self.vrp.max_truck_capacity)
            colony_vrp.append(vrp)

        return colony_vrp

    def calc_colony_fitness(self, colony_vrp):
        """
        Calculates the fitness function for every ant VRP in the colony.
        :param list colony_vrp: A list of lists with ant VRPs.
        :return: A list with the ant fitness values.
        """
        # Store the fitness values of the ants in the colony
        colony_fitness = []

        for i in range(len(colony_vrp)):
            # Create the VRP based on the Ant
            # We use the original distance matrix to calculate the fitness, because the distance_matrix_copy has
            # modified values
            ant_fitness = self.vrp.fitness_function(colony_vrp[i], self.vrp.fitness_distance_matrix)
            colony_fitness.append(ant_fitness)

        return colony_fitness

    def update_best_ant(self, colony_fitness):
        """
        Finds the ant with the best fitness value in the colony and updates the best_vrp value.
        :param list colony_fitness: A list with the ant fitness values.
        :return: Nothing. Updates the best_vrp value.
        """
        # Find the best ant in the colony and update best_vrp
        best = min(colony_fitness)  # The minimum colony fitness value is the best value in the colony
        best_index = colony_fitness.index(best)  # The index of the best fitness value equals the index of the best ant
        best_ant = self.colony_vrp[best_index]  # The tour of the best ant

        if best < self.vrp.best_vrp_fitness:  # Update the best ant so far
            self.vrp.best_vrp_fitness = best
            self.vrp.best_vrp = best_ant

        return

    def update_pheromone_matrix(self, colony, colony_fitness):
        """
        Update the Pheromone matrix values based on the length of the path that every ant has taken.
        :param list colony: A list of ants/tours.
        :param list colony_fitness: A list with the ant fitness values.
        :return: Nothing. Updates the Pheromone Matrix τ.
        """
        # Update the Pheromone matrix values based on the length of the path that every ant has taken
        for i in range(len(colony)):  # Iterate the colony
            tour = colony[i]  # Take the ith ant
            for j in range(len(tour) - 1):
                # Iterate every edge of the tour
                # Find the pheromone lvl of each edge from the pheromone matrix
                # Update every pheromone lvl by adding (1 / Lk), where Lk is the length of the path
                # Because the graph is undirected we have to update and the symmetric values of the pheromone matrix
                self.tau[tour[j], tour[j + 1]] += 1 / colony_fitness[i]
                self.tau[tour[j + 1], tour[j]] += 1 / colony_fitness[i]

        return

    def refine_vrp(self, vrp, max_sa_iterations=1000):
        """
        Optimizes the defined truck tours by solving the TSP for each on of them using Simulated Annealing.
        :param list vrp: A list of truck tours.
        :param int max_sa_iterations: The number of iterations for the Simulated Annealing
        :return:
        """
        refined_vrp = []  # The refined truck tours list
        self.sa.max_iter = max_sa_iterations  # The number of iterations of the Simulated Annealing

        for i in range(len(vrp)):
            tour = vrp[i]  # Current truck tour
            if len(tour) > 3:  # Check if the tour has only one node and the depot
                new_tour = self.sa.run_tsp(tour)  # Solve the TSP using SA
                refined_vrp.append(new_tour)  # Append the refined tour to the refined_vrp
            else:  # If tour has only one node, append as it is
                refined_vrp.append(tour)

        return refined_vrp

    def run(self, with_refining=False, max_sa_iterations=1000):
        """
        Runs the Ant Colony Optimization Algorithm. There is also the option to refine the final output using
        Simulated Annealing.
        :param bool with_refining: If True refining is performed. Default False.
        :param int max_sa_iterations: The number of SA iterations for refining.
        :return: Nothing. It modifies the class's attributes based on the given problem.
        """
        # Check if an SA instance has been given in case refining has been asked
        if self.sa is None:
            with_refining = False

        # Perform the ACO algorithm
        if self.vrp.with_sensor_data:
            print("Solving VRP with ACO algorithm and Sensor Data...")
        else:
            print("Solving VRP with ACO algorithm...")

        start = time.time()  # Calculate time to perform the algorithm

        for i in range(self.max_iter):
            # Create a new colony
            self.colony = self.create_colony()

            # Create the VRPs of the colony
            self.colony_vrp = self.create_colony_vrp(self.colony)

            # Calculate the fitness values of the colony
            self.colony_fitness = self.calc_colony_fitness(self.colony_vrp)

            # Update best ant
            self.update_best_ant(self.colony_fitness)

            # Update pheromone matrix
            self.update_pheromone_matrix(self.colony, self.colony_fitness)

            # Apply Evaporation
            self.tau = (1 - self.rho) * self.tau

            # Add the best colony fitness to all_fitness list
            self.vrp.all_fitness.append(self.vrp.best_vrp_fitness)

        end = time.time()  # End ACO time calculation
        self.aco_time = end - start

        print(f"Runtime of the ACO is: {self.aco_time}")
        print(f"Best Tour: {self.vrp.best_vrp_fitness}")
        print(f"Total demand = {self.vrp.total_demand}")  # Print total demand
        # Print every truck demand
        for i in range(len(self.vrp.best_vrp)):  # Iterate the best vrp
            truck_demands = []  # Store the demands collected from each node
            for j in range(len(self.vrp.best_vrp[i])):  # Iterate all the nodes a truck has visited
                truck_demands.append(self.vrp.demand[self.vrp.best_vrp[i][j]])  # Find the corresponding demand
            total_truck_demand = sum(truck_demands)  # Sum the demands
            print(f"Truck{i + 1} demand = {total_truck_demand}")

        if with_refining:
            start = time.time()  # Calculate time to perform the refining
            print()
            print("Refining...")
            self.refined_vrp = self.refine_vrp(self.vrp.best_vrp,
                                               max_sa_iterations=max_sa_iterations)  # Perform refining
            self.refined_vrp_fitness = self.vrp.fitness_function(
                self.refined_vrp, self.vrp.fitness_distance_matrix)  # Calculate refined vrp fitness

            end = time.time()  # End refining time calculation
            self.refining_time = end - start
            self.aco_time_with_refining = self.aco_time + self.refining_time

            print(f"Refining time: {self.refining_time}")
            print(f"Refined Best Tour: {self.refined_vrp_fitness}")
            print(f"Runtime of the ACO with refining is: {self.aco_time_with_refining}")

        print()  # An empty line for cleaner visualizing

        return
