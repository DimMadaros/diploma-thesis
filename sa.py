"""
Created on Sun Jan 8 15:51:22 2023

@author: Dimitris Madaros
"""
import time
import numpy as np
from numpy.random import default_rng


class SA:
    def __init__(self, vrp, max_iter=1000):
        self.vrp = vrp  # An instance of a VRP problem class
        self.distance_matrix = vrp.distance_matrix_copy
        self.max_iter = max_iter  # Max number of iterations
        self.t0 = 1  # Initial Temperature
        self.t = self.t0  # Temperature
        self.a = 0.99  # Cooling Factor Alpha
        self.sa_time = 0  # The performance time of the SA algorithm

    # noinspection PyMethodMayBeStatic
    def create_neighbour(self, tour):
        """
        Changes the position of two random nodes of the tour with one another.
        :param list tour: A list of nodes in the order visited.
        :return: The neighbouring tour.
        """
        rng = default_rng()  # Create a new instance of numpy Generator for random methods

        # Choose two random positions from the tour without replacement
        pos = rng.choice(len(tour) - 1, 2, replace=False)  # -1 because the first and the last node are the same

        # Swap the two positions
        new_tour = np.array(tour).copy()  # Create a copy from an array to avoid making a reference
        new_tour[pos[0]] = tour[pos[1]]
        new_tour[pos[1]] = tour[pos[0]]
        new_tour[-1] = new_tour[0]

        return new_tour

    def run(self):
        """
        Runs the Simulated Annealing Algorithm.
        :return: Nothing. It modifies the class's attributes based on the given problem.
        """
        # Perform the SA algorithm
        x0 = self.vrp.x0  # Initial Tour
        vrp0 = self.vrp.init_vrp  # Initial VRP from Initial Tour x0

        if self.vrp.with_sensor_data:
            print("Solving VRP with SA algorithm and Sensor Data...")
        else:
            print("Solving VRP with SA algorithm...")

        start = time.time()  # Calculate time to perform the algorithm

        for i in range(self.max_iter):
            # Calculate the fitness of the current VRP
            vrp0_fitness = self.vrp.fitness_function(vrp0, self.distance_matrix)

            # Create a random neighbour
            x1 = self.create_neighbour(x0)  # Neighbouring Tour
            vrp1 = self.vrp.create_vrp(x1, self.vrp.demand,
                                       self.vrp.max_truck_capacity)  # Create VRP from Neighbouring Tour x1
            vrp1_fitness = self.vrp.fitness_function(vrp1, self.distance_matrix)

            # ---------- Simulated Annealing ---------- #

            delta = (-1) * (vrp1_fitness - vrp0_fitness)  # delta > 0 -> 'good solution'. (-1) because of minimization

            if delta > 0:
                # If the fitness of the potential solution is better (less)
                x0 = x1
                vrp0 = vrp1
            else:
                p = np.exp(delta / self.t)  # Probability to choose a "bad solution"
                r = np.random.random()  # If r < p, then we accept the "bad solution"

                if r < p:
                    x0 = x1
                    vrp0 = vrp1

            # Update Global Best
            if vrp0_fitness < self.vrp.best_vrp_fitness:
                self.vrp.best_vrp = vrp0
                self.vrp.best_vrp_fitness = vrp0_fitness

            # Update Temperature
            self.t = self.a * self.t
            self.vrp.all_fitness.append(vrp0_fitness)

        end = time.time()
        self.sa_time = end - start
        print(f"Runtime of the SA is: {self.sa_time}")
        print(f"Best Tour: {self.vrp.best_vrp_fitness}")
        print(f"Total demand = {self.vrp.total_demand}")  # Print total demand
        # Print every truck demand
        for i in range(len(self.vrp.best_vrp)):  # Iterate the best vrp
            truck_demands = []  # Store the demands collected from each node
            for j in range(len(self.vrp.best_vrp[i])):  # Iterate all the nodes a truck has visited
                truck_demands.append(self.vrp.demand[self.vrp.best_vrp[i][j]])  # Find the corresponding demand
            total_truck_demand = sum(truck_demands)  # Sum the demands
            print(f"Truck{i+1} demand = {total_truck_demand}")
        print()

        return

    def run_tsp(self, tour):
        """
        Given a random tour, find the best one keeping the same depot.
        :param list tour: A list of nodes.
        :return: The minimum cost tour.
        """
        # Set up initial values
        x0 = tour.copy()  # Avoid reference
        x0_cost = self.vrp.calc_tour_cost(x0, self.distance_matrix)

        # Set up best values
        best_tour = x0.copy()
        best_cost = x0_cost

        for i in range(self.max_iter):
            # Calculate the cost of the current tour
            x0_cost = self.vrp.calc_tour_cost(x0, self.distance_matrix)

            # Create a random neighbour
            depot = x0[0]  # Keep the start (and end) of the tour
            x_temp = x0.copy()  # Avoid reference
            x_temp.pop(0)  # Remove the start of the tour (we don't want it to be replaced in create_neighbour)

            x1 = list(self.create_neighbour(x_temp))
            x1 = [depot] + x1  # Add the start again
            x1[-1] = x1[0]  # Make the start the same as the end to complete the tour
            x1_cost = self.vrp.calc_tour_cost(x1, self.distance_matrix)

            # ---------- Simulated Annealing ---------- #

            delta = (-1) * (x1_cost - x0_cost)  # Delta > 0 -> 'good solution'. Also (-1) because of minimization

            if delta > 0:
                # If the cost of the potential solution is better (less)
                x0 = x1.copy()
                x0_cost = x1_cost
            else:
                p = np.exp(delta / self.t)  # Probability to choose a "bad solution"
                r = np.random.random()  # If r < P, then we accept the "bad solution"

                if r < p:
                    x0 = x1.copy()
                    x0_cost = x1_cost

            # Update Global Best
            if x0_cost < best_cost:
                best_tour = x0.copy()
                best_cost = x0_cost

            # Update Temperature
            self.t = self.a * self.t

        return best_tour
