"""
Created on Fri Dec 16 17:36:10 2022

@author: Dimitris Madaros
"""
import matplotlib.pyplot as plt

class Visual:
    def __init__(self, dataset, sensor_dataset):
        # Dataset Related #
        self.dataset = dataset
        self.sensor_dataset = sensor_dataset
        self.x = dataset["X"].values  # A NumPy array of the x coordinates of the nodes
        self.y = dataset["Y"].values  # A NumPy array of the y coordinates of the nodes
        self.x_sensor =\
            sensor_dataset["X"].values  # A NumPy array of the x coordinates of the nodes if sensors are used
        self.y_sensor =\
            sensor_dataset["Y"].values  # A NumPy array of the y coordinates of the nodes if sensors are used

    def plot_nodes(self, x=None, y=None, ax=None, color='red', marker_size=7):
        # Check if a value of x or y has been given
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if ax is None:
            # Create a figure containing a single axes.
            fig, ax = plt.subplots()

        # Plot the nodes
        ax.plot(x, y, 'o', color=color, markersize=marker_size)
        plt.show()
        return

    def plot_edges(self, x=None, y=None, ax=None, color='blue', l_width=1):
        # Check if a value of x or y has been given
        if x is None:
            x = self.x
        if x is None:
            y = self.y
        if ax is None:
            # Create a figure containing a single axes.
            fig, ax = plt.subplots()

        for i in range(len(x)):
            for j in range(len(x)):
                if j > i:
                    ax.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=l_width)

        ax.plot(x, y, 'o', color='red', markersize=8)

        plt.show()
        return

    def plot_graph(self, x=None, y=None, ax=None):
        # Check if a value of x, y or ax has been given
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if ax is None:
            # Create a figure containing a single axes.
            fig, ax = plt.subplots()

        self.plot_edges(x, y, ax)
        self.plot_nodes(x, y, ax)

        plt.show()
        return

    def plot_tour(self, tour, ax=None, color='blue'):
        # Check if a value ax has been given
        if ax is None:
            # Create a figure containing a single axes.
            fig, ax = plt.subplots()

        # x, y coordinates of the tour nodes
        x = []
        y = []

        for index in tour:
            # Tour is calculated from dist_matrix indexes
            x.append(self.x[index])
            y.append(self.y[index])

        # Find the x, y coordinates of the edge between each two consecutive nodes in the tour
        edge_x = [0] * 2
        edge_y = [0] * 2

        for i in range(len(tour) - 1):
            # The x coordinates between two consecutive nodes
            edge_x[0] = x[i]
            edge_x[1] = x[i + 1]

            # The y coordinates between two consecutive nodes
            edge_y[0] = y[i]
            edge_y[1] = y[i + 1]

            # Plot the edge
            ax.plot(edge_x, edge_y, color=color, linewidth=1)

        # Plot the nodes of the tour
        self.plot_nodes(x, y, ax)

        plt.show()
        return

    def plot_sa(self, initial_solution, global_best, fitness):
        # Create a figure with 2x2 grid
        fig = plt.figure(figsize=(9, 6), layout="constrained")
        spec = fig.add_gridspec(2, 2)
        # Create Subplots
        ax0 = fig.add_subplot(spec[0, 0])
        ax1 = fig.add_subplot(spec[0, 1])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[1, 1])

        # Subplot#1 - Graph
        self.plot_graph(self.x, self.y, ax0)
        ax0.set_title("Dataset")

        # Subplot#2 - Initial Solution
        self.plot_tour(initial_solution, ax1)
        ax1.set_title("Initial Solution")

        # Subplot#3 - Iteration vs Fitness
        ax2.semilogx(fitness)
        ax2.set_title("Iteration v Fitness")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Fitness")

        # Subplot#4 - Global Best
        self.plot_tour(global_best, ax3)
        ax3.set_title("Best Solution Found")

        fig.suptitle("Simulated Annealing - SA")

        plt.show()
        return

    def plot_aco(self, initial_solution, global_best, fitness):
        # Create a figure with 2x2 grid
        fig = plt.figure(figsize=(9, 6), layout="constrained")
        spec = fig.add_gridspec(2, 2)
        # Create Subplots
        ax0 = fig.add_subplot(spec[0, 0])
        ax1 = fig.add_subplot(spec[0, 1])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[1, 1])

        # Subplot#1 - Graph
        self.plot_graph(self.x, self.y, ax0)
        ax0.set_title("Dataset")

        # Subplot#2 - Initial Solution
        self.plot_tour(initial_solution, ax1)
        ax1.set_title("Initial Solution")

        # Subplot#3 - Iteration vs Fitness
        ax2.semilogx(fitness)
        ax2.set_title("Iteration v Fitness")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Fitness")

        # Subplot#4 - Global Best
        self.plot_tour(global_best, ax3)
        ax3.set_title("Best Solution Found")

        fig.suptitle("Ant Colony Optimization - ACO")

        plt.show()
        return

    def plot_vrp(self, vrp, ax=None):
        # Create a list of colors to make the graph readable
        colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors += colors  # Make the list bigger to cover more trucks
        # Check if an ax has been given
        if ax is None:
            # Create a figure containing a single axes.
            fig, ax = plt.subplots()

        # Iterate through all the trucks
        for i in range(len(vrp)):
            self.plot_tour(vrp[i], ax, colors[i])

        # Color depot differently
        self.plot_nodes(self.x[0], self.y[0], ax, color='orange')

        plt.show()
        return

    def plot_vrp_pycharm(self, vrp, with_sensor_data=False):
        # Check if sensor data will be considered
        if with_sensor_data:
            data_x = self.x_sensor
            data_y = self.y_sensor
        else:
            data_x = self.x
            data_y = self.y

        # Create a list of colors to make the graph readable
        colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors += colors  # Make the list bigger to cover more trucks

        # Create a figure containing a single axes.
        # fig, ax = plt.subplots()
        # Create a figure with 2x2 grid
        fig = plt.figure(figsize=(9, 6), layout="constrained")
        spec = fig.add_gridspec(2, 2)
        # Create Subplots
        ax0 = fig.add_subplot(spec[0, 0])
        ax1 = fig.add_subplot(spec[0, 1])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[1, 1])

        for i in range(len(vrp)):
            # x, y coordinates of the tour nodes
            x = []
            y = []

            for index in vrp[i]:
                # Tour is calculated from dist_matrix indexes
                x.append(data_x[index])
                y.append(data_y[index])

            # Find the x, y coordinates of the edge between each two consecutive nodes in the tour
            edge_x = [0] * 2
            edge_y = [0] * 2

            for j in range(len(vrp[i]) - 1):
                # The x coordinates between two consecutive nodes
                edge_x[0] = x[j]
                edge_x[1] = x[j + 1]

                # The y coordinates between two consecutive nodes
                edge_y[0] = y[j]
                edge_y[1] = y[j + 1]

                # Plot the edge
                ax0.plot(edge_x, edge_y, color=colors[i], linewidth=1)

            ax0.plot(x, y, 'o', color='red', markersize=8)
            ax0.plot(x[0], y[0], 'o', color='orange', markersize=8)

        plt.show()

        return

    def sa_presentation(self, sa, sa_sensor):
        # Create a list of colors to make the graph readable
        colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors += colors  # Make the list bigger to cover more trucks

        # Put the VRP instances in a list to iterate
        vrp_list = [sa.vrp, sa.vrp, sa_sensor.vrp]  # First for initial vrp, second for best vrp, third for sensor vrp

        # Create a figure with 2x2 grid
        fig = plt.figure(figsize=(9, 6), layout="constrained")
        spec = fig.add_gridspec(2, 2)
        # Create Subplots
        ax0 = fig.add_subplot(spec[0, 0])
        ax1 = fig.add_subplot(spec[0, 1])
        ax2 = fig.add_subplot(spec[1, 0])
        ax3 = fig.add_subplot(spec[1, 1])

        axes = [ax0, ax1, ax2]

        for i in range(len(vrp_list)):
            # Take the VRP instance
            vrp_instance = vrp_list[i]

            # Check if Sensor Data are being used
            if vrp_instance.with_sensor_data:
                data_x = self.x_sensor
                data_y = self.y_sensor
            else:
                data_x = self.x
                data_y = self.y

            # Choose between the VRPs in the VRP instance
            if i == 0:
                vrp = vrp_instance.init_vrp
            else:
                vrp = vrp_instance.best_vrp

            for j in range(len(vrp)):
                # x, y coordinates of the tour nodes
                x = []
                y = []

                for index in vrp[j]:
                    # Tour is calculated from dist_matrix indexes
                    x.append(data_x[index])
                    y.append(data_y[index])

                # Find the x, y coordinates of the edge between each two consecutive nodes in the tour
                edge_x = [0] * 2
                edge_y = [0] * 2

                for k in range(len(vrp[j]) - 1):
                    # The x coordinates between two consecutive nodes
                    edge_x[0] = x[k]
                    edge_x[1] = x[k + 1]

                    # The y coordinates between two consecutive nodes
                    edge_y[0] = y[k]
                    edge_y[1] = y[k + 1]

                    # Plot the edge
                    axes[i].plot(edge_x, edge_y, color=colors[j], linewidth=0.5)

                if vrp_instance.with_sensor_data:
                    axes[i].plot(self.x, self.y, 'o', zorder=2, color='green', markersize=5)

                axes[i].plot(x, y, 'o', zorder=3, color='red', markersize=5)
                axes[i].plot(x[0], y[0], 'o', zorder=4, color='orange', markersize=5)

        # Subplot#1 - Initial VRP
        ax0.set_title("Initial VRP")

        # Subplot#2 - Best VRP
        ax1.set_title("Best VRP")

        # Subplot#3 - VRP with Sensor Data
        ax2.set_title("VRP with Sensor Data")

        # Subplot#4 - Iteration vs Fitness
        ax3.semilogx(sa.vrp.all_fitness)
        ax3.set_title("Iteration v Fitness")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Fitness")

        fig.suptitle("Simulated Annealing - SA")

        plt.show()

        return

    def aco_presentation(self, aco, aco_sensor):
        # Create a list of colors to make the graph readable
        colors = ['blue', 'red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors += colors  # Make the list bigger to cover more trucks

        # Put the VRP instances in a list to iterate
        # 1: initial vrp, 2: best vrp, 3: refined vrp, 4: vrp with sensor data, 5: refined vrp with sensor data
        aco_list = [aco, aco, aco, aco_sensor, aco_sensor]

        # Create a figure with 3x3 grid
        fig = plt.figure(figsize=(16, 10), layout="constrained")
        spec = fig.add_gridspec(3, 3)
        # Create Subplots
        ax0 = fig.add_subplot(spec[0, 0])
        ax1 = fig.add_subplot(spec[0, 1])
        ax2 = fig.add_subplot(spec[0, 2])
        ax3 = fig.add_subplot(spec[1, 0])
        ax4 = fig.add_subplot(spec[1, 1])
        ax5 = fig.add_subplot(spec[1, 2])

        axes = [ax0, ax1, ax2, ax3, ax4, ax5]

        for i in range(len(aco_list)):
            # Take the VRP instance
            vrp_instance = aco_list[i].vrp

            # Check if Sensor Data are being used
            if vrp_instance.with_sensor_data:
                data_x = self.x_sensor
                data_y = self.y_sensor
            else:
                data_x = self.x
                data_y = self.y

            # Choose between the VRPs in the VRP instance
            if i == 0:
                vrp = vrp_instance.init_vrp
            elif i == 1 or i == 3:
                vrp = vrp_instance.best_vrp
            else:
                vrp = aco_list[i].refined_vrp

            for j in range(len(vrp)):
                # x, y coordinates of the tour nodes
                x = []
                y = []

                for index in vrp[j]:
                    # Tour is calculated from dist_matrix indexes
                    x.append(data_x[index])
                    y.append(data_y[index])

                # Find the x, y coordinates of the edge between each two consecutive nodes in the tour
                edge_x = [0] * 2
                edge_y = [0] * 2

                for k in range(len(vrp[j]) - 1):
                    # The x coordinates between two consecutive nodes
                    edge_x[0] = x[k]
                    edge_x[1] = x[k + 1]

                    # The y coordinates between two consecutive nodes
                    edge_y[0] = y[k]
                    edge_y[1] = y[k + 1]

                    # Plot the edge
                    axes[i].plot(edge_x, edge_y, color=colors[j], linewidth=0.5)

                if vrp_instance.with_sensor_data:
                    axes[i].plot(self.x, self.y, 'o', zorder=2, color='green', markersize=5)

                axes[i].plot(x, y, 'o', zorder=3, color='red', markersize=5)
                axes[i].plot(x[0], y[0], 'o', zorder=4, color='orange', markersize=5)

        # Subplot#1 - Initial VRP
        ax0.set_title("Initial VRP")

        # Subplot#2 - Best VRP
        ax1.set_title("Best VRP")

        # Subplot#3 - Refined VRP / Hybrid ACO-SA VRP
        ax2.set_title("Refined VRP / Hybrid ACO-SA VRP")

        # Subplot#4 - VRP with Sensor Data
        ax3.set_title("VRP with Sensor Data")

        # Subplot#5 - Refined VRP with Sensor Data
        ax4.set_title(" Refined VRP with Sensor Data / Hybrid ACO-SA VRP with Sensor Data")

        # Subplot#6 - Iteration vs Fitness
        ax5.semilogx(aco.vrp.all_fitness)
        ax5.set_title("Iteration v Fitness")
        ax5.set_xlabel("Iteration")
        ax5.set_ylabel("Fitness")

        fig.suptitle("Ant Colony Optimization - ACO")

        plt.show()

        return

    def plot_datasets(self, list_of_datasets):

        # Create a figure with 3x3 grid
        fig = plt.figure(figsize=(16, 10), layout="constrained")
        spec = fig.add_gridspec(2, 3)
        # Create Subplots
        ax0 = fig.add_subplot(spec[0, 0])
        ax1 = fig.add_subplot(spec[0, 1])
        ax2 = fig.add_subplot(spec[0, 2])
        ax3 = fig.add_subplot(spec[1, 0])
        ax4 = fig.add_subplot(spec[1, 1])

        axes = [ax0, ax1, ax2, ax3, ax4]

        for i in range(len(list_of_datasets)):
            dataset = list_of_datasets[i]
            x = dataset["X"].values
            y = dataset["Y"].values

            axes[i].plot(x, y, 'o', zorder=3, color='red', markersize=5)
            axes[i].plot(x[0], y[0], 'o', zorder=4, color='orange', markersize=5)

        ax0.set_title("Dataset-1")
        ax1.set_title("Dataset-2")
        ax2.set_title("Dataset-3")
        ax3.set_title("Dataset-4")
        ax4.set_title("Dataset-5")

        fig.suptitle("DATASETS")

        plt.show()
