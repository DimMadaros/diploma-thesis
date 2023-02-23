"""
Created on Mon Dec 9 15:41:49 2022

@author: Dimitris Madaros
"""
from vrp import VRP
from sa import SA
from aco import ACO
from visual import Visual

"""
- When using Simulated Annealing, with_depot must be set to False.
- When using Ant Colony Optimization, with_depot must be set to True.
- If Pycharm is used, there are specific visualization functions in the Visual class that work with it.
- When creating a Visual class instance, it doesn't matter if the datasets are from the SA or ACO VRP instances.
- We pass an SA in the ACO in order to perform refining at the end of the VRP calculation.
"""
# ---------- VRP INSTANCES ---------- #

# Initialize VRP instances for SA
sa_vrp_1 = VRP(dataset_no=1, with_depot=False, with_random_demand=False, with_sensor_data=False)
sa_vrp_2 = VRP(dataset_no=2, with_depot=False, with_random_demand=False, with_sensor_data=False)
sa_vrp_3 = VRP(dataset_no=3, with_depot=False, with_random_demand=False, with_sensor_data=False)
sa_vrp_4 = VRP(dataset_no=4, with_depot=False, with_random_demand=False, with_sensor_data=False)
sa_vrp_5 = VRP(dataset_no=5, with_depot=False, with_random_demand=False, with_sensor_data=False)

# Initialize VRP instances for ACO
aco_vrp_1 = VRP(dataset_no=1, with_depot=True, with_random_demand=False, with_sensor_data=False)
aco_vrp_2 = VRP(dataset_no=2, with_depot=True, with_random_demand=False, with_sensor_data=False)
aco_vrp_3 = VRP(dataset_no=3, with_depot=True, with_random_demand=False, with_sensor_data=False)
aco_vrp_4 = VRP(dataset_no=4, with_depot=True, with_random_demand=False, with_sensor_data=False)
aco_vrp_5 = VRP(dataset_no=5, with_depot=True, with_random_demand=False, with_sensor_data=False)

# ---------- VRP INSTANCES WITH SENSOR DATA ---------- #

# Initialize VRP instances for SA with Sensor Data
sa_vrp_1_sensor = VRP(dataset_no=1, with_depot=False, with_random_demand=False, with_sensor_data=True)
sa_vrp_2_sensor = VRP(dataset_no=2, with_depot=False, with_random_demand=False, with_sensor_data=True)
sa_vrp_3_sensor = VRP(dataset_no=3, with_depot=False, with_random_demand=False, with_sensor_data=True)
sa_vrp_4_sensor = VRP(dataset_no=4, with_depot=False, with_random_demand=False, with_sensor_data=True)
sa_vrp_5_sensor = VRP(dataset_no=5, with_depot=False, with_random_demand=False, with_sensor_data=True)

# Initialize VRP instances for ACO with Sensor Data
aco_vrp_1_sensor = VRP(dataset_no=1, with_depot=True, with_random_demand=False, with_sensor_data=True)
aco_vrp_2_sensor = VRP(dataset_no=2, with_depot=True, with_random_demand=False, with_sensor_data=True)
aco_vrp_3_sensor = VRP(dataset_no=3, with_depot=True, with_random_demand=False, with_sensor_data=True)
aco_vrp_4_sensor = VRP(dataset_no=4, with_depot=True, with_random_demand=False, with_sensor_data=True)
aco_vrp_5_sensor = VRP(dataset_no=5, with_depot=True, with_random_demand=False, with_sensor_data=True)

# ---------- VISUALIZATION ENVIRONMENTS ---------- #

# Initialize a visualization environment
vs1 = Visual(dataset=sa_vrp_1.dataset, sensor_dataset=sa_vrp_1.sensor_dataset)
vs2 = Visual(dataset=sa_vrp_2.dataset, sensor_dataset=sa_vrp_2.sensor_dataset)
vs3 = Visual(dataset=sa_vrp_3.dataset, sensor_dataset=sa_vrp_3.sensor_dataset)
vs4 = Visual(dataset=sa_vrp_4.dataset, sensor_dataset=sa_vrp_4.sensor_dataset)
vs5 = Visual(dataset=sa_vrp_5.dataset, sensor_dataset=sa_vrp_5.sensor_dataset)

# ---------- INITIALIZE SIMULATED ANNEALING ---------- #

# Initialize Simulated Annealing for each VRP
sa1 = SA(vrp=sa_vrp_1, max_iter=5000)
sa2 = SA(vrp=sa_vrp_2, max_iter=5000)
sa3 = SA(vrp=sa_vrp_3, max_iter=5000)
sa4 = SA(vrp=sa_vrp_4, max_iter=5000)
sa5 = SA(vrp=sa_vrp_5, max_iter=5000)

# Initialize Simulated Annealing for each VRP with sensor data
sa1_sensor = SA(vrp=sa_vrp_1_sensor, max_iter=5000)
sa2_sensor = SA(vrp=sa_vrp_2_sensor, max_iter=5000)
sa3_sensor = SA(vrp=sa_vrp_3_sensor, max_iter=5000)
sa4_sensor = SA(vrp=sa_vrp_4_sensor, max_iter=5000)
sa5_sensor = SA(vrp=sa_vrp_5_sensor, max_iter=5000)

# ---------- INITIALIZE ANT COLONY OPTIMIZATION ---------- #

# Initialize Ant Colony Optimization for each VRP
aco1 = ACO(vrp=aco_vrp_1, max_iter=200, num_of_ants=13, sa=sa1)
aco2 = ACO(vrp=aco_vrp_2, max_iter=200, num_of_ants=13, sa=sa2)
aco3 = ACO(vrp=aco_vrp_3, max_iter=200, num_of_ants=13, sa=sa3)
aco4 = ACO(vrp=aco_vrp_4, max_iter=200, num_of_ants=13, sa=sa4)
aco5 = ACO(vrp=aco_vrp_5, max_iter=200, num_of_ants=13, sa=sa5)

# Initialize Ant Colony Optimization for each VRP
aco1_sensor = ACO(vrp=aco_vrp_1_sensor, max_iter=200, num_of_ants=13, sa=sa1_sensor)
aco2_sensor = ACO(vrp=aco_vrp_2_sensor, max_iter=200, num_of_ants=13, sa=sa2_sensor)
aco3_sensor = ACO(vrp=aco_vrp_3_sensor, max_iter=200, num_of_ants=13, sa=sa3_sensor)
aco4_sensor = ACO(vrp=aco_vrp_4_sensor, max_iter=200, num_of_ants=13, sa=sa4_sensor)
aco5_sensor = ACO(vrp=aco_vrp_5_sensor, max_iter=200, num_of_ants=13, sa=sa5_sensor)

# ---------- Dataset 1 ---------- #
# # Run and Plot SA
# sa1.run()
# sa1_sensor.run()
# vs1.sa_presentation(sa1, sa1_sensor)
#
# # Run and Plot ACO
# aco1.run(with_refining=True, max_sa_iterations=5000)
# aco1_sensor.run(with_refining=True, max_sa_iterations=5000)
# vs1.aco_presentation(aco1, aco1_sensor)

# ---------- Dataset 2 ---------- #
# Run and Plot SA
# sa2.run()
# sa2_sensor.run()
# vs2.sa_presentation(sa2, sa2_sensor)
#
# # Run and Plot ACO
# aco2.run(with_refining=True, max_sa_iterations=5000)
# aco2_sensor.run(with_refining=True, max_sa_iterations=5000)
# vs2.aco_presentation(aco2, aco2_sensor)

# ---------- Dataset 3 ---------- #
# Run and Plot SA
# sa3.run()
# sa3_sensor.run()
# vs3.sa_presentation(sa3, sa3_sensor)
#
# # Run and Plot ACO
# aco3.run(with_refining=True, max_sa_iterations=5000)
# aco3_sensor.run(with_refining=True, max_sa_iterations=5000)
# vs3.aco_presentation(aco3, aco3_sensor)

# ---------- Dataset 4 ---------- #
# Run and Plot SA
# sa4.run()
# sa4_sensor.run()
# vs4.sa_presentation(sa4, sa4_sensor)
#
# # Run and Plot ACO
# aco4.run(with_refining=True, max_sa_iterations=5000)
# aco4_sensor.run(with_refining=True, max_sa_iterations=5000)
# vs4.aco_presentation(aco4, aco4_sensor)
#
# # ---------- Dataset 5 ---------- #
# Run and Plot SA
sa5.run()
sa5_sensor.run()
vs5.sa_presentation(sa5, sa5_sensor)

# Run and Plot ACO
aco5.run(with_refining=True, max_sa_iterations=5000)
aco5_sensor.run(with_refining=True, max_sa_iterations=5000)
vs5.aco_presentation(aco5, aco5_sensor)

# Plot the datasets together
# list_of_datasets = [vs1.dataset, vs2.dataset, vs3.dataset, vs4.dataset, vs5.dataset]
# vs1.plot_datasets(list_of_datasets)
