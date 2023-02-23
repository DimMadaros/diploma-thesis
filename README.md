# diploma-thesis
Metaheuristics for the Waste Collection Routing Problem in Smart Cities using Sensor Data

- Conducted a study on optimizing waste collection routes in smart cities using sensor data.

- This study simulates data from smart bins equipped with sensors that measure waste levels to determine the minimum number of bins to be visited based on a threshold level.

- Two Metaheuristic Algorithms (MAs) - Simulated Annealing (SA) and Ant Colony Optimization (ACO), as well as a Hybridization of these two algorithms, are considered to solve the VRP in waste management.

- A Discrete Particle Swarm Optimization Algorithm was also developed, but eventually considered unsuitable.

- A key selection method was borrowed from the Genetic Algorithm, Roulette Wheel Selection, for the development of the ACO algorithm.

- The algorithms were tested on five datasets of different input scales, with the largest being 100 nodes.

- Implemented using Python and Pandas, NumPy, Matplotlib libraries.

# HOW TO RUN

pip install -r requirements.txt

python main.py
