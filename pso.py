"""
Created on Thu Oct 27 18:40:44 2022

@author: Dimitris Madaros
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

'''

For the Binary PSO:
    1) We have to change the upper bounds to 1 and the lower bounds to 0.
    2) We initialize the position of the particle in the Particle class with 0 and 1 values.
    3) Instead of updating the next position by calculating X1 = X0 + V1, we map the velocity vector to the
       sigmoid transfer function, in order to get binary values.

'''

# Create a new instance of numpy Generator for random methods
rng = default_rng()


class Best:
    # A class that stores and updates best value
    def __init__(self, n_var):
        self.x = np.zeros((1, n_var))  # Initialize Position Vector
        self.o = np.inf  # Initialize Objective Value

    def update(self, position, objective):
        # Update G-best or P-best
        if objective < self.o:
            self.x = position.copy()  # To avoid shallow copy's
            self.o = objective


class Particle:
    def __init__(self, n_var, upper_b, lower_b):
        # Position Vector. Use rng.random(random) for continuous. Use rng.integers(low=0,high=2,nVar) for Binary PSO
        self.x = (upper_b - lower_b) * rng.integers(0, 2, n_var) + lower_b
        self.v = np.zeros(n_var)  # Velocity Vector
        self.o = np.inf  # Objective Value
        self.pbest = Best(n_var)  # Personal Best Position and Value - Personal Best


class Swarm:
    def __init__(self, n_particle, n_var, upper_b, lower_b):
        # Initialize a list of Particle objects
        self.particles = [Particle(n_var, upper_b, lower_b) for _ in range(n_particle)]
        self.gbest = Best(n_var)  # Best value between all particles - Global Best
        self.history = []  # All G-Bests for plotting


# Objective Function - f(x)
def f(value):
    return np.sum(value ** 2)  # Sphere Function


# Define the details of the OF
nVar = 20
ub = np.ones(nVar)
lb = np.zeros(nVar)

# Define the PSO parameters #
noP = 500
maxIter = 1500
wMax = 0.9  # Max inertia value
wMin = 0.2  # Min inertia value
c1 = 2
c2 = 2
vMax = (ub - lb) * 0.2
vMin = -vMax

# Initialize Swarm #
swarm = Swarm(noP, nVar, ub, lb)

# ---------- PSO ---------- #
for i in range(maxIter):

    for p in range(noP):
        # Calculate Objective and update PBEST & GBEST
        X = swarm.particles[p].x  # Get Particle's Position

        swarm.particles[p].o = f(X)  # Calculate the Objective Value of the Particle
        O = swarm.particles[p].o  # Get Particle's Objective Value

        swarm.particles[p].pbest.update(X, O)  # Update PBEST
        swarm.gbest.update(X, O)  # Update GBEST

    # Update current Velocity (V) and Particle's next Position (X)
    w = wMax - (i * (wMax - wMin) / maxIter)  # Update inertia (from 0.9 to 0.2 linearly)

    for p in range(noP):
        #  For best readability
        X = swarm.particles[p].x  # Get Particle's Position
        V = swarm.particles[p].v  # Get current Velocity
        P = swarm.particles[p].pbest.x  # Get Particle's personal BEST position until now
        G = swarm.gbest.x  # Get Swarm's BEST position among the Particle's

        # Create random variables r1, r2
        r1 = rng.random(nVar)
        r2 = rng.random(nVar)

        # Update Velocity
        swarm.particles[p].v = w * V + c1 * r1 * (P - X) + c2 * r2 * (G - X)
        V = swarm.particles[p].v  # Get new Velocity

        # Check if velocities are inside boundaries (We use numpy fancy indexing)
        swarm.particles[p].v[V > vMax] = vMax[V > vMax]
        swarm.particles[p].v[V < vMin] = vMin[V < vMin]

        # # Calculate next Position / Not for Binary PSO. Calculate  Sigmoid Transfer function instead
        # swarm.particles[p].X = X + V
        # X = swarm.particles[p].X  # Get new Position

        # # Check if positions are inside boundaries / Not for Binary PSO
        # swarm.particles[p].X[X > ub] = ub[X > ub]
        # swarm.particles[p].X[X < lb] = ub[X < lb]

        # Sigmoid Transfer function
        T = 1 / (1 + np.exp(swarm.particles[p].v))

        # Get random numbers for each velocity dimension
        # r = rng.random(nVar)

        # Update Position (Binary)
        for j in range(nVar):
            r = rng.random()
            if r < T[j]:
                swarm.particles[p].x[j] = 1
            else:
                swarm.particles[p].x[j] = 0

    # Append the GBEST to the history list to plot later
    print(f"Iteration# {i} GBEST = {swarm.gbest.o}")
    swarm.history.append(swarm.gbest.o)

print(f"GBEST = {swarm.gbest.x}")

# PLOT
Y = swarm.history
X = np.arange(len(Y))

fig, ax = plt.subplots()
ax.semilogy(X, Y)
