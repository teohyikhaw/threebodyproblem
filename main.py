import itertools
import numpy as np
from equations import *
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import os

if __name__ == '__main__':
    particles = [
        Particle("alpha cen A", 1.1, np.array([-0.5, 0, 0], dtype="float64"),
                 np.array([0.01, 0.01, 0], dtype="float64")),
        Particle("alpha cen B", 0.907, np.array([0.5, 0, 0], dtype="float64"),
                 np.array([-0.05, 0, -0.1], dtype="float64")),
        Particle("alpha cen C", 1.0, np.array([0, 1, 0], dtype="float64"), np.array([0, -0.01, 0], dtype="float64"))
    ]

    solutions = run_simulation(particles, 20000, 0.01)
    positions = solutions[:, :3, :]
    r1_sol = positions[:, 0, :]  # Positions for body 1
    r2_sol = positions[:, 1, :]  # Positions for body 2
    r3_sol = positions[:, 2, :]  # Positions for body 3

    # Plot trajectories for each body
    fig = plt.figure(figsize=(15, 15))
    # Create 3D axes
    ax = fig.add_subplot(111, projection="3d")
    # print(r1_sol)
    # Plot the orbits
    ax.plot(r1_sol[:, 0], r1_sol[:, 1], r1_sol[:, 2], color="darkblue")
    ax.plot(r2_sol[:, 0], r2_sol[:, 1], r2_sol[:, 2], color="tab:red")
    ax.plot(r3_sol[:, 0], r3_sol[:, 1], r3_sol[:, 2], color="orange")
    # Plot the final positions of the stars
    ax.scatter(r1_sol[-1, 0], r1_sol[-1, 1], r1_sol[-1, 2], color="darkblue", marker="o", s=100,
               label="Alpha Centauri A")
    ax.scatter(r2_sol[-1, 0], r2_sol[-1, 1], r2_sol[-1, 2], color="tab:red", marker="o", s=100,
               label="Alpha Centauri B")
    ax.scatter(r3_sol[-1, 0], r3_sol[-1, 1], r3_sol[-1, 2], color="orange", marker="o", s=100,
               label="Alpha Centauri C")
    # Add a few more bells and whistles
    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylabel("y-coordinate", fontsize=14)
    ax.set_zlabel("z-coordinate", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a three-body system\n", fontsize=14)
    ax.legend(loc="upper left", fontsize=14)
    plt.show()