import itertools
import numpy as np
from equations import Particle, total_potential_energy, centre_of_mass, velocity_of_centre_of_mass, n_body_odes, \
    generate_masses_array, generate_rv_vectors, evolve_rungekutta4
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import os

if __name__ == '__main__':
    particles = [
        Particle("alpha cen A", 1000000, np.array([2000, 0, 0], dtype="float64"), np.array([2, 0, 0], dtype="float64")),
        Particle("alpha cen B", 1000000, np.array([2, 1000, 32], dtype="float64"), np.array([0, 2, 2], dtype="float64")),
        Particle("alpha cen C", 500000, np.array([2, 5000, 90], dtype="float64"), np.array([1, 0, 1], dtype="float64"))
    ]

    # Test individual functions
    # print(total_potential_energy(particles))
    # print(velocity_of_centre_of_mass(particles))

    # Call n_body_odes
    # print(n_body_odes(init_params, t, *generate_masses_array(particles)))

    # Call ode solution
    init_params = generate_rv_vectors(particles).flatten()
    t = np.linspace(0, 20, 500)
    solutions = scipy.integrate.odeint(n_body_odes, init_params, t, args=(tuple(generate_masses_array(particles))))
    # print(solutions)
    # print(generate_rv_vectors(particles))
    # evolve_rungekutta4(particles, 0.1)
    # print(generate_rv_vectors(particles))


    ### Plotting code
    r1_sol = solutions[:,:3]
    r2_sol = solutions[:,3:6]
    r3_sol = solutions[:,6:9]
    os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/universal-darwin/'
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    # Create 3D axes
    ax = fig.add_subplot(111, projection="3d")
    print(r1_sol[:, 1])
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
               label="Alpha Centauri B")
    # Add a few more bells and whistles
    ax.set_xlabel("x-coordinate", fontsize=14)
    ax.set_ylabel("y-coordinate", fontsize=14)
    ax.set_zlabel("z-coordinate", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a three-body system\n", fontsize=14)
    ax.legend(loc="upper left", fontsize=14)
    plt.show()