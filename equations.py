import itertools
from scipy.constants import G
import numpy as np

# Reference quantities
m_nd = 1.989e+30  # kg #mass of the sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 30000  # m/s #relative velocity of earth around the sun
t_nd = 79.91 * 365 * 24 * 3600 * 0.51  # s #orbital period of Alpha Centauri
# Net constants
K1 = G * t_nd * m_nd / (r_nd ** 2 * v_nd)
K2 = v_nd * t_nd / r_nd


class Particle:
    def __init__(self, name, mass, init_velocity, position):
        self._name = name
        self._mass = mass
        self._velocity = init_velocity
        self._position = position

    def mass(self):
        return self._mass

    def position(self):
        return self._position

    def velocity(self):
        return self._velocity

    def name(self):
        return self._name

    def update_position(self, pos):
        self._position = pos

    def update_velocity(self, v):
        self._velocity = v


def potential_energy(m1, m2, r):
    return -G * m1 * m2 / r


def distance(r1, r2):
    return np.linalg.norm(r1 - r2)


def total_potential_energy(particles):
    # This function takes in an array of particle objects and the distance matrix to compute the total potential energy
    # of the system
    input_array = [x for x in range(len(particles))]
    combinations = itertools.combinations(input_array, 2)
    return -G * sum(potential_energy(particles[pair[0]].mass(), particles[pair[1]].mass(),
                                     distance(particles[pair[0]].position(), particles[pair[1]].position()))
                    for pair in list(combinations))


def centre_of_mass(particles):
    return sum(particle.mass() * particle.position() for particle in particles) / sum(
        particle.mass() for particle in particles)


def velocity_of_centre_of_mass(particles):
    return sum(particle.mass() * particle.velocity() for particle in particles) / sum(
        particle.mass() for particle in particles)


def generate_rv_vectors(particles):
    n = len(particles)
    vectors = []
    for i in range(n):
        vectors.append(particles[i].position())
    for i in range(n):
        vectors.append(particles[i].velocity())
    return np.array(vectors)


def generate_masses_array(particles):
    return [particle.mass() for particle in particles]


def drdt(t, v):
    return K2 * v


# Dynamically take in dvdt
def dvdt(t, r, masses):
    # Pass in all of r's
    n = len(r)
    dv_ndt = np.zeros(3)
    for i in range(1, n):
        temp_vec = r[i] - r[0]
        dv_ndt += 1 * masses[i] * temp_vec / (np.linalg.norm(temp_vec)) ** 3
    return K1 * dv_ndt


def n_body_odes(t, vectors, *args):
    n = int(len(vectors) / 2)
    # First half are r vectors and 2nd half are velocity vectors
    masses = [arg for arg in args]
    r_vectors = vectors[:n]

    v_derivatives = []
    for i in range(n):
        r_temp = [r_vectors[(j + i) % n] for j in range(n)]
        v_derivatives.append(dvdt(0, r_temp, masses))

    r_derivatives = np.array([drdt(t, vec) for vec in v_derivatives])
    return np.concatenate((r_derivatives, v_derivatives))


def runge_kutta4_onestep(f_t: callable, tau, y, t, *args):
    k1 = f_t(t, y, *args[0]) * tau
    k2 = f_t(t + tau / 2, y + 0.5 * k1, *args[0]) * tau
    k3 = f_t(t + tau / 2, y + 0.5 * k2, *args[0]) * tau
    k4 = f_t(t + tau, y + k3, *args[0]) * tau
    return y + k1 / 6 + k2 / 3 + k4 / 6


def run_simulation(particles, num_steps, tau=0.01):
    n = len(particles)
    init_params = generate_rv_vectors(particles)
    masses = generate_masses_array(particles)
    solutions = np.zeros((num_steps, 2 * n, 3))
    solutions[0] = runge_kutta4_onestep(n_body_odes, tau, init_params, 0, masses)
    for i in range(1, num_steps):
        solutions[i] = runge_kutta4_onestep(n_body_odes, tau, solutions[i-1], 0, masses)
    return solutions


def evolve_rungekutta4(particles, timestep):
    masses = generate_masses_array(particles)
    n = len(particles)
    r_vectors = generate_rv_vectors(particles)[:n]

    for i in range(n):
        r_temp = [r_vectors[(j + i) % n] for j in range(n)]
        particles[i].update_velocity(runge_kutta4_onestep(dvdt, timestep, r_temp, 0, masses))
    for particle in particles:
        particle.update_position(runge_kutta4_onestep(drdt, timestep, particle.position(), 0))
