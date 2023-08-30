import itertools
from scipy.constants import G


class Particle:
    def __init__(self, name, mass, init_velocity):
        self._name = name
        self._mass = mass
        self._init_velocity = init_velocity

    def mass(self):
        return self._mass

def potential_energy(m1, m2, r):
    return -G * m1 * m2 / r

def total_potential_energy(particles, distances):
    # This function takes in an array of particle objects and the distance matrix to compute the total potential energy
    # of the system
    input_array = [x for x in range(len(particles))]
    combinations = itertools.combinations(input_array, 2)
    return -G * sum(potential_energy(particles[pair[0]].mass(), particles[pair[1]].mass(), distances[pair[0]][pair[1]])
                    for pair in list(combinations))




