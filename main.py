import itertools
import numpy as np
from equations import Particle, total_potential_energy

if __name__ == '__main__':
    particles = [
        Particle("alpha cen A", 1000, 100),
        Particle("alpha cen B", 1000, 100),
        Particle("alpha cen C", 500000, 100)
    ]
    radii = np.array([
        [-1, 10, 10],
        [10, -1, 100],
        [10, 100, -1],
    ])
    print(total_potential_energy(particles, radii))
