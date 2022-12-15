# main function
import argparse

from core import Grasp3DCore, Grasp3DViewer
from shapes import Box, Ellipsoid, HShape
import numpy as np
from scipy.spatial.transform import Rotation as R


def main(*args, **kwargs):
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Start the simulation server')
    parser.add_argument('--ui', type=bool, default=False, help='Start the UI')
    parser.add_argument('--input', type=str, default=None, help='Input file')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--log', type=str, default=None, help='Log file')
    parser.add_argument('--loglevel', type=str, default='INFO', help='Log level')

    sim = Grasp3DCore()
    sim.reset_experiment()
    for i in range(3):
        center = np.random.randint(20, 180, 3)
        rotation = R.random()
        box = Box()
        box.dimensions = np.random.randint(10, 20, 3)
        box.color = random_color()
        box.core = sim
        box.plot(center, rotation)
    for i in range(3):
        center = np.random.randint(20, 180, 3)
        rotation = R.random()
        ellips = Ellipsoid()
        ellips.dimensions = np.random.randint(10, 20, 3)
        ellips.color = random_color()
        ellips.core = sim
        ellips.plot(center, rotation)
    for i in range(3):
        h = HShape()
        center = np.random.randint(50, 150, 3)
        rotation = R.random()
        h.core = sim
        h.center = center
        h.rotation = rotation
        h.dimensions = np.random.randint(30, 50, 3)
        h.color = random_color()
        h.plot(center, rotation)

    viewer = Grasp3DViewer(sim)

    viewer.plot_views()


def random_color():
    # get a random versor
    color = np.random.rand(3)
    # normalize it
    color /= np.linalg.norm(color)
    # scale it to the range [0.5, 1]
    color = 0.5 * color + 0.5
    return color


if __name__ == '__main__':
    main()
