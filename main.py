# main function
import argparse

from core import Grasp3DCore, Grasp3DViewer


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
    sim.generate_random_shape(size=[50, 50, 50], type='cube', color=[1, 0, 0])
    sim.generate_random_shape(size=[30, 30, 30], type='cube', color=[0, 1, 0])
    viewer = Grasp3DViewer(sim)

    viewer.plot_views()


if __name__ == '__main__':
    main()
