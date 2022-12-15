from typing import Protocol

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.spatial.transform import Rotation as R


class Grasp3DCore(object):

    def __init__(self, width=200, height=200, depth=200, safe_area=10):
        self.experiment = None
        self.width = width
        self.height = height
        self.depth = depth
        self.safe_area = safe_area

    def reset_experiment(self):
        self.experiment = np.zeros((self.width, self.height, self.depth, 3))


class Shape3D(Protocol):
    dimensions: ndarray
    color: ndarray
    core: Grasp3DCore
    canvas: ndarray
    _experiment: ndarray

    # if the canvas is not None, the object is painted on the canvas else on the experiment
    def plot(self, position: ndarray, rotation: R, experiment: ndarray = None) -> None:
        ...

    def get_hitbox(self, position: ndarray, rotation:ndarray) -> ndarray:
        ...


class Grasp3DViewer(object):
    def __init__(self, core: Grasp3DCore):
        self.core = core

    def view_from_axis(self, axis, top: bool, reference_axis=True):
        # first we copy the experiment to a new array
        view = self.core.experiment.copy()
        if reference_axis:
            # if we want to see the reference axis we paint the axis on the view
            view[2:50, 2:8, 2:8] = [1, 0, 0]
            view[2:8, 2:50, 2:8] = [0, 1, 0]
            view[2:8, 2:8, 2:50] = [0, 0, 1]
        # then we make the colors decay with the distance from the top
        distance =self. core.depth if axis == 2 else self.core.width if axis == 0 else self.core.height
        decay = np.linspace(1, 0.2, distance)[:, np.newaxis, np.newaxis, np.newaxis]
        # reorder decay to match the axis
        decay = np.moveaxis(decay, 0, axis)
        view = view * decay
        # then we find the first non zero element from the top if top is true else from the bottom
        # then we add a z-channel to the array to order the pixels by z only for the non zero elements
        view = np.concatenate([view, np.zeros_like(view[:, :, :, 0:1])], axis=3)
        # now we fill the z-channel with the coordinate of the non zero elements along the axis selected
        z_channel = np.arange(distance)[:, np.newaxis, np.newaxis]
        # reorder z_channel to match the axis
        z_channel = np.moveaxis(z_channel, 0, axis)
        view[:, :, :, 3] = z_channel
        # now we set to zero the z-channel of the zero elements
        view[np.all(view[:, :, :, 0:3] == 0, axis=3), 3] = 0
        # we now take the argmax of the z-channel to find the first non zero element from the top if top is true else from the bottom excluding 0
        first_non_zero = np.argmax(view[:, :, :, 3], axis=axis) if top else np.argmin(np.where(view[:, :, :, 3] > 0, view[:, :, :, 3], 10e5), axis=axis)
        # we now take the first non zero element from the top discarding the z-channel
        if axis == 2:
            view = view[np.arange(self.core.width)[:, None],
                   np.arange(self.core.height)[None, :],
                   first_non_zero, 0:3]
            if top:
                view = np.flip(view, axis=0)
        elif axis == 0:
            view = view[first_non_zero,
                   np.arange(self.core.height)[:, None],
                   np.arange(self.core.depth)[None, :], 0:3]
            # swap y and z axis
            view = np.swapaxes(view, 0, 1)
            if not top:
                view = np.flip(view, axis=0)

        else:
            view = view[np.arange(self.core.width)[:, None],
                   first_non_zero,
                   np.arange(self.core.depth)[None, :], 0:3]
            if not top:
                view = np.flip(view, axis=1)
        return view

    def plot_views(self):
        # plot the views from the top, front, left and right
        fig, axes = plt.subplots(2, 3, figsize=(10, 10))
        axes[0, 1].imshow(self.view_from_axis(0, False))
        axes[0, 1].set_title('Top')
        axes[1, 1].imshow(self.view_from_axis(2, False))
        axes[1, 1].set_title('Front')
        axes[1, 0].imshow(self.view_from_axis(1, False))
        axes[1, 0].set_title('Left')
        axes[1, 2].imshow(self.view_from_axis(1, True))
        axes[1, 2].set_title('Right')
        plt.show()
