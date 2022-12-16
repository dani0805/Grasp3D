import timeit
from typing import Protocol

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy import ndimage

DEBUG_TIME = False

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

    def view_from_axis(self, axis, top: bool, reference_axis=True, view=None):
        if view is None:
            # first we copy the experiment to a new array
            view = self.core.experiment.copy()
        bbox = np.array([view.shape[0], view.shape[1], view.shape[2]])

        if reference_axis:
            # if we want to see the reference axis we paint the axis on the view
            view[2:50, 2:8, 2:8] = [1, 0, 0]
            view[2:8, 2:50, 2:8] = [0, 1, 0]
            view[2:8, 2:8, 2:50] = [0, 0, 1]
        # then we make the colors decay with the distance from the top
        distance = bbox[axis]
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
            view = view[np.arange(bbox[0])[:, None],
                   np.arange(bbox[1])[None, :],
                   first_non_zero, 0:3]
            if top:
                view = np.flip(view, axis=0)
        elif axis == 0:
            view = view[first_non_zero,
                   np.arange(bbox[1])[:, None],
                   np.arange(bbox[2])[None, :], 0:3]
            # swap y and z axis
            view = np.swapaxes(view, 0, 1)
            if not top:
                view = np.flip(view, axis=0)

        else:
            view = view[np.arange(bbox[0])[:, None],
                   first_non_zero,
                   np.arange(bbox[2])[None, :], 0:3]
            if not top:
                view = np.flip(view, axis=1)
        return view

    def camera_view(self, camera_position:ndarray, camera_rotation:R, camera_fov:ndarray, camera_height:int, camera_width:int, camera_depth_resolution:int):
        # record the time
        if DEBUG_TIME:
            start = timeit.default_timer()
        # first we copy the experiment to a new array
        view = self.core.experiment.copy()
        # find the indices of the first 3 axes of the non zero elements
        non_zero_indices = np.nonzero(view)
        # ignore axe 3 and remove duplicates
        non_zero_indices = np.unique(non_zero_indices[0:3], axis=1)
        # copy the non zero indices to a new array to save the original cartesian coordinates
        non_zero_indices_cartesian = non_zero_indices.copy()
        # offset the indices by the camera position
        non_zero_indices = non_zero_indices - camera_position[:, np.newaxis]
        # rotate the indices by the camera rotation
        rotation_matrix = camera_rotation.as_matrix()
        non_zero_indices = rotation_matrix @ non_zero_indices
        # find the distance of the non zero elements from the camera
        distance = np.linalg.norm(non_zero_indices, axis=0)
        # find the latitude of the non zero elements from the camera
        latitude = np.arctan2(non_zero_indices[1], non_zero_indices[0])
        # find the longitude of the non zero elements from the camera
        longitude = np.arctan2(non_zero_indices[2], (non_zero_indices[0] ** 2 + non_zero_indices[1] ** 2) ** 0.5)
        # stack distance, latitude and longitude
        non_zero_indices = np.stack([distance, latitude, longitude], axis=0)

        # we now compute the latitude and longitude difference between each pixel of the view
        latitude_step = camera_fov[0] / camera_width
        longitude_step = camera_fov[1] / camera_height
        depth_step = (camera_fov[3] - camera_fov[2]) / camera_depth_resolution
        # we now assign to the polar grid the color of the maximum color of the view for pixels within half a step of the polar grid
        polar_view = np.zeros((camera_depth_resolution, camera_height, camera_width, 3))
        # we first find for each pixel in the cartesian grid the nearest pixel in the polar grid

        # we now round the non zero indices to the nearest pixel in the polar grid
        # first we offset the indices to have the minimum value of the polar grid at 0
        non_zero_indices = non_zero_indices - np.array([camera_fov[2], -camera_fov[0] / 2, -camera_fov[1] / 2])[:, np.newaxis]
        # now we divide by the step to find the nearest pixel in the polar grid
        non_zero_indices = non_zero_indices / np.array([depth_step, latitude_step, longitude_step])[:, np.newaxis]
        # now we round to the nearest pixel in the polar grid
        non_zero_indices = np.round(non_zero_indices).astype(int)
        # create a mask that selects only the indeces inside the polar grid bounds
        mask = np.all(np.logical_and(non_zero_indices >= 0, non_zero_indices < np.array([camera_depth_resolution, camera_height, camera_width])[:, np.newaxis]), axis=0)
        # apply the mask to the non zero indices
        non_zero_indices = non_zero_indices[:, mask]
        # apply the mask to the non zero indices cartesian
        non_zero_indices_cartesian = non_zero_indices_cartesian[:, mask]
        # create a mask that selects the unique indices
        mask = np.unique(non_zero_indices, axis=1, return_index=True)[1]
        # apply the mask to the non zero indices
        non_zero_indices = non_zero_indices[:, mask]
        # apply the mask to the non zero indices cartesian
        non_zero_indices_cartesian = non_zero_indices_cartesian[:, mask]
        # we now assign to the polar grid at the non zero indices the color of the view at the corresponding non zero indices cartesian
        polar_view[non_zero_indices[0], non_zero_indices[1], non_zero_indices[2]] = view[non_zero_indices_cartesian[0], non_zero_indices_cartesian[1], non_zero_indices_cartesian[2]]

        return self.view_from_axis(0, False, view=polar_view, reference_axis=False)



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
        # plot the views from the camera from the front top left and front top right pointing at the center
        # front top left camera
        camera_position = np.array([-30, -30, -30])
        # the rotation points at the center of the experiment which corresponds to [width/2, height/2, depth/2]
        # the euler angles from the [-30, -30, -30] to [width/2, height/2, depth/2]
        # are [atan2(width/2 +30, depth/2 + 30), atan2(height/2 + 30, depth/2 + 30), 0]
        camera_rotation = R.from_euler('xyz', [
                                           np.arctan2(self.core.width/2 + 30, self.core.depth/2 + 30),
                                           np.arctan2(np.sqrt((self.core.width/2 + 30) ** 2 + (self.core.depth/2 + 30) ** 2), self.core.height/2 + 30),
                                           0
                                       ], degrees=False)
        camera_fov = np.array([np.pi/3, np.pi/3, 30, 300])
        camera_height = 100
        camera_width = 100
        camera_depth_resolution = 100
        axes[0, 0].imshow(self.camera_view(camera_position, camera_rotation, camera_fov, camera_height, camera_width, camera_depth_resolution))
        axes[0, 0].set_title('Front top left')
        # front top right camera
        camera_position = np.array([-30, 230, -30])
        # the rotation points at the center of the experiment which corresponds to [width/2, height/2, depth/2]
        # the euler angles from the [30, -30, -30] to [width/2, height/2, depth/2]
        # are [atan2(width/2 -30, depth/2 + 30), atan2(height/2 + 30, depth/2 + 30), 0]
        camera_rotation = R.from_euler('xyz', [
                                            -np.arctan2(self.core.width/2 + 30, self.core.depth/2 + 30),
                                            np.arctan2(np.sqrt((self.core.width/2 + 30) ** 2 + (self.core.depth/2 + 30) ** 2), self.core.height/2 + 30),
                                            0
                                        ], degrees=False)
        axes[0, 2].imshow(self.camera_view(camera_position, camera_rotation, camera_fov, camera_height, camera_width, camera_depth_resolution))
        axes[0, 2].set_title('Front top right')


        plt.show()
