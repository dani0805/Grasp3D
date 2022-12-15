from typing import Protocol

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.spatial.transform import Rotation as R
from scipy import ndimage


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
        # first we copy the experiment to a new array
        view = self.core.experiment.copy()
        # we now find in polar coordinates around the camera the bounding box of the view
        # we first find the distance from the camera to each point of the view
        # we create a grid of the same size as the view
        x, y, z = np.meshgrid(np.arange(self.core.width).astype(float), np.arange(self.core.height).astype(float), np.arange(self.core.depth).astype(float))
        # now we transform the grid to have the camera at the origin
        x -= camera_position[0]
        y -= camera_position[1]
        z -= camera_position[2]
        # now we rotate the grid to have the camera looking in the z direction
        grid = np.stack([x, y, z], axis=3)
        # first we get the rotation as 3x3 matrix
        rotation_matrix = camera_rotation.as_matrix()
        # then we rotate the grid by multiplying the inverse rotation matrix by the grid
        grid = np.einsum("ij, lmnj-> lmni", rotation_matrix, grid)
        # now we find the distance from the camera to each point of the view
        distance = np.sqrt(grid[:,:,:,0] ** 2 + grid[:,:,:,1] ** 2 + grid[:,:,:,2] ** 2)
        # now we find the latitude and longitude of each point of the view
        latitude = np.arctan2(grid[:,:,:,1], grid[:,:,:,0])
        longitude = np.arctan2(grid[:,:,:,2], np.sqrt(grid[:,:,:,0] ** 2 + grid[:,:,:,1] ** 2))
        # now we find the bounding box of the view
        #bbox = np.array([np.min(distance), np.max(distance), np.min(latitude), np.max(latitude), np.min(longitude), np.max(longitude)])
        # we now compute the latitude and longitude difference between each pixel of the view
        latitude_step = camera_fov[0] / camera_width
        longitude_step = camera_fov[1] / camera_height
        depth_step = (camera_fov[3] - camera_fov[2]) / camera_depth_resolution
        # now we create a polar grid matching the camera field of view
        rho, theta, phi = np.meshgrid(
            np.arange(camera_fov[2], camera_fov[3], depth_step).astype(float),
            np.arange(-camera_fov[0]/2, camera_fov[0]/2, latitude_step).astype(float),
            np.arange(-camera_fov[1]/2, camera_fov[1]/2, longitude_step).astype(float)
            )
        polar_grid = np.stack([rho, theta, phi], axis=3)
        polar_view = np.zeros_like(polar_grid)
        # we now assign to the polar grid the color of the maximum color of the view for pixels within half a step of the polar grid
        # we first find for each pixel in the cartesian grid the nearest pixel in the polar grid
        grid = np.stack([distance, latitude, longitude], axis=3)
        # we now round the grid to the nearest pixel in the polar grid
        # first we offset the grid to have the minimum value of the polar grid at 0
        grid -= np.array([camera_fov[2], -camera_fov[0]/2, -camera_fov[1]/2])
        # now we round the grid to the nearest pixel in the polar grid
        grid = np.round(grid / np.array([depth_step, latitude_step, longitude_step])).astype(int)
        # we now set any grid value outside the polar grid to 0
        grid[grid < 0] = 0
        grid[grid[:, :, :, 0] >= camera_depth_resolution, 0] = camera_depth_resolution - 1
        grid[grid[:, :, :, 1] >= camera_width, 1] = camera_width - 1
        grid[grid[:, :, :, 2] >= camera_height, 2] = camera_height - 1

        polar_view[grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]] = view
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
