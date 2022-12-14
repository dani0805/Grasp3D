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

    def generate_random_shape(self, size=None, type="cube", color=None):
        if color is None:
            color = np.array([0., 1., 0.])
        if size is None:
            size = [10, 10, 10]
        if type == "cube":
            return self.generate_random_cube(size, color=color)

    def generate_random_cube(self, size, color: ndarray):
        x_center = np.random.randint(self.safe_area + size[0], self.width - self.safe_area - size[0])
        y_center = np.random.randint(self.safe_area + size[1], self.height - self.safe_area - size[1])
        z_center = np.random.randint(self.safe_area + size[2], self.depth - self.safe_area - size[2])
        rotation = R.random()
        center_to_north_face = np.array([0, 0, size[2] / 2])
        center_to_south_face = np.array([0, 0, -size[2] / 2])
        center_to_east_face = np.array([size[0] / 2, 0, 0])
        center_to_west_face = np.array([-size[0] / 2, 0, 0])
        center_to_front_face = np.array([0, size[1] / 2, 0])
        center_to_back_face = np.array([0, -size[1] / 2, 0])
        faces = [
            center_to_north_face,
            center_to_south_face,
            center_to_east_face,
            center_to_west_face,
            center_to_front_face,
            center_to_back_face
        ]

        faces = [rotation.apply(face) for face in faces]
        # copy from self.experiment the area where the object is going to be drawn
        canvas = self.experiment[
                 x_center - size[0]:x_center + size[0],
                 y_center - size[1]:y_center + size[1],
                 z_center - size[2]:z_center + size[2],
                 :].copy()
        # create a grid of points labeled with their x,y,z coordinates relative to the center of the cube
        x, y, z = np.meshgrid(
            np.arange(-size[0], size[0]),
            np.arange(-size[1], size[1]),
            np.arange(-size[2], size[2]),
            indexing='ij'
        )
        # use the labels to create a grid of points
        points = np.stack([x, y, z], axis=3)
        # check which points have dot product with the faces smaller than the distance from the center to the face squared
        # this is a quick way to check if a point is inside a cube
        # unqueeze the faces by 3 extra dimensions to make the dot product broadcastable
        faces = np.expand_dims(faces, axis=(1, 2, 3))
        # unqueeze the points by 1 extra dimension to make the dot product broadcastable
        points = np.expand_dims(points, axis=0)
        # compute the dot product between the points and the faces and take the maximum
        dot_products = np.sum(points * faces, axis=4)
        # compute the distance from the center to the face
        face_distances = np.linalg.norm(faces, axis=4)
        # check if the dot product is smaller than the distance from the center to the face squared
        # this is a quick way to check if a point is inside a cube
        inside = dot_products < face_distances ** 2
        # check if the point is inside any of the faces
        inside = inside.all(axis=0)
        # if inside color the canvas with the color of the object
        canvas[inside] = color
        # paste the canvas on the experiment
        self.experiment[
        x_center - size[0]:x_center + size[0],
        y_center - size[1]:y_center + size[1],
        z_center - size[2]:z_center + size[2],
        :] = canvas


class Grasp3DViewer(object):
    def __init__(self, core: Grasp3DCore):
        self.core = core

    def view_from_axis(self, axis, top: bool):
        # first we copy the experiment to a new array
        view = self.core.experiment.copy()
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
            view = view[np.arange(self.core.width)[:, None], np.arange(self.core.height)[None, :], first_non_zero, 0:3]
        elif axis == 0:
            view = view[first_non_zero, np.arange(self.core.height)[:, None], np.arange(self.core.depth)[None, :], 0:3]
        else:
            view = view[np.arange(self.core.width)[:, None], first_non_zero, np.arange(self.core.depth)[None, :], 0:3]
        return view

    def plot_views(self):
        # plot the views from the top, front, left and right
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(self.view_from_axis(2, True))
        axes[0, 0].set_title('Top')
        axes[0, 1].imshow(self.view_from_axis(0, True))
        axes[0, 1].set_title('Front')
        axes[1, 0].imshow(self.view_from_axis(1, True))
        axes[1, 0].set_title('Left')
        axes[1, 1].imshow(self.view_from_axis(1, False))
        axes[1, 1].set_title('Right')
        plt.show()
