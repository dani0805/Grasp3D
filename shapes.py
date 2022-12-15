import math

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation as R

from core import Shape3D


class AbstractShape3D(Shape3D):
    _experiment:ndarray = None

    def init_grid(self) -> ndarray:
        x, y, z = np.meshgrid(
            np.arange(-self.canvas_radius, self.canvas_radius),
            np.arange(-self.canvas_radius, self.canvas_radius),
            np.arange(-self.canvas_radius, self.canvas_radius),
            indexing='ij'
        )
        # use the labels to create a grid of points
        return np.stack([x, y, z], axis=3).astype(float)

    def init_canvas(self, position):
        pos = np.round(position).astype(int)
        self.canvas = self.experiment[
                 pos[0] - self.canvas_radius:pos[0] + self.canvas_radius,
                 pos[1] - self.canvas_radius:pos[1] + self.canvas_radius,
                 pos[2] - self.canvas_radius:pos[2] + self.canvas_radius,
                 :].copy()

    @property
    def canvas_radius(self):
        # get the radius of the canvas based on the dimensions of the shape
        return np.ceil(np.max(self.dimensions) / math.sqrt(3) + 1).astype(int)

    def paint_object_in_sim(self, center, mask):
        # if inside color the canvas with the color of the object
        self.canvas[mask[0:self.canvas.shape[0], 0:self.canvas.shape[1], 0:self.canvas.shape[2]]] = self.color
        # paste the canvas on the experiment
        c = np.round(center).astype(int)
        self.experiment[
        c[0] - self.canvas_radius:c[0] + self.canvas_radius,
        c[1] - self.canvas_radius:c[1] + self.canvas_radius,
        c[2] - self.canvas_radius:c[2] + self.canvas_radius,
        :] = self.canvas

    @property
    def experiment(self):
        if self._experiment is not None:
            return self._experiment
        return self.core.experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment


class Box(AbstractShape3D):

    def plot(self, position: ndarray, rotation:R, canvas:ndarray = None) -> None:
        faces = [
            np.array([0, 0, self.dimensions[2] / 2]),
            np.array([0, 0, -self.dimensions[2] / 2]),
            np.array([self.dimensions[0] / 2, 0, 0]),
            np.array([-self.dimensions[0] / 2, 0, 0]),
            np.array([0, self.dimensions[1] / 2, 0]),
            np.array([0, -self.dimensions[1] / 2, 0])
        ]

        faces = [rotation.apply(face) for face in faces]
        # copy from self.experiment the area where the object is going to be drawn
        self.init_canvas(position)
        points = self.init_grid()

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
        self.paint_object_in_sim(position, inside)


class Ellipsoid(AbstractShape3D):

    def plot(self, position: ndarray, rotation:R, canvas:ndarray = None) -> None:
        # copy the relevant part of the core.experiment
        self.init_canvas(position)
        # create a grid of points labeled with their x,y,z coordinates relative to the center of the ellipsoid
        points = self.init_grid()
        # create the main axes of the ellipsoid
        v_x = np.array([1, 0, 0])*self.dimensions[0]/2
        v_y = np.array([0, 1, 0])*self.dimensions[1]/2
        v_z = np.array([0, 0, 1])*self.dimensions[2]/2
        # rotate the axes
        v_x = rotation.apply(v_x)
        v_y = rotation.apply(v_y)
        v_z = rotation.apply(v_z)
        # compute the dot product between the points and the axes
        dot_products = ((points @ v_x) / np.sum(v_x ** 2)) ** 2 + \
                       ((points @ v_y) / np.sum(v_y ** 2)) ** 2 + \
                       ((points @ v_z) / np.sum(v_z ** 2)) ** 2
        # check if the dot product is smaller than 1
        inside = dot_products < 1
        self.paint_object_in_sim(position, inside)


class CompositeShape(AbstractShape3D):
    # each primitive used to create the composite shape is defined by a shape, a position, a rotation
    # and the dimensions relative to a standard 100x100x100 cube
    shapes: list[(type, ndarray, R, ndarray)]

    def plot(self, position: ndarray, rotation:R, canvas:ndarray = None) -> None:
        # copy the relevant part of the core.experiment
        self.init_canvas(position)
        # create a grid of points labeled with their x,y,z coordinates relative to the center of the composite shape
        for shape in self.shapes:
            # compute the position of the primitive relative to the center of the composite shape
            primitive_position = position + rotation.apply(shape[1] * self.dimensions/100)
            # compute the rotation of the primitive relative to the rotation of the composite shape
            primitive_rotation = rotation * shape[2]
            # create the primitive
            primitive = shape[0]()
            primitive.core = self.core
            primitive.color = self.color
            primitive.dimensions = shape[3] * self.dimensions/100
            primitive.rotation = primitive_rotation
            # plot the primitive
            primitive.plot(primitive_position, primitive_rotation, self.canvas)

    def add_primitive(self, shape:type, position:ndarray, rotation:R, dimensions:ndarray):
        self.shapes.append((shape, position, rotation, dimensions))


class HShape(CompositeShape):
    def __init__(self):
        super().__init__()
        self.shapes = []
        self.add_primitive(Box, np.array([49, 49, 49]), R.identity(), np.array([60, 20, 20]))
        self.add_primitive(Box, np.array([9, 49, 49]), R.identity(), np.array([20, 100, 20]))
        self.add_primitive(Box, np.array([89, 49, 49]), R.identity(), np.array([20, 100, 20]))
