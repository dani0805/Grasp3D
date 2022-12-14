import math
from abc import abstractmethod

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

    def init_canvas(self, position: ndarray) -> ndarray:
        pos = np.round(position).astype(int)
        self.canvas = self.experiment[
                 pos[0] - self.canvas_radius:pos[0] + self.canvas_radius,
                 pos[1] - self.canvas_radius:pos[1] + self.canvas_radius,
                 pos[2] - self.canvas_radius:pos[2] + self.canvas_radius,
                 :].copy()
        return self.canvas

    @abstractmethod
    def get_base_imitation_grasps(self,
                                  gripper_width: float,
                                  gripper_length: float,
                                  gripper_aperture: float
                                  ) -> ndarray:
        # abstract method that returns the unrotated uncentered grasps for the object
        # return the grasps as 3x4 SE(3) matrices shaped as (n, 3, 4)
        # must be implemented by the child class
        pass

    def get_imitation_grasps(self,
                             position: ndarray,
                             rotation: ndarray,
                             gripper_width: float,
                             gripper_length: float,
                             gripper_aperture: float
                             ) -> ndarray:
        # get the base grasps
        grasps = self.get_base_imitation_grasps(gripper_width, gripper_length, gripper_aperture)
        # rotate the grasps
        grasps_r = np.einsum('ik, lkj -> lij', rotation, grasps[:, :3, :3])
        # translate the grasps
        grasps_t = grasps[:, :3, 3] + position
        # return the grasps
        return np.concatenate([grasps_r, grasps_t[:, :, np.newaxis]], axis=2)


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

    def get_base_imitation_grasps(self,
                                  gripper_width: float,
                                  gripper_length: float,
                                  gripper_aperture: float
                                  ) -> ndarray:
        # return the grasps as 3x4 SE(3) matrices shaped as (n, 3, 4)
        # must be implemented by the child class
        grasps = []
        if gripper_aperture > self.dimensions[0]:
            # if the gripper aperture is bigger than the width of the box
            # then the gripper can grasp the box from the left and right faces looking at it from
            # the front, the back, the top and the bottom
            grasps.append(np.array([
                [0, 0, 1, np.minimum( self.dimensions[0] / 2 - gripper_length, 0)],
                [0, 1, 0, 0],
                [-1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, -1, np.minimum( self.dimensions[0] / 2 - gripper_length, 0)],
                [0, 1, 0, 0],
                [1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, 1, np.minimum( self.dimensions[0] / 2 - gripper_length, 0)],
                [0, -1, 0, 0],
                [-1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, -1, np.minimum( self.dimensions[0] / 2 - gripper_length, 0)],
                [0, -1, 0, 0],
                [1, 0, 0, 0]
            ]))
        if gripper_aperture > self.dimensions[1]:
            # if the gripper aperture is bigger than the height of the box
            # then the gripper can grasp the box from the top and bottom faces looking at it from
            # the front, the back, the left and the right
            grasps.append(np.array([
                [0, 0, 1, 0],
                [0, 1, 0, np.minimum( self.dimensions[1] / 2 - gripper_length, 0)],
                [-1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, -1, 0],
                [0, 1, 0, np.minimum( self.dimensions[1] / 2 - gripper_length, 0)],
                [1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, 1, 0],
                [0, -1, 0, np.minimum( self.dimensions[1] / 2 - gripper_length, 0)],
                [-1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, -1, 0],
                [0, -1, 0, np.minimum( self.dimensions[1] / 2 - gripper_length, 0)],
                [1, 0, 0, 0]
            ]))
        if gripper_aperture > self.dimensions[2]:
            # if the gripper aperture is bigger than the depth of the box
            # then the gripper can grasp the box from the front and back faces looking at it from
            # the top, the bottom, the left and the right
            grasps.append(np.array([
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, np.minimum( self.dimensions[2] / 2 - gripper_length, 0)]
            ]))
            grasps.append(np.array([
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, np.minimum( self.dimensions[2] / 2 - gripper_length, 0)]
            ]))
            grasps.append(np.array([
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [-1, 0, 0, np.minimum( self.dimensions[2] / 2 - gripper_length, 0)]
            ]))
            grasps.append(np.array([
                [0, 0, -1, 0],
                [0, -1, 0, 0],
                [1, 0, 0, np.minimum( self.dimensions[2] / 2 - gripper_length, 0)]
            ]))
        return np.array(grasps)


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

    def get_base_imitation_grasps(self,
                                  gripper_width: float,
                                  gripper_length: float,
                                  gripper_aperture: float
                                  ) -> ndarray:
        # return the grasps as 3x4 SE(3) matrices shaped as (n, 3, 4)
        # must be implemented by the child class
        grasps = []
        if gripper_aperture > self.dimensions[0]:
            # if the gripper aperture is bigger than the width of the ellipsoid
            # then the gripper can grasp the ellipsoid from the left and right faces looking at it from
            # the front, the back, the top and the bottom
            grasps.append(np.array([
                [0, 0, 1, np.minimum( self.dimensions[0] / 2 - gripper_length, 0)],
                [0, 1, 0, 0],
                [-1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, -1, np.minimum( self.dimensions[0] / 2 - gripper_length, 0)],
                [0, 1, 0, 0],
                [1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, 1, np.minimum( self.dimensions[0] / 2 - gripper_length, 0)],
                [0, -1, 0, 0],
                [-1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, -1, np.minimum( self.dimensions[0] / 2 - gripper_length, 0)],
                [0, -1, 0, 0],
                [1, 0, 0, 0]
            ]))
        if gripper_aperture > self.dimensions[1]:
            # if the gripper aperture is bigger than the height of the ellipsoid
            # then the gripper can grasp the ellipsoid from the top and bottom faces looking at it from
            # the front, the back, the left and the right
            grasps.append(np.array([
                [0, 0, 1, 0],
                [0, 1, 0, np.minimum( self.dimensions[1] / 2 - gripper_length, 0)],
                [-1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, -1, 0],
                [0, 1, 0, np.minimum( self.dimensions[1] / 2 - gripper_length, 0)],
                [1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, 1, 0],
                [0, -1, 0, np.minimum( self.dimensions[1] / 2 - gripper_length, 0)],
                [-1, 0, 0, 0]
            ]))
            grasps.append(np.array([
                [0, 0, -1, 0],
                [0, -1, 0, np.minimum( self.dimensions[1] / 2 - gripper_length, 0)],
                [1, 0, 0, 0]
            ]))
        if gripper_aperture > self.dimensions[2]:
            # if the gripper aperture is bigger than the depth of the ellipsoid
            # then the gripper can grasp the ellipsoid from the front and back faces looking at it from
            # the top, the bottom, the left and the right
            grasps.append(np.array([
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, np.minimum( self.dimensions[2] / 2 - gripper_length, 0)]
            ]))
            grasps.append(np.array([
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, np.minimum( self.dimensions[2] / 2 - gripper_length, 0)]
            ]))
            grasps.append(np.array([
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [-1, 0, 0, np.minimum( self.dimensions[2] / 2 - gripper_length, 0)]
            ]))
            grasps.append(np.array([
                [0, 0, -1, 0],
                [0, -1, 0, 0],
                [1, 0, 0, np.minimum( self.dimensions[2] / 2 - gripper_length, 0)]
            ]))
        return np.array(grasps)


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

    def get_base_imitation_grasps(self,
                                  gripper_width: float,
                                  gripper_length: float,
                                  gripper_aperture: float
                                  ) -> ndarray:
        # return the grasps as 3x4 SE(3) matrices shaped as (n, 3, 4)
        # must be implemented by the child class
        grasps = []
        for shape in self.shapes:
            # add the grasps of each primitive
            primitive = shape[0]()
            primitive.core = self.core
            primitive.dimensions = shape[3] * self.dimensions/100
            primitive_rotation = shape[2]
            primitive_position = shape[1] * self.dimensions/100
            primitive_grasps = primitive.get_imitation_grasps(primitive_position, primitive_rotation.as_matrix(), gripper_width, gripper_length, gripper_aperture)
            for grasp in primitive_grasps:
                # transform the grasp from the primitive frame to the composite shape frame
                grasp = np.linalg.inv(primitive_rotation.as_matrix()) @ grasp
                grasp[:, 3] = primitive_position - grasp[:, 3]
                grasps.append(grasp)
        # check for all the grasps if the back of the gripper is colliding with the composite shape
        # if so remove the grasp
        grasps = np.array(grasps)
        remove_grasps = []
        for i in range(len(grasps)):
            # compute the position of the back of the gripper
            gripper_position = grasps[i, :, 3] - grasps[i, :, 2] * gripper_length
            # check if any primitive occupies the cube of size gripper_length /2 on the gripper position
            for shape in self.shapes:
                bbox = [shape[1] * self.dimensions/100 - shape[3] * self.dimensions/100/2,
                        shape[1] * self.dimensions/100 + shape[3] * self.dimensions/100/2]
                if np.all(gripper_position + gripper_length/2 > bbox[0]) and np.all(gripper_position - gripper_length/2 < bbox[1]):
                    remove_grasps.append(i)
                    break
        grasps = np.delete(grasps, remove_grasps, axis=0)
        return grasps




            # check if the gripper is colliding with the composite shape

        return np.array(grasps)



class HShape(CompositeShape):
    def __init__(self):
        super().__init__()
        self.shapes = []
        self.add_primitive(Box, np.array([49, 49, 49]), R.identity(), np.array([60, 20, 20]))
        self.add_primitive(Box, np.array([9, 49, 49]), R.identity(), np.array([20, 100, 20]))
        self.add_primitive(Box, np.array([89, 49, 49]), R.identity(), np.array([20, 100, 20]))
