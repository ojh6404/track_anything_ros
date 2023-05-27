#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from collections import defaultdict
from enum import IntEnum

import geometry_msgs.msg
import jsk_recognition_msgs.msg
from jsk_topic_tools import ConnectionBasedTransport
import matplotlib.pyplot as plt
import message_filters
import numpy as np
import rospy
import shapely.geometry
import skrobot
from skrobot.coordinates import rotate_points


class BoundingBoxVertexType(IntEnum):
    FrontTopRight = 0
    FrontTopLeft = 1
    FrontBottomLeft = 2
    FrontBottomRight = 3
    RearTopRight = 4
    RearTopLeft = 5
    RearBottomLeft = 6
    RearBottomRight = 7
    Center = 8
    TotalCornerVertexCount = 8
    # Corner vertexes doesn't include the center point
    TotalVertexCount = 9


class BoundingBox(skrobot.coordinates.CascadedCoords):

    def __init__(self, *args, **kwargs):
        self.dimensions = kwargs.pop('dimensions', [1.0, 1.0, 1.0])
        super(BoundingBox, self).__init__(*args, **kwargs)

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dim):
        self._dimensions = np.array(dim, 'f')

    @property
    def vertices_coords(self):
        center_coords = self.copy_worldcoords()
        dx, dy, dz = self.dimensions / 2.0
        add_y = 1.0
        vertices_coords = [
            # front top right
            center_coords.copy_worldcoords().translate((dx, dy, dz)),
            # front top left
            center_coords.copy_worldcoords().translate((dx, -dy - add_y, dz)),
            # front bottom left
            center_coords.copy_worldcoords().translate((dx, -dy - add_y, -dz)),
            # front bottom right
            center_coords.copy_worldcoords().translate((dx, dy, -dz)),
            # rear top right
            center_coords.copy_worldcoords().translate((-dx, dy,  dz)),
            # rear top left
            center_coords.copy_worldcoords().translate((-dx, -dy - add_y, dz)),
            # rear bottom left
            center_coords.copy_worldcoords().translate(
                (-dx, -dy - add_y, -dz)),
            # rear bottom right
            center_coords.copy_worldcoords().translate((-dx, dy, -dz)),
            # center
            center_coords.copy_worldcoords(),
        ]
        return vertices_coords

    @property
    def vertices(self):
        return np.array([c.worldpos() for c in self.vertices_coords],
                        dtype=np.float64)

    @staticmethod
    def from_ros_message(box_msg):
        trans = (box_msg.pose.position.x,
                 box_msg.pose.position.y,
                 box_msg.pose.position.z)
        q = (box_msg.pose.orientation.w,
             box_msg.pose.orientation.x,
             box_msg.pose.orientation.y,
             box_msg.pose.orientation.z)
        box = BoundingBox(
            pos=trans, rot=q,
            dimensions=(box_msg.dimensions.x,
                        box_msg.dimensions.y,
                        box_msg.dimensions.z))
        return box





def random_points_within(poly, num_points, box_polygons=None):
    box_polygons = box_polygons or []
    try:
        min_x, min_y, max_x, max_y = poly.bounds
    except ValueError:
        return []

    points = []

    while len(points) < num_points:
        random_point = shapely.geometry.Point(
            [np.random.uniform(min_x, max_x),
             np.random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            valid = True
            for box_polygon in box_polygons:
                if random_point.within(box_polygon):
                    valid = False
                    break
            if valid:
                points.append(random_point)

    return points


def points_within(poly, box_polygons=None):
    box_polygons = box_polygons or []
    try:
        min_x, min_y, max_x, max_y = poly.bounds
    except ValueError:
        return []
    points = []

    for y in np.arange(min_y, max_y, 0.05):
        for x in np.arange(min_x, max_x, 0.05):
            point = shapely.geometry.Point([x, y])
            if not point.within(poly):
                continue
            valid = True
            for box_polygon in box_polygons:
                if point.within(box_polygon):
                    valid = False
                    break
            if valid:
                points.append(point)

    return points


class PlacementFinder(ConnectionBasedTransport):

    def __init__(self):
        self.plot = rospy.get_param('~plot', False)
        super(PlacementFinder, self).__init__()
        self.poses_pub = self.advertise(
            '~output/poses',
            geometry_msgs.msg.PoseArray,
            queue_size=1)
        self.i = 0

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_polygon = message_filters.Subscriber(
            '~input/polygons',
            jsk_recognition_msgs.msg.PolygonArray,
            queue_size=1)
        sub_coefficients = message_filters.Subscriber(
            '~input/coefficients',
            jsk_recognition_msgs.msg.ModelCoefficientsArray,
            queue_size=1)

        if rospy.get_param('~with_boxes', False):
            sub_boxes = message_filters.Subscriber(
                '~input/boxes',
                jsk_recognition_msgs.msg.BoundingBoxArray,
                queue_size=1)
            self.subs = [sub_polygon, sub_coefficients, sub_boxes]
            if rospy.get_param('~approximate_sync', False):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    fs=self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    fs=self.subs, queue_size=queue_size)
            sync.registerCallback(self._cb_with_boxes)
        else:
            self.subs = [sub_polygon, sub_coefficients]
            if rospy.get_param('~approximate_sync', False):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    fs=self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    fs=self.subs, queue_size=queue_size)
            sync.registerCallback(self._cb)

    def unsubscribe(self):
        for s in self.subs:
            s.unregister()

    def _cb(self, polygons_msg, coeffs_msg):
        print(polygons_msg)
        print(coeffs_msg)
        for coeff, polygon in zip(
                coeffs_msg.coefficients,
                polygons_msg.polygons):
            a, b, c, d = coeff.values
            points = np.array(
                [[point.x, point.y, point.z]
                 for point in polygon.polygon.points],
                dtype=np.float32)
            normal = [a, b, c]
            projected_points = rotate_points(
                points,
                normal,
                [0, 0, 1])
            shapely_polygon = shapely.geometry.Polygon(
                projected_points)
            x, y = shapely_polygon.exterior.xy
            x, y = shapely_polygon.buffer(0.05).exterior.xy
            x, y = shapely_polygon.buffer(-0.05).exterior.xy
            points = random_points_within(shapely_polygon.buffer(-0.05), 10)

    def _cb_with_boxes(self, polygons_msg, coeffs_msg, boxes_msg):
        if polygons_msg.header.frame_id != boxes_msg.header.frame_id:
            raise ValueError
        plane_id_to_boxes = defaultdict(list)
        for box in boxes_msg.boxes:
            plane_id_to_boxes[box.label].append(box)

        pose_array_msg = geometry_msgs.msg.PoseArray()
        pose_array_msg.header = polygons_msg.header
        for index, (coeff, polygon) in enumerate(zip(
                coeffs_msg.coefficients,
                polygons_msg.polygons)):
            a, b, c, d = coeff.values
            points = np.array(
                [[point.x, point.y, point.z]
                 for point in polygon.polygon.points],
                dtype=np.float32)

            # for j in range(len(points)):
            #     for k in range(j + 1, len(points)):
            #         for l in range(k + 1, len(points)):
            #             nnn = np.cross(points[j] - points[l], points[k] - points[l])
            #             nnn = nnn / np.linalg.norm(nnn)
            #             ddd = np.dot(nnn, points[l])

            normal = np.array([a, b, c])
            projected_points = rotate_points(
                points,
                normal,
                [0, 0, 1])
            shapely_polygon = shapely.geometry.Polygon(
                projected_points)
            boxes = plane_id_to_boxes[index]

            if self.plot:
                plt.clf()
                x, y = shapely_polygon.exterior.xy
                plt.plot(x, y)
                x, y = shapely_polygon.buffer(0.05).exterior.xy
                plt.plot(x, y)
                x, y = shapely_polygon.buffer(-0.05).exterior.xy
                plt.plot(x, y)

            shapely_box_polygons = []
            for box in boxes:
                box = BoundingBox.from_ros_message(box)
                projected_vertices = rotate_points(
                    box.vertices,
                    normal,
                    [0, 0, 1])
                box_polygon = shapely.geometry.MultiPoint(
                    projected_vertices).convex_hull

                shapely_box_polygons.append(box_polygon.buffer(0.05))

                x, y = box_polygon.exterior.xy
                if self.plot:
                    plt.plot(x, y)

            points = points_within(shapely_polygon.buffer(-0.05),
                                   box_polygons=shapely_box_polygons)

            orientation_coords = skrobot.coordinates.Coordinates()
            skrobot.coordinates.geo.orient_coords_to_axis(
                orientation_coords, - normal)
            q_wxyz = orientation_coords.quaternion
            for p in points:
                pose_msg = geometry_msgs.msg.Pose()

                if self.plot:
                    plt.scatter(p.x, p.y, color='r')

                x, y, z = rotate_points(
                    np.array([p.x, p.y, -d]),
                    [0, 0, 1],
                    normal)[0]
                pose_msg.position.x = x
                pose_msg.position.y = y
                pose_msg.position.z = z
                pose_msg.orientation.w = q_wxyz[0]
                pose_msg.orientation.x = q_wxyz[1]
                pose_msg.orientation.y = q_wxyz[2]
                pose_msg.orientation.z = q_wxyz[3]
                pose_array_msg.poses.append(pose_msg)

            if self.plot:
                plt.savefig('/tmp/hoge-{0:08d}.png'.
                            format(self.i))
                self.i += 1
        self.poses_pub.publish(pose_array_msg)

if __name__ == '__main__':
    rospy.init_node('placement_finder')
    act = PlacementFinder()
    rospy.spin()
