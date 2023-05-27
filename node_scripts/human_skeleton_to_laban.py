#!/usr/bin/env python
# -*- coding:utf-8 -*-

from jsk_recognition_msgs.msg import HumanSkeletonArray
import jsk_robocup_common_msgs.msg
from jsk_topic_tools import ConnectionBasedTransport
from labanotation.labanotation_utils import calculate_labanotation
import numpy as np
import rospy


class HumanSkeletonToLaban(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.pose_pub = self.advertise(
            '~output/skeleton',
            jsk_robocup_common_msgs.msg.HumanSkeletonArray, queue_size=1)

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 1)
        self.sub_skeleton = rospy.Subscriber(
            '~input/skeleton',
            HumanSkeletonArray,
            callback=self._cb,
            queue_size=queue_size, buff_size=2**24)

    def unsubscribe(self):
        self.sub_skeleton.unregister()
        self.sub_skeleton = None

    def _cb(self, skeleton_msg):
        skeleton_with_laban_msg = jsk_robocup_common_msgs.msg.HumanSkeletonArray(
            header=skeleton_msg.header,
            human_ids=skeleton_msg.human_ids)
        for skeleton in skeleton_msg.skeletons:
            limb_to_pose = {}
            out_skeleton_msg = jsk_robocup_common_msgs.msg.HumanSkeleton(
                header=skeleton.header,
                bone_names=skeleton.bone_names,
                bones=skeleton.bones)
            for bone_name, bone in zip(skeleton.bone_names,
                                       skeleton.bones):
                a, b = bone_name.split('->')
                limb_to_pose[a] = np.array([bone.start_point.x, bone.start_point.y, bone.start_point.z])
                limb_to_pose[b] = np.array([bone.end_point.x, bone.end_point.y, bone.end_point.z])

            if (('left_hip' in limb_to_pose and 'right_hip' in limb_to_pose)
                or ('left_waist' in limb_to_pose and 'right_waist' in limb_to_pose)) \
               and 'left_shoulder' in limb_to_pose and 'right_shoulder' in limb_to_pose \
               and 'left_elbow' in limb_to_pose and 'right_elbow' in limb_to_pose \
               and 'left_wrist' in limb_to_pose and 'right_wrist' in limb_to_pose:
                if 'left_hip' in limb_to_pose:
                    spine_position = (limb_to_pose['left_hip'] + limb_to_pose['right_hip']) / 2.0
                else:
                    spine_position = (limb_to_pose['left_waist'] + limb_to_pose['right_waist']) / 2.0
                _, _, _, _, elbow_r, elbow_l, wrist_r, wrist_l = calculate_labanotation(spine_position,
                                       limb_to_pose['right_shoulder'],
                                       limb_to_pose['left_shoulder'],
                                       limb_to_pose['right_elbow'],
                                       limb_to_pose['left_elbow'],
                                       limb_to_pose['right_wrist'],
                                       limb_to_pose['left_wrist'])
                out_skeleton_msg.right_elbow_laban = ' '.join(elbow_r)
                out_skeleton_msg.right_wrist_laban = ' '.join(wrist_r)
                out_skeleton_msg.left_elbow_laban = ' '.join(elbow_l)
                out_skeleton_msg.left_wrist_laban = ' '.join(wrist_l)
            skeleton_with_laban_msg.skeletons.append(out_skeleton_msg)
        self.pose_pub.publish(skeleton_with_laban_msg)


if __name__ == '__main__':
    rospy.init_node('human_skeleton_to_laban')
    HumanSkeletonToLaban()
    rospy.spin()
