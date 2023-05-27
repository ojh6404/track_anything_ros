#!/usr/bin/env python

import copy

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from jsk_recognition_msgs.msg import HumanSkeleton
from jsk_recognition_msgs.msg import HumanSkeletonArray
from jsk_recognition_msgs.msg import Segment
from jsk_topic_tools import ConnectionBasedTransport
import numpy as np
import onnxruntime
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image


def draw_human_skeletons(image, results, connections):
    debug_image = copy.deepcopy(image)

    for result in results:
        keypoints_iter = iter(result['keypoints'])

        keypoint_list = []
        for cx, cy, _ in zip(keypoints_iter, keypoints_iter, keypoints_iter):
            cx = int(cx)
            cy = int(cy)
            keypoint_list.append([cx, cy])

        for keypoint in keypoint_list:
            cx = keypoint[0]
            cy = keypoint[1]
            if cx > 0 and cy > 0:
                cv2.circle(debug_image, (cx, cy), 3, (0, 255, 0),
                           -1, lineType=cv2.LINE_AA,)

        for connect in connections:
            cx1 = keypoint_list[connect[0]][0]
            cy1 = keypoint_list[connect[0]][1]
            cx2 = keypoint_list[connect[1]][0]
            cy2 = keypoint_list[connect[1]][1]
            if cx1 > 0 and cy1 > 0 and cx2 > 0 and cy2 > 0:
                cv2.line(debug_image, (cx1, cy1), (cx2, cy2),
                         (0, 255, 0), 2, lineType=cv2.LINE_AA,)
    return debug_image


def run_inference(onnx_session, input_name, input_size, image, score_th=0.5):
    # ONNX Infomation
    input_width = input_size[3]
    input_height = input_size[2]
    image_height, image_width = image.shape[0], image.shape[1]

    # Pre process:Resize, BGR->RGB, float32 cast
    input_image = cv2.resize(image, dsize=(input_width, input_height))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')

    # Inference
    results = onnx_session.run(None, {input_name: input_image})

    # Post process
    kpt, pv = results
    pv = np.reshape(pv[0], [-1])
    kpt = kpt[0][pv >= score_th]
    kpt[:, :, -1] *= image_height
    kpt[:, :, -2] *= image_width
    kpt[:, :, -3] *= 2
    skeletons = []
    for human, score in zip(kpt, pv):
        mask = np.stack(
            [(human[:, 0] >= score_th).astype(np.float32)],
            axis=-1,
        )
        human *= mask
        human = np.stack([human[:, _ii] for _ii in [1, 2, 0]], axis=-1)
        skeletons.append({
            'keypoints': np.reshape(human, [-1]).tolist(),
            'category_id': 1,
            'score': score.tolist(),
        })

    return skeletons



class PeoplePoseEstimation(ConnectionBasedTransport):

    def __init__(self):
        super(PeoplePoseEstimation, self).__init__()

        self.names = [
            'nose',
            'left_eye',
            'right_eye',
            'left_ear',
            'right_ear',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
            'left_waist',
            'right_waist',
            'left_knee',
            'right_knee',
            'left_ankle',
            'right_ankle',
        ]

        self.connections = [
            [0, 1],  # 00:鼻(nose) -> 01:左目(left eye)
            [0, 2],  # 00:鼻(nose) -> 02:右目(right eye)
            [1, 3],  # 01:左目(left eye) -> 03:左耳(left ear)
            [2, 4],  # 02:右目(right eye) -> 04:右耳(right ear)
            [3, 5],  # 03:左耳(left ear) -> 05:左肩(left shoulder)
            [4, 6],  # 04:右耳(right ear) -> 06:右肩(right shoulder)
            [5, 6],  # 05:左肩(left shoulder) -> 06:右肩(right shoulder)
            [5, 7],  # 05:左肩(left shoulder)  -> 07:左肘(left elbow)
            [7, 9],  # 07:左肘(left elbow) -> 09:左手首(left wrist)
            [6, 8],  # 06:右肩(right shoulder) -> 08:右肘(right elbow)
            [8, 10],  # 08:右肘(right elbow) -> 10:右手首(right wrist)
            [5, 11],  # 05:左肩(left shoulder) -> 11:左腰(left waist)
            [6, 12],  # 06:右肩(right shoulder) -> 12:右腰(right waist)
            [11, 12],  # 11:左腰(left waist) -> 12:右腰(right waist)
            [11, 13],  # 11:左腰(left waist) -> 13:左膝(left knee)
            [13, 15],  # 13:左膝(left knee) -> 15:左足首(left ankle)
            [12, 14],  # 12:右腰(right waist) -> 14:右膝(right knee),
            [14, 16],  # 14:右膝(right knee) -> 16:右足首(right ankle)
        ]


        providers = [
            'CPUExecutionProvider',
        ]
        model_path = rospy.get_param('~model_path')
        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.input_size = self.onnx_session.get_inputs()[0].shape
        self.score_th = 0.4

        self.bridge = CvBridge()
        self.pub_img = self.advertise(
            '~output/viz', Image, queue_size=1)
        self.pub_img_compressed = self.advertise(
            '~output/viz/compressed',
            CompressedImage, queue_size=1)
        self.skeleton_pub = self.advertise(
            '~output/skeleton', HumanSkeletonArray, queue_size=1)

    def subscribe(self):
        self.sub = rospy.Subscriber(
            '~input',
            Image, self.callback,
            queue_size=1, buff_size=2**24)

    def unsubscribe(self):
        self.sub.unregister()

    def callback(self, img_msg):
        bridge = self.bridge
        image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

        # Inference execution
        skeletons = run_inference(
            self.onnx_session,
            self.input_name,
            self.input_size,
            image,
            score_th=self.score_th,
        )

        skeleton_msgs = HumanSkeletonArray(header=img_msg.header)
        pose_list = []
        for skeleton in skeletons:
            keypoints_iter = iter(skeleton['keypoints'])
            keypoint_list = []
            pose = {}
            for idx, (cx, cy, _) in enumerate(zip(keypoints_iter, keypoints_iter, keypoints_iter)):
                cx = int(cx)
                cy = int(cy)
                keypoint_list.append([cx, cy])
                pose[self.names[idx]] = np.array([cx, cy, 0.0])
            pose_list.append(pose)

        for pose in pose_list:
            skeleton_msg = HumanSkeleton(header=img_msg.header)
            for a, b in self.connections:
                a = self.names[a]
                b = self.names[b]
                if not (a in pose and b in pose):
                    continue
                bone_name = '{}->{}'.format(a, b)
                bone = Segment(
                    start_point=Point(*pose[a]),
                    end_point=Point(*pose[b]))
                skeleton_msg.bones.append(bone)
                skeleton_msg.bone_names.append(bone_name)
            skeleton_msgs.skeletons.append(skeleton_msg)
        self.skeleton_pub.publish(skeleton_msgs)

        if self.pub_img.get_num_connections() > 0 or self.pub_img_compressed.get_num_connections() > 0:
            image = draw_human_skeletons(
                image,
                skeletons,
                self.connections)

        if self.pub_img.get_num_connections() > 0:
            out_img_msg = bridge.cv2_to_imgmsg(
                image, encoding='bgr8')
            out_img_msg.header = img_msg.header
            self.pub_img.publish(out_img_msg)

        if self.pub_img_compressed.get_num_connections() > 0:
            # publish compressed http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber  # NOQA
            vis_compressed_msg = CompressedImage()
            vis_compressed_msg.header = img_msg.header
            # image format https://github.com/ros-perception/image_transport_plugins/blob/f0afd122ed9a66ff3362dc7937e6d465e3c3ccf7/compressed_image_transport/src/compressed_publisher.cpp#L116  # NOQA
            vis_compressed_msg.format = 'bgr8' + '; jpeg compressed bgr8'
            vis_compressed_msg.data = np.array(
                cv2.imencode('.jpg', image)[1]).tobytes()
            self.pub_img_compressed.publish(vis_compressed_msg)


if __name__ == '__main__':
    rospy.init_node('people_pose_estimation')
    node = PeoplePoseEstimation()
    rospy.spin()
