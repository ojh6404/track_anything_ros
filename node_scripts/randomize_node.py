#!/usr/bin/env python3

import rospy
import cv2
import cv_bridge
import numpy as np
import message_filters

from sensor_msgs.msg import Image, Joy


class MaskRandomizeNode(object):
    def __init__(self):
        # self.hz = rospy.get_param("~hz", 10)
        self.bridge = cv_bridge.CvBridge()

        h_deg = 180  # 色相(Hue)の回転度数
        # self.h_deg = np.random.uniform(low=-h_deg, high=h_deg)
        # self.h_deg = np.random.uniform(low=0, high=h_deg)
        self.h_deg = np.random.uniform(low=0, high=h_deg, size=2)
        self.s_mag = np.random.uniform(low=0.8, high=1.2)
        self.v_mag = np.random.uniform(low=0.8, high=1.2)
        rospy.loginfo(
            "h_deg: {}, s_mag: {}, v_mag: {}".format(self.h_deg, self.s_mag, self.v_mag)
        )

        self.topic_list = [
            "/track_anything/processed_image_color",
            "/track_anything/segmentation_mask",
        ]
        rospy.loginfo("subscribing to {}".format(self.topic_list))
        # synced subscribe
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [
                message_filters.Subscriber(
                    topic,
                    Image,
                    queue_size=1000000,
                    buff_size=2**24,
                )
                for topic in self.topic_list
            ],
            100,
            0.2,
            #     rospy.get_param("~mf_queue_size"),
            #     rospy.get_param("~mf_slop"),
        )
        self.ts.registerCallback(self.mf_callback)

        self.pub_randomized_img = rospy.Publisher(
            "/track_anything/randomized_image",
            Image,
            queue_size=1,
        )

        self.sub_joy = rospy.Subscriber(
            "/controller_LHR_F7AFBF47/joy",
            Joy,
            self.joy_callback,
            queue_size=1,
        )

    def joy_callback(self, msg):
        if msg.buttons[3] == 1:
            self.h_deg = np.random.uniform(low=0, high=180, size=2)
            self.s_mag = np.random.uniform(low=0.8, high=1.2)
            self.v_mag = np.random.uniform(low=0.8, high=1.2)
            rospy.loginfo(
                "changed to h_deg: {}, s_mag: {}, v_mag: {}".format(
                    self.h_deg, self.s_mag, self.v_mag
                )
            )

    def decompose_mask(self, mask_image):
        decomposed_masks = []
        mask_list = np.unique(mask_image)  # 2 objects -> [0,1,2], 0 is bg
        for i in range(1, len(mask_list)):  # 1,2
            decomposed_mask = np.where(mask_image == i, 1, 0)
            decomposed_masks.append(decomposed_mask)
        return decomposed_masks

    def mf_callback(self, *msgs):
        img_msg = msgs[0]
        mask_msg = msgs[1]

        original_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        mask_image = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="32SC1")

        randomized_image = np.zeros_like(original_image)
        randomized_images = []
        extracted_image = original_image.copy()

        for mask_idx in range(1, len(np.unique(mask_image))):
            randomized_image = cv2.cvtColor(original_image.copy(), cv2.COLOR_RGB2HSV)
            # randomized_image[:, :, 0] = np.clip(
            #     randomized_image[:, :, 0] + self.h_deg[mask_idx - 1], 0, 180
            # )
            randomized_image[:, :, 0] = self.h_deg[mask_idx - 1]
            randomized_image[:, :, 1] = np.clip(
                randomized_image[:, :, 1] * self.s_mag, 0, 255
            )
            randomized_image[:, :, 2] = np.clip(
                randomized_image[:, :, 2] * self.v_mag, 0, 255
            )
            randomized_image = cv2.cvtColor(randomized_image, cv2.COLOR_HSV2RGB)
            randomized_images.append(randomized_image)

        for mask_idx in range(1, len(np.unique(mask_image))):
            for channel_idx in range(original_image.shape[2]):
                extracted_image[:, :, channel_idx] = np.where(
                    mask_image == mask_idx,
                    randomized_images[mask_idx - 1][:, :, channel_idx],
                    extracted_image[:, :, channel_idx],
                )

        randomized_img_msg = self.bridge.cv2_to_imgmsg(extracted_image, encoding="rgb8")
        randomized_img_msg.header = img_msg.header
        self.pub_randomized_img.publish(randomized_img_msg)

        # extracted_image = np.zeros_like(original_image)
        # for i in range(original_image.shape[2]):
        #     extracted_image[:, :, i] = np.where(
        #         mask_image > 0, original_image[:, :, i], 0
        #     )
        # extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_RGB2HSV)

        # extracted_image[:, :, 0] = extracted_image[:, :, 0] + self.h_deg  # 色相の計算
        # extracted_image[:, :, 1] = extracted_image[:, :, 1] * self.s_mag  # 彩度の計算
        # extracted_image[:, :, 2] = extracted_image[:, :, 2] * self.v_mag  # 明度の計算
        # extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_HSV2RGB)

        # for i in range(original_image.shape[2]):
        #     extracted_image[:, :, i] = np.where(
        #         mask_image > 0, extracted_image[:, :, i], original_image[:, :, i]
        #     )

        # debug_img_msg = self.bridge.cv2_to_imgmsg(extracted_image, encoding="rgb8")
        # debug_img_msg.header = img_msg.header
        # self.pub_debug_img.publish(debug_img_msg)


if __name__ == "__main__":
    rospy.init_node("randomize_node")
    node = MaskRandomizeNode()
    rospy.spin()
