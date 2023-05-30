#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import rospy
import rospkg
import numpy as np

# from informatized_body_msgs.msg import Float32MultiArrayStamped
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import torch
import torchvision
from vc_models.models.vit import model_utils

import time

# from image_utils import ImageUtils
# from image_autoencoder import ImageAutoEncoder

# from informatized_body_utils.image_utils import ImageUtils
# from informatized_body_utils.image_autoencoder_torch import ImageAutoEncoder


class ImageLatentPublisher:
    def __init__(self):
        rospy.init_node("image_latent_publisher")

        self.bridge = CvBridge()

        (
            self.model,
            self.embd_size,
            self.model_transforms,
            self.model_info,
        ) = model_utils.load_model(model_utils.VC1_BASE_NAME)

        self.model.to("cuda:0")

        self.img_transform = torchvision.transforms.ToTensor()

        self.latent_pub = rospy.Publisher("image_latent", Float32, queue_size=1)
        rospy.Subscriber(
            "/zed/zed_node/rgb/image_rect_color",
            Image,
            self.image_callback,
            queue_size=1000,
            buff_size=2**24,
        )

    def calc_z(self, image_data):
        img = self.img_transform(image_data)
        img = img.unsqueeze(0)
        transformed_img = self.model_transforms(img).to("cuda:0")
        print(transformed_img.shape)
        with torch.no_grad():
            z = self.model(transformed_img)
        print(z.device)
        # z = self.model(transformed_img)
        return z

    def image_callback(self, msg):
        # self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image_data = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        print("test")
        print(msg.encoding)
        # if ("r" in msg.encoding) and ("g" in msg.encoding) and ("b" in msg.encoding):
        #     image_data = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        # else:
        #     image_data = self.bridge.imgmsg_to_cv2(msg, "mono8")[:, :, np.newaxis]

        z = self.calc_z(image_data)
        print(z.shape)

        latent_msg = Float32()
        latent_msg.data = 2.5
        self.latent_pub.publish(latent_msg)


def main():
    print("# initialize... <= {}".format(__file__.split("/")[-1]))
    imageLatentPublisher = ImageLatentPublisher()
    print("# start... <= {}".format(__file__.split("/")[-1]))
    rospy.spin()


if __name__ == "__main__":
    main()
