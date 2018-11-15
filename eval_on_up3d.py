"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px.

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os import listdir
from os.path import join
from absl import flags
import numpy as np
import cv2

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(folder_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    images = sorted([f for f in listdir(folder_path) if f.endswith('image.png')])
    annotated_joints = sorted([f for f in listdir(folder_path) if f.endswith('.npy')])
    abs_errors = []
    num_joints = 0
    num_accepted = 0

    images_joints_dict = dict(zip(images, annotated_joints))
    print('Predicting on all png images in folder.')
    for image in images:
        print('Image:', image)
        img_path = join(folder_path, image)
        input_img, proc_param, img = preprocess_image(img_path, json_path)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)

        cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
            proc_param, verts[0], cams[0], joints[0], img_size=img.shape[:2])

        annotation_array = np.load(join(folder_path, images_joints_dict[image]))
        annotation_array = zip(annotation_array[0], annotation_array[1])
        annotation_array = (np.rint(annotation_array)).astype(int)
        joints_orig = (np.rint(joints_orig)).astype(int)
        joints_orig = joints_orig[:14]

        for i in range(len(annotation_array)):
            annotation = (annotation_array[i][0], annotation_array[i][1])
            prediction = (joints_orig[i][0], joints_orig[i][1])
            # cv2.circle(img, annotation, 2,
            #            (0, 255, 0), thickness=5)
            # cv2.circle(img, prediction, 2,
            #            (0, 0, 255), thickness=5)
            # cv2.imshow('win', img)
            # cv2.waitKey(0)
            error = np.linalg.norm(np.subtract(annotation, prediction))
            # print(error)
            abs_errors.append(error)
            if error < 30:
                num_accepted += 1
            num_joints += 1

    print("MAE", np.mean(abs_errors), 'Fraction Accepted', float(num_accepted)/num_joints)







if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)
