from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os import listdir
from os.path import join
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf
import pickle

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def load_gt_pose(fname):
    gt_smpl_dir = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/gt_smpl"
    with open(join(gt_smpl_dir, fname), 'rb') as f:
        data = pickle.load(f)
        gt_pose = data['pose']
        gt_pose_no_glob_rot = gt_pose[3:]

    return gt_pose_no_glob_rot


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
    folder_path = "/Users/Akash_Sengupta/Documents/4th_year_project_datasets/up-s31/s31_padded_small_glob_rot/val_masks/val"

    images = [f for f in sorted(listdir(folder_path)) if f.endswith('.png')]
    print('Predicting on all png images in folder.')
    squared_errors = []
    for image in images:
        print('Image:', image)
        img_path = join(folder_path, image)
        input_img, proc_param, img = preprocess_image(img_path, json_path)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)

        fnumber = image[:5]
        gt_smpl_fname = fnumber + "_body.pkl"
        gt_pose_no_glob_rot = load_gt_pose(gt_smpl_fname)
        pred_pose_no_glob_rot = theta[0, 6:75]

        error = np.square(gt_pose_no_glob_rot - pred_pose_no_glob_rot)
        squared_errors.append(error)

    squared_errors = np.concatenate(squared_errors)
    print(squared_errors.shape)
    print("MSE pose params", np.mean(squared_errors))



if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)