from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from absl import flags
import numpy as np
import cv2
from tqdm import tqdm

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if np.max(img.shape[:2]) != config.img_size:
        # print('Resizing so the max image size is %d..' % config.img_size)
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def convert_bbox_centre_hw_to_corners(centre, height, width):
    x1 = centre[0] - height/2.0
    x2 = centre[0] + height/2.0
    y1 = centre[1] - width/2.0
    y2 = centre[1] + width/2.0

    return np.array([x1, y1, x2, y2])


def main(dataset_path, paired=True, path_correction=False):
    """
    This function isn't really doing evaluation on H3.6M - it just runs HMR on each H3.6M frame and stores the output.
    There is (or will be) a separate script in the pytorch_indirect_learning repo that will do the evaluation and metric
    computations.
    """
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    save_path = '/data/cvfs/as2562/hmr/evaluations/ssp3d'
    if not paired:
        save_path += '_unpaired'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    data = np.load(os.path.join(dataset_path, 'labels.npz'))
    fname_per_frame = data['fnames']
    bbox_centres = data['bbox_centres']
    bbox_whs = data['whs']

    pose_per_frame = []
    shape_per_frame = []
    verts_per_frame = []
    cam_per_frame = []

    for i in tqdm(range(len(fname_per_frame))):
        fpath = os.path.join(dataset_path, 'images', fname_per_frame[i])
        img = io.imread(fpath)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        # Crop inputs to bounding box
        # Annoyingly, for sports videos dataset, the frames in the cropped_frames dir are not
        # actually the cropped frames that are input into the network... Need to do further
        # cropping and resizing.
        bbox_centre = bbox_centres[i]
        bbox_wh = bbox_whs[i] * 1.2
        bbox_corners = convert_bbox_centre_hw_to_corners(bbox_centre, bbox_wh, bbox_wh)
        top_left = bbox_corners[:2].astype(np.int16)
        bottom_right = bbox_corners[2:].astype(np.int16)
        top_left[top_left < 0] = 0
        bottom_right[bottom_right < 0] = 0
        img = img[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1]]
        img = cv2.resize(img, (config.img_size, config.img_size), interpolation=cv2.INTER_LINEAR)
        input_img = 2 * ((img / 255.) - 0.5)
        # input_img, proc_param, img = preprocess_image(image_path)

        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        joints, verts, cams, joints3d, thetas = model.predict(input_img, get_theta=True)
        poses = thetas[:, 3:3+72]
        shapes = thetas[:, 3+72:]

        pose_per_frame.append(poses)
        shape_per_frame.append(shapes)
        verts_per_frame.append(verts)
        cam_per_frame.append(cams)

    fname_per_frame = np.array(fname_per_frame)
    np.save(os.path.join(save_path, 'fname_per_frame.npy'), fname_per_frame)

    pose_per_frame = np.concatenate(pose_per_frame, axis=0)
    np.save(os.path.join(save_path, 'pose_per_frame.npy'), pose_per_frame)

    shape_per_frame = np.concatenate(shape_per_frame, axis=0)
    np.save(os.path.join(save_path, 'shape_per_frame.npy'), shape_per_frame)

    verts_per_frame = np.concatenate(verts_per_frame, axis=0)
    np.save(os.path.join(save_path, 'verts_per_frame.npy'), verts_per_frame)

    cam_per_frame = np.concatenate(cam_per_frame, axis=0)
    np.save(os.path.join(save_path, 'cam_per_frame.npy'), cam_per_frame)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    if src.config.PRETRAINED_MODEL.endswith('model.ckpt-667589'):
        paired = True
    elif src.config.PRETRAINED_MODEL.endswith('model.ckpt-99046'):
        paired = False
        print('Using unpaired model!')

    config.batch_size = 1
    print('Images from', config.img_path)
    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
    path_correction = False
    main(dataset_path=config.img_path, paired=paired, path_correction=path_correction)
