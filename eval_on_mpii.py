from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os import listdir
from os.path import join, splitext, isdir
from absl import flags

import numpy as np
import skimage.io as io
import tensorflow as tf
import cv2
import scipy.io as spio

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


def preprocess_image(img, json_path=None):
    """
    Crops and rescales image - this function was given (my own bb crop code is separate)
    """
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


def render_bb_joints_verts(bb_img, proc_param, joints, verts, cam, image_path, person_num,
                           visualise=False, save=False):
    """
    Renders the result in original image coordinate frame FOR EACH BOUNDING BOX.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=bb_img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(bb_img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=bb_img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=bb_img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=bb_img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=bb_img.shape[:2])

    if visualise or save:
        import matplotlib.pyplot as plt
        # plt.ion()
        plt.figure(1)
        plt.clf()
        plt.subplot(231)
        plt.imshow(bb_img)
        plt.title('input')
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(skel_img)
        plt.title('joint projection')
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(rend_img_overlay)
        plt.title('3D Mesh overlay')
        plt.axis('off')
        plt.subplot(234)
        plt.imshow(rend_img)
        plt.title('3D mesh')
        plt.axis('off')
        plt.subplot(235)
        plt.imshow(rend_img_vp1)
        plt.title('diff vp')
        plt.axis('off')
        plt.subplot(236)
        plt.imshow(rend_img_vp2)
        plt.title('diff vp')
        plt.axis('off')
        plt.draw()
        if visualise:
            plt.show()
        if save:
            plt.savefig(splitext(image_path)[0] + "_hmr_result_person" + str(person_num) + ".png",
                    format='png')

    return skel_img, rend_img_overlay


def recreate_scene_from_bbs(orig_img, bbs, skel_bb_imgs, rend_bb_img_overlays, image_path):
    """
    Piece together original image from bb, with joints and verts overlayed on top of each bb.
    :param orig_img:
    :param bbs:
    :param skel_bb_imgs:
    :param rend_bb_img_overlays:
    :param image_path:
    :return:
    """
    skel_img = np.copy(orig_img)
    rend_img_overlay = np.dstack((orig_img, 255*np.ones(orig_img.shape[:2])))
    rend_img_overlay = rend_img_overlay.astype(int)

    for i in range(len(bbs)):
        y1, x1, y2, x2 = bbs[i]
        skel_img[y1:y2, x1:x2] = skel_bb_imgs[i]
        rend_img_overlay[y1:y2, x1:x2] = rend_bb_img_overlays[i]

    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(skel_img)
    plt.axis('off')
    plt.savefig(splitext(image_path)[0] + "_hmr_result_joints.png",
                format='png')
    plt.clf()
    plt.imshow(rend_img_overlay)
    plt.axis('off')
    plt.savefig(splitext(image_path)[0] + "_hmr_result_render_overlay.png",
                format='png')


import scipy
import numpy as np


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def main(mpii_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    images_path = mpii_path + "/trial_images/"
    annotations_path = mpii_path + "/annotations/mpii_human_pose_v1_u12_1.mat"

    images = sorted([f for f in listdir(images_path) if f.endswith('.jpg')])
    # print('Predicting on all jpg images in folder.')

    annotations = loadmat(annotations_path)
    annotations = annotations['RELEASE']

    annotated_images_dict = {}
    joints_predicts_dict = {}
    for i in range(len(annotations['annolist'])):
        if annotations['img_train'][i] == 1:
            name = annotations['annolist'][i]['image']['name']
            if name in images:
                annotated_images_dict[name] = annotations['annolist'][i]['annorect']

    print(annotated_images_dict)


    #TODO Complete this code to test 2D joint locations on MPII dataset
    for image in images:
        print('Image:', image)
        img_path = join(images_path, image)
        orig_image = cv2.imread(img_path)
        # Change to RGB (cv2 loads BGR)
        orig_image = np.fliplr(orig_image.reshape(-1, 3)).reshape(orig_image.shape)

        # Crop out bounding boxes
        bb_file = images_path + splitext(image)[0] + "_bb_coords.pkl"
        bbs = pickle.load(open(bb_file, 'rb'))

        # Predict joints and vertices for each bounding box.

        for bb in bbs:
            y1, x1, y2, x2 = bb
            bb_image = orig_image[y1:y2, x1:x2, :]

            input_crop_img, proc_param, bb_img = preprocess_image(bb_image, json_path)
            # Add batch dimension: 1 x D x D x 3
            input_crop_img = np.expand_dims(input_crop_img, 0)

            joints, verts, cams, joints3d, theta = model.predict(
                input_crop_img, get_theta=True)

            joints = joints[0]
            cam = cams[0]
            verts = verts[0]

            cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
                proc_param, verts, cam, joints, img_size=bb_img.shape[:2])

            # cv2.imshow('window', orig_image)
            # cv2.waitKey(0)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
    main(config.img_path, config.json_path)
