from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os import listdir
from os.path import join, splitext, isdir, basename
from absl import flags

import numpy as np
import skimage.io as io
import tensorflow as tf
import cv2

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
                           visualise=False, save=False, outfile=None):
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
            if outfile is None:
                plt.savefig(splitext(image_path)[0] + "_hmr_result_person" + str(person_num) + ".png",
                            format='png')
            else:
                plt.savefig(outfile + "_hmr_result_person" + str(person_num) + ".png",
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


def write_ply_file(fpath, verts, colour):
    ply_header = '''ply
                    format ascii 1.0
                    element vertex {}
                    property float x
                    property float y
                    property float z
                    property uchar red
                    property uchar green
                    property uchar blue
                    end_header
                   '''
    num_verts = verts.shape[0]
    color_array = np.tile(np.array(colour), (num_verts, 1))
    verts_with_colour = np.concatenate([verts, color_array], axis=-1)
    with open(fpath, 'w') as f:
        f.write(ply_header.format(num_verts))
        np.savetxt(f, verts_with_colour, '%f %f %f %d %d %d')


def main(input_path, json_path=None, crop_height_add=20):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    out_folder = "predictions/sports_videos/00001"
    if isdir(input_path):
        images = [f for f in sorted(listdir(input_path)) if f.endswith('.png')]
        print('Predicting on all png images in folder.')

        for image in images:
            print('Image:', image)
            img_path = join(input_path, image)
            orig_image = cv2.imread(img_path)
            # Change to RGB (cv2 loads BGR)
            orig_image = np.fliplr(orig_image.reshape(-1, 3)).reshape(orig_image.shape)

            # Crop out bounding boxes
            bb_file = input_path + splitext(image)[0] + "_bb_coords.pkl"
            bbs = pickle.load(open(bb_file, 'rb'))

            # Predict joints and vertices + render them for each bounding box.
            # Recreate scene from bounding box outputs.
            skel_bb_images = []
            rend_bb_image_overlays = []
            person_num = 1
            for bb in bbs:
                y1, x1, y2, x2 = bb
                y1 = min(0, y1 - 20)
                y2 = min(orig_image.shape[0], y2 + 20)
                bb_image = orig_image[y1:y2, x1:x2, :]

                input_crop_img, proc_param, bb_img = preprocess_image(bb_image, json_path)
                # Add batch dimension: 1 x D x D x 3
                input_crop_img = np.expand_dims(input_crop_img, 0)

                joints, verts, cams, joints3d, theta = model.predict(
                    input_crop_img, get_theta=True)

                # Saving joints in pkl file
                # joints_file_path = splitext(img_path)[0] + "_joints_coords.pkl"
                # with open(joints_file_path, 'wb') as outfile:
                #     pickle.dump(joints, outfile,
                #                 protocol=2)  # protocol=2 for python2 (HMR uses this)
                # print("Joints saved to ", joints_file_path)

                # Saving verts in pkl file
                # verts_file_path = splitext(img_path)[0] + "_verts_coords.pkl"
                # with open(verts_file_path, 'wb') as outfile:
                #     pickle.dump(verts, outfile,
                #                 protocol=2)  # protocol=2 for python2 (HMR uses this)
                # print("Verts saved to ", verts_file_path)

                outfile = join(out_folder, splitext(image)[0])
                print('Saving to:', outfile)
                # Saving verts in ply file
                write_ply_file(outfile + "_verts.ply", verts[0], [255, 0, 0])

                # Save joints and render plots
                skel_bb_image, rend_bb_image_overlay = render_bb_joints_verts(bb_img,
                                                                              proc_param,
                                                                              joints[0],
                                                                              verts[0],
                                                                              cams[0],
                                                                              img_path,
                                                                              person_num,
                                                                              save=True,
                                                                              outfile=outfile)

                skel_bb_images.append(skel_bb_image)
                rend_bb_image_overlays.append(rend_bb_image_overlay)
                person_num += 1

            # recreate_scene_from_bbs(orig_image, bbs, skel_bb_images, rend_bb_image_overlays,
            #                         img_path)

    else:
        orig_image = cv2.imread(input_path)
        # Change to RGB (cv2 loads BGR)
        orig_image = np.fliplr(orig_image.reshape(-1, 3)).reshape(orig_image.shape)
        # Crop out bounding boxes
        bb_file = splitext(input_path)[0] + "_bb_coords.pkl"
        bbs = pickle.load(open(bb_file, 'rb'))

        skel_bb_images = []
        rend_bb_image_overlays = []
        person_num = 1
        for bb in bbs:
            y1, x1, y2, x2 = bb
            y1 = min(0, y1-20)
            y2 = min(orig_image.shape[0], y2+20)
            bb_image = orig_image[y1:y2, x1:x2, :]

            input_crop_img, proc_param, bb_img = preprocess_image(bb_image, json_path)
            # Add batch dimension: 1 x D x D x 3
            input_crop_img = np.expand_dims(input_crop_img, 0)

            joints, verts, cams, joints3d, theta = model.predict(
                input_crop_img, get_theta=True)

            skel_bb_image, rend_bb_image_overlay = render_bb_joints_verts(bb_img,
                                                                          proc_param,
                                                                          joints[0],
                                                                          verts[0],
                                                                          cams[0],
                                                                          input_path,
                                                                          person_num)

            skel_bb_images.append(skel_bb_image)
            rend_bb_image_overlays.append(rend_bb_image_overlay)
            person_num += 1

        recreate_scene_from_bbs(orig_image, bbs, skel_bb_images, rend_bb_image_overlays,
                                input_path)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
    main(config.img_path, config.json_path)
