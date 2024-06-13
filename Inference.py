import os
import argparse
import numpy as np
import scipy.io as sio
import scipy.misc
import tensorflow as tf

from utils import *
from model_zalando_mask_content import create_generator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_dir", type=str, default="data/pose/", help="Directory containing poses.")
    parser.add_argument("--segment_dir", type=str, default="data/segment/", help="Directory containing human segmentations.")
    parser.add_argument("--image_dir", type=str, default="data/women_top/", help="Directory containing product and person images.")
    parser.add_argument("--test_label", type=str, default="data/viton_test_pairs.txt", help="File containing labels for testing.")
    parser.add_argument("--result_dir", type=str, default="results/", help="Folder containing the results of testing.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--begin", type=int, default=0, help="Start index for testing.")
    parser.add_argument("--end", type=int, default=2032, help="End index for testing.")
    return parser.parse_args()

def _process_image(image_name, product_image_name, resize_width=192, resize_height=256):
    image_id = image_name[:-4]
    image = scipy.misc.imread(FLAGS.image_dir + image_name)
    prod_image = scipy.misc.imread(FLAGS.image_dir + product_image_name)
    segment_raw = sio.loadmat(os.path.join(FLAGS.segment_dir, image_id))["segment"]
    segment_raw = process_segment_map(segment_raw, image.shape[0], image.shape[1])
    pose_raw = sio.loadmat(os.path.join(FLAGS.pose_dir, image_id))
    pose_raw = extract_pose_keypoints(pose_raw)
    pose_raw = extract_pose_map(pose_raw, image.shape[0], image.shape[1])
    pose_raw = np.asarray(pose_raw, np.float32)

    body_segment, prod_segment, skin_segment = extract_segmentation(segment_raw)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    prod_image = tf.image.convert_image_dtype(prod_image, dtype=tf.float32)

    image = tf.image.resize(image, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR)
    prod_image = tf.image.resize(prod_image, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR)
    body_segment = tf.image.resize(body_segment, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    skin_segment = tf.image.resize(skin_segment, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    prod_segment = tf.image.resize(prod_segment, size=[resize_height, resize_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = (image - 0.5) * 2.0
    prod_image = (prod_image - 0.5) * 2.0

    skin_segment = skin_segment * image

    return image, prod_image, pose_raw, body_segment, prod_segment, skin_segment

def main():
    args = parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "images/"), exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, "tps/"), exist_ok=True)

    batch_size = 1
    prod_image_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
    body_segment_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 1])
    skin_segment_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 3])
    pose_map_holder = tf.placeholder(tf.float32, shape=[batch_size, 256, 192, 18])

    outputs = create_generator(prod_image_holder, body_segment_holder, skin_segment_holder, pose_map_holder, 4)

    images = np.zeros((batch_size, 256, 192, 3))
    prod_images = np.zeros((batch_size, 256, 192, 3))
    body_segments = np.zeros((batch_size, 256, 192, 1))
    skin_segments = np.zeros((batch_size, 256, 192, 3))
    pose_raws = np.zeros((batch_size, 256, 192, 18))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        if checkpoint is None:
            checkpoint = args.checkpoint
        print(checkpoint)
        step = int(checkpoint.split('-')[-1])

        saver.restore(sess, checkpoint)

        test_info = open(args.test_label).read().splitlines()
        for i in range(args.begin, args.end, batch_size):
            image_names = []
            product_image_names = []

            for j in range(i, i + batch_size):
                info = test_info[j].split()
                print(info)
                image_name = info[0]
                product_image_name = info[1]
                image_names.append(image_name)
                product_image_names.append(product_image_name)
                (image, prod_image, pose_raw, body_segment, prod_segment, skin_segment) = _process_image(image_name, product_image_name)

                images[j-i] = image
                prod_images[j-i] = prod_image
                body_segments[j-i] = body_segment
                skin_segments[j-i] = skin_segment
                pose_raws[j-i] = pose_raw

            feed_dict = {
                prod_image_holder: prod_images,
                body_segment_holder: body_segments,
                skin_segment_holder: skin_segments,
                pose_map_holder: pose_raws,
            }

            [image_and_mask_output] = sess.run([outputs], feed_dict=feed_dict)

            mask_output = image_and_mask_output[:, :, :, :1]
            image_output = image_and_mask_output[:, :, :, 1:]

            for j in range(batch_size):
                scipy.misc.imsave(os.path.join(args.result_dir, "images/%08d_" % step + image_names[j] + "_" + product_image_names[j] + '.png'), (image_output[j] / 2.0 + 0.5))
                scipy.misc.imsave(os.path.join(args.result_dir, "images/%08d_" % step + image_names[j] + "_" + product_image_names[j] + '_mask.png'), np.squeeze(mask_output[j]))
                scipy.misc.imsave(os.path.join(args.result_dir, "images/" + image_names[j]), (images[j] / 2.0 + 0.5))
                scipy.misc.imsave(os.path.join(args.result_dir, "images/" + product_image_names[j]), (prod_images[j] / 2.0 + 0.5))
                sio.savemat(os.path.join(args.result_dir, "tps/%08d_" % step + image_names[j] + "_" + product_image_names[j] + "_mask.mat"), {"mask": np.squeeze(mask_output[j])})

            index_path = os.path.join(args.result_dir, "index.html")
            if os.path.exists(index_path):
                index = open(index_path, "a")
            else:
                index = open(index_path, "w")
                index.write("<html><body><table><tr>")
                index.write("<th>step</th>")
                index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")
            for j in range(batch_size):
                index.write("<tr>")
                index.write("<td>%d %d</td>" % (step, i + j))
                index.write("<td>%s %s</td>" % (image_names[j], product_image_names[j]))
                index.write("<td><img src='images/%s'></td>" % image_names[j])
                index.write("<td><img src='images/%s'></td>" % product_image_names[j])
                index.write("<td><img src='images/%08d_%s'></td>" % (step, image_names[j] + "_" + product_image_names[j] + '.png'))
                index.write("<td><img src='images/%08d_%s'></td>" % (step, image_names[j] + "_" + product_image_names[j] + '_mask.png'))
                index.write("</tr>")

if __name__ == "__main__":
    main()
