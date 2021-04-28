import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import time
import glob
import cv2
import requests

from io import StringIO
from PIL import Image

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils import visualization_utils as vis_util
from utils import label_map_util

from multiprocessing.dummy import Pool as ThreadPool
def process(path):
    
    MAX_NUMBER_OF_BOXES = 10
    MINIMUM_CONFIDENCE = 0.9

    PATH_TO_LABELS = 'annotations/label_map.pbtxt'
    PATH_TO_TEST_IMAGES_DIR = path

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=10, use_display_name=True)
    print("categories=",categories)
    CATEGORY_INDEX = label_map_util.create_category_index(categories)
    print("categories index=",CATEGORY_INDEX)

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    MODEL_NAME = 'output_inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def detect_objects(image_path):
        image = Image.open('test_images/'+image_path)
        g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

        img = cv2.imread('test_images/'+image_path)
        cv2.imshow('Original Image', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

        
        cv2.imshow('Filter Image', filtered_img)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            CATEGORY_INDEX,
            min_score_thresh=MINIMUM_CONFIDENCE,
            use_normalized_coordinates=True,
            line_thickness=8)
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.imshow(image_np, aspect = 'auto')
        plt.savefig('output//{}'.format(image_path), dpi = 62)
        plt.close(fig)
        img = cv2.imread('output1//{}'.format(image_path))
        cv2.imshow('Output Image', img)
        k = cv2.waitKey(0)
        msg="Garbage Detected  amount of garbage is"+str(num)
        print("Msg",msg)
        Adminmobilenumber="7975945667"
        print("mobile number",Adminmobilenumber)
        url = "https://www.fast2sms.com/dev/bulk"
        payload = "sender_id=FSTSMS&message="+msg+"&language=english&route=p&numbers="+Adminmobilenumber
        headers = {'authorization':"DUZVeSgEtRMh92b1sikHmrT6GAP7xY8CLBuNldFacX30nwQOqov7N82AtJE0dGBzROrVeI1XnTZHcQ5w",'Content-Type': "application/x-www-form-urlencoded",'Cache-Control': "no-cache",}
        response = requests.request("POST", url, data=payload, headers=headers)
        print(response.text)

        


    # TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image-{}.jpg'.format(i)) for i in range(1, 4) ]
    print(PATH_TO_TEST_IMAGES_DIR)
    TEST_IMAGE_PATHS = os.listdir('test_images')


    print(TEST_IMAGE_PATHS)
    # Load model into memory
    print('Loading model...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    from tqdm import tqdm
    print('detecting...')
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            for image_path in tqdm(TEST_IMAGE_PATHS):

                detect_objects(image_path)
