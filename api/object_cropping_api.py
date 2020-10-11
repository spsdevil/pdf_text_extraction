import tensorflow as tf
import cv2
import os
import glob
from PIL import Image
import numpy as np
from utils import visualization_utils as vis_util


def object_cropping(detection_graph, category_index):

        # Grab path to current working directory
        CWD_PATH = os.getcwd()

        # Path to image
        IMAGE_NAME = 'pdf_pages/*'
        PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

        # Output cropped
        CROP_FOLDER = 'crop'
        PATH_TO_CROP = os.path.join(CWD_PATH, CROP_FOLDER)

        folder = glob.glob(PATH_TO_IMAGE)
        for data in folder:
            # print(data)
            
            with detection_graph.as_default():
              with tf.Session(graph=detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Load image using OpenCV and
                # expand image dimensions to have shape: [1, None, None, 3]
                # i.e. a single-column array, where each item in the column has the pixel RGB value
                image = cv2.imread(data)
                image_expanded = np.expand_dims(image, axis=0)

                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_expanded})


                # Draw the results of the detection (aka 'visulaize the results')

                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=0.80)

                # Crop the Detected Part from image..

                (frame_height, frame_width) = image.shape[:2]
                count = 1
                for i in range(len(np.squeeze(scores))):
                            
                    # print(np.squeeze(boxes)[i])
                    # print(np.squeeze(scores)[i])
                    if np.squeeze(scores)[i] > 0.8:
                        ymin = int((np.squeeze(boxes)[i][0]*frame_height))
                        xmin = int((np.squeeze(boxes)[i][1]*frame_width))
                        ymax = int((np.squeeze(boxes)[i][2]*frame_height))
                        xmax = int((np.squeeze(boxes)[i][3]*frame_width))
                        img = Image.open(data)
                        x = img.crop((xmin, ymin, xmax, ymax))
                        x.save(PATH_TO_CROP + "/crop_{}.jpg".format(count))
                        count +=1
                    else:
                        print("Cropping Done..")
                        break