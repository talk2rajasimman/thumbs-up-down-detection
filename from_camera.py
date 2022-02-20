import cv2
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append("..")

def GetClassName(data):
    for cl in data:
        return cl['name']

# What model to download.
MODEL_NAME = 'model'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    '', 'labelmap.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
skip = 1

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

frame = 0

thumbs_up_frame_count = 0
thumbs_down_frame_count = 0
no_hand_detected = 0

tu_count = 0
td_count = 0

tu_text = "Thumbs up"
td_text = "Thumbs Down"

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            
            frame = frame+1
            
            ret, image_np = cap.read()
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(image_np,(x,y),(x+w,y+h),(255,255,0),2) 
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image_np[y:y+h, x:x+w]
            
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            
            min_score_thresh = 0.60
            bboxes = boxes[scores > min_score_thresh]
            im_width, im_height = image_np.shape[1::-1]
            
            max_boxes_to_draw = 20
            
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=5)
            
            # The following code replaces the 'print ([category_index...' statement
            objects = []
            detection_class_name = "N/A"
            
            data = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.9]

            class_name = GetClassName(data)
            
            if class_name == "thumbs_up":
                thumbs_up_frame_count = thumbs_up_frame_count + 1
            elif class_name == "thumbs_down":
                thumbs_down_frame_count = thumbs_down_frame_count + 1
            elif class_name == 'None':
                no_hand_detected = no_hand_detected + 1
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            count = 1
            
            if frame == 31:
                frame = 0
                
                if (thumbs_up_frame_count > thumbs_down_frame_count) and (thumbs_up_frame_count > no_hand_detected):
                    tu_count = tu_count + 1
                    tu_text = "Thumbs Up -", str(tu_count)
                elif (thumbs_down_frame_count > thumbs_up_frame_count) and (thumbs_down_frame_count > no_hand_detected):
                    td_count = td_count + 1
                    td_text = "Thumbs Down -", str(td_count)
                elif (no_hand_detected > thumbs_up_frame_count) and (no_hand_detected > thumbs_down_frame_count):
                    print("No hand detected")
                                    
                thumbs_up_frame_count = 0
                thumbs_down_frame_count = 0
                no_hand_detected = 0
                
            cv2.putText(image_np, str(tu_text), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(image_np, str(td_text), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            
        cap.release()
