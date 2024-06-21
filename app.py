import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
from ultralytics import YOLO
import cv2
# Load the YOLO model
model = YOLO("yolov9c-seg.pt")


def detect_objects(image):
    results = model(image)
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    classes = results[0].boxes.cls
    masks=results[0].masks.xy
    return boxes, scores, classes,masks

def draw_boxes(image, boxes, scores, classes, class_names):
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i]
        confidence = scores[i]
        class_id = int(classes[i])
        label = class_names[class_id]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), f'{label} {confidence:.2f}', fill="red")
    return image
def draw_segmentation_map(image, masks, labels):
    alpha = 1.0
    beta = 0.5  
    gamma = 0  
    image = np.array(image)  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    for mask, label in zip(masks, labels):
        color = (0, 0, 255)  # Green color for visualization
        segmentation_map = np.zeros_like(image)
 
        if mask is not None and len(mask) > 0:
            poly = np.array(mask, dtype=np.int32)
            cv2.fillPoly(segmentation_map, [poly], color)
 
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
    return image
st.title("Object Detection App")
st.write("Upload an image and detect objects in it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Analyse Image'):
        st.write("Detecting objects...")
        # Detect objects
        boxes, scores, classes,masks = detect_objects(np.array(image))

        # Convert tensors to numpy arrays for easier handling
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        #masks = masks.cpu().numpy()
        # Class names for COCO dataset
        class_names = model.names

        # Draw bounding boxes
        image_with_boxes = draw_boxes(image, boxes, scores, classes, class_names)
        image_with_boxes_seg=draw_segmentation_map(image_with_boxes,masks,classes)
        image_with_boxes_seg=cv2.cvtColor(image_with_boxes_seg, cv2.COLOR_RGB2BGR)
        st.image(image_with_boxes_seg, caption='Processed Image.', use_column_width=True)
        st.write("Detected objects:")
        for i in range(len(boxes)):
            st.write(f"{class_names[int(classes[i])]} with confidence {scores[i]:.2f}")
