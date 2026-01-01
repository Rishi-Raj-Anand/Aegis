import cv2
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os
import time

class FaceRedactor:
    def __init__(self):
        """Initialize the MTCNN detector once to save overhead."""
        print("Initializing MTCNN Detector...")
        self.detector = MTCNN()

    def load_image(self, image_path):
        """Loads an image and converts it to RGB for MTCNN/Matplotlib."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        return img

    def get_detections(self, image):
        """Performs face detection and returns bounding boxes."""
        # MTCNN needs RGB
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_img)
        return [d['box'] for d in detections]

    def plot_preview(self, image, boxes):
        """Displays the image with ID labels using Matplotlib."""
        display_img = image.copy()
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"ID: {i}"
            cv2.putText(display_img, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        plt.figure(figsize=(12, 8))
        plt.imshow(display_img)
        plt.title(f"Detected {len(boxes)} faces")
        plt.axis('off')
        plt.show()

    def apply_redaction(self, image, boxes, indices_to_blur):
        """Applies Gaussian blur to selected bounding boxes."""
        final_img = image.copy()
        for idx in indices_to_blur:
            if 0 <= idx < len(boxes):
                x, y, w, h = boxes[idx]
                x, y = max(0, x), max(0, y)
                
                roi = final_img[y:y+h, x:x+w]
                if roi.size > 0:
                    # Adaptive kernel size: roughly half the face width, must be odd
                    ksize = (int(w//2) | 1, int(h//2) | 1)
                    ksize = (max(31, ksize[0]), max(31, ksize[1]))
                    
                    blurred_roi = cv2.GaussianBlur(roi, ksize, 30)
                    final_img[y:y+h, x:x+w] = blurred_roi
            else:
                print(f"Warning: ID {idx} is out of range.")
        return final_img



    