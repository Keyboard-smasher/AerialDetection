from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from ultralytics import YOLO


class YOLOInference:
    def __init__(self, model_path, classes, imgsize=(640, 640)):
        self.model = YOLO(model_path)
        self.classes = classes
        self.imgsize = imgsize

    def predict(self, image_path, conf=0.5):
        results = self.model(image_path, conf=conf)
        return results[0].boxes

    def visualize(self, image_path, results, output_path="result.jpg"):
        img = Image.open(image_path)
        prev_size = img.size
        img = np.array(img.resize(self.imgsize))

        for box in results:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = self.classes[cls_id]
            conf = box.conf[0]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        img = cv2.resize(img, prev_size)

        plt.imshow(img)
        plt.show()


def inference(cfg):
    # Example usage
    classes = ["bicycle", "bus", "car", "motorcycle", "person", "truck"]
    print(Path(cfg.inference.inference.data_path).exists())

    infer = YOLOInference(cfg.inference.inference.model_path, classes)
    results = infer.predict(cfg.inference.inference.data_path)
    print(results)
    infer.visualize(cfg.inference.inference.data_path, results)
