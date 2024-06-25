import streamlit as st
import cv2
import numpy as np
import tempfile

class YOLO_Pred:
    def __init__(self, model_path, config_path, classes_path):
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.classes = []
        with open(classes_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def predictions(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outputs = self.net.forward(output_layers)
        return self._draw_predictions(image, outputs)

    def _draw_predictions(self, image, outputs):
        height, width = image.shape[:2]
        boxes, confidences, class_ids = [], [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
        for i in indices:
            i = i[0]
            box = boxes[i]
            (x, y, w, h) = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if class_ids[i] < len(self.classes):  # Check if class_id is within bounds
                text = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

def process_video(video_path, yolo):
    cap = cv2.VideoCapture(video_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_output.name, fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pred_frame = yolo.predictions(frame)
        out.write(pred_frame)

    cap.release()
    out.release()

    return temp_output.name

def main():
    st.header("USB")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            temp_video.write(uploaded_video.read())
            output_video_path = process_video(temp_video.name, yolo)
            st.video(output_video_path)

if __name__ == "__main__":
    yolo = YOLO_Pred('./best.onnx', './best.yaml', './classes.txt')
    main()
