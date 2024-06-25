import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

class YOLO_Pred:
    def __init__(self, model_path, data_path):
        self.net = cv2.dnn.readNet(model_path)
        self.classes = self._load_classes(data_path)

    def _load_classes(self, data_path):
        with open(data_path, 'r') as f:
            return f.read().strip().split("\n")

    def predictions(self, image):
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
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
    output_video_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pred_frame = yolo.predictions(frame)
        if out is None:
            h, w = pred_frame.shape[:2]
            out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (w, h))
        out.write(pred_frame)

    cap.release()
    out.release()
    return output_video_path

def main():
    st.title("YOLOv5 Object Detection")

    # Load YOLO model
    yolo = YOLO_Pred('./best.onnx', './data.yaml')

    tab1, tab2, tab3, tab4 = st.tabs(["Detect from Image", "Detect from Video", "USB", "Info"])

    # Tab for image detection
    with tab1:
        st.header("Image")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
        if uploaded_image:
            image = Image.open(uploaded_image)
            image_array = np.array(image)
            pred_img = yolo.predictions(image_array)
            st.image(pred_img, caption='Processed Image', use_column_width=True)

    # Tab for video detection
    with tab2:
        st.header("Video")
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False) as temp_video:
                temp_video.write(uploaded_video.read())
                temp_video_path = temp_video.name

            st.video(uploaded_video)
            if st.button('Process Video', key="process_video_btn"):
                output_video_path = process_video(temp_video_path, yolo)
                st.success("Video processed successfully!")
                st.video(output_video_path)

    # Conteúdo da página "USB"
    with tab3:
        st.header("USB")
        st.subheader("Video from USB or Webcam")
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="usb_video_uploader")
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False) as temp_video:
                temp_video.write(uploaded_video.read())
                temp_video_path = temp_video.name

            st.video(uploaded_video)
            if st.button('Process Video from USB', key="process_usb_video_btn"):
                output_video_path = process_video(temp_video_path, yolo)
                st.success("Video processed successfully!")
                st.video(output_video_path)

    # Tab for additional info
    with tab4:
        st.subheader("| A Classe Myxozoa")

if __name__ == "__main__":
    main()
