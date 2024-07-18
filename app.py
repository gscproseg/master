import streamlit as st
from yolov5_predictions import YOLO_Pred
from PIL import Image
import numpy as np
import cv2
import tempfile
from pathlib import Path
import time
from streamlit_webrtc import webrtc_streamer
import av

# Configuração da página
st.set_page_config(
    page_title="MLens",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Criação das guias
tab1, tab2, tab3, tab4, tab5 = st.tabs(["MLens", "Detecção em Imagens", "Detecção em Vídeo", "Tempo Real - USB", "Quem somos"])

# Conteúdo da página "Home"
with tab1:
    st.subheader("Bem-vindo ao MLens")
    st.image("MLens.png", use_column_width=True)

st.write("Desenvolvido por LIMT-Ufra e [Carneiro, G.S](http://lattes.cnpq.br/3771047626259544)")

# Função para upload de imagem
with tab2:

def upload_image():
    image_file = st.file_uploader(label='Enviar Imagem')
    if image_file is not None:
        size_mb = image_file.size / (1024 ** 2)
        file_details = {"filename": image_file.name, "filetype": image_file.type, "filesize": "{:,.2f} MB".format(size_mb)}
        if file_details['filetype'] in ('image/png', 'image/jpeg'):
            st.success('Tipo de arquivo imagem VALIDO (png ou jpeg)')
            return {"file": image_file, "details": file_details}
        else:
            st.error('Tipo de arquivo de imagem INVALIDO')
            st.error('Envie apenas arquivos nos formatos png, jpg e jpeg')
            return None

# Função principal para detecção em imagens
def main():
    st.header("Detecção em Imagens")
    st.write('Por favor, carregue a imagem para obter a identificação')

    with st.spinner('Por favor, aguarde enquanto analisamos a sua imagem'):
        yolo = YOLO_Pred(onnx_model='./best.onnx', data_yaml='./data.yaml')

    object = upload_image()

    if object:
        prediction = False
        image_obj = Image.open(object['file'])

        col1, col2 = st.columns(2)

        with col1:
            st.info('Pré-visualização da imagem')
            st.image(image_obj)

        with col2:
            st.subheader('Confira abaixo os detalhes do arquivo')
            st.json(object['details'])
            button = st.button('Detectar Myxozoário na Imagem')
            if button:
                with st.spinner("Obtendo as Detecções dos Myxozoários na imagem. Aguarde"):
                    image_array = np.array(image_obj)

                    if len(image_array.shape) == 2:
                        input_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                    else:
                        input_image = image_array

                    # Realiza a predição e obtém as contagens das classes
                    pred_img, class_counts = yolo.predictions(input_image)
                    pred_img_obj = Image.fromarray(pred_img)
                    prediction = True

        if prediction:
            st.subheader("Resultado da Detecção e Contagem")

            # Mostra a imagem com as detecções e contagens lado a lado
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Imagem com a Detecção")
                st.image(pred_img_obj, use_column_width=True)

            # Adiciona a contagem das classes na imagem detectada
            pred_img_counts = pred_img.copy()  # Copia a imagem para adicionar as contagens
            y_offset = 20
            for class_name, count in class_counts.items():
                text = f'{class_name}: {count}'
                cv2.putText(pred_img_counts, text, (20, y_offset), cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 1, (255, 255, 255), 1)
                y_offset += 30
            
            # Mostra a imagem com o texto das contagens
            with col2:
                st.subheader("Contagem das Classes Detectadas")
                st.image(pred_img_counts, channels='BGR', use_column_width=True)

if __name__ == "__main__":
    main()
    
with tab3:
    st.header("Detecção em Vídeo")
    st.write('Por favor, utilize vídeos curtos para obter as identificações')

    def detect_video(upload_file):
        yolo = YOLO_Pred(onnx_model='./best.onnx', data_yaml='./data.yaml')

        temp_video_path = Path(tempfile.NamedTemporaryFile().name)
        with open(temp_video_path, 'wb') as temp_file:
            temp_file.write(upload_file.read())

        video_capture = cv2.VideoCapture(str(temp_video_path))

        start_time = time.time()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_frame, class_counts = yolo.predictions(frame_rgb)

            # Adicionar contagem das classes detectadas no frame
            for class_name, count in class_counts.items():
                text = f'{class_name}: {count}'
                cv2.putText(pred_frame, text, (10, 30 + 30 * list(class_counts.keys()).index(class_name)),
                            cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 1, (255, 0, 0), 2, cv2.LINE_AA)

            st.image(pred_frame, channels='RGB', use_column_width=True)

            if time.time() - start_time > 30:
                break

        video_capture.release()

    uploaded_file = st.file_uploader(label='Enviar Vídeo', type=['mp4', 'avi'])
    if uploaded_file is not None:
        if st.button('Iniciar Detecção em Vídeo'):
            detect_video(uploaded_file)


with tab4:

    st.header("Detecção em Tempo Real")
    st.write('Por favor, selecione o equipamente de imageamento para obter as identificações')

    yolo = YOLO_Pred('./best.onnx', './data.yaml')

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")

        # Reduzir a resolução para 640x480 para processamento mais rápido
        img = cv2.resize(img, (640, 480))

        pred_img, class_counts = yolo.predictions(img)

        # Overlay class counts on the video frame
        for class_name, count in class_counts.items():
            text = f'{class_name}: {count}'
            cv2.putText(pred_img, text, (10, 30 + 30 * list(class_counts.keys()).index(class_name)),
                        cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

    webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False})



with tab5:
    st.subheader("Use o Qr para saber mais sobre nossos integrantes")
    st.image("qr.png", use_column_width=True)
