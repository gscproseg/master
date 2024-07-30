

import streamlit as st

# Configuração da página
st.set_page_config(
    page_title= "MLens",
    page_icon= "🧠",  # Defina o ícone da página como um emoji de tubarão
    layout="wide",  # Defina o layout como "wide" para aproveitar melhor o espaço na tela
    initial_sidebar_state="collapsed"  # Defina a barra lateral como colapsada
)

# Criação das guias
tab1, tab2, tab3, tab4, tab5 = st.tabs(["MLens", "Detecção em Imagens", "Detecção em Vídeo", "Tempo Real - USB", "Quem somos"])

# Conteúdo da página "Home"
with tab1:

    st.subheader("Bem-vindo ao MLens")
    # Adicionar a imagem ao espaço em branco
    st.image("MLens.png", use_column_width=True)  #  caminho da sua imagem

# Adicione as informações adicionais
st.write("Desenvolvido por LIMT-Ufra e [Carneiro, G.S](http://lattes.cnpq.br/3771047626259544)")

#######################################################

with tab2:

    import streamlit as st
    from yolov5_predictions import YOLO_Pred
    from PIL import Image
    import numpy as np
    import cv2

    def upload_image():
        # Upload Image
        image_file = st.file_uploader(label='Enviar Imagem')
        if image_file is not None:
            size_mb = image_file.size / (1024 ** 2)
            file_details = {"filename": image_file.name,
                            "filetype": image_file.type,
                            "filesize": "{:,.2f} MB".format(size_mb)}
            # validate file
            if file_details['filetype'] in ('image/png', 'image/jpeg'):
                st.success('Tipo de arquivo imagem VALIDO (png ou jpeg)')
                return {"file": image_file,
                        "details": file_details}
            else:
                st.error('Tipo de arquivo de imagem INVALIDO')
                st.error('Envie apenas arquivos nos formatos png, jpg e jpeg')
                return None

    def main():
        st.header("Detecção em Imagens")
        st.write('Por favor, carregue a imagem para obter a identificação')

        with st.spinner('Por favor, aguarde enquanto analisamos a sua imagem'):
            yolo = YOLO_Pred(onnx_model='./bestL.onnx',
                            data_yaml='./data.yaml')

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
                        # Converta a imagem para array
                        image_array = np.array(image_obj)

                        # Verifica se a imagem está em escala de cinza ou RGB
                        if len(image_array.shape) == 2:
                            # Imagem em escala de cinza
                            input_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                        else:
                            # Imagem em RGB
                            input_image = image_array

                        # Faça a predição
                        pred_img = yolo.predictions(input_image)
                        pred_img_obj = Image.fromarray(pred_img)
                        prediction = True

            if prediction:
                st.subheader("Imagem com a possível detecção")
                st.caption("Detecção de Myxozoários")
                st.image(pred_img_obj)

    if __name__ == "__main__":
        main()


with tab3:

    st.header("Detecção em Vídeo")
    st.write('Por favor, utilize videos curtos para obter as identificações')

    # Função para detecção em vídeo
    def detect_video(upload_file):
        # Importações necessárias
        import cv2
        import tempfile
        from pathlib import Path

        # Carrega o modelo YOLO e outros recursos
        yolo = YOLO_Pred(onnx_model='./bestL.onnx', data_yaml='./data.yaml')

        # Salva o vídeo temporariamente
        temp_video_path = Path(tempfile.NamedTemporaryFile().name)
        with open(temp_video_path, 'wb') as temp_file:
            temp_file.write(upload_file.read())

        # Inicia a captura de vídeo
        video_capture = cv2.VideoCapture(str(temp_video_path))

        start_time = time.time()  # Marca o tempo de início da detecção

        while True:
            ret, frame = video_capture.read()

            # Verifica se a captura de vídeo foi bem-sucedida
            if not ret:
                break

            # Converte o frame para o formato esperado pela função de detecção
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Realiza a detecção de objetos
            pred_frame = yolo.predictions(frame_rgb)

            # Exibe o frame com as detecções no Streamlit
            st.image(pred_frame, channels='RGB', use_column_width=True)

            # Limita o tempo de execução do loop (por exemplo, 30 segundos)
            if time.time() - start_time > 30:
                break

        # Libera a captura de vídeo
        video_capture.release()

    # Upload de arquivo de vídeo
    uploaded_file = st.file_uploader(label='Enviar Vídeo', type=['mp4', 'avi'])

    # Botão para iniciar a detecção em vídeo
    if uploaded_file is not None:
        if st.button('Iniciar Detecção em Vídeo'):
            detect_video(uploaded_file)

pass

#########################################################################################
with tab4:
    
    st.header("Detecção em Tempo Real")
    st.write('Por favor, selecione o equipamente de imageamento para obter as identificações')

    from streamlit_webrtc import webrtc_streamer
    import av
    from yolo_predictions import YOLO_Pred

    # load yolo model
    yolo = YOLO_Pred('./bestL.onnx',
                    './data.yaml')


    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        # any operation 
        #flipped = img[::-1,:,:]
        pred_img = yolo.predictions(img)

        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


    webrtc_streamer(key="example", 
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video":True,"audio":False})

pass



with tab5:

    st.subheader("Use o Qr para saber mais sobre nossos integrates")
    # Adicionar a imagem ao espaço em branco
    st.image("qr.png", use_column_width=True)  #  caminho da sua imagem
