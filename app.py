

import streamlit as st

# Configuração da página
st.set_page_config(
    page_title= "MyxoNet",
    page_icon= "🧠",  # Defina o ícone da página como um emoji de tubarão
    layout="wide",  # Defina o layout como "wide" para aproveitar melhor o espaço na tela
    initial_sidebar_state="collapsed"  # Defina a barra lateral como colapsada
)

# Criação das guias
tab1, tab2, tab3, tab4 = st.tabs(["Home", "MixoNet", "VIDEO", "USB"])

# Conteúdo da página "Home"
with tab1:
    st.subheader("| A Classe Myxozoa")
    # Use uma única coluna para posicionar a imagem e o texto na mesma linha
    col1, col2 = st.columns([1,0.85])  # Defina a largura da primeira coluna

    with col1:
        # Adicione a imagem ao espaço em branco
        st.image("./images/sera.png", width=638)
        # Adicione a legenda da imagem
        st.caption("""Courtesy W.L. Current
                   Myxobolus/Myxosoma sp.
                   """, unsafe_allow_html=True)  
        # Adicione um espaçamento para criar espaço entre a imagem e o texto
        st.text("")  # Ajuste o espaço conforme necessário

    with col2:
        # Ajuste a largura da coluna 2 (texto)
        st.markdown(""*20)  # Isso cria um espaço em branco para ajustar a largura
        intro_text = """
        Os myxozoários são parasitas com ciclos de vida complexos, pertencentes ao filo Cnidaria, como águas-vivas e medusas.
        Com mais de 65 gêneros e 2.200 espécies, a maioria parasita peixes, causando doenças graves e alta mortalidade.
        Myxobolus é o gênero mais conhecido, especialmente a espécie Myxobolus cerebralis, responsável pela "Doença do rodopio"
        em salmonídeos e danos à aquicultura e populações de peixes selvagens. Outros gêneros notáveis são Henneguya, Kudoa
        e Ellipsomyxa. Alguns myxozoários já foram relatados em humanos, causando surtos após o consumo de peixe cru infectado
        no Japão. O ciclo de vida envolve hospedeiros intermediários (peixes) e definitivos (anelídeos). Apesar da importância
        zoonótica, esses parasitas não são inspecionados no pescado brasileiro, ao contrário dos Estados Unidos. A abordagem 
        da Saúde Única promove a saúde sustentável de pessoas, animais e ecossistemas, reconhecendo sua interdependência e
        envolvendo vários setores para enfrentar ameaças à saúde, ecossistemas, segurança alimentar e mudanças climáticas,
        contribuindo para o desenvolvimento sustentável.
        """
        #st.markdown(intro_text)
        
        st.write(f'<p style="color:#9c9d9f">{intro_text}</p>', unsafe_allow_html=True)
        audio_file = open("images/p_9841290_826.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mpeg")

        st.subheader("| Seu ciclo de vida")
        st.write(
            '<p style="color:#9c9d9f">Seu ciclo de vida é indireto, envolvendo hospedeiros intermediários (peixes) e definitivos (anelídeos)</p>',
            unsafe_allow_html=True,
            )
        st.subheader("| Saúde Única")
        st.write(
            '<p style="color:#9c9d9f">A abordagem da Saúde Única promove uma visão integrada e multissetorial da saúde, envolvendo humanos, animais e ecossistemas. Reconhece a interdependência desses elementos e mobiliza diversos setores e disciplinas para promover o bem-estar e lidar com ameaças à saúde e aos ecossistemas, incluindo água limpa, segurança alimentar, mudanças climáticas e desenvolvimento sustentável.</p>',
            unsafe_allow_html=True,
            )

# Adicione as informações adicionais
st.write("Desenvolvido por [Carneiro, G.S](http://lattes.cnpq.br/3771047626259544)")

#######################################################

with tab2:

    st.header("MyxoNet")
    
    from yolo_predictions import YOLO_Pred
    from PIL import Image
    import numpy as np

    st.write('Por favor, carregue a imagem para obter a identificação')

    with st.spinner('Por favor, aguarde enquanto analisamos a sua imagem'):
        yolo = YOLO_Pred(onnx_model='./best.onnx',
                        data_yaml='./data.yaml')
        #st.balloons()

    def upload_image():
        # Upload Image
        image_file = st.file_uploader(label='Enviar Imagem')
        if image_file is not None:
            size_mb = image_file.size / (1024 ** 2)
            file_details = {"filename": image_file.name,
                            "filetype": image_file.type,
                            "filesize": "{:,.2f} MB".format(size_mb)}
            #st.json(file_details)
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
        object = upload_image()

        if object:
            prediction = False
            image_obj = Image.open(object['file'])

            col1, col2 = st.columns(2)#

            with col1:
                st.info('Pré-visualização da imagem')
                st.image(image_obj)#

            with col2:
                st.subheader('Confira abaixo os detalhes do arquivo')
                st.json(object['details'])
                button = st.button('Descubra qual o Myxozoário pode estar presente em sua imagem')
                if button:
                    with st.spinner("""
                    Obtendo Objets de imagem. Aguarde
                    """):
                        # below command will convert
                        # obj to array
                        image_array = np.array(image_obj)
                        pred_img = yolo.predictions(image_array)
                        pred_img_obj = Image.fromarray(pred_img)
                        prediction = True

            if prediction:
                st.subheader("Imagem com a possivel detecção")
                st.caption("Detecção de Myxozoários")
                st.image(pred_img_obj)

    if __name__ == "__main__":
         main()

pass



# Conteúdo da página "Video"
import time  # Importe o módulo time

with tab3:
    st.header("Detecção em Vídeo")

    # Função para detecção em vídeo
    def detect_video(upload_file):
        # Importações necessárias
        import cv2
        import tempfile
        from pathlib import Path

        # Carrega o modelo YOLO e outros recursos
        yolo = YOLO_Pred(onnx_model='./best.onnx', data_yaml='./data.yaml')

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
    
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
    import av
    import cv2
    import numpy as np
    import streamlit as st
    
    # Classe para carregar e utilizar o modelo de detecção
    class ObjectDetectionModel:
        def __init__(self, model_path, config_path):
            self.model_path = model_path
            self.config_path = config_path
            self.model = cv2.dnn.readNetFromONNX(self.model_path)
    
        def detect_objects(self, frame):
            img = frame.to_ndarray(format="bgr24")
    
            # Preprocessamento da imagem, se necessário
            img_resized = cv2.resize(img, (640, 640))
    
            # Detecção de objetos
            blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (640, 640), swapRB=True, crop=False)
            self.model.setInput(blob)
            outputs = self.model.forward()
    
            # Processa as saídas para exibir as detecções na imagem
            for output in outputs:
                for detection in output:
                    confidence = detection[4]
                    if confidence > 0.5:  # Confiança mínima para exibir a detecção
                        x, y, w, h = detection[:4] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
            return img
    
    # Inicialização do modelo de detecção de objetos
    object_detector = ObjectDetectionModel(model_path='./model.onnx', config_path='./config.yaml')
    
    # Função de callback para processar cada quadro de vídeo recebido
    def video_frame_callback(frame):
        # Realiza a detecção de objetos usando o modelo
        processed_frame = object_detector.detect_objects(frame)
    
        # Retorna o quadro processado para exibição
        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
    
    # Configuração do WebRTC Streamer com streamlit-webrtc
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False}
    )
    
    # Exibe uma mensagem indicando que o streamer está ativo
    if webrtc_ctx.video_transformer:
        st.write("WebRTC streamer is running...")




