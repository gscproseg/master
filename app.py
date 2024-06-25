

import streamlit as st

# Configuração da página
st.set_page_config(
    page_title= "MyxoNet",
    page_icon= "🧠",  # Defina o ícone da página como um emoji de tubarão
    layout="wide",  # Defina o layout como "wide" para aproveitar melhor o espaço na tela
    initial_sidebar_state="collapsed"  # Defina a barra lateral como colapsada
)

# Criação das guias
tab1, tab2, tab3, tab4 = st.tabs(["Home", "MixoNet", "USB", "Informações"])

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


# Conteúdo da página "USB"
with tab3:

    import streamlit as st
    import cv2
    import numpy as np
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    
    # Classe para fazer as previsões do YOLOv5
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
                text = f"{self.classes[class_ids[i]]}: {confidences[i]:.2f}"
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return image
    
    # Definindo o processador de vídeo para Streamlit WebRTC
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.yolo = YOLO_Pred('./best.onnx', './data.yaml')
    
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            pred_img = self.yolo.predictions(img)
            return av.VideoFrame.from_ndarray(pred_img, format="bgr24")
    
    # Função principal do Streamlit
    def main():
        st.title("YOLOv5 Real-Time Object Detection with Webcam")
    
        webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )
    
    if __name__ == "__main__":
        main()


    


#####################################################################################
