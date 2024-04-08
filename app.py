import streamlit as st

# Configuração da página
    
st.set_page_config(
    page_title= "πFINDER",
    page_icon= "🔬",  # Defina o ícone da página como um emoji de tubarão
    layout="wide",  # Defina o layout como "wide" para aproveitar melhor o espaço na tela
    initial_sidebar_state="collapsed"  # Defina a barra lateral como colapsada
)

# Criação das guias
tab1, tab2, tab3 = st.tabs(["Home", "🖼️- image", "📸-Cameras"])

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
st.write("Desenvolvido por [Carneiro, G.S]( http://lattes.cnpq.br/3771047626259544) em colaboração com o com o LIM²T-Ufra")

pass
#######################################################

# Conteúdo da página "MyxoDetect"
with tab2:

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
                st.error('Upload only png, jpg, jpeg')
                return None

    def main():
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


#################################
with tab3:

    import streamlit as st 
    from streamlit_webrtc import (
        VideoTransformerBase,
        RTCConfiguration,
        webrtc_streamer,
    )
    import av
    from yolo_predictions import YOLO_Pred
    
    # Carregue o modelo YOLO
    yolo = YOLO_Pred('./models/best.onnx', './models/data.yaml')
    
    # Definir configuração RTC (WebRTC)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            pred_img = yolo.predictions(img)
            return av.VideoFrame.from_ndarray(pred_img, format="bgr24")
    
    # Configurar e iniciar a transmissão WebRTC
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=YOLOVideoTransformer,
        rtc_configuration=rtc_configuration,
        async_transform=True,
        media_stream_constraints={"video": True, "audio": False},
    )
    
    # Exibir a interface do Streamlit
    if webrtc_ctx.video_transformer:
        st.write("Streaming de vídeo com detecção de objetos está ativo.")
    else:
        st.write("Aguardando a transmissão de vídeo começar...")












    
