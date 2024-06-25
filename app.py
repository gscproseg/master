import streamlit as st
from PIL import Image
import numpy as np
from yolov5_predictions import YOLO_Pred  # Importe sua classe YOLO_Pred corretamente

# Configuração da página
st.set_page_config(
    page_title="πFINDER",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Criação das guias
tab1, tab2, tab3 = st.tabs(["Home", "🖼️- image", "📸-Cameras"])

# Conteúdo da página "Home"
with tab1:
    st.subheader("| A Classe Myxozoa")
    st.image("./images/sera.png", width=638)
    st.caption("""Courtesy W.L. Current
                   Myxobolus/Myxosoma sp.
                   """, unsafe_allow_html=True)
    st.text("")

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

# Conteúdo da página "MyxoDetect"
with tab2:
    st.write('Por favor, carregue a imagem para obter a identificação')

    def upload_image():
        # Upload Image
        image_file = st.file_uploader(label='Enviar Imagem')
        if image_file is not None:
            size_mb = image_file.size / (1024 ** 2)
            file_details = {"filename": image_file.name,
                            "filetype": image_file.type,
                            "filesize": "{:,.2f} MB".format(size_mb)}
            st.json(file_details)
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
                    with st.spinner("Obtendo Objetos de imagem. Aguarde"):
                        # Converta o objeto de imagem para uma matriz numpy
                        image_array = np.array(image_obj)
                        yolo = YOLO_Pred(onnx_model='./best.onnx', data_yaml='./data.yaml')
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

# Conteúdo da página "MyxoCam"
with tab3:
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    import av
    import asyncio

    # Define yolocam como uma variável global
    yolocam = None

    class YOLOVideoProcessor(VideoProcessorBase):
        async def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            global yolocam  # Acessar a variável global yolocam
            try:
                img_cam = frame.to_ndarray(format="bgr24")
                pred_img_video = yolocam.predictions(img_cam)
                return av.VideoFrame.from_ndarray(pred_img_video, format="bgr24")
            except Exception as e:
                print(f"Error processing frame: {e}")
                return frame  # Retornar o frame original em caso de erro

    async def start_webrtc_stream():
        global yolocam  # Acessar a variável global yolocam
        # Carregar o modelo YOLO dentro da função
        yolocam = YOLO_Pred(onnx_model='./best.onnx', data_yaml='./data.yaml')

        # Configurar e iniciar o stream WebRTC de forma assíncrona
        webrtc_ctx = webrtc_streamer(
            key="example",
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": 640,
                    "height": 640
                },
                "audio": False
            },
        )
        if webrtc_ctx.state == "running":
            print("Video streaming with object detection is active.")
        else:
            print("Waiting for the video stream to start...")

    def main():
        asyncio.run(start_webrtc_stream())

    if __name__ == "__main__":
        main()

pass
