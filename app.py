import streamlit as st

# Configuração da página
    
st.set_page_config(
    page_title= "πFINDER",
    page_icon= "🔬",  # Defina o ícone da página como um emoji de tubarão
    layout="wide",  # Defina o layout como "wide" para aproveitar melhor o espaço na tela
    initial_sidebar_state="collapsed"  # Defina a barra lateral como colapsada
)

# Criação das guias
tab1, tab2, tab3 = st.tabs(["Home", "📸- image","📸- Cam"])

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

    import streamlit as st
    from yolo_predictions import YOLO_Pred
    from PIL import Image
    import numpy as np
    import requests
    from io import BytesIO
    
    st.title('Detecção de Myxozoários')
    
    # Carregar o modelo YOLO
    with st.spinner('Por favor, aguarde enquanto o modelo é carregado...'):
        yolo = YOLO_Pred(onnx_model='./best.onnx',
                         data_yaml='./data.yaml')
    
    def load_image_from_url(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image_url = Image.open(BytesIO(response.content))
                return image_url
            else:
                return None
        except Exception as e:
            st.error('Erro ao carregar a imagem via URL. Certifique-se de que o URL é válido.')
            return None
    
    def main():
        # Botão para carregar imagem via URL
        st.subheader('Carregar imagem via URL')
        url = st.text_input('Digite o URL da imagem:')
        button_url = st.button('Carregar Imagem via URL')
        if button_url:
            image_url = load_image_from_url(url)
            if image_url is not None:
                st.image(image_url, caption='Imagem carregada via URL', use_column_width=True)
                with st.spinner('Analisando a imagem...'):
                    image_array = np.array(image_url)
                    pred_img = yolo.predictions(image_array)
                    pred_img_obj = Image.fromarray(pred_img)
                    st.subheader('Imagem com a possível detecção de Myxozoários')
                    st.image(pred_img_obj, caption='Detecção de Myxozoários', use_column_width=True)
    
        # Botão para carregar imagem localmente
        st.subheader('Carregar imagem localmente')
        image_file = st.file_uploader('Selecione uma imagem')
        button_local = st.button('Carregar Imagem Localmente')
        if button_local and image_file is not None:
            image_obj = Image.open(image_file)
            st.image(image_obj, caption='Imagem carregada localmente', use_column_width=True)
            with st.spinner('Analisando a imagem...'):
                image_array = np.array(image_obj)
                pred_img = yolo.predictions(image_array)
                pred_img_obj = Image.fromarray(pred_img)
                st.subheader('Imagem com a possível detecção de Myxozoários')
                st.image(pred_img_obj, caption='Detecção de Myxozoários', use_column_width=True)
    
    if __name__ == "__main__":
        main()



#################################
with tab3:

    import streamlit as st 
    from streamlit_webrtc import webrtc_streamer
    import av


# Conteúdo da página "📸- Cam"
with tab3:
    import streamlit as st
    from streamlit_webrtc import webrtc_streamer
    from yolo_predictions import YOLO_Pred  # Importe sua classe YOLO_Pred
    import cv2
    import numpy as np

    
    # Função de callback para processar o vídeo da webcam
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Faça a detecção de objetos com o modelo YOLO
        pred_img = yolo.predictions(img)
        
        return pred_img
    
    def main():
        st.title("Detecção de Objetos em Vídeo com YOLO")
        
        # Componente do Streamlit para captura de vídeo da webcam e processamento
        webrtc_streamer(key="example",
                        video_frame_callback=video_frame_callback,
                        media_stream_constraints={"video": True, "audio": False})
    
    if __name__ == "__main__":
        main()







    
