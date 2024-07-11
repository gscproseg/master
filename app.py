

import streamlit as st

# Configuração da página
st.set_page_config(
    page_title= "MLens",
    page_icon= "🧠",  # Defina o ícone da página como um emoji de tubarão
    layout="wide",  # Defina o layout como "wide" para aproveitar melhor o espaço na tela
    initial_sidebar_state="collapsed"  # Defina a barra lateral como colapsada
)

# Criação das guias
tab1, tab2, tab3, tab4 = st.tabs(["MLens", "Detecção em Imagens", "Detecção em Vídeo", "Detecão em RT"])

# Conteúdo da página "Home"
with tab1:
    st.subheader("Bem-vindo ao MLens")
    # Adicionar a imagem ao espaço em branco
    st.image("./Mlens.png", use_column_width=True)  #  caminho da sua imagem

# Adicione as informações adicionais
st.write("Desenvolvido por LIMT - Ufra e [Carneiro, G.S](http://lattes.cnpq.br/3771047626259544)")

#######################################################

with tab2:

    st.header("Detecção em Imagens")
    
    from yolov5_predictions import YOLO_Pred
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
    st.header("Detecção em Video")

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


with tab4:
    st.subheader("Não disponível nesta Versão, Aguarde!")

    import cv2
    import numpy as np
    import yaml
    
    # Função para realizar a pré-processamento da imagem
    def preprocess(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image
    
    # Função para realizar a pós-processamento da saída do modelo
    def postprocess(outputs, img_shape):
        # A função de pós-processamento pode variar com base no formato da saída do seu modelo.
        # Aqui você deve converter as saídas em caixas delimitadoras, confidências e classes.
        pass  # Implementar de acordo com seu modelo
    
    # Carregar o modelo ONNX
    session = onnxruntime.InferenceSession('best.onnx')
    
    # Carregar o arquivo data.yaml
    with open('data.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    # Inicializar a captura de vídeo
    cap = cv2.VideoCapture(0)  # 0 para a webcam padrão
    
    # Verificar se a captura de vídeo foi inicializada corretamente
    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        exit()
    
    while True:
        # Capturar um quadro da webcam
        ret, frame = cap.read()
    
        # Verificar se o quadro foi capturado corretamente
        if not ret:
            print("Erro ao capturar o quadro.")
            break
    
        # Pré-processar o quadro
        input_image = preprocess(frame)
    
        # Realizar a inferência
        outputs = session.run(None, {session.get_inputs()[0].name: input_image})
    
        # Pós-processar a saída
        boxes, confidences, class_ids = postprocess(outputs, frame.shape)
    
        # Desenhar as caixas delimitadoras no quadro
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{data['names'][class_id]}: {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
        # Exibir o quadro com as detecções
        cv2.imshow('Webcam - YOLOv5', frame)
    
        # Pressionar 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    cv2.destroyAllWindows()


#########################################################################################


