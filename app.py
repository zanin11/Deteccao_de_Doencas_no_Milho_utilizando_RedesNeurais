import streamlit as st
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = load_model('modelo_milho.h5')  # Substitua pelo caminho correto do modelo

# Função para preparar a imagem
def prepare_image(image_path):
    # Carregar a imagem com tamanho ajustado
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)  # Converter para array NumPy
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma dimensão extra
    img_array = img_array / 255.0  # Normalizar os valores da imagem
    return img_array

# Função para fazer a previsão
def predict_image(image_path):
    img_array = prepare_image(image_path)
    predictions = model.predict(img_array)  # Fazer a previsão
    predicted_class = np.argmax(predictions)  # Obter a classe com maior probabilidade
    return predicted_class

# Função para mostrar a imagem
def show_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Recarregar a imagem
    plt.imshow(img)
    plt.axis('off')
    st.pyplot()  # Exibir a imagem no Streamlit

# Definir o título do aplicativo
st.title('Detecção de Doenças no Milho utilizando Redes Neurais')

# Criar uma barra lateral com duas opções de abas
option = st.sidebar.selectbox(
    "Escolha uma opção:",
    ("Informações Gerais", "Classificar Imagem")
)

# Aba de Informações Gerais
if option == "Informações Gerais":
    st.header("Informações Acadêmicas")
    st.write("""Sistema desenvolvido como método avaliativo da matéria de Inteligência Artificial.
    """)
    st.write("Guilherme Zanin - RA: 221026479.")
    st.write("Ciência da Computação - Unesp/Bauru.")
    st.write("Novembro de 2024.")
    st.header("Sobre o DataSet")
    st.write("O Corn or Maize Leaf Disease Dataset, disponível no Kaggle, é um conjunto de dados usado para treinar modelos de aprendizado de máquina para a classificação de doenças em folhas de milho. Este dataset contém imagens de folhas de milho saudáveis e infectadas por diferentes doenças. Ele é amplamente utilizado para desenvolver sistemas de detecção de doenças em plantas com o objetivo de melhorar a produtividade agrícola e auxiliar no monitoramento de lavouras.")
    st.header("Sobre o Sistema")
    st.write("""
        Este sistema utiliza redes neurais para detectar doenças em folhas de milho. 
        Ele classifica a folha em uma das seguintes categorias:
        - **Blight**: Uma doença que causa manchas nas folhas.
        - **Common Rust**: Manchas características causadas por fungos.
        - **Gray Leaf Spot**: Mancha de folhas cinzas, causada por fungos.
        - **Healthy**: Folha saudável sem sinais de doenças.
        
        Carregue uma imagem de uma folha de milho e o sistema retornará a classificação da condição da folha.
    """)
    st.header("Explicação das Doenças")
    st.subheader("Blight (Pinta ou Mancha Escura)")
    st.write("""
        Blight é uma doença fúngica causada por fungos como *Helminthosporium* e *Cochliobolus*. Ela se caracteriza por manchas escuras nas folhas, que rapidamente se expandem e podem levar à morte celular. Em casos graves, a planta pode morrer.
    """)
    
    st.subheader("Common Rust (Ferrugem Comum)")
    st.write("""
        Causada pelo fungo *Puccinia sorghi*, a ferrugem comum resulta em manchas alaranjadas ou vermelhas nas folhas, prejudicando a fotossíntese e reduzindo a produtividade das plantas.
    """)
    
    st.subheader("Gray Leaf Spot (Mancha de Folha Cinza)")
    st.write("""
        A doença é provocada pelo fungo *Cercospora zeae-maydis*, que causa manchas cinza nas folhas. Ela pode se espalhar rapidamente em condições de umidade e reduzir o rendimento do milho.
    """)
    
    st.subheader("Healthy (Saudável)")
    st.write("""
        Folhas saudáveis são verdes e sem lesões, com boa capacidade de realizar a fotossíntese. Plantas saudáveis têm maior resistência a doenças e estresses ambientais.
    """)
    st.header("Sobre o Modelo Utilizado")
    st.write("""
        O modelo utilizado é baseado na arquitetura **MobileNetV2**, que foi treinado para identificar doenças nas folhas de milho. 
        A MobileNetV2 foi escolhida por ser uma rede neural eficiente para dispositivos móveis e para aplicações que exigem uma boa precisão, mas com um modelo leve. 
        
        As camadas principais do modelo incluem:
        - **Base do modelo MobileNetV2**: A base do modelo é carregada com pesos pré-treinados do ImageNet.
        - **GlobalAveragePooling2D**: Camada de pooling para reduzir as dimensões dos dados.
        - **Camada densa de 128 neurônios com ReLU**: Camada densa intermediária para capturar padrões complexos.
        - **Camada de saída com 4 neurônios e função de ativação softmax**: Para classificar as folhas nas 4 classes possíveis (Blight, Common Rust, Gray Leaf Spot, Healthy).
        
        O modelo foi treinado por **10 épocas** para realizar a tarefa de classificação de imagens.
    """)

# Aba de Classificação de Imagem
if option == "Classificar Imagem":
    st.header("Classificação de Folha de Milho")
    st.write("Envie uma imagem de uma folha de milho para verificar se está saudável ou com alguma doença.")

    # Carregar o uploader de imagens
    uploaded_image = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

    # Se o usuário carregar uma imagem
    if uploaded_image is not None:
        # Exibir a imagem carregada no Streamlit
        st.image(uploaded_image, caption='Imagem carregada', use_column_width=True)
        
        # Salvar a imagem temporariamente
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Previsão da classe
        predicted_class = predict_image(temp_image_path)
        
        # Mapear o índice para o nome da classe
        class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']  # Ajuste os nomes conforme seu modelo
        predicted_class_name = class_names[predicted_class]
        
        # Exibir o resultado da previsão
        st.write(f"A planta foi classificada como: **{predicted_class_name}**")
