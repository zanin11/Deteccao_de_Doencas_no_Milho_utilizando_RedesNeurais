import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import imagepi
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model = load_model('modelo_milho.h5')  # Carregar o modelo salvo no formato .h5

# Função para preparar a imagem
def prepare_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma dimensão extra
    img_array = img_array / 255.0  # Normalizar a imagem
    return img_array

# Função para fazer a previsão
def predict_image(image_path):
    img_array = prepare_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Função para mostrar a imagem
def show_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    plt.imshow(img)
    plt.axis('off')
    st.pyplot()

# Definir o título do aplicativo
st.title('Detecção de Doenças no Milho utilizando Redes Neurais')

# Descrição
st.write("Envie uma imagem de uma folha de milho para verificar se está saudável ou com alguma doença.")

# Carregar o uploader de imagens
uploaded_image = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

# Se o usuário carregar uma imagem
if uploaded_image is not None:
    # Exibir a imagem carregada
    st.image(uploaded_image, caption='Imagem carregada', use_column_width=True)
    
    # Salvar a imagem temporariamente para fazer a previsão
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_image.getbuffer())
    
    # Previsão da classe
    predicted_class = predict_image("temp_image.jpg")
    
    # Mapear o índice para o nome da classe
    class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']  # Defina conforme suas classes
    predicted_class_name = class_names[predicted_class]
    
    # Exibir o resultado da previsão
    st.write(f"A imagem foi classificada como: **{predicted_class_name}**")
    
    # Mostrar a imagem
    show_image("temp_image.jpg")
