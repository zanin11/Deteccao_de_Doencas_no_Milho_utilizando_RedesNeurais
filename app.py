# Aba de Informações Gerais
if option == "Informações Gerais":
    st.header("Informações Acadêmicas")
    st.write("Guilherme Zanin")
    st.write("RA: 221026479")
    st.write("Novembro de 2024")
    st.write("Unesp/Bauru")
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

    st.header("Por que utilizamos a função ReLU?")
    st.write("""
        A função de ativação ReLU (Rectified Linear Unit) foi utilizada devido a seus vários benefícios para redes neurais:
        - **Simplicidade e eficiência computacional**: É computacionalmente eficiente, realizando apenas uma operação simples de verificar se o valor é maior que zero.
        - **Evita saturação**: Ao contrário de outras funções de ativação como sigmoid e tanh, a ReLU não satura em sua região positiva, permitindo um aprendizado mais rápido.
        - **Esparsidade**: Zera os valores negativos, criando uma representação esparsa, o que pode ajudar na eficiência e na interpretação dos dados.
        - **Melhora o aprendizado em redes profundas**: Reduz o problema do gradiente desaparecendo, comum em redes com muitas camadas.
    """)
