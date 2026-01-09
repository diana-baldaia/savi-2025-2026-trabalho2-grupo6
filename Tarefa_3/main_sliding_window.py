import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np
from model import ModelBetterCNN # Importa o teu modelo da Tarefa 1

def sliding_window_detection(image_path, model_path):
    
    # -----------------------------------------------------------
    # 1. Configurações (Hiperparâmetros da Deteção)
    # -----------------------------------------------------------
    WINDOW_SIZE = 28       # O tamanho que o modelo sabe ler (28x28)
    STRIDE = 4             # O "Passo". Avança 4 pixels de cada vez.
                           # Passo pequeno = Mais lento, mais preciso.
                           # Passo grande = Rápido, pode falhar o centro.
    
    CONFIDENCE_THRESHOLD = 0.8 # Só aceita se tiver 95% de certeza.
                                # Como o modelo não conhece "fundo preto", 
                                # ele tende a inventar. Precisamos de filtrar.

    # -----------------------------------------------------------
    # 2. Preparar o Modelo (Cérebro da Tarefa 1)
    # -----------------------------------------------------------
    print(f"A carregar modelo de: {model_path}")
    
    # Inicializar a arquitetura (tem de ser igual à usada no treino!)
    model = ModelBetterCNN()
    
    # Carregar os pesos (o ficheiro .pkl)
    # Adicionamos weights_only=False para resolver o erro do PyTorch 2.6+
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # O checkpoint tem várias coisas (epoch, loss...), queremos só o 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Caso tenhas gravado o modelo diretamente sem o dicionário (menos provável no teu trainer)
        model.load_state_dict(checkpoint)
        
    model.eval() # IMPORTANTE: Coloca em modo de avaliação (desliga Dropout, fixa Batch Norm)

    # -----------------------------------------------------------
    # 3. Preparar a Imagem (Cena da Tarefa 2)
    # -----------------------------------------------------------
    print(f"A carregar imagem de: {image_path}")
    full_image = Image.open(image_path).convert('L') # Converter para cinzento
    width, height = full_image.size

    # Transformação: A mesma usada no treino (apenas ToTensor)
    to_tensor = transforms.ToTensor()

    # -----------------------------------------------------------
    # 4. O Ciclo da Janela Deslizante (O "Scanner")
    # -----------------------------------------------------------
    detected_boxes = [] # Vamos guardar aqui: (x, y, label, probabilidade)

    print("A iniciar varrimento (isto pode demorar um pouco)...")
    
    # Loop Y (Linhas)
    for y in range(0, height - WINDOW_SIZE + 1, STRIDE):
        # Loop X (Colunas)
        for x in range(0, width - WINDOW_SIZE + 1, STRIDE):
            
            # ... dentro dos ciclos for x, for y ...

            # A. Recortar a Janela
            crop = full_image.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
            
            crop_tensor = to_tensor(crop)
            
            # --- ALTERAÇÃO AQUI ---
            # Comenta estas linhas. Vamos obrigar o modelo a olhar para o preto
            # e a decidir com base na confiança, como pede o professor.
            # if torch.sum(crop_tensor) < 0.5: 
            #     continue 
            # ----------------------
            if crop_tensor.std() < 0.01:  
                continue
            # C. Preparar para o Modelo
            input_tensor = crop_tensor.unsqueeze(0)

            # D. Previsão e Softmax
            with torch.no_grad(): 
                output = model(input_tensor)
                
                # O Softmax transforma os números brutos (logits) em percentagens (0.0 a 1.0)
                probabilities = torch.softmax(output, dim=1) 
                
                max_prob, predicted_class = torch.max(probabilities, 1)
                
                prob_value = max_prob.item()
                label = predicted_class.item()

            # E. Decisão (Thresholding)
            # É AQUI que se cumpre o ponto 2 da Tarefa 3
            # Se a certeza for baixa, ignoramos.
            if prob_value > CONFIDENCE_THRESHOLD:
                detected_boxes.append((x, y, label, prob_value))
            
            
    print(f"Concluído! Foram detetadas {len(detected_boxes)} caixas potenciais.")

    # -----------------------------------------------------------
    # 5. Visualização
    # -----------------------------------------------------------
    fig, ax = plt.subplots(1)
    ax.imshow(full_image, cmap='gray')

    # Desenhar todas as caixas detetadas
    for (x, y, label, prob) in detected_boxes:
        # Criar retângulo vermelho
        rect = patches.Rectangle((x, y), WINDOW_SIZE, WINDOW_SIZE, 
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Escrever o número e a confiança
        ax.text(x, y - 2, f"{label} ({prob:.2f})", color='yellow', fontsize=6, weight='bold')

    plt.title(f"Sliding Window Detection (Stride={STRIDE})")
    plt.axis('off')
    
    output_filename = 'sliding_window_result.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Resultado salvo em: {output_filename}")
    plt.show()

if __name__ == "__main__":
    
    # --- CAMINHOS (AJUSTA AQUI SE NECESSÁRIO) ---
    # root_path = '/home/baldaia/Desktop/savi-2025-2026-trabalho2-grupo5/Tarefa_3'
    
    # 1. Onde está o modelo treinado (Task 1)?
    # Procura pelo ficheiro 'best.pkl' dentro da pasta Experiments
    # model_path = os.path.join(root_path, 'Experiments', 'best.pkl')
    model_path = 'Tarefa_1/Experiments/best.pkl'

    # 2. Qual imagem queres testar (Task 2)?
    # Vamos testar uma imagem da versão FÁCIL (A) primeiro para ver se funciona
    # Escolhe uma imagem que saibas que existe na pasta mnist_detection_A/test/images
    # image_to_test = os.path.join(root_path, 'mnist_detection_A', 'test', 'images', 'scene_00001.jpg')
    image_to_test = 'Tarefa_2/mnist_detection_A/test/images/scene_00001.jpg'

    # Se quiseres testar a difícil, descomenta esta:
    # image_to_test = os.path.join(root_path, 'mnist_detection_D', 'test', 'images', 'scene_00001.jpg')
    # image_to_test = 'Tarefa_2/mnist_detection_D/test/images/scene_00001.jpg'

    # Verificar se os ficheiros existem antes de correr
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em {model_path}")
        print("Certifica-te que treinaste o modelo na Tarefa 1 e que o ficheiro best.pkl existe.")
    elif not os.path.exists(image_to_test):
        print(f"ERRO: Imagem não encontrada em {image_to_test}")
        print("Verifica se geraste o dataset na Tarefa 2.")
    else:
        # Correr a função
        sliding_window_detection(image_to_test, model_path)