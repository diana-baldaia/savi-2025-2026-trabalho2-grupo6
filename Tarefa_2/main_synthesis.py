import os
import random
import torch
from torchvision import datasets
from PIL import Image
from tqdm import tqdm


'''
Código responsável por criar imagens mais complexas (128x128),
onde vamos ter números com diferentes dimensões e mais que um número por imagem
'''


def check_overlap(new_box, existing_boxes):
    """
    Verifica se a nova caixa se sobrepõe a alguma caixa existente, ou seja,
    se algum número não sobrepõe outro
    """
    new_x, new_y, new_w, new_h = new_box
    
    for (ex_x, ex_y, ex_w, ex_h, _) in existing_boxes:
        # Verifica se a nova caixa sobrepõe alguma existente
        if (new_x + new_w < ex_x or  # Novo está à esquerda
            new_x > ex_x + ex_w or   # Novo está à direita
            new_y + new_h < ex_y or  # Novo está acima
            new_y > ex_y + ex_h):    # Novo está abaixo
            continue    # Se alguma das condições for verdadeira podemos avançar
        else:
            return True # FALHA: Não está livre em nenhum lado, logo, tocam-se
    return False        # Sucesso: Não tocou em ninguém

def generate_dataset(root_folder, dataset_name, num_images, is_train=True, 
                     min_digit_size=28, max_digit_size=28,
                     min_digits=1, max_digits=1):
    
    # 1. Configurações Físicas
    # Definimos que a imagem tem 128x128 pixeis
    width_canvas = 128
    height_canvas = 128

    # 2. Preparar pastas (Usa o dataset_name para separar as versões)
    split_name = 'train' if is_train else 'test'
    output_folder_images = os.path.join(root_folder, dataset_name, split_name, 'images')
    output_folder_labels = os.path.join(root_folder, dataset_name, split_name, 'labels')
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_labels, exist_ok=True)

    print(f"\n--- A gerar {dataset_name} ({split_name}) ---")
    print(f"Config: {min_digits}-{max_digits} dígitos por img | Tamanho: {min_digit_size}-{max_digit_size}px")

    # 3. Carregar MNIST Original
    mnist_data = datasets.MNIST(root='./data', train=is_train, download=True, transform=None)

    # 4. Ciclo de Geração de imagem
    for i in tqdm(range(num_images)):
        
        # Crias um fundo preto vazio
        canvas = Image.new('L', (width_canvas, height_canvas), color=0)
        placed_digits = []
        
        # Sortei quantos números vamos pôr nesta imagem
        num_digits_this_image = random.randint(min_digits, max_digits)
        
        for _ in range(num_digits_this_image):
            
            # Tenta 20 vezes encontrar um espaço livre
            # Se a imagem já estiver cheia, desiste para não bloquear o programa
            for attempt in range(20): 
                
                # Pega num dígito aleatório do MNIST original
                rand_idx = random.randint(0, len(mnist_data) - 1)
                digit_img, digit_label = mnist_data[rand_idx]
                
                # Re-escala: Transforma o tamanho (ex: aumenta para 36px ou diminui para 22px)
                # Image.BILINEAR é o método matemático para redimensionar sem ficar pixelizado
                new_size = random.randint(min_digit_size, max_digit_size)
                digit_img = digit_img.resize((new_size, new_size), Image.BILINEAR)
                
                # Calcula coordenadas aleatórias (x, y) garantindo que cabe dentro da imagem
                max_x = width_canvas - new_size
                max_y = height_canvas - new_size
                
                if max_x < 0 or max_y < 0: continue 

                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                # Verifica se há colisões
                new_box = (x, y, new_size, new_size)

                if not check_overlap(new_box, placed_digits):
                    # Se tiver espaço coloca o digito na imagem
                    canvas.paste(digit_img, (x, y))
                    # Guarda a posição
                    placed_digits.append((x, y, new_size, new_size, digit_label))
                    break 
        
        # 4. Exportação
        # Guarda a imagem final
        image_filename = f"scene_{i:05d}.jpg"
        canvas.save(os.path.join(output_folder_images, image_filename))
        
        # Guarda o ficheiro de texto com as labels
        # Formato: Classe X Y largura altura
        label_filename = f"scene_{i:05d}.txt"
        with open(os.path.join(output_folder_labels, label_filename), 'w') as f:
            for (x, y, w, h, label) in placed_digits:
                f.write(f"{label} {x} {y} {w} {h}\n")

if __name__ == "__main__":
    
    # Define a pasta base
    root_path = '/home/baldaia/Desktop/savi-2025-2026-trabalho2-grupo5/Tarefa_2' 
    
    # --- GERAR VERSÃO A (FÁCIL) ---
    # Requisito: 1 dígito, posição aleatória (tamanho fixo ou pouco variável)
    # Tamanho 28 fixo, igual ao original.
    generate_dataset(root_path, dataset_name='mnist_detection_A', 
                     num_images=1000, is_train=True,
                     min_digit_size=28, max_digit_size=28,
                     min_digits=1, max_digits=1)

    generate_dataset(root_path, dataset_name='mnist_detection_A', 
                     num_images=200, is_train=False,
                     min_digit_size=28, max_digit_size=28,
                     min_digits=1, max_digits=1)

    # --- GERAR VERSÃO D (DIFÍCIL) ---
    # Requisito: 3 a 5 dígitos, escalas 22x22 a 36x36
    generate_dataset(root_path, dataset_name='mnist_detection_D', 
                     num_images=1000, is_train=True,
                     min_digit_size=22, max_digit_size=36,
                     min_digits=3, max_digits=5)

    generate_dataset(root_path, dataset_name='mnist_detection_D', 
                     num_images=200, is_train=False,
                     min_digit_size=22, max_digit_size=36,
                     min_digits=3, max_digits=5)
    
    print("\nGeração das versões A e D concluída!")