import os
import random
import torch
from torchvision import datasets
from PIL import Image
from tqdm import tqdm

def check_overlap(new_box, existing_boxes):
    """
    Verifica se a nova caixa se sobrepõe a alguma caixa existente.
    new_box: (x, y, largura, altura)
    """
    new_x, new_y, new_w, new_h = new_box
    
    for (ex_x, ex_y, ex_w, ex_h, _) in existing_boxes:
        if (new_x + new_w < ex_x or  # Novo está à esquerda
            new_x > ex_x + ex_w or   # Novo está à direita
            new_y + new_h < ex_y or  # Novo está acima
            new_y > ex_y + ex_h):    # Novo está abaixo
            continue 
        else:
            return True # Tocam (Há sobreposição)
    return False 

def generate_dataset(root_folder, dataset_name, num_images, is_train=True, 
                     min_digit_size=28, max_digit_size=28,
                     min_digits=1, max_digits=1):
    
    # 1. Configurações Físicas
    width_canvas = 128
    height_canvas = 128

    # 2. Preparar pastas (Agora usa o dataset_name para separar as versões)
    split_name = 'train' if is_train else 'test'
    output_folder_images = os.path.join(root_folder, dataset_name, split_name, 'images')
    output_folder_labels = os.path.join(root_folder, dataset_name, split_name, 'labels')
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_labels, exist_ok=True)

    print(f"\n--- A gerar {dataset_name} ({split_name}) ---")
    print(f"Config: {min_digits}-{max_digits} dígitos por img | Tamanho: {min_digit_size}-{max_digit_size}px")

    # 3. Carregar MNIST Original
    mnist_data = datasets.MNIST(root='./data', train=is_train, download=True, transform=None)

    # 4. Ciclo de Geração
    for i in tqdm(range(num_images)):
        
        canvas = Image.new('L', (width_canvas, height_canvas), color=0)
        placed_digits = []
        
        # Agora a quantidade é variável conforme os argumentos
        num_digits_this_image = random.randint(min_digits, max_digits)
        
        for _ in range(num_digits_this_image):
            
            for attempt in range(20): 
                rand_idx = random.randint(0, len(mnist_data) - 1)
                digit_img, digit_label = mnist_data[rand_idx]
                
                # Agora o tamanho é variável conforme os argumentos
                new_size = random.randint(min_digit_size, max_digit_size)
                digit_img = digit_img.resize((new_size, new_size), Image.BILINEAR)
                
                max_x = width_canvas - new_size
                max_y = height_canvas - new_size
                
                if max_x < 0 or max_y < 0: continue 

                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                new_box = (x, y, new_size, new_size)
                if not check_overlap(new_box, placed_digits):
                    canvas.paste(digit_img, (x, y))
                    placed_digits.append((x, y, new_size, new_size, digit_label))
                    break 
        
        # 5. Salvar Imagem
        image_filename = f"scene_{i:05d}.jpg"
        canvas.save(os.path.join(output_folder_images, image_filename))
        
        # 6. Salvar Labels
        label_filename = f"scene_{i:05d}.txt"
        with open(os.path.join(output_folder_labels, label_filename), 'w') as f:
            for (x, y, w, h, label) in placed_digits:
                f.write(f"{label} {x} {y} {w} {h}\n")

if __name__ == "__main__":
    
    # Define a tua pasta base aqui
    root_path = '/home/baldaia/Desktop/savi-2025-2026-trabalho2-grupo5/Tarefa_2' 
    
    # --- GERAR VERSÃO A (FÁCIL) ---
    # Requisito: 1 dígito, posição aleatória (tamanho fixo ou pouco variável)
    # Vou pôr tamanho 28 fixo, igual ao original.
    generate_dataset(root_path, dataset_name='mnist_detection_A', 
                     num_images=1000, is_train=True, # Reduzi para 1000 para testares rápido
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