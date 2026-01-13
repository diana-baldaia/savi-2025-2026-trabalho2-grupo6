import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
import numpy as np
from collections import Counter

'''
Código responsável por analisar e validar os datasets criados no main_synthesis.
'''


def visualize_dataset_stats(root_folder, dataset_name, split_name='train'):
    
    '''
    Função de Controlo de Qualidade.
    Gera estatístias e visualizções para confirmar se o dataset foi bem gerado.
    '''

    # -----------------------------
    # 1. Caminho para o ficheiro
    # -----------------------------
    base_path = os.path.join(root_folder, dataset_name, split_name)
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')
    
    print(f"\n--- A analisar: {dataset_name} ({split_name}) ---")

    # Vai buscar todos os ficheiros de texto ordenados
    label_files = glob.glob(os.path.join(labels_path, "*.txt"))
    label_files.sort()
    
    if len(label_files) == 0:
        print("Erro: Não encontrei ficheiros. Verifica se geraste o dataset primeiro!")
        return

    # -----------------------------
    # 2. Recolha de Dados
    # -----------------------------
    # Lemos apenas os ficheiros de texto para ser mais rápido
    all_classes = []            # Guarda todos os números encontrados (ex: 0,0,7,2...)
    digits_per_image = []       # Guarda quantos números há em cada imagem (ex: 3,5,4...)
    digit_sizes = []            # Guarda a altura dos dígitos para verificar o resize
    
    for l_file in label_files:
        with open(l_file, 'r') as f:
            lines = f.readlines()
            digits_per_image.append(len(lines)) # Quantos dígitos tem a imagem (ex: 5 linhas são 5 digitos)
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    cls = int(parts[0])     # Classe (0-9)
                    h = float(parts[4])     # Altura
                    all_classes.append(cls) 
                    digit_sizes.append(h)

    # ------------------------------------
    # 3. Relatório Estatístico (Gráficos)
    # ------------------------------------

    # Cria uma janela para caberem 3 gráficos
    plt.figure(figsize=(15, 5))

    # Gráfico A: Distribuição de Classes
    # Verifica se o dataset está balanceado, por exemplo, não queremos 1000 zeros e apenas 10 noves
    plt.subplot(1, 3, 1)
    counts = Counter(all_classes)
    plt.bar(counts.keys(), counts.values(), color='skyblue', edgecolor='black')
    plt.title('Frequência das Classes (0-9)')
    plt.xlabel('Dígito')
    plt.ylabel('Quantidade')
    plt.xticks(range(10))

    # Gráfico B: Densidade de Objetos
    # Confirma as regras do enunciado
    # No Dataset A deve ser sempre 1. No D deve variar entre 3 e 5
    plt.subplot(1, 3, 2)
    plt.hist(digits_per_image, bins=range(1, 7), align='left', rwidth=0.8, color='orange', edgecolor='black')
    plt.title('Histograma: Dígitos por Imagem')
    plt.xlabel('Número de Dígitos')
    plt.ylabel('Imagens')

    # Gráfico C: Verificação de Escala
    # Confirma se os dígitos foram redimensionados (Resize)
    plt.subplot(1, 3, 3)
    plt.hist(digit_sizes, bins=20, color='green', edgecolor='black')
    plt.title(f'Tamanhos dos Dígitos (Média: {np.mean(digit_sizes):.1f}px)')
    plt.xlabel('Altura (pixels)')
    plt.ylabel('Frequência')

    plt.tight_layout()
    stats_file = f'stats_{dataset_name}_{split_name}.png'
    plt.savefig(stats_file)
    print(f"Gráficos estatísticos salvos em: {stats_file}")
    # plt.show() # Descomenta se quiseres ver janelas a abrir

    # -------------------------------
    # 4. Validação Visual (Mosaico)
    # -------------------------------
    # Desenhamos as caixas sobre as imagens para garantir que as coordenadas estão certas
    print("A gerar mosaico...")
    num_rows, num_cols = 4, 4
    num_images = num_rows * num_cols
    
    # Escolher imagens aleatórias, para não ver sempre as mesmas iniciais
    sample_files = random.sample(label_files, num_images)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    fig.suptitle(f'Exemplo de Amostras: {dataset_name}', fontsize=16)

    for i, ax in enumerate(axes.flat):
        label_file = sample_files[i]
        
        # Substituir .txt por .jpg para encontrar a imagem correspondente
        base_name = os.path.basename(label_file).replace('.txt', '.jpg')
        img_path = os.path.join(images_path, base_name)
        
        # Carregar a imagem
        im = Image.open(img_path)
        ax.imshow(im, cmap='gray')
        ax.axis('off')
        
        # Desenhar as caixas
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls = int(parts[0])

                # Converte strings para floats (x, y, largura, altura)
                x, y, w, h = map(float, parts[1:])
                
                # Cria o retângulo
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                
                # Texto a dizer a qual número corresponde
                ax.text(x, y-2, str(cls), color='yellow', fontsize=8, weight='bold')

    mosaic_file = f'mosaic_{dataset_name}_{split_name}.png'
    plt.savefig(mosaic_file)
    print(f"Mosaico salvo em: {mosaic_file}")
    # plt.show()

if __name__ == "__main__":
    
    root_path = '/home/baldaia/Desktop/savi-2025-2026-trabalho2-grupo5/Tarefa_2/' 
    
    # 1. Analisar a Versão A (Fácil)
    # Deve mostrar: 1 dígito por imagem, classes equilibradas, tamanho fixo.
    if os.path.exists(os.path.join(root_path, 'mnist_detection_A')):
        visualize_dataset_stats(root_path, 'mnist_detection_A', 'train')
    
    # 2. Analisar a Versão D (Difícil)
    # Deve mostrar: 3 a 5 dígitos, classes equilibradas, tamanhos variados.
    if os.path.exists(os.path.join(root_path, 'mnist_detection_D')):
        visualize_dataset_stats(root_path, 'mnist_detection_D', 'train')