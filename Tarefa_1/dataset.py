import glob
import os
import zipfile
import numpy as np
import requests
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, is_train):

        # Armazena os argumentos nos parâmetros da classe
        self.args = args
        self.train = is_train

        # -----------------------------------------
        # Prepara os inputs (caminhos das imagens)
        # -----------------------------------------
        
        # Cria o caminho para a pasta de imagens
        print(args['dataset_folder'])
        split_name = 'train' if is_train else 'test'
        image_path = os.path.join(args['dataset_folder'], split_name, 'images/')

        print('image path is: ' + image_path)

        # Procura todos os ficheiros .jpg na pasta de imagens
        self.image_filenames = glob.glob(image_path + "/*.jpg")
        # Ordena os ficheiros para garantir que estão na ordem correta
        self.image_filenames.sort()

        # print("image_filenames= " + str(self.image_filenames))


        # --------------------------------
        # Prepara as labels
        # --------------------------------

        # Cria o caminho para o ficheiro de labels
        self.labels_filename = os.path.join(
            args['dataset_folder'], split_name, 'labels.txt')
        
        # Lista que guardará as labels na mesma ordem que as imagens
        self.labels = []

        # Abre o ficheiro de labels e lê linha a linha
        with open(self.labels_filename, "r") as f:  # type: ignore
            for line in f:
                # print("line= " + line)
                parts = line.strip().split()   # divide por espaços
                # print('parts= ' + str(parts))
                label = float(parts[1])    # extrai a label (segunda parte) e converte para float
                # print('label= ' + label)
                self.labels.append(label)


        # Seleciona a percentagem de exemplos especificada nos args
        num_examples = round(len(self.image_filenames) * args['percentage_examples'])

        # Reduz o tamanho dos caminhos das imagens e das labels
        self.image_filenames = self.image_filenames[0:num_examples]
        self.labels = self.labels[0:num_examples]

        # Para converter para tensores, usado pelo PyTorch
        self.to_tensor = transforms.ToTensor()


    def __len__(self):
        # Retorna o número de exemplos no dataset
        return len(self.image_filenames)


    def __getitem__(self, idx):
        # Recebe um índice 'idx' de exemplo como input e retorna o input e output correspondentes
        # O retorno será um tuple (input, output), mas o input e output serão tensores

        # ----------------------------
        # Obtém a label
        # ----------------------------

        # Obtém o índice
        label_index = int(self.labels[idx])

        # Cria um vetor de 10 posições para dígitos de 0 a 9
        # ex.: se o valor for 2, é colocado '1' na posição correspondente: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        label = [0]*10
        label[label_index] = 1

        # Converte para um tensor de números decimais
        label_tensor = torch.tensor(label, dtype=torch.float)


        # ----------------------------
        # Obtém a imagem como tensor
        # ----------------------------
        # Obtém o caminho da imagem correspondente ao índice
        image_filename = self.image_filenames[idx]
        # Abre a imagem e converte para escala de cinza (L)
        image = Image.open(image_filename).convert('L')
        # Converte a imagem num tensor
        image_tensor = self.to_tensor(image)

        return image_tensor, label_tensor