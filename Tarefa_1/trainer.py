import glob
import os
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import requests
import seaborn
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm
from sklearn import metrics # <--- NOVIDADE: A ferramenta profissional de métricas

class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):

        # Storing arguments in class properties
        self.args = args
        self.model = model

        # Create the dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args['batch_size'],
            shuffle=False)
        
        # Setup loss function (Mantemos MSE como combinado, ou CrossEntropy se mudaste)
        self.loss = nn.MSELoss() 

        # Define optimizer
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)

        # Start from scratch or resume training
        if self.args['resume_training']:
            self.loadTrain()
        else:
            self.train_epoch_losses = []
            self.test_epoch_losses = []
            self.epoch_idx = 0

    def train(self):
        print('Training started. Max epochs = ' + str(self.args['num_epochs']))

        for i in range(self.epoch_idx, self.args['num_epochs']): 
            self.epoch_idx = i
            print('\nEpoch index = ' + str(self.epoch_idx))
            
            # --- TRAIN ---
            self.model.train()
            train_batch_losses = []
            num_batches = len(self.train_dataloader)
            
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(enumerate(self.train_dataloader), total=num_batches):
                label_pred_tensor = self.model.forward(image_tensor)
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                train_batch_losses.append(batch_loss.item())

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            # --- TEST ---
            self.model.eval()
            test_batch_losses = []
            num_batches = len(self.test_dataloader)
            
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(enumerate(self.test_dataloader), total=num_batches):
                label_pred_tensor = self.model.forward(image_tensor)
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                test_batch_losses.append(batch_loss.item())

            # End of epoch
            print('Finished epoch ' + str(i) + ' out of ' + str(self.args['num_epochs']))
            
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)

            self.draw()
            self.saveTrain()

        print('Training completed.')
        self.evaluate() # <--- Chama a avaliação automaticamente no fim

    def loadTrain(self):
        print('Resuming training from last checkpoint.')
        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        if not os.path.exists(checkpoint_file):
            raise ValueError('Checkpoint file not found: ' + checkpoint_file)
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        self.epoch_idx = checkpoint['epoch_idx']
        self.train_epoch_losses = checkpoint['train_epoch_losses']
        self.test_epoch_losses = checkpoint['test_epoch_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def saveTrain(self):
        checkpoint = {}
        checkpoint['epoch_idx'] = self.epoch_idx
        checkpoint['train_epoch_losses'] = self.train_epoch_losses
        checkpoint['test_epoch_losses'] = self.test_epoch_losses
        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        torch.save(checkpoint, checkpoint_file)

        if self.test_epoch_losses[-1] == min(self.test_epoch_losses):
            best_file = os.path.join(self.args['experiment_full_name'], 'best.pkl')
            torch.save(checkpoint, best_file)

    def draw(self):
        plt.figure(1)
        plt.clf()
        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        axis = plt.gca()
        axis.set_xlim([1, self.args['num_epochs']+1])
        axis.set_ylim([0, 0.1])
        
        xs = range(1, len(self.train_epoch_losses)+1)
        plt.plot(xs, self.train_epoch_losses, 'r-', linewidth=2)
        plt.plot(xs, self.test_epoch_losses, 'b-', linewidth=2)

        best_epoch_idx = int(np.argmin(self.test_epoch_losses))
        plt.plot([best_epoch_idx, best_epoch_idx], [0, 0.5], 'g--', linewidth=1)
        plt.legend(['Train', 'Test', 'Best'], loc='upper right')
        plt.savefig(os.path.join(self.args['experiment_full_name'], 'training.png'))

    def evaluate(self):
        # -----------------------------------------
        # Recolher todas as previsões
        # -----------------------------------------
        self.model.eval()
        num_batches = len(self.test_dataloader)
        gt_classes = []
        predicted_classes = []

        print('Evaluating model on test set...')
        for batch_idx, (image_tensor, label_gt_tensor) in tqdm(enumerate(self.test_dataloader), total=num_batches):
            # Verdade
            batch_gt_classes = label_gt_tensor.argmax(dim=1).tolist()
            # Previsão
            label_pred_tensor = self.model.forward(image_tensor)
            label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)
            batch_predicted_classes = label_pred_probabilities_tensor.argmax(dim=1).tolist()

            gt_classes.extend(batch_gt_classes)
            predicted_classes.extend(batch_predicted_classes)

        # -----------------------------------------
        # 1. Matriz de Confusão (Usando sklearn)
        # -----------------------------------------
        confusion_matrix = metrics.confusion_matrix(gt_classes, predicted_classes)
        
        plt.figure(2)
        class_names = [str(i) for i in range(10)]
        seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args['experiment_full_name'], 'confusion_matrix.png'))

        # -----------------------------------------
        # 2. Métricas Avançadas (Precision, Recall, F1)
        # -----------------------------------------
        # O classification_report faz tudo o que o professor pediu:
        # Calcula metricas por classe E a média macro/global
        report = metrics.classification_report(gt_classes, predicted_classes, output_dict=True)
        
        # Imprime bonito no terminal
        print(metrics.classification_report(gt_classes, predicted_classes))

        # Guarda em JSON
        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(report, f, indent=4)
        
        print('Statistics saved to ' + json_filename)