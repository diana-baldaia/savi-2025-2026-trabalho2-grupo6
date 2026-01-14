import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import random
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm


# Para correr sem o aviso do NNPACK (apenas após correr sem erros)
# python3 main_improved_detection.py 2>/dev/null


# ==========================================
# ARQUITETURA DO MODELO
# ==========================================
class ModelImproved(nn.Module):
    def __init__(self):
        super(ModelImproved, self).__init__()

        # Primeira camada: Procura padrões simples (linhas, curvas)
        # Normalização (bn1) para o treino ser mais rápido e estável
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Segunda camada: Combina padrões simples em padrões mais complexos
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Reduz o tamanho da imagem para focar no mais importante
        self.pool = nn.MaxPool2d(2, 2)

        # Evita que o modelo decore o dataset (overfitting)
        self.dropout = nn.Dropout(0.25)

        # Camadas finais: decide qual é o número ou se é fundo
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 11)       # 11 saídas (Classes dos dígitos + Classe 10 para fundo)
        self.relu = nn.ReLU()       # Para lógica não-linear

    def forward(self, x):
        # Fluxo dos dados pelas camadas
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)       # Achata (Flatten) a matriz para a camada linear
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# DATASET MELHORADO (HARD NEGATIVES)
# ==========================================
class ComplexSceneDataset(Dataset):
    """Esta classe serve para ensinar o que não é um número"""

    def __init__(self, root_folder, dataset_name, split='train', transform=None):
        self.images_dir = os.path.join(root_folder, dataset_name, split, 'images')
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))
        self.transform = transform
        self.crop_size = 28
        
    def __len__(self):
        # Multiplicar para criar várias amostras de cada imagem
        # O modelo verá 15 exemplos diferentes por imagem em cada época
        return len(self.image_files) * 15

    def __getitem__(self, idx):
        # Seleciona qual imagem abrir com base no índice
        img_idx = idx // 15
        img_path = self.image_files[img_idx % len(self.image_files)]
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        
        # Abre em escala de cinzas
        image = Image.open(img_path).convert('L')
        width, height = image.size
        
        # CLê as coordenadas reais dos números presentes na imagem
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    # Cada linha contém: classe_do_número, x, y, largura, altura
                    gt_boxes.append(list(map(float, line.strip().split())))


        choice = random.random()
        
        # Caso 1: Exemplo Positivo (40% de probabilidade)
        # O objetivo é ensinar a reconhecer o número
        if choice < 0.4 and len(gt_boxes) > 0:
            cls, bx, by, bw, bh = random.choice(gt_boxes)
            
            # Pequeno jitter para não viciar no número sempre centrdo
            offset_x = random.randint(-2, 2)
            offset_y = random.randint(-2, 2)
            cx = int(bx) + offset_x
            cy = int(by) + offset_y
            
            # Garante que o recorte não sai para fora dos limites da imagem
            cx = max(0, min(cx, width - self.crop_size))
            cy = max(0, min(cy, height - self.crop_size))
            
            crop = image.crop((cx, cy, cx + self.crop_size, cy + self.crop_size))
            label = int(cls)

        # Caso 2: Hard Negative (30% de probabilidade)
        # O objetivo é ensinar a não confundir pedaços com o dígito real    
        elif choice < 0.7 and len(gt_boxes) > 0:
            # Seleciona um número real
            target_gt = random.choice(gt_boxes)
            cls, bx, by, bw, bh = target_gt
            
            # Força um corte deslocado (apanha só a ponta)
            shift_x = random.choice([-10, 10])
            shift_y = random.choice([-10, 10])
            rx = int(bx) + shift_x
            ry = int(by) + shift_y
            
            rx = max(0, min(rx, width - self.crop_size))
            ry = max(0, min(ry, height - self.crop_size))
            
            crop = image.crop((rx, ry, rx + self.crop_size, ry + self.crop_size))
            
            # Um pedaço de número é classificado como fundo (10)
            label = 10

        # Caso 3: Fundo puro (30% de probabilidade) 
        # O objetivo é ensinar o modelo a ignorar o VAZIO
        else:
            # Tenta encontrar uma área aleatória que não sobreponha a nenhum número
            for _ in range(20):
                rx = random.randint(0, width - self.crop_size)
                ry = random.randint(0, height - self.crop_size)
                overlap = False
                for gt in gt_boxes:
                    g_cls, gx, gy, gw, gh = gt
                    # Se tocar muito no número, não é fundo puro
                    if (rx < gx + gw and rx + self.crop_size > gx and
                        ry < gy + gh and ry + self.crop_size > gy):
                        overlap = True
                        break
                if not overlap:
                    crop = image.crop((rx, ry, rx + self.crop_size, ry + self.crop_size))
                    label = 10 
                    break
            else:
                crop = Image.new('L', (28, 28), color=0)
                label = 10

        if self.transform:
            crop = self.transform(crop)
        
        return crop, label

# ==========================================
# TREINO DO MODELO
# ==========================================
def train_robot(root_path, dataset_name='Tarefa_2/mnist_detection_D'):
    print(f"\n--- 1. A TREINAR O ROBÔ (COM HARD NEGATIVES) ---")
    # Carregar Dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = ComplexSceneDataset(root_path, dataset_name, 'train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = ModelImproved()
    # O otimizador Adam ajusta a velocidade do treino automaticamente
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Função de perda
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Ciclo de aprendizagem (5 épocas)
    for epoch in range(5): 
        model.train()
        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/5", file=sys.stdout)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()               # Dá reset à memória do otimizador
            outputs = model(images)             # Tenta adivinhar (Forward)
            loss = criterion(outputs, labels)   # Calcula-se o erro
            loss.backward()                     # Aprende com o erro
            optimizer.step()                    # Ajusta os neurónios (step)
            # Estatísticas
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=total_loss/len(train_loader), acc=100*correct/total)

    # Guarda o treino
    save_path = os.path.join(root_path, 'Tarefa_4/Experiments', 'best_improved.pkl')
    torch.save(model.state_dict(), save_path)
    return save_path


# ==========================================
# LIMPEZA DE REDUNDÂNCIAS
# ==========================================

def calculate_iou(box1, box2):
    """
    Mede quão sobrepostas as caixas estão (o--> não se tocam; 1--> sobrepostas)
    """
    x1, y1, s1 = box1[0], box1[1], box1[4]
    x2, y2, s2 = box2[0], box2[1], box2[4]

    # Encontrar as coordenadas do retângulo de interseção
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+s1, x2+s2)
    yi2 = min(y1+s1, y2+s2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Área total das duas menos a parte que se sobrepõe
    box1_area = s1 * s1
    box2_area = s2 * s2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def is_contained(inner_box, outer_box):
    """
    Verifica se uma caixa está 'dentro' de outra. 
    Isto evita que detete o 'buraco do 8' como sendo um '0'.
    """
    ix, iy, isize = inner_box[0], inner_box[1], inner_box[4]
    ox, oy, osize = outer_box[0], outer_box[1], outer_box[4]
    
    xi1 = max(ix, ox)
    yi1 = max(iy, oy)
    xi2 = min(ix+isize, ox+osize)
    yi2 = min(iy+isize, oy+osize)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    inner_area = isize * isize
    
    # Se mais de 80% da caixa pequena está dentro da grande -> True
    if inner_area > 0 and (inter_area / inner_area) > 0.80:
        return True
    return False

def advanced_nms(boxes, iou_threshold=0.2):
    """
    Non-Maximum Suppression (NMS)
    Quando vê o mesmo número 50 vezes, esta função apaga as 49 
    caixas piores e deixa apenas a mais confiante.
    """
    if not boxes: return []
    
    # Ordenar por confiança (do maior para o menor)
    boxes = sorted(boxes, key=lambda x: x[3], reverse=True)
    
    keep = []
    while boxes:
        current = boxes.pop(0)  # Pega na melhor
        keep.append(current)    # Guarda-a como definitiva
        
        rest_boxes = []
        for candidate in boxes:
            # Se o candidato estiver muito em cima da definitiva (IoU alto)
            # ou se estiver contido nela, é descartado
            iou = calculate_iou(current, candidate)
            contained = is_contained(candidate, current) or is_contained(current, candidate)
            
            if iou < iou_threshold and not contained:
                rest_boxes.append(candidate)
        
        boxes = rest_boxes  # Repete para o que resta
        
    return keep



def check_hit_advanced(pred_box, gt_boxes, pred_class):
    # Calcula o centro da caixa detetada
    px, py, size = pred_box
    pcx, pcy = px + size/2, py + size/2
    
    for idx, gt in enumerate(gt_boxes):
        gt_cls, gx, gy, gw, gh = gt
        if int(gt_cls) != pred_class: continue
        # Guarda se o centro da detetada estiver dentro da real
        if (gx <= pcx <= gx + gw) and (gy <= pcy <= gy + gh):
            return idx 
    return -1

# ==========================
# AVALIAÇÃO
# ==========================
def test_robot_metrics(root_path, model_path, dataset_name='Tarefa_2/mnist_detection_D', num_eval=200, num_vis=5):
    print(f"\n--- 2. AVALIAÇÃO FINAL (NMS AVANÇADO) ---")
    
    model = ModelImproved()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    model.eval()    # O modo de avaliação desliga o dropout
    
    test_images_dir = os.path.join(root_path, dataset_name, 'test', 'images')
    all_images = sorted(glob.glob(os.path.join(test_images_dir, "*.jpg")))
    eval_samples = all_images[:num_eval]
    
    to_tensor = transforms.ToTensor()

    # São usadas 3 escalas diferentes para ver números de tamanho diferentes
    SCALES = [28, 36, 48]
    
    total_predictions = 0
    total_real_digits = 0
    total_hits = 0
    
    vis_results = []
    
    print(f"A avaliar {len(eval_samples)} imagens...")
    
    for i, image_path in enumerate(tqdm(eval_samples, file=sys.stdout)):
        full_image = Image.open(image_path).convert('L')
        width, height = full_image.size
        
        # Carrega as labels reais
        label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    gt_boxes.append(list(map(float, line.strip().split())))
        
        total_real_digits += len(gt_boxes)
        raw_detections = []
        
        # Sliding Window com janelas de diferentes escalas
        for current_size in SCALES:
            STRIDE = 6 
            for y in range(0, height - current_size + 1, STRIDE):
                for x in range(0, width - current_size + 1, STRIDE):
                    crop = full_image.crop((x, y, x + current_size, y + current_size))
                    crop_resized = crop.resize((28, 28), Image.BILINEAR)
                    input_tensor = to_tensor(crop_resized).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        max_prob, predicted_class = torch.max(probs, 1)
                        label = predicted_class.item()
                        conf = max_prob.item()
                        
                        # Threshold mais alto para reduzir falsos positivos
                        if label < 10 and conf > 0.98:
                            raw_detections.append((x, y, label, conf, current_size))

        # NMS: Limpa as deteções repetidas
        final_detections = advanced_nms(raw_detections, iou_threshold=0.1) # IoU muito agressivo
        total_predictions += len(final_detections)
        
        # Verifica se o previsto corresponde ao real
        matched_gt_indices = set()
        for det in final_detections:
            x, y, label, conf, size = det
            pred_box = (x, y, size)
            hit_idx = check_hit_advanced(pred_box, gt_boxes, label)
            if hit_idx != -1:
                matched_gt_indices.add(hit_idx)
        
        total_hits += len(matched_gt_indices)

        # Guarda algumas imagens
        if i < num_vis:
            vis_results.append((full_image, final_detections, os.path.basename(image_path)))

    # Cálculo de métricas
    precision = (total_hits / total_predictions * 100) if total_predictions > 0 else 0
    recall = (total_hits / total_real_digits * 100) if total_real_digits > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*40)
    print(f" RELATÓRIO OFICIAL (HARD NEGATIVES + ADVANCED NMS)")
    print("="*40)
    print(f"Total Números Reais:  {total_real_digits}")
    print(f"Total Deteções:       {total_predictions}")
    print(f"Números Encontrados:  {total_hits}")
    print("-" * 40)
    print(f"PRECISÃO: {precision:.2f}%")
    print(f"RECALL:   {recall:.2f}%")
    print(f"F1-SCORE: {f1:.2f}%")
    print("="*40)

    fig, axes = plt.subplots(1, num_vis, figsize=(15, 5))
    if num_vis == 1: axes = [axes]
    
    for idx, (img, boxes, name) in enumerate(vis_results):
        ax = axes[idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(name)
        ax.axis('off')
        for (x, y, label, prob, size) in boxes:
            rect = patches.Rectangle((x, y), size, size, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f"{label}", color='lime', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig("final_metrics_visualization.png")
    plt.show()

if __name__ == "__main__":
    root = '/home/baldaia/Desktop/savi-2025-2026-trabalho2-grupo5/' 
    
    # IMPORTANTE: VOLTA A TREINAR (Descomenta) para aprender os Hard Negatives!
    # model_path = train_robot(root, dataset_name='Tarefa_2/mnist_detection_D')
    
    # Se já tiveres treinado e quiseres só testar, comenta a linha de cima e usa esta:
    model_path = os.path.join(root, 'Tarefa_4/Experiments', 'best_improved.pkl')

    test_robot_metrics(root, model_path, dataset_name='Tarefa_2/mnist_detection_D', num_eval=200, num_vis=5)