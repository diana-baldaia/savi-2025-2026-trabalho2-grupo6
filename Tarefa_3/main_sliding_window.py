
import torch
import torchvision.transforms as transforms
import os
import sys
from model import ModelBetterCNN
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm

# Para correr sem o aviso do NNPACK
# python3 main_sliding_window.py 2>/dev/null

def nms(boxes, iou_threshold):
    """
    Non-Maximum Suppression: Remove caixas sobrepostas que detetam o mesmo objeto.
    boxes: lista de (x, y, w, h, label, score)
    """
    if not boxes:
        return []

    # Converter para formato utilizável (x1, y1, x2, y2, score)
    b = np.array([[box[0], box[1], box[0]+box[2], box[1]+box[3], box[5]] for box in boxes])
    labels = np.array([box[4] for box in boxes])
    
    x1, y1, x2, y2, scores = b[:,0], b[:,1], b[:,2], b[:,3], b[:,4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return [boxes[i] for i in keep]


def detect_and_save(image_path, model, device, output_path, stride=4, threshold=0.98):
    """Processa uma única imagem e guarda o resultado visual."""
    WINDOW_SIZE = 28
    full_image = Image.open(image_path).convert('L')
    width, height = full_image.size
    to_tensor = transforms.ToTensor()
    
    raw_detections = []
    
    # Sliding Window
    for y in range(0, height - WINDOW_SIZE + 1, stride):
        for x in range(0, width - WINDOW_SIZE + 1, stride):
            crop = full_image.crop((x, y, x + WINDOW_SIZE, y + WINDOW_SIZE))
            crop_tensor = to_tensor(crop).unsqueeze(0).to(device)

            if crop_tensor.std() < 0.1: continue        # Ignora zonas vazias (pretas)

            # Os recortes são submetidos ao modelo ModelBetterCNN
            with torch.no_grad():
                output = model(crop_tensor)
                probs = torch.softmax(output, dim=1)
                max_prob, pred = torch.max(probs, 1)

                # Apenas guarda deteções acima do threshold
                if max_prob.item() > threshold:
                    raw_detections.append((x, y, WINDOW_SIZE, WINDOW_SIZE, pred.item(), max_prob.item()))

    # Aplica Non-Maximum Suppression para filtrar deteções redundantes
    final_detections = nms(raw_detections, iou_threshold=0.2)

    # Visualização e Gravação
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(full_image, cmap='gray')
    for (x, y, w, h, label, prob) in final_detections:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, f"{label}({prob:.2f})", color='lime', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    plt.title(f"Detection: {os.path.basename(image_path)}")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig) # IMPORTANTE: Fecha o gráfico para não encher a memória RAM


# --- SCRIPT PRINCIPAL ---

def process_full_dataset(base_data_path, model_path, versions=['A', 'D'], num_images_per_ver=20):

    # Carregar Modelo
    model = ModelBetterCNN()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    # Criar pasta raiz para resultados
    results_root = "Tarefa_3_Results"
    os.makedirs(results_root, exist_ok=True)

    for ver in versions:
        print(f"\n--- Processando Versão {ver} ---")
        img_dir = os.path.join(base_data_path, f"mnist_detection_{ver}", "test", "images")
        save_dir = os.path.join(results_root, f"detections_{ver}")
        os.makedirs(save_dir, exist_ok=True)

        if not os.path.exists(img_dir):
            print(f"Aviso: Pasta {img_dir} não encontrada. Ignorando...")
            continue

        # Listar imagens e limitar quantidade
        all_images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        images_to_process = all_images[:num_images_per_ver]

        for img_name in tqdm(images_to_process, desc=f"Versão {ver}", file=sys.stdout):
            input_path = os.path.join(img_dir, img_name)
            output_path = os.path.join(save_dir, f"result_{img_name}")
            detect_and_save(input_path, model, 'cpu', output_path, stride=6, threshold=0.98)

    print(f"\nConcluído! Resultados guardados em: {results_root}")

if __name__ == "__main__":
    # CONFIGURAÇÕES
    MODEL_FILE = '/home/baldaia/Desktop/savi-2025-2026-trabalho2-grupo5/Tarefa_1/Experiments_MBCNN/best.pkl' # Caminho do teu modelo
    DATASET_PATH = '/home/baldaia/Desktop/savi-2025-2026-trabalho2-grupo5/Tarefa_2'                         # Pasta onde estão as subpastas mnist_detection_X

    # Corre para todas as versões, processando 20 imagens de cada para teste rápido
    process_full_dataset(DATASET_PATH, MODEL_FILE, versions=['A', 'D'], num_images_per_ver=20)