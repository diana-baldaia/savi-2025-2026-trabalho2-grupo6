# savi-2025-2026-trabalho2-grupo5


## Tarefa 1

### 1. Objetivo
Neste trabalho pretende-se desenvolver um classificador de imagens baseado em Redes Neuronais Convolucionais (CNN) para identificar dígitos manuscritos (0 a 9). 
Para a primeira tarefa, o objetivo é otimizar o classificador desenvolvido nas aulas, de modo a lidar com a totalidade do dataset MNIST (60.000 imagens de treino e 10.000 de teste), garantindo uma solução escalável e precisa.


### 2. Metodologia

A metodologia adotada para a Tarefa 1 seguiu um fluxo de trabalho de *Deep Learning* ponta-a-ponta, ou seja, desde o tratamento de dados até à obtenção de métricas que permitam uma análise dos resultados obtidos. Para tal, foram reaproveitados os códigos desenvolvidos em sala de aula (main.py, model.py, trainer.py e dataset.py), a função de cada um e as alterações a que foram sujeitos encontram-se detalhadas de seguida.

#### 2.1. Tratamento de Dados e Pré-processamento (dataset.py)

O tratamento e pré-processamento das imagens e respetivas legendas é feito pela classe Dataset, permitindo tornar o código principal mais limpo e organizado e o processamento mais rápido e eficiente. Esta classe permite obter a imagem e respetiva legenda já processadas e no devido formato exigido pelo PyTorch (tensor) com apenas um índice.
A classe é composta por três funções, __init__, __len__ e __getitem__.

__init__
Esta função é executada uma única vez. O seu objetivo é preparar o índice dos dados, realizando as seguintes tarefas:
- Gestão de Caminhos (paths):
    Verifica se é treino ou teste (is_train) e constrói o caminho para a pasta correta (dataset_folder/train/images ou dataset_folder/test/images).
- Listagem das Imagens:
    glob.glob(...): Procura todos os ficheiros .jpg.
    self.image_filenames.sort(): Ordena os nomes dos ficheiros alfabeticamente. Isto garante que a ordem das imagens corresponde exatamente à ordem das labels no ficheiro de texto, uma vez que este possui em cada linha o nome do ficheiro seguido da respetiva legenda, igualmente organizado alfabeticamente.
- Leitura de Labels:
    Lê o ficheiro labels.txt. Cada linha tem o formato nome_imagem label. O código faz parts[1] para ignorar o nome e pegar apenas na classificação numérica (a legenda).
- Quantidade a usar:
    O código permite ainda definir apenas uma percentagem do dataset a ser usada (args['percentage_examples']). Isto é útil para efetuar testes rápidos ao código sem ter de carregar 60.000 imagens.
    
__len__
Retorna o tamanho total do dataset a ser usado. O DataLoader precisa disto para saber quantas iterações (batches) fará numa época (epoch), ou seja, quantos lotes de imagens terá por cada vez que percorre todo o dataset disponibilizado.

__getitem__
Esta função é chamada repetidamente durante o treino. Cada vez que o DataLoader pede um exemplo, este método é executado para o índice (idx) pedido.

- Processamento da Label (One-Hot Encoding)
    Recebe o índice e obtém a legenda (dígito real) correspondente criando um vetor de 10 posições para dígitos de 0 a 9 (ex.: se o dígito for 2, é colocado '1' na posição correspondente: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]). Esta lista diz à rede que a probabilidade correta para o dígito 2 é 100% (1) e para todos os outros é 0% (0).
    Contudo, o PyTorch não trabalha com listas de números inteiros, mas sim com tensores de números decimais, sendo necessária a sua conversão (label_tensor = torch.tensor(label, dtype=torch.float)).
    Assim, a previsão da rede será algo como [0.01, 0.02, 0.85, ... , 0.01, 0.02], um tensor com as probabilidades previstas, que será comparado com o tensor real de modo a que se obtenha um valor de erro a ser minimizado através da função perda MSE.

- Processamento da Imagem
    Obtém o caminho da imagem correspondente ao índice, abre a imagem e converte-a para escala de cinza (image = Image.open(image_filename).convert('L')), garantindo que a imagem conta com apenas 1 canal relativo a cor.
    A imagem de 28x28 píxeis é também convertida para um tensor float normalizando automaticamente os valores para o intervalo [0, 1].





#### 2.2. Arquitetura da Rede Neuronal (model.py)

A definição da arquitetura da rede neuronal é realizada no ficheiro model.py. Embora o ficheiro contenha versões incrementais (como ModelFullyconnected e ModelConvNet), o trabalho foca-se no desenvolvimento da classe ModelBetterCNN, que herda da classe base nn.Module do PyTorch.

A classe é constituída fundamentalmente por duas funções descritas abaixo: __init__ e forward. Conta ainda com a função getNumberOfParameters que retorna o número total de elementos de todos os tensores de parâmetros da rede que estejam configurados para serem usados pelo otimizador. Esta função permite medir a complexidade do modelo e perceber se está a ocorrer overfitting caso o valor seja absurdamente superior ao número de exemplos.

__init__ 
Esta função estabelece a arquitetura da rede, sendo responsável pela inicialização das camadas com parâmetros treináveis. O objetivo é construir uma arquitetura hierárquica capaz de extrair características visuais complexas.

- Blocos Convolucionais: Foram definidos três blocos sequenciais de extração de características. A profundidade dos filtros aumenta progressivamente (32 → 64 → 128), permitindo que a rede aprenda desde arestas simples nas primeiras camadas até formas complexas nas últimas. Cada bloco é composto por:
    nn.Conv2d: Esta é a camada convolucional e tem como parâmetros: 
    - o número de canais da imagem de entrada, como é em tons de cinza apenas tem 1 canal no primeiro bloco, posteriormente passa a ter 32 e no último 64; 
    - o número de "mapas" de características a obter, 32 no primeiro, 64 no segundo e por fim 128;
    - kernel_size=3, indicando o tamanho da janela que desliza pela imagem, no caso é uma matriz 3x3;
    - padding=1, adiciona uma "moldura" de 1 píxel de zeros à volta da imagem original, de modo a que o resultado seja da mesma dimensão da imagem de entrada (ex.: 28x28 no primeiro bloco)
    nn.BatchNorm2d: Normaliza as saídas de cada canal convolucional recebendo os resultados dos mapas de características e ajustando os valores para que a média seja próxima de 0 e o desvio padrão próximo de 1. Isto estabiliza a distribuição das ativações, permitindo treinos mais rápidos e com taxas de aprendizagem mais elevadas.
    nn.MaxPool2d: Divide a imagem em blocos de 2x2 pixéis (kernel_size=2) e, de cada bloco, escolhe apenas o valor mais alto. O parâmetro stride=2 induz um salto de 2 em 2 pixéis de modo a que não haja sobreposição. Isto permite que a imagem seja reduzida para metade da sua dimensão (28x28 --> 14x14 --> 7x7 --> 3x3). Esta redução não só diminui o custo computacional como também faz com que a rede se foque nas características mais relevantes, ignorando ruído ou variações de posição.

- Classificador: Após a extração de características, define-se o classificador final.
    nn.Linear: Primeiramente, os dados precisam de ser "achatados" (flatten) para poderem passar para a camada Linear. Assim, o primeiro parâmetro é o resultado da multiplicação entre o número de canais que saíram da última camada convolucional (128) e o tamanho final (3x3), resultando em 1152 sinais. Estes sinais são então combinados para formar 256 novos conceitos (neurónios - o segundo parâmetro a indicar).
    nn.Dropout: O Dropout é importante para evitar o Overfitting. Durante o treino ele coloca a zero ("desliga") 50% dos neurónios (parâmetro a definir) de forma aleatória em cada iteração. Isto evita a elevada dependência de apenas alguns neurónios, forçando a rede a encontrar padrões alternativos para chegar à resposta correta. Melhora também a generalização e escalabilidade do modelo.

Por fim, é de novo usado o nn.Linear para receber os 256 conceitos da camada anterior e reduzir para as 10 classes finais.
    
    

__forward__

Esta função define como os dados fluem através da rede, ou seja, conecta as camadas definidas no __init__. É executada sempre que se passa um lote de imagens para a rede.

Pipeline de Ativação: Os dados passam sequencialmente pelos três blocos definidos. Em cada bloco, aplica-se a ordem: Convolução → Batch Normalization → ReLU → Pooling. A função torch.relu(x) é aplicada explicitamente aqui para introduzir não-linearidade no sistema. Sem isto, a rede comportar-se-ia como uma simples regressão linear, independentemente da sua profundidade.

Achatamento (Flatten): A saída do último bloco convolucional é um tensor 3D (128 canais x 3 x 3). A função x.view(-1, ...) transforma este tensor num vetor unidimensional (achatamento), formato necessário para alimentar as camadas densas (Linear) seguintes.

Classificação Final: O vetor resultante passa pela camada densa, pela ativação ReLU e pelo Dropout. Finalmente, a última camada fc2 projeta o resultado em 10 saídas (correspondentes aos dígitos 0-9). Note-se que não se aplica Softmax aqui, pois a função de perda utilizada no treino (CrossEntropy ou similar no PyTorch) geralmente já inclui essa operação internamente ou espera os logits "crus", embora para a inferência final seja interpretado como probabilidades.
  
        
#### 2.3. Otimização e Treino --trainer

### 2.4. main_classification
