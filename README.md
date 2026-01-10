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
    
    Blocos Convolucionais: Foram definidos três blocos sequenciais de extração de características. A profundidade dos filtros aumenta progressivamente (32 → 64 → 128), permitindo que a rede aprenda desde arestas simples nas primeiras camadas até formas complexas nas últimas.
    Cada bloco é composto por:

    nn.Conv2d: Esta é a camada convolucional e tem como parâmetros: 
    - o número de canais da imagem de entrada, como é em tons de cinza apenas tem 1 canal no primeiro bloco, posteriormente passa a ter 32 e no último 64; 
    - o número de "mapas" de características a obter, 32 no primeiro, 64 no segundo e por fim 128;
    - kernel_size=3, indicando o tamanho da janela que desliza pela imagem, no caso é uma matriz 3x3;
    - padding=1, adiciona uma "moldura" de 1 píxel de zeros à volta da imagem original, de modo a que o resultado seja da mesma dimensão da imagem de entrada (ex.: 28x28 no primeiro bloco)

    nn.BatchNorm2d: Normaliza as saídas de cada canal convolucional recebendo os resultados dos mapas de características e ajustando os valores para que a média seja próxima de 0 e o desvio padrão próximo de 1. Isto estabiliza a distribuição das ativações, permitindo treinos mais rápidos e com taxas de aprendizagem mais elevadas.

    nn.MaxPool2d: Divide a imagem em blocos de 2x2 pixéis (kernel_size=2) e, de cada bloco, escolhe apenas o valor mais alto. O parâmetro stride=2 induz um salto de 2 em 2 pixéis de modo a que não haja sobreposição. Isto permite que a imagem seja reduzida para metade da sua dimensão (28x28 --> 14x14 --> 7x7 --> 3x3). Esta redução não só diminui o custo computacional como também faz com que a rede se foque nas características mais relevantes, ignorando ruído ou variações de posição.


    Classificador: Após a extração de características, define-se o classificador final.

    nn.Linear: Primeiramente, os dados precisam de ser "achatados" (flatten) para poderem passar para a camada Linear. Assim, o primeiro parâmetro é o resultado da multiplicação entre o número de canais que saíram da última camada convolucional (128) e o tamanho final (3x3), resultando em 1152 sinais. Estes sinais são então combinados para formar 256 novos conceitos (neurónios - o segundo parâmetro a indicar).

    nn.Dropout: O Dropout é importante para evitar o Overfitting. Durante o treino ele coloca a zero ("desliga") 50% dos neurónios (parâmetro a definir) de forma aleatória em cada iteração. Isto evita a elevada dependência de apenas alguns neurónios, forçando a rede a encontrar padrões alternativos para chegar à resposta correta. Melhora também a generalização e escalabilidade do modelo.
    
    Por fim, é de novo usado o nn.Linear para receber os 256 conceitos da camada anterior e reduzir para as 10 classes finais.
    

__forward__
Esta função define como os dados fluem através da rede, ou seja, conecta as camadas definidas no __init__. É executada sempre que se passa um lote de imagens para a rede e conta com os seguintes processos:

    Sequência de Ativação: Os dados passam sequencialmente pelos três blocos definidos. Em cada bloco, aplica-se a ordem: Convolução → Batch Normalization → ReLU → Pooling. A função torch.relu(x) é aplicada explicitamente aqui para introduzir não-linearidade no sistema. Sem isto, a rede comportar-se-ia como uma simples regressão linear, incapaz de combinar os sinais de forma inteligente para detetar padrões complexos.

    Achatamento (Flatten): A saída do último bloco convolucional é um tensor 3D (128 canais x 3 x 3). A função x.view(-1, ...) transforma este tensor num vetor unidimensional (achatamento), formato necessário para alimentar as camadas densas (Linear) seguintes.

    Classificação Final: O vetor resultante passa pela camada densa (fc1), pela ativação ReLU e pelo Dropout. Finalmente, a última camada fc2 projeta o resultado em 10 saídas (correspondentes aos dígitos 0-9). 

        
#### 2.3. Ciclo de Treino (trainer.py)
O processo de aprendizagem é realizado através da classe Trainer. Esta classe não herda de módulos do PyTorch, funcionando antes como um controlador que integra os dados (datasets), o modelo (model) e as regras de otimização. A classe é estruturada em torno das funções principais __init__ e train, métodos auxiliares e visualização (saveTrain, loadTrain, draw) e, finalmente, a função de avaliação evaluate.

__init__ 
Este método estabelece a infraestrutura necessária para o ciclo de treino, inicializando os componentes responsáveis pelo processamento dos dados e pela quantificação do erro.

    DataLoader: A classe torch.utils.data.DataLoader é fundamental para a gestão eficiente da memória. Em vez de carregar todas as imagens de uma vez, ela fornece os dados em pequenos lotes (batches) definidos pelo argumento batch_size. Para o conjunto de treino, utiliza-se shuffle=True, o que baralha os dados a cada época (epoch). Isto é crucial para garantir que o modelo não memoriza a ordem das imagens e para que os gradientes sejam mais representativos da distribuição geral dos dados. Para o conjunto de teste, o shuffle=False mantém a ordem, permitindo uma avaliação determinística.

    nn.MSELoss: Define a função de custo (Loss Function). Neste caso, optou-se pelo Erro Quadrático Médio (Mean Squared Error) que calcula a média das diferenças ao quadrado entre o vetor de probabilidade previsto pela rede e o vetor real. O objetivo do treino será minimizar este valor.

    torch.optim.Adam: A otimização é feita através do algoritmo Adam (Adaptive Moment Estimation) que atualiza os pesos da rede com base nos gradientes calculados. A taxa de aprendizagem (lr=0.001) define o tamanho do passo que o otimizador dá em direção ao mínimo do erro; um valor equilibrado é essencial para evitar que o treino estagne ou oscile excessivamente.


__train__ 
Esta função contém o ciclo principal (Loop) de treino e validação, iterando pelo número de épocas definido (num_epochs).

    Modo de Treino e Backpropagation: Ao invocar self.model.train(), a rede ativa camadas específicas como Dropout e Batch Normalization. Para cada lote de imagens realiza-se a previsão (forward), calcula-se o erro (loss), o loss.backward() calcula os gradientes (a direção em que os pesos devem mudar) e o optimizer.step() atualiza efetivamente os pesos.

    Modo de Avaliação: Em cada época, o modelo é também testado (self.model.eval()) com dados que nunca viu. Aqui, o cálculo de gradientes é desligado para poupar memória, servindo apenas para monitorizar a capacidade de generalização da rede.

    Monitorização (Draw e Save): No final de cada época, a função draw utiliza o matplotlib para atualizar o gráfico das curvas de perda (treino vs teste), permitindo detetar visualmente fenómenos de Overfitting (quando o erro de treino desce, mas o de teste sobe). Simultaneamente, o saveTrain guarda o estado do modelo (checkpoint.pkl) e, caso o erro de teste seja o menor histórico, guarda uma cópia de "melhor modelo" (best.pkl).


__evaluate__ 
Executada automaticamente após o fim do treino, esta função utiliza a biblioteca sklearn.metrics para uma análise rigorosa da performance do modelo, indo além da simples precisão.

    Agregação de Previsões: A função percorre todo o conjunto de teste para compilar dois vetores globais: as classes reais (gt_classes) e as classes previstas (predicted_classes), obtidas através da função argmax que seleciona o neurónio com maior ativação na saída.

    Matriz de Confusão: Utiliza-se a metrics.confusion_matrix em conjunto com a biblioteca seaborn para gerar um mapa de calor (Heatmap). Esta visualização permite identificar quais as classes que o modelo confunde entre si (ex.: se o dígito 3 é frequentemente classificado como 8).

    Relatório de Métricas: A função metrics.classification_report calcula métricas detalhadas para cada classe individualmente e a sua média global (macro average):
    -Precision: De todas as vezes que a rede previu o dígito X, quantas vezes estava correta?
    -Recall: De todos os dígitos X que realmente existiam, quantos a rede encontrou?
    -F1-Score: Uma média harmónica entre Precision e Recall, ideal para avaliar o equilíbrio do modelo. Estes dados são exportados para um ficheiro JSON (statistics.json).


#### 2.4. Execução (main_classification.py)
Este módulo assume um papel central na coordenação do sistema: processa os argumentos de entrada, prepara o ambiente experimental, instancia as classes fundamentais (Dataset, Modelo, Trainer) e despoleta o ciclo de treino inicia o processo de aprendizagem.

O código está estruturado numa função main() e conta com um mecanismo de interrupção (sigintHandler) para garantir que o programa encerra corretamente caso o utilizador pressione Ctrl+C, evitando a corrupção de ficheiros ou logs pendentes.

Os principais argumentos definidos (através do argparse) são:
    --num_epochs: O número de vezes que a rede verá o dataset completo (ciclos de treino).
    --batch_size: A quantidade de imagens processadas simultaneamente antes de uma atualização de pesos.
    --experiment_path: O diretório onde serão guardados os resultados (gráficos, checkpoints e estatísticas).
    --resume_training: Uma flag booleana que, se ativada, instrui o sistema a procurar um checkpoint existente e continuar o treino a partir desse ponto.

Antes de iniciar o processamento, o script prepara o sistema de ficheiros. Utiliza a função os.makedirs para garantir que o diretório de destino (experiment_full_name) existe. Isto garante que os ficheiros gerados (gráficos de perda, matrizes de confusão) ficam salvaguardados e organizados como pretendido.

Posteriormente, são defenidos os datasets de treino e teste através da classe Dataset e é elecionado o modelo a utilizar, neste caso, o ModelBetterCNN.

Após isto realiza-se a verificação de integridade (Sanity Check), na qual o código executa um teste preliminar manual. Extrai-se uma única imagem do dataset e aplica-se a função unsqueeze(0). Este passo é essencial uma vez que as redes em PyTorch esperam receber dados em lotes com 4 dimensões (Batch, Canais, Altura, Largura), mas uma imagem isolada tem apenas 3. O unsqueeze adiciona artificialmente a dimensão do lote (tamanho 1). A imagem é passada pela função forward do modelo. Se este passo não gerar erros de dimensão ou memória, confirma-se que a arquitetura está compatível com os dados de entrada.

Por fim, dá-se início à classe Trainer, recebendo os argumentos, os datasets e o modelo. O método trainer.train() é invocado para iniciar o loop de aprendizagem. Após a conclusão das épocas, o método trainer.evaluate() é chamado explicitamente para correr o modelo no conjunto de teste e gerar as métricas finais de desempenho (Precisão, Recall, F1-Score e Matriz de Confusão).
