# savi-2025-2026-trabalho2-grupo6


## Tarefa 1

### 1. Objetivo
Neste trabalho pretende-se desenvolver um classificador de imagens baseado em Redes Neuronais Convolucionais (CNN) para identificar dígitos manuscritos (0 a 9). 

Para a primeira tarefa, o objetivo é otimizar o classificador desenvolvido nas aulas, de modo a lidar com a totalidade do _dataset_ MNIST (60.000 imagens de treino e 10.000 de teste), garantindo uma solução escalável e precisa.


### 2. Metodologia

A metodologia adotada para a Tarefa 1 seguiu um fluxo de trabalho de *Deep Learning* ponta-a-ponta, ou seja, desde o tratamento de dados até à obtenção de métricas que permitam uma análise dos resultados obtidos. Para tal, foram reaproveitados os códigos desenvolvidos em sala de aula (_main.py, model.py, trainer.py e dataset.py_), a função de cada um e as alterações a que foram sujeitos encontram-se detalhadas de seguida.

#### 2.1. Tratamento de Dados e Pré-processamento (_dataset.py_)

O tratamento e pré-processamento das imagens e respetivas legendas é feito pela classe _Dataset_, permitindo tornar o código principal mais limpo e organizado e o processamento mais rápido e eficiente. Esta classe permite obter a imagem e respetiva legenda já processadas e no devido formato exigido pelo PyTorch (tensor) com apenas um índice.

A classe é composta por três funções, `__init__`, `__len__` e `__getitem__`.

`__init__`

Esta função é executada uma única vez. O seu objetivo é preparar o índice dos dados, realizando as seguintes tarefas:
- Gestão de Caminhos (_paths_):
    Verifica se é treino ou teste (_is_train_) e constrói o caminho para a pasta correta (_dataset_folder/train/images ou dataset_folder/test/images_).
- Listagem das Imagens:
    _glob.glob()_: Procura todos os ficheiros .jpg.
    _self.image_filenames.sort()_: Ordena os nomes dos ficheiros alfabeticamente. Isto garante que a ordem das imagens corresponde exatamente à ordem das _labels_ no ficheiro de texto, uma vez que este possui em cada linha o nome do ficheiro seguido da respetiva legenda, igualmente organizado alfabeticamente.
- Leitura de _labels_:
    Lê o ficheiro labels.txt. Cada linha tem o formato nome_imagem _label_. O código faz parts[1] para ignorar o nome e pegar apenas na classificação numérica (a legenda).
- Quantidade a usar:
    O código permite ainda definir apenas uma percentagem do _dataset_ a ser usada (args['percentage_examples']). Isto é útil para efetuar testes rápidos ao código sem ter de carregar 60.000 imagens.
    
`__len__`

Retorna o tamanho total do _dataset_ a ser usado. O _DataLoader_ precisa disto para saber quantas iterações (_batches_) fará numa época (_epoch_), ou seja, quantos lotes de imagens terá por cada vez que percorre todo o _dataset_ disponibilizado.

`__getitem__`

Esta função é chamada repetidamente durante o treino. Cada vez que o _DataLoader_ pede um exemplo, este método é executado para o índice (_idx_) pedido.

- Processamento da _Label_ (_One-Hot Encoding_)
    Recebe o índice e obtém a legenda (dígito real) correspondente criando um vetor de 10 posições para dígitos de 0 a 9 (ex.: se o dígito for 2, é colocado '1' na posição correspondente: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]). Esta lista diz à rede que a probabilidade correta para o dígito 2 é 100% (1) e para todos os outros é 0% (0).
    Contudo, o _PyTorch_ não trabalha com listas de números inteiros, mas sim com tensores de números decimais, sendo necessária a sua conversão (`label_tensor = torch.tensor(label, dtype=torch.float)`).
    Assim, a previsão da rede será algo como [0.01, 0.02, 0.85, ... , 0.01, 0.02], um tensor com as probabilidades previstas, que será comparado com o tensor real de modo a que se obtenha um valor de erro a ser minimizado através da função perda MSE.

- Processamento da Imagem
    Obtém o caminho da imagem correspondente ao índice, abre a imagem e converte-a para escala de cinza (`image = Image.open(image_filename).convert('L')`), garantindo que a imagem conta com apenas 1 canal relativo a cor.
    A imagem de 28x28 píxeis é também convertida para um tensor float normalizando automaticamente os valores para o intervalo [0, 1].


#### 2.2. Arquitetura da Rede Neuronal (_model.py_)

A definição da arquitetura da rede neuronal é realizada no ficheiro _model.py_. Embora o ficheiro contenha versões incrementais (como _ModelFullyconnected_ e _ModelConvNet_), o trabalho foca-se no desenvolvimento da classe ___ModelBetterCNN___, que herda da classe base _nn.Module_ do _PyTorch_.

A classe é constituída fundamentalmente por duas funções descritas abaixo: `__init__` e `forward`. Conta ainda com a função `getNumberOfParameters` que retorna o número total de elementos de todos os tensores de parâmetros da rede que estejam configurados para serem usados pelo otimizador. Esta função permite medir a complexidade do modelo e perceber se está a ocorrer _overfitting_ caso o valor seja absurdamente superior ao número de exemplos.

`__init__` 

Esta função estabelece a arquitetura da rede, sendo responsável pela inicialização das camadas com parâmetros treináveis. O objetivo é construir uma arquitetura hierárquica capaz de extrair características visuais complexas.
    
**Blocos Convolucionais**: Foram definidos três blocos sequenciais de extração de características. A profundidade dos filtros aumenta progressivamente (32 → 64 → 128), permitindo que a rede aprenda desde arestas simples nas primeiras camadas até formas complexas nas últimas.
Cada bloco é composto por:

- **nn.Conv2d**: Esta é a camada convolucional e tem como parâmetros: 
-- o número de canais da imagem de entrada, como é em tons de cinza apenas tem 1 canal no primeiro bloco, posteriormente passa a ter 32 e no último 64; 
-- o número de "mapas" de características a obter, 32 no primeiro, 64 no segundo e por fim 128;
-- `kernel_size=3`, indicando o tamanho da janela que desliza pela imagem, no caso é uma matriz 3x3;
-- `padding=1`, adiciona uma "moldura" de 1 píxel de zeros à volta da imagem original, de modo a que o resultado seja da mesma dimensão da imagem de entrada (ex.: 28x28 no primeiro bloco)

- **nn.BatchNorm2d**: Normaliza as saídas de cada canal convolucional recebendo os resultados dos mapas de características e ajustando os valores para que a média seja próxima de 0 e o desvio padrão próximo de 1. Isto estabiliza a distribuição das ativações, permitindo treinos mais rápidos e com taxas de aprendizagem mais elevadas.

- **nn.MaxPool2d**: Divide a imagem em blocos de 2x2 pixéis (`kernel_size=2`) e, de cada bloco, escolhe apenas o valor mais alto. O parâmetro stride=2 induz um salto de 2 em 2 pixéis de modo a que não haja sobreposição. Isto permite que a imagem seja reduzida para metade da sua dimensão (28x28 --> 14x14 --> 7x7 --> 3x3). Esta redução não só diminui o custo computacional como também faz com que a rede se foque nas características mais relevantes, ignorando ruído ou variações de posição.


**Classificador**: Após a extração de características, define-se o classificador final.
- **nn.Linear**: Primeiramente, os dados precisam de ser "achatados" (_flatten_) para poderem passar para a camada Linear. Assim, o primeiro parâmetro é o resultado da multiplicação entre o número de canais que saíram da última camada convolucional (128) e o tamanho final (3x3), resultando em 1152 sinais. Estes sinais são então combinados para formar 256 novos conceitos (neurónios - o segundo parâmetro a indicar).
- **nn.Dropout**: O _Dropout_ é importante para evitar o _Overfitting_. Durante o treino ele coloca a zero ("desliga") 50% dos neurónios (parâmetro a definir) de forma aleatória em cada iteração. Isto evita a elevada dependência de apenas alguns neurónios, forçando a rede a encontrar padrões alternativos para chegar à resposta correta. Melhora também a generalização e escalabilidade do modelo.

Por fim, é de novo usado o _nn.Linear_ para receber os 256 conceitos da camada anterior e reduzir para as 10 classes finais.
    

`forward`

Esta função define como os dados fluem através da rede, ou seja, conecta as camadas definidas no `__init__. É executada sempre que se passa um lote de imagens para a rede e conta com os seguintes processos:

- Sequência de Ativação: Os dados passam sequencialmente pelos três blocos definidos. Em cada bloco, aplica-se a ordem: **Convolução → _Batch Normalization_ → _ReLU_ → _Pooling_**. A função `torch.relu()` é aplicada explicitamente aqui para introduzir não-linearidade no sistema. Sem isto, a rede comportar-se-ia como uma simples regressão linear, incapaz de combinar os sinais de forma inteligente para detetar padrões complexos.
- Achatamento (_Flatten_): A saída do último bloco convolucional é um tensor 3D (128 canais x 3 x 3). A função `x.view(-1, ...)` transforma este tensor num vetor unidimensional (achatamento), formato necessário para alimentar as camadas densas (Linear) seguintes.
- Classificação Final: O vetor resultante passa pela camada densa (fc1), pela ativação _ReLU_ e pelo _Dropout_. Finalmente, a última camada fc2 projeta o resultado em 10 saídas (correspondentes aos dígitos 0-9). 

        
#### 2.3. Ciclo de Treino (_trainer.py_)
O processo de aprendizagem é realizado através da classe _Trainer_. Esta classe não herda de módulos do _PyTorch_, funcionando antes como um controlador que integra os dados (_datasets_), o modelo (_model_) e as regras de otimização. A classe é estruturada em torno das funções principais `__init__` e `train`, métodos auxiliares e visualização (`saveTrain`, `loadTrain`, `draw`) e, finalmente, a função de avaliação `evaluate`.

`__init__`

Este método estabelece a infraestrutura necessária para o ciclo de treino, inicializando os componentes responsáveis pelo processamento dos dados e pela quantificação do erro.
A classe `torch.utils.data.DataLoader` é fundamental para a gestão eficiente da memória. Em vez de carregar todas as imagens de uma vez, ela fornece os dados em pequenos lotes (_batches_) definidos pelo argumento `batch_size`. 

Para o conjunto de treino, utiliza-se `shuffle=True`, o que baralha os dados a cada época (_epoch_). Isto é crucial para garantir que o modelo não memoriza a ordem das imagens e para que os gradientes sejam mais representativos da distribuição geral dos dados. 

Para o conjunto de teste, o `shuffle=False` mantém a ordem, permitindo uma avaliação determinística.

- _nn.MSELoss_: Define a função de custo (_Loss Function_). Neste caso, optou-se pelo Erro Quadrático Médio (_Mean Squared Error_) que calcula a média das diferenças ao quadrado entre o vetor de probabilidade previsto pela rede e o vetor real. O objetivo do treino será minimizar este valor.

- _torch.optim.Adam_: A otimização é feita através do algoritmo Adam (_Adaptive Moment Estimation_) que atualiza os pesos da rede com base nos gradientes calculados. A taxa de aprendizagem (lr=0.001) define o tamanho do passo que o otimizador dá em direção ao mínimo do erro; um valor equilibrado é essencial para evitar que o treino estagne ou oscile excessivamente.


`train`

Esta função contém o ciclo principal (_Loop)_ de treino e validação, iterando pelo número de épocas definido (_num_epochs_).

- **Modo de Treino e _Backpropagation_**: Ao invocar `self.model.train()`, a rede ativa camadas específicas como _Dropout_ e _Batch Normalization_. Para cada lote de imagens realiza-se a previsão (_forward_), calcula-se o erro (_loss_), o `loss.backward()` calcula os gradientes (a direção em que os pesos devem mudar) e o `optimizer.step()` atualiza efetivamente os pesos.

- **Modo de Avaliação**: Em cada época, o modelo é também testado (`self.model.eval()`) com dados que nunca viu. Aqui, o cálculo de gradientes é desligado para poupar memória, servindo apenas para monitorizar a capacidade de generalização da rede.

- **Monitorização (_Draw_ e _Save_)**: No final de cada época, a função `draw` utiliza o _matplotlib_ para atualizar o gráfico das curvas de perda (treino vs teste), permitindo detetar visualmente fenómenos de _Overfitting_ (quando o erro de treino desce, mas o de teste sobe). Simultaneamente, o `saveTrain` guarda o estado do modelo (_checkpoint.pkl_) e, caso o erro de teste seja o menor histórico, guarda uma cópia de "melhor modelo" (_best.pkl_).


`evaluate`

Executada automaticamente após o fim do treino, esta função utiliza a biblioteca `sklearn.metrics` para uma análise rigorosa da performance do modelo, indo além da simples precisão.

- **Agregação de Previsões**: A função percorre todo o conjunto de teste para compilar dois vetores globais: as classes reais (_gt_classes_) e as classes previstas (_predicted_classes_), obtidas através da função _argmax_ que seleciona o neurónio com maior ativação na saída.

- **Matriz de Confusão**: Utiliza-se a `metrics.confusion_matrix` em conjunto com a biblioteca _seaborn_ para gerar um mapa de calor (_Heatmap_). Esta visualização permite identificar quais as classes que o modelo confunde entre si (ex.: se o dígito 3 é frequentemente classificado como 8).

- **Relatório de Métricas**: A função `metrics.classification_report` calcula métricas detalhadas para cada classe individualmente e a sua média global (_macro average_):
-- _Precision_: De todas as vezes que a rede previu o dígito X, quantas vezes estava correta?
-- _Recall_: De todos os dígitos X que realmente existiam, quantos a rede encontrou?
-- _F1-Score_: Uma média harmónica entre _Precision_ e _Recall_, ideal para avaliar o equilíbrio do modelo. Estes dados são exportados para um ficheiro JSON (_statistics.json_).


#### 2.4. Execução (_main_classification.py_)
Este módulo assume um papel central na coordenação do sistema: processa os argumentos de entrada, prepara o ambiente experimental, instancia as classes fundamentais (_Dataset, Modelo, Trainer_) e despoleta o ciclo de treino iniciando o processo de aprendizagem.

O código está estruturado numa função `main()` e conta com um mecanismo de interrupção (_sigintHandler_) para garantir que o programa encerra corretamente caso o utilizador pressione Ctrl+C, evitando a corrupção de ficheiros ou _logs_ pendentes.

Os principais argumentos definidos (através do _argparse_) são:
--_num_epochs_: O número de vezes que a rede verá o _dataset_ completo (ciclos de treino).
--_batch_size_: A quantidade de imagens processadas simultaneamente antes de uma atualização de pesos.
--_experiment_path_: O diretório onde serão guardados os resultados (gráficos, _checkpoints_ e estatísticas).
--_resume_training_: Uma _flag_ booleana que, se ativada, instrui o sistema a procurar um _checkpoint_ existente e continuar o treino a partir desse ponto.

Antes de iniciar o processamento, o _script_ prepara o sistema de ficheiros. Utiliza a função _os.makedirs_ para garantir que o diretório de destino (`experiment_full_name`) existe. Isto garante que os ficheiros gerados (gráficos de perda, matrizes de confusão) ficam salvaguardados e organizados como pretendido.

Posteriormente, são defenidos os _datasets_ de treino e teste através da classe _Dataset_ e é elecionado o modelo a utilizar, neste caso, o _ModelBetterCNN_.

Após isto realiza-se a verificação de integridade (_Sanity Check_), na qual o código executa um teste preliminar manual. Extrai-se uma única imagem do _dataset_ e aplica-se a função `unsqueeze(0)`. Este passo é essencial uma vez que as redes em _PyTorch_ esperam receber dados em lotes com 4 dimensões (_Batch_, Canais, Altura, Largura), mas uma imagem isolada tem apenas 3. O _unsqueeze_ adiciona artificialmente a dimensão do lote (tamanho 1). A imagem é passada pela função _forward_ do modelo. Se este passo não gerar erros de dimensão ou memória, confirma-se que a arquitetura está compatível com os dados de entrada.

Por fim, dá-se início à classe _Trainer_, recebendo os argumentos, os _datasets_ e o modelo. O método `trainer.train()` é invocado para iniciar o _loop_ de aprendizagem. Após a conclusão das épocas, o método `trainer.evaluate()` é chamado explicitamente para correr o modelo no conjunto de teste e gerar as métricas finais de desempenho (Precisão, _Recall, F1-Score_ e Matriz de Confusão).

### 3. Resultados
Nesta secção, apresentam-se os resultados obtidos durante a fase de treino e teste. A análise foca-se sobretudo no desempenho do modelo final proposto (_ModelBetterCNN_), seguida de um estudo comparativo com os modelos iterativas anteriores para demonstrar a evolução do desempenho.

#### 3.1. _ModelBetterCNN_
O processo de treino foi monitorizado ao longo de 10 épocas. A figura abaixo ilustra a evolução da função de perda (_Loss_) nos conjuntos de treino e de teste.

<div align="center">
  <img src="https://github.com/user-attachments/assets/6e692fb6-6ca8-47e1-b0a6-740207e494b7" width="500">
  <br>
  <em>Figura 1: Evolução do erro (Loss) durante as épocas de treino e teste.</em>
  <br><br>
</div>


Observa-se uma convergência rápida e estável do modelo. O erro no conjunto de teste (linha azul) acompanha a descida do erro de treino (linha vermelha), estabilizando em valores próximos de zero. A ausência de uma divergência significativa entre as duas curvas indica que o modelo possui uma boa capacidade de generalização e que não ocorreu _Overfitting_ significativo. O melhor modelo (indicado pela linha tracejada verde) foi obtido na época 8, onde o erro de teste atingiu o seu mínimo global.

A avaliação final no conjunto de teste, detalhada na tabela seguinte, revela um ótimo desempenho da rede, alcançando uma exatidão (_Accuracy_) global de 99%.

| Classe (Dígito) | Precision | Recall | F1-Score | Suporte |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0.99 | 1.00 | 0.99 | 980 |
| 1 | 1.00 | 0.99 | 1.00 | 1135 |
| 2 | 1.00 | 0.99 | 1.00 | 1032 |
| 3 | 0.99 | 1.00 | 0.99 | 1010 |
| 4 | 1.00 | 0.99 | 0.99 | 982 |
| 5 | 0.99 | 0.99 | 0.99 | 892 |
| 6 | 0.99 | 0.99 | 0.99 | 958 |
| 7 | 0.99 | 0.99 | 0.99 | 1028 |
| 8 | 0.99 | 0.99 | 0.99 | 974 |
| 9 | 0.99 | 0.98 | 0.99 | 1009 |
| **Média / Total** | **0.99** | **0.99** | **0.99** | **10000** |


A análise das métricas por classe demonstra uma consistência notável, com a precisão e a sensibilidade (_recall_) acima de 0.98 para todos os dígitos. O dígito '1' e '2' destacam-se com um _F1-Score_ perfeito de 1.00.

Para compreender a natureza dos erros residuais, analisou-se a Matriz de Confusão apresentada na Figura 2.

<div align="center">
  <img src="https://github.com/user-attachments/assets/67130f0d-ebe1-48b0-9893-d959d234391f" width="500">
  <br>
  <em>Figura 2: Matriz de Confusão do ModelBetterCNN no conjunto de teste.</em>
  <br><br>
</div>

A matriz apresenta uma diagonal dominante, corroborando a alta taxa de acerto. Os erros (valores fora da diagonal) são esporádicos e semanticamente justificáveis. Destaca-se, por exemplo, uma ligeira confusão entre os dígitos 4 e 9 (o modelo classificou 5 vezes um '9' real como sendo '4', e 8 vezes um '5' real como sendo '9' ou '3'). Estas falhas devem-se à semelhança geométrica entre estes caracteres em certas caligrafias manuscritas. Contudo, dado o volume total de dados (10.000 imagens), estes erros são estatisticamente irrelevantes.

#### 3.2. Comparação entre Modelos
Para validar a eficiência do modelo _ModelBetterCNN_, comparou-se o seu desempenho e complexidade computacional com as abordagens implementadas anteriormente. A tabela seguinte resume estes dados.

| Modelo | Nº de Parâmetros | Accuracy | F1-Score (Macro) |
| :--- | :---: | :---: | :---: |
| ModelFullyConnected | 7850 | 93% | 0.93 |
| ModelConvNet | 421642 | 98% | 0.98 |
| ModelConvNet3 | 159626 | 98% | 0.98 |
| ModelBetterCNN | 390858 | 99% | 0.99 |

Análise Comparativa:

- Limitações da Abordagem Densa (_ModelFullyConnected_): Sendo o modelo mais simples e apresentando o menor número de parâmetros do conjunto, o _ModelFullyConnected_ revelou limitações estruturais significativas. Ao "achatar" a imagem num vetor unidimensional logo na entrada, a rede ignora a componente espacial e as relações de vizinhança entre os pixéis. Esta perda de informação traduz-se numa convergência muito mais tardia durante o treino e numa _accuracy_ final inferior às abordagens convolucionais.

- A _ModelConvNet_ destaca-se por utilizar o maior número de parâmetros de todas as redes testadas. Este excesso de complexidade, contudo, revelou-se contraproducente, levando a uma estagnação rápida da aprendizagem e a uma elevada variação entre a precisão de treino e de teste, indiciando dificuldades de generalização.

- Por outro lado, a _ModelConvNet3_ apresenta-se como uma rede substancialmente mais profunda, mas, devido a uma arquitetura mais eficiente, possui menos parâmetros que a _ConvNet_. Esta estrutura permite uma melhor compreensão das características dos dados. No entanto, observou-se que a rede ainda carecia de estabilidade, necessitando de mecanismos adicionais para combater o _overfitting_.

- A arquitetura final, _ModelBetterCNN_, possui uma elevada profundidade, mas distingue-se pela introdução de camadas de regularização cruciais: _Batch Normalization_ e _Dropout_. O _Batch Normalization_, ao normalizar os pesos durante o treino, induz uma convergência muito mais rápida e estável. Já o _Dropout_ permite mitigar eficazmente o _overfitting_, tornando a precisão no conjunto de teste muito mais próxima da obtida no treino. Em suma, estes mecanismos permitiram treinar uma rede complexa e profunda de forma robusta, alcançando os 99% de exatidão reportados.

### 4. Conclusão
Destaca-se, em particular, o desempenho da arquitetura proposta, _ModelBetterCNN_. Os resultados obtidos demonstram que o aumento da profundidade da rede, quando acompanhado por mecanismos de regularização e normalização adequados, é determinante para a performance do modelo. 

A introdução de _Batch Normalization_ foi crucial para acelerar a convergência e estabilizar o treino, enquanto o _Dropout_ desempenhou um papel vital na prevenção de _overfitting_, garantindo que a rede generalizasse corretamente para dados não vistos.

Com uma exatidão final de 99% no conjunto de teste e uma matriz de confusão que apresenta erros residuais apenas em casos de elevada ambiguidade gráfica, conclui-se que o modelo desenvolvido é robusto e eficiente. Este trabalho consolida, assim, a importância do equilíbrio entre a complexidade da arquitetura e as técnicas de otimização no desenvolvimento de soluções de visão computacional de alto desempenho.

#
## Tarefa 2

### 1. Objetivo
O objetivo desta tarefa consiste na criação de um _dataset_ sintético mais complexo, transitando de um cenário de classificação simples (onde o dígito está centrado e isolado) para um cenário de deteção de objetos. As imagens geradas simulam "cenas" onde os dígitos do _dataset_ MNIST são posicionados aleatoriamente num fundo maior, podendo variar em escala e quantidade, introduzindo desafios de localização espacial e múltiplas instâncias.

### 2. Metodologia
A metodologia adotada divide-se em duas fases distintas: a geração sintética dos dados e a sua subsequente validação estatística. Para tal, conta com um _script_ responsável por criar as imagens mais complexas, `main_synthesis.py`, e um outro, `main_dataset_stats.py`, responsável por analisar e validar os _datasets_ criados.

#### 2.1. Criação das imagens (_main_synthesis.py_)
A sua função é transformar o _dataset_ MNIST original (destinado a classificação simples) num _dataset_ mais complexo para deteção de objetos. O código opera através da composição sintética de imagens, colocando dígitos em posições aleatórias sobre um fundo negro, gerindo simultaneamente a escala, a quantidade e a não-sobreposição dos elementos.

O código organiza-se em duas funções e um bloco de execução principal:

`check_overlap`

Esta função implementa a lógica geométrica crucial para garantir a integridade da "_Ground Truth_". O objetivo é impedir que dois dígitos sejam desenhados um em cima do outro, o que tornaria a deteção ambígua ou impossível.

- Entrada: Recebe as coordenadas da caixa proposta (new_box) e a lista de caixas já colocadas (existing_boxes).

- Lógica: Verifica se existe intersecção entre retângulos. A função valida se a nova caixa está estritamente à esquerda, direita, acima ou abaixo das existentes. Se nenhuma destas condições for verdadeira, assume-se que há colisão e retorna True (sobreposição detetada), sinalizando que a posição deve ser descartada.


`generate_dataset`

Esta é a função central que orquestra todo o processo de criação, configurável através de parâmetros como o tamanho dos dígitos (_min/max_digit_size_) e a densidade de objetos (_min/max_digits_).

Configuração do Ambiente: Define a dimensão do "canvas" (fundo) como 128x128 pixéis e cria automaticamente a estrutura de diretorias para separar imagens (/images) e legendas (/_labels_), tanto para treino como para teste.

Aquisição de Dados: Utiliza a biblioteca `torchvision.datasets` para descarregar e carregar o _dataset_ MNIST original em memória.

Ciclo de Geração: Para cada imagem a ser gerada:

- Inicialização: Cria uma imagem vazia (preta) utilizando a biblioteca PIL (Image.new).

- Determinação da Complexidade: Sorteia aleatoriamente o número de dígitos a inserir na cena atual (ex.: entre 3 e 5).

- Processamento Individual dos Dígitos:
-- Seleciona um dígito aleatório do MNIST.
-- Redimensionamento: Aplica uma transformação de escala aleatória dentro dos limites definidos. Utiliza-se a interpolação bilinear (`Image.BILINEAR`) para redimensionar o dígito, garantindo que a imagem mantém a suavidade e não fica pixelizada ou distorcida ao ser aumentada ou diminuída.
-- Posicionamento e Validação: Gera coordenadas aleatórias (x,y). Antes de "colar" o dígito, invoca a função `check_overlap`.
-- Heurística de Tentativa: Implementa um ciclo de persistência com 20 tentativas. Se o algoritmo não encontrar um espaço livre após 20 tentativas (devido ao congestionamento da imagem), desiste de colocar esse dígito específico, evitando _loops_ infinitos e garantindo a diversidade das cenas.

Exportação:
- A imagem final é guardada em formato .jpg.
- As anotações são guardadas num ficheiro de texto correspondente (.txt). Cada linha representa um objeto no formato: [Classe] [X] [Y] [Largura] [Altura].


`__main__`

No final do script, o código instancia a criação de duas versões de _dataset_:
- Versão A: Gera imagens contendo estritamente 1 dígito com tamanho fixo de 28x28 (igual ao original), mas em posição aleatória.
- Versão D: Gera imagens complexas contendo entre 3 a 5 dígitos, onde cada dígito sofre uma variação de escala aleatória entre 22x22 e 36x36 pixéis. Esta versão testa a capacidade do modelo de lidar com múltiplas instâncias e variabilidade de tamanho.


#### 2.2. Verificação e validação (_main_dataset_stats.py_)
Uma vez que o _dataset_ é criado sinteticamente, é imperativo garantir que os dados gerados cumprem as especificações estatísticas (equilíbrio entre classes e densidade de objetos) e geométricas (precisão das _bounding boxes)_. Este _script_ automatiza essa verificação através de métodos analíticos e visuais.

A arquitetura do código centra-se na função `visualize_dataset_stats`, que executa a análise em três fases distintas:

1. Agregação de Metadados - Numa primeira fase, o algoritmo percorre todos os ficheiros de anotação (.txt) presentes na pasta de _labels_. Esta abordagem é computacionalmente mais eficiente do que carregar as imagens, permitindo uma análise rápida mesmo em datasets com milhares de exemplos. Para cada anotação, extraem-se três vetores de informação:
    - Classes: A identificação do dígito (0-9), permitindo verificar o equilíbrio do _dataset_.
    - Densidade: O número de linhas no ficheiro de texto, que corresponde diretamente ao número de objetos na imagem.
    - Dimensões: A altura das _bounding boxes_, usada para validar se o redimensionamento aleatório (na Versão D) ocorreu conforme esperado.

2. Análise Estatística (Gráficos Descritivos) - Com os dados recolhidos, o _script_ utiliza a biblioteca _matplotlib_ para gerar um painel composto por três histogramas complementares:
    - Frequência das Classes: Um gráfico de barras que valida se todos os dígitos (0 a 9) estão representados de forma equitativa. Um desequilíbrio aqui poderia enviesar o treino da rede neuronal futura.
    - Histograma de Densidade: Valida se o número de objetos por imagem respeita as regras da versão gerada (ex.: confirma se a Versão D contém estritamente entre 3 e 5 dígitos).
    - Distribuição de Escalas: Um histograma que mostra a variabilidade dos tamanhos dos dígitos, confirmando a média e o intervalo de redimensionamento (ex.: verificar a presença de dígitos entre 22px e 36px).

3. Validação Visual (_Ground Truth Mosaics_) Para além das métricas abstratas, é fundamental a inspeção visual humana. O código seleciona aleatoriamente 16 imagens do _dataset_ e gera um mosaico 4x4. Para cada imagem, o ficheiro de anotação correspondente é lido. Utilizando a biblioteca `matplotlib.patches`, são desenhados retângulos vermelhos (_bounding boxes_) sobre as coordenadas anotadas. A classe do objeto é impressa sobre a caixa (texto amarelo). Esta visualização serve como "prova de conceito", permitindo detetar erros graves como coordenadas desfasadas, caixas com tamanho incorreto ou falhas no algoritmo de anti-sobreposição que as estatísticas puras poderiam não revelar.


O bloco `__main__` aplica esta lógica sequencialmente às pastas geradas (**_mnist_detection_A_** e **_mnist_detection_D_**), guardando os resultados como ficheiros de imagem. Isto permite uma verificação rápida e documental da qualidade dos dados antes de se avançar para a fase de treino de modelos de deteção.

### 3. Resultados
#### 3.1. Análise da Versão A

As imagens da versão A devem conter apenas um objeto por imagem com dimensões fixas mas posições variáveis. A figura seguinte apresenta um mosaico de amostras aleatórias do conjunto criado.

<div align="center">
  <img src="https://github.com/user-attachments/assets/caac103b-83bf-4de7-b8ce-939c81f83998" width="700">
  <br>
  <em>Figura 3: Mosaico de validação visual da Versão A.</em>
  <br><br>
</div>

A inspeção visual confirma que o gerador posicionou corretamente os dígitos dentro da área de 128x128 pixéis. As caixas delimitadoras (bounding boxes) envolvem os dígitos com precisão, e a classe associada (texto a amarelo) corresponde ao dígito visível.

As métricas globais do _dataset_ são apresentadas na Figura 4.

<div align="center">
  <img src="https://github.com/user-attachments/assets/73140630-2b1e-408f-8e95-674045883c70" width="900">
  <br>
  <em>Figura 4: Estatísticas da Versão A.</em>
  <br><br>
</div>

A análise dos gráficos permite concluir que existe uma distribuição uniforme entre os dígitos 0 e 9, garantindo que o modelo não será enviesado para uma classe específica. O histograma central confirma o cumprimento estrito do requisito, apresentando uma barra única em N=1, indicando que 100% das imagens contêm exatamente um dígito. Já o histograma de tamanhos apresenta um pico isolado em torno dos 28 pixéis, validando a manutenção da escala original do MNIST.

#### 3.2. Análise da Versão D

A Versão D introduz variabilidade de escala e múltiplas instâncias, testando a robustez do algoritmo de anti-sobreposição.

A Figura 5 ilustra a complexidade das cenas geradas.

<div align="center">
  <img src="https://github.com/user-attachments/assets/97ab388f-c542-4149-af58-d42480f286bf" width="700">
  <br>
  <em>Figura 5: Mosaico de validação visual da Versão D.</em>
  <br><br>
</div>

Neste cenário, verifica-se a eficácia do algoritmo de verificação de colisões. Apesar da densidade elevada de objetos, as bounding boxes não se intersetam, garantindo que todos os dígitos são totalmente visíveis. É também perceptível a variação de tamanho entre os diferentes dígitos na mesma imagem.

A conformidade com os parâmetros de configuração é evidenciada na Figura 6.

<div align="center">
  <img src="https://github.com/user-attachments/assets/a32ea461-af54-4581-a7b7-3a015ec1cf55" width="900">
  <br>
  <em>Figura 6: Estatísticas da Versão D.</em>
  <br><br>
</div>

A análise quantitativa demonstra que mesmo com a inserção aleatória de múltiplos objetos, o equilíbrio entre as classes (gráfico da esquerda) mantém-se estável. O histograma central confirma que todas as imagens geradas contêm entre 3 e 5 dígitos, respeitando os limites impostos. Quanto ao gráfico da direita, este exibe uma distribuição de tamanhos espalhada uniformemente pelo intervalo configurado [22px, 36px], confirmando que o redimensionamento aleatório funcionou conforme o esperado.

### 4. Conclusão
Conclui-se, assim, que os dados gerados possuem a qualidade e a diversidade necessárias para servir de base fiável ao treino e teste de modelos de deteção de objetos nas etapas subsequentes do projeto.

#
## Tarefa 3

### 1. Objetivo
O objetivo desta tarefa é transitar da classificação isolada para a deteção de objetos em cenários complexos, sem a necessidade de re-treinar a rede neuronal. Para tal, implementa-se a técnica de Janela Deslizante _(Sliding Window_). Esta abordagem reutiliza o modelo convolucional (`ModelBetterCNN`), treinado na Tarefa 1, para varrer as imagens geradas na Tarefa 2, classificando recortes locais e identificando a presença e localização dos dígitos no espaço.

### 2. Metodologia
O código desenvolvido (`main_sliding_window.py`) orquestra o processo de deteção através de três componentes lógicos fundamentais: o algoritmo de varrimento (`detect_and_save`), a filtragem de resultados (`nms`) e o processamento em lote (`process_full_dataset`).

`detect_and_save`

Esta função trata do processamento da imagem, seguindo os passos apresentados de seguida:
- Varrimento (_Sliding Window_): Percorre a imagem (128x128) com uma janela de tamanho fixo de 28x28 pixéis (dimensão nativa do modelo), aplicando um passo (_stride_) de 6 pixéis.
- Otimização de pré-processamento: Para evitar classificar zonas de fundo irrelevantes (preto absoluto), implementou-se uma verificação de desvio padrão (`crop_tensor.std() < 0.1`). Se o recorte for uniforme (vazio), é ignorado imediatamente, poupando recursos computacionais.
- Os recortes válidos são submetidos ao modelo (`ModelBetterCNN`). Apenas as previsões com um grau de confiança extremamente elevado (> 98%) são consideradas, minimizando falsos positivos.
- Visualização: Após a aplicação do NMS, as caixas finais são desenhadas sobre a imagem original utilizando a biblioteca matplotlib, indicando a classe prevista e a probabilidade associada.

`nms`

Ao aplicar a técnica de janela deslizante, esta gera múltiplas deteções redundantes para o mesmo objeto (várias caixas sobrepostas em torno do mesmo dígito). Para resolver este problema, implementou-se a função NMS (_Non-Maximum Suppression_).

O algoritmo ordena todas as caixas detetadas pela sua probabilidade (confiança). Seleciona a caixa com maior _score_ e elimina todas as outras que tenham uma sobreposição significativa com esta.

A sobreposição é calculada através da IoU (_Intersection over Union_). Se a IoU for superior a um limiar de 0.2 (definido na chamada da função), considera-se que as caixas referem-se ao mesmo objeto e a redundante é descartada.

`process_full_dataset`

A função principal gere a execução sistemática do detetor nas diferentes versões do _dataset_ criado anteriormente.

O sistema carrega o estado do modelo "_Best Checkpoint_" da Tarefa 1, garantindo que o detetor utiliza a versão mais precisa da rede. Posteriormente, o código processa automaticamente as pastas de teste das versões 'A' e 'D', limitando a execução a 20 imagens por versão para uma validação rápida. As imagens resultantes, contendo as _bounding boxes_ e as etiquetas, são guardadas numa diretoria dedicada (Tarefa_3_Results), permitindo uma inspeção visual posterior da eficácia do algoritmo.

### 3. Resultados
#### 3.1. Resultados da versão A
Os resultados foram visualmente analisados de modo a compreender a eficácia do método da janela deslizante. 

É percetível nos resultados a tendência deste método a classificar apenas frações dos dígitos como um dígito em si, tipicamento como '1'. Ainda assim, em muitos dos casos o método foi capaz de identificar o dígito presente na imagem. É possível verificar casos com pouco ruído (deteções de dígitos que não existem na imagem), contendo apenas 2 deteções extras, no entanto, há também exemplos que apresentam imenso ruído, tornando inclusive difícil a identificação da _bounding box_ correspondente à deteção correta.
Em escassas situações, o método não foi capaz de identificar o dígito presente na imagem, identificando apenas dígitos provenientes de frações do dígito real.

Na figura seguinte temos 3 exemplos dos resultados.


<table>
  <tr align="center">
    <td>
      <img src="https://github.com/user-attachments/assets/f1059c7c-0632-4055-b0a6-538731a5ccec" alt="3 pouco ruido" width="300">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/b2821c9b-e9ac-4b3e-bc4a-b96f6a285e3a" alt="2 n detetado" width="300">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/76e6e576-b209-4eec-9b94-6871be2e868f" alt="6 muito ruido" width="300">
    </td>
  </tr>
  <tr align="center">
    <td><em> </em></td>
    <td><em>Figura 7: Exemplos dos resultados da versão A</em></td>
    <td><em> </em></td>
  </tr>
</table>

Na primeira imagem podemos verificar que a rede foi capaz de identificar o dígito, contudo gerou também algum ruído em torno, sendo ainda assim um resultado satisfatório.

Na segunda imagem vemos um dos escassos casos em que a rede não foi capaz de identificar o número, identificando apenas fragmentos deste como se do dígito '1' se tratasse.

Já na última imagem vemos um dos casos em que a rede fez uma correta identificação, contudo com imenso ruído, classificando demasiados fragmentos como dígito '1'.


#### 3.2. Resultados da versão D
Na versão D, onde cada imagem é composta por mais que um dígito, torna-se ainda mais frequente a identificação de dígitos falsos através de fragmentos dos dígitos reais. Isto resulta num elevado ruído presente em maior parte das imagens. São vários os exemplos onde o método não foi capaz de identificar todos os dígitos, sobretudo em imagens com 5 dígitos. 

<table>
  <tr align="center">
    <td>
      <img src="https://github.com/user-attachments/assets/a97a1f4a-a8f9-4f91-b234-0aa0aebf4ce6" alt="tudo identificado" width="300">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/f8cddd79-4177-4466-a11f-2a141bd533e2" alt="2/5 identificados" width="300">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/1fec6b83-d4aa-4f98-b182-9bc5951783c0" alt="N sem id" width="300">
    </td>
  </tr>
  <tr align="center">
    <td><em> </em></td>
    <td><em>Figura 8: Exemplos dos resultados da versão D</em></td>
    <td><em> </em></td>
  </tr>
</table>

Na primeira imagem da figura acima podemos verificar um dos casos em que a rede teve sucesso na identificação de todos os dígitos, contudo a quantidade de falsos positivos é elevada, corroborando a problemática da deteção de fragmentos.

A imagem central revela um caso em que a rede apenas foi capaz de identificar dois dos cinco dígitos existentes, classificando imensos fragmentos erradamente.

Na última imagem esiste um dígito sem qualquer bounding box (um falso negativo) revelando a ineficiência deste método em identificar dígitos de diferentes tamanhos, que não estejam centrados na janela de varrimento e com semelhanças a outros números.

### 4. Conclusão
Através dos resultados obtidos verifica-se que o método embora seja capaz de detetar maior parte dos dígitos, devido ao ruído que gera, não é um bom método para identificação de objetos. O método da janela deslizante teria de sofrer melhorias consideráveis para produzir resulatos favoráveis.

#
## Tarefa 4

### 1. Objetivo
A Tarefa 4 visa refinar o sistema de deteção desenvolvido anteriormente, atacando as suas limitações principais: a elevada taxa de falsos positivos em áreas de fundo complexo e a incapacidade de detetar objetos de dimensões variadas com precisão. 

Para tal, implementou-se uma estratégia dupla:
- Re-treino com _Hard Negative Mining_: O modelo é reeducado para distinguir ativamente entre "dígitos completos" e "fragmentos de dígitos" (falsos positivos difíceis).
- Abordagem Multi-Escala: O algoritmo passa a varrer a imagem com janelas de diferentes tamanhos, garantindo que objetos maiores ou menores que o padrão sejam capturados corretamente.

### 2. Metodologia
O código (`main_improved_detection.py`) apresenta uma evolução significativa face à Tarefa 3, introduzindo lógica avançada tanto na preparação dos dados como no pós-processamento. É composto por duas classes, `ModelImproved` que define a arquitetura do modelo e `ComplexSceneDataset` que "cria" o _dataset_ melhorado. Contém ainda funções que definem o treino, funções que filtram os resultados e funções para a avaliação do modelo.

#### 2.1. Arquitetura da Rede (_ModelImproved_)
A classe _ModelImproved_ representa uma evolução da arquitetura CNN utilizada nas tarefas anteriores. Embora mantenha a filosofia de extração de características hierárquica, esta versão foi ligeiramente compactada e ajustada para o problema de classificação de 11 classes (10 dígitos + 1 classe de "Fundo/Ruído").

A classe herda de _nn.Module_ e estrutura-se fundamentalmente em duas funções:

`__init__` 

Esta função define os blocos construtivos da rede, inicializando as camadas com parâmetros treináveis.

- Bloco Convolucional 1 (conv1, bn1):
-- `nn.Conv2d(1, 32, ...)`: A entrada é uma imagem em tons de cinza (1 canal). A camada aplica 32 filtros distintos para extrair características de baixo nível (arestas, curvas). O `padding=1` garante que a dimensão espacial se mantém (28x28).
-- `nn.BatchNorm2d(32)`: Normaliza as saídas dos 32 canais, estabilizando o treino e permitindo taxas de aprendizagem mais altas.

- Bloco Convolucional 2 (conv2, bn2):
-- `nn.Conv2d(32, 32, ...)`: Refina as características extraídas anteriormente. Mantém-se a profundidade de 32 mapas de características.
-- `nn.BatchNorm2d(32)`: Nova normalização para garantir a estabilidade do sinal em profundidade.

- Camadas de Processamento e Regularização:
-- `nn.MaxPool2d(2, 2)`: Reduz a dimensionalidade espacial para metade (de 28x28 para 14x14), condensando a informação e reduzindo o custo computacional.
-- `nn.Dropout(0.25)`: "Desliga" aleatoriamente 25% dos neurónios durante o treino. Esta técnica é essencial para evitar o _Overfitting_, forçando a rede a não depender de caminhos neuronais específicos.

- Classificador _Densen_ (fc1, fc2):
-- `nn.Linear(32 * 14 * 14, 128)`: A camada de entrada recebe o tensor "achatado". O valor 32×14×14 corresponde ao volume total de dados que saem da camada de _Pooling_. Estes são projetados para 128 neurónios.
-- `nn.Linear(128, 11)`: A camada final possui agora 11 saídas em vez de 10. As primeiras 10 correspondem aos dígitos (0-9) e a 11.ª saída (índice 10) representa a classe "Fundo". Isto permite ao modelo aprender explicitamente a rejeitar imagens que não contêm dígitos completos.

`forward` 

Esta função define o fluxo de dados durante a inferência e treino.

A entrada x passa sequencialmente pelas camadas convolucionais. Em cada etapa, aplica-se a função de ativação não-linear _ReLU_ após a normalização (Convolução → _Batch Norm_ → _ReLU_).
Após o segundo bloco convolucional, os dados passam pelo `MaxPool2d` (reduzindo o tamanho) e pelo `Dropout`. A instrução `x.view(x.size(0), -1)` transforma os mapas de características 3D (Batch, 32, 14, 14) num vetor 1D, compatível com as camadas densas. 

O vetor passa então pela primeira camada densa (fc1) com ativação _ReLU_. Finalmente, a camada fc2 projeta o resultado nas 11 classes possíveis (_logits_), sem função de ativação explícita no final (pois a função de perda `CrossEntropyLoss` usada no treino já aplica internamente o _Softmax_).

#### 2.2. Gestão do _Dataset_ (_ComplexSceneDataset_)
Esta classe substitui a abordagem simples de carregar imagens inteiras por uma estratégia de amostragem dinâmica. Em vez de fornecer a imagem completa de 128x128 pixéis à rede, esta classe recorta pequenas janelas de 28x28 pixéis, classificando-as de acordo com o seu conteúdo. Este processo é fundamental para ensinar a rede a distinguir não apenas "dígito" de "fundo", mas também "dígito completo" de "fragmento de dígito".

A classe estrutura-se em dois métodos principais:

`__init__` e `__len__`

Estas funções configuram o acesso aos ficheiros e definem a dimensão virtual do dataset.
- Localiza todas as imagens de treino (.jpg) e prepara o acesso às respetivas anotações (.txt).
- Ao retornar `len(self.image_files) * 15`, o _dataset_ "finge" ser 15 vezes maior do que o número de imagens físicas. Isto garante que, em cada época, a rede vê 15 recortes diferentes extraídos da mesma imagem original, aumentando drasticamente a diversidade de treino sem necessitar de mais dados brutos.

`__getitem__`

Para cada pedido de dados, o algoritmo carrega a imagem original e as bounding boxes reais ("_Ground Truth_") e decide probabilisticamente que tipo de exemplo gerar:

- Exemplos Positivos (40% de probabilidade):
-- O sistema seleciona aleatoriamente uma das caixas de dígitos presentes na imagem.
-- Aplica-se um pequeno deslocamento aleatório (_jitter_) ao centro do recorte (±2 pixéis). Isto torna a rede robusta a pequenos erros de centralização durante a deteção real.
-- O recorte resultante recebe a etiqueta (_label_) correta do dígito (0-9).

- _Hard Negatives_ (30% de probabilidade):
-- Esta é a componente crítica para a redução de falsos positivos. O algoritmo seleciona um dígito real, mas força propositadamente um recorte deslocado (desvio de ±10 pixéis).
-- O resultado é uma imagem que contém apenas uma fração do dígito (ex.: apenas a curva superior de um '8' ou a barra vertical de um '7').
-- A este recorte é atribuída a etiqueta 10 (Fundo). Isto força a rede neuronal a aprender que "meio dígito" não serve; ela deve penalizar ativações em zonas que não contenham o objeto na sua totalidade.

- Fundo Puro (30% de probabilidade):
-- O algoritmo procura uma região da imagem que não intersete significativamente nenhuma _bounding box_ de dígitos.
-- Caso encontre (após 20 tentativas), extrai esse recorte de espaço vazio.
-- Atribui-se a etiqueta 10 (Fundo), reforçando o conhecimento do que é vazio.

Este equilíbrio probabilístico (40/30/30) garante que o modelo é exposto a uma variedade de exemplos equilibrados, resultando num classificador muito mais discriminativo.

#### 2.3. Ciclo de Treino (_train_robot_)
A função `train_robot` operacionaliza o ciclo de treino, estabelecendo a ponte entre a arquitetura da rede (_ModelImproved_) e a estratégia de dados dinâmicos (_ComplexSceneDataset_). Diferente do treino simples da Tarefa 1, este foi ajustado para lidar com a complexidade acrescida da rejeição de fundo e fragmentos.

O dataset é encapsulado num _DataLoader_ com `batch_size=64` e `shuffle=True`. O parâmetro _shuffle_ é crucial uma vez que, como o _dataset_ gera exemplos probabilísticos (positivos/negativos/fundo), baralhar os dados garante que cada lote contém uma mistura representativa de todas as situações, estabilizando o cálculo do gradiente. Mantém-se o algoritmo _Adam_ (lr=0.001) e define-se a nn.CrossEntropyLoss como função perda. 

Esta função é adequada para classificação multi-classe, lidando agora com as 11 classes possíveis (0-9 + Fundo). Ela penaliza o modelo não apenas por errar o dígito, mas por confundir "fundo" com "dígito".

O ciclo de treino possui 5 épocas. Devido à natureza dinâmica do _dataset_ (que gera recortes diferentes a cada acesso), mais épocas significam que a rede é exposta a uma maior variabilidade de exemplos "difíceis" (_Hard Negatives_), refinando progressivamente a sua fronteira de decisão.

Dentro do _loop_, executa-se:
- _Forward Pass_: A rede processa o lote de imagens.
- _Loss_: Calcula-se o erro entre a previsão e a etiqueta real.
- _Backward Pass_: `loss.backward()` calcula os gradientes (as correções necessárias para os pesos).
- _Weight Update_: `optimizer.step()` aplica as correções.

Ao final do processo, os pesos otimizados da rede são guardados no ficheiro _best_improved.pkl_. Este ficheiro guarda o "conhecimento" adquirido pelo modelo sobre como distinguir dígitos completos de fundo e fragmentos, ficando pronto para a fase de teste.

#### 2.4. Filtragem e Validação
O sucesso da abordagem multi-escala depende inteiramente da capacidade de filtrar o excesso de caixas geradas. Como o modelo varre a imagem com janelas de 28, 36 e 48 pixéis, é comum que o mesmo dígito seja detetado múltiplas vezes e em tamanhos diferentes. O conjunto de funções abaixo implementa a lógica necessária para consolidar estas previsões numa única deteção coerente.

`calculate_iou`

Esta função implementa a métrica Intersection over Union (IoU).

O seu objetivo è quantificar geometricamente o quanto duas caixas (box1 e box2) partilham o mesmo espaço. Para tal, determina-se a área do retângulo de interseção (onde as caixas se cruzam) e divide-se pela área da união (a área total ocupada por ambas). O resultado é um valor entre 0 (sem contacto) e 1 (sobreposição perfeita). Este valor é utilizado como critério primário para identificar deteções duplicadas do mesmo objeto.

`is_contained` 

Esta função introduz uma regra de filtragem específica para lidar com problemas de escala, onde o IoU tradicional falha.

Em deteção multi-escala, uma janela pequena pode detetar corretamente uma parte de um número (ex: o círculo superior de um '8') enquanto uma janela grande deteta o '8' completo. A área de união entre elas é grande (corresponde à àrea da caixa maior) enquanto que a àrea de interseção será apenas a àrea da caixa menor, resultando num IoU baixo, o que impediria o NMS normal de eliminar a caixa pequena.

Assim, esta função verifica se uma caixa (_inner_box_) está substancialmente contida dentro de outra (_outer_box_). Para tal calcula-se a interseção e divide-se pela área da caixa menor. Se mais de 80% da caixa menor estiver dentro da maior, a função retorna True. Isto sinaliza ao sistema que a caixa pequena é redundante e deve ser suprimida em favor da deteção maior e mais completa.

`advanced_nms` 

Esta é a função que combina as métricas anteriores para limpar os resultados brutos. Diferente do NMS clássico, esta versão considera tanto a sobreposição como a contenção.

O algoritmo começa por ordenar todas as caixas detetadas pela probabilidade dada pelo modelo (_score_), da maior para a menor. Assume-se que a caixa com maior confiança é a "correta". Posteriormente, esta é comparada com todas as outras restantes na lista (candidatas). Uma caixa candidata é eliminada se se verificar qualquer uma das seguintes condições:
- Alta Sobreposição: Tem um IoU superior ao limiar definido (0.2) com a caixa "correta".
- Contenção: Está contida dentro da caixa "correta" (ou vice-versa), indicando que é um fragmento ou uma deteção concêntrica redundante.

Este processo repete-se iterativamente até restarem apenas as deteções únicas e não sobrepostas.

`check_hit_advanced`

Esta função é responsável pela avaliação de desempenho, verificando se uma previsão corresponde à realidade (_Ground Truth_).

Esta função utiliza uma abordagem baseada no centroide da caixa prevista e na caixa real. Assim, calcula-se o ponto central da caixa prevista (pcx, pcy) e considera-se um "_Hit_" (Verdadeiro Positivo) se:
- O centro da caixa prevista cair geometricamente dentro de uma caixa real.
- A classe prevista (ex: dígito '3') for igual à classe real anotada.

Esta função retorna o índice da caixa real encontrada ou -1 em caso de erro (Falso Positivo), permitindo a contagem precisa para o cálculo da Precisão e _Recall_.

#### 2.5. Avaliação (_test_robot_metrics_)
A função `test_robot_metrics` trata da avaliação final do modelo em questão. Aqui o modelo percorre imagens desconhecidas, deteta objetos de vários tamanhos e fornece métricas quantitativas de desempenho. Diferente das funções de avaliação anteriores, esta integra a lógica Multi-Escala.

O fluxo de processamento divide-se em três etapas lógicas:
- Varredura Multi-Escala: A função percorre a imagem com janelas de 3 escalas distintas, SCALES = [28, 36, 48], percorrendo a totalidade da imagem 3 vezes. Como a rede neuronal (_ModelImproved_) tem uma entrada fixa de 28x28 pixéis, qualquer recorte extraído (seja ele de 36px ou 48px) é imediatamente redimensionado para 28x28 usando interpolação bilinear. Isto permite que a rede "veja" e classifique objetos grandes utilizando a mesma arquitetura de pesos aprendida.
- Previsão e Verificação: Para cada recorte extraído nas várias escalas, o modelo emite uma previsão. A primeira verificação é a classe prevista. Se o modelo classificar o recorte como Classe 10 (Fundo), a deteção é imediatamente descartada. Para além disso, apenas previsões onde a rede tem mais de 98% de certeza são aceites. Este rigor é necessário para minimizar os falsos positivos.
- Consolidação e Cálculo de Métricas: Após recolher todas as deteções "brutas" das três escalas, a função invoca o `advanced_nms` para fundir as caixas redundantes. De seguida, compara as deteções finais com o _Ground Truth_ para calcular as métricas de Precisão, Recall e F1-Score

Finalmente, a função gera visualizações qualitativas (desenhando as caixas verdes sobre as imagens), permitindo ao utilizador validar visualmente se a estratégia multi-escala está a funcionar corretamente (ex.: verificar se um dígito grande está a ser envolvido por uma caixa grande e não por várias pequenas).


### 3. Resultados
A avaliação do sistema otimizado foi realizada num subconjunto de 200 imagens de teste da Versão D. Os resultados quantitativos e qualitativos demonstram a eficácia das técnicas de _Hard Negative Mining_ e deteção multi-escala.

A tabela seguinte resume as métricas obtidas.

| Métrica | Valor Obtido | Significado |
| :--- | :---: | :--- |
| **Total de Deteções** | 637 | Número de caixas geradas pelo modelo após filtragem. |
| **Deteções Corretas** | 621 | Número de caixas que correspondem a dígitos reais (*True Positives*). |
| **Precisão** | 97.49% | De tudo o que o modelo detetou, 97.5% eram realmente dígitos. |
| **Recall** | 78.21% | De todos os dígitos existentes, o modelo encontrou 78.2%. |
| **F1-Score** | 86.79% | Média harmónica, indicando o equilíbrio global do sistema. |

Estes valores demonstram uma elevada precisão (97.49%), indicando que a estratégia de _Hard Negative Mining_ foi extremamente bem-sucedida. O modelo aprendeu eficazmente a rejeitar "ruído" (fundo e fragmentos de dígitos), gerando muito poucos falsos positivos (apenas 16 erros em 637 deteções). 

Já o valor da sensibilidade (78.21%) é uma consequência de um _threshold_ elevado (> 0.98). Este threshold foi definido para garantir uma precisão elevada. Isto significa que o modelo opta por "ignorar" dígitos ambíguos ou difíceis em vez de arriscar uma classificação errada.

A figura abaixo apresenta exemplos visuais das deteções finais.

<div align="center">
  <img src="https://github.com/user-attachments/assets/8b6e16e5-aa5b-47ae-9cde-2569c76bfba7" width="900">
  <br>
  <em>Figura 9: Resultados visuais na Versão D.</em>
  <br><br>
</div>

A inspeção visual corrobora os dados numéricos. Ao contrário da abordagem de Janela Deslizante simples (Tarefa 3), não se observam múltiplas caixas a rodear o mesmo objeto. O NMS Avançado eliminou eficazmente as redundâncias.

O sistema demonstra ainda capacidade de detetar corretamente tanto dígitos pequenos (ex.: o '1' na quarta imagem) como dígitos maiores, validando a abordagem multi-escalas (28px, 36px, 48px).

Nas amostras apresentadas, não existem caixas desenhadas em zonas vazias ou que contenham apenas fragmentos dos números, confirmando que a classe "Fundo" (Classe 10) foi bem treinada.

### 4. Conclusão
Em conclusão, o sistema evoluiu de um detetor rudimentar e cheio de falhas para uma solução robusta e precisa, ideal para aplicações onde o custo de um falso positivo é elevado.

#
## Reflexão
O projeto desenvolvido percorreu o ciclo fundamental de um sistema de Visão Computacional, evoluindo da classificação de dígitos isolados (Tarefa 1) para a deteção de objetos em cenários complexos (Tarefa 4). Esta progressão permitiu validar a eficácia das CNNs, mas expôs simultaneamente as limitações das abordagens clássicas de deteção.

É de notar que a arquitetura do modelo _ModelBetterCNN_ é insuficiente sem uma estratégia de gestão de dados robusta. A Tarefa 1 comprovou a utilidade do _Batch Normalization_ e _Dropout_. Contudo a implementação de _Hard Negative Mining_ na Tarefa 4 foi o fator decisivo, elevando a precisão para 97.5% ao forçar a rede a distinguir ativamente entre dígitos completos, fragmentos e fundo.

A implementação do pós-processamento apresentou os maiores desafios técnicos, entre os quais a remoção de caixas redundantes na Tarefa 3 e a obtenção e calibragem dos filtros de rejeição na Tarefa 4. Quanto à Tarefa 3, o problema não foi totalmente resolvido, sendo possível verificar a presença de imenso ruído nas imagens exemplo resultantes. Já na Tarefa 4, foi possível resolver o problema do ruído e dos falsos positivos mas abdicando da deteção de dígitos nos quais a rede tinha "baixa" confiança, resultando num baixo _Recall_, em prol de um _F1-Score_ equilibrado.
