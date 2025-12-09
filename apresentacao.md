---
title: "Análise Comparativa de Arquiteturas de Deep Learning (LSTM vs. 1D-CNN) no Reconhecimento de Atividades Humanas"
subtitle: "Tópicos Especiais em Estatística Computacional"
author: "Ramon Lima de Oliveira Tavares"
institute: "Universidade Federal de Pernambuco (UFPE)"
date: "08/12/2025"
format:
  revealjs:
    theme: simple 
    slide-number: true
    show-slide-number: all
    transition: slide
    background-transition: fade
    logo: "figuras_artigo/Brasão_da_UFPE.png"
    width: 1280
    height: 720
    center: false
    margin: 0.05
    auto-stretch: true
    navigation-mode: linear
    css: styles.css
    embed-resources: true  # <--- ADICIONE ISTO AQUI!
editor: visual
---

## Agenda

1.  Contextualização e Problemática
2.  Fundamentação Teórica e Dados
3.  Metodologia e Arquiteturas (Detalhamento Matemático)
4.  Análise de Resultados
5.  Conclusão e Trabalhos Futuros

------------------------------------------------------------------------

# 1. Introdução

## O Problema: HAR (*Human Activity Recognition*)

::::::: columns
:::: {.column width="60%"}
**Contexto:** A computação onipresente gera fluxos massivos de dados inerciais (*Smartphones*, *Smartwatches*).

**O Desafio:** Classificar atividades complexas a partir de séries temporais ruidosas e estocásticas.

. . .

**Abordagem Proposta:** Substituir a *Feature Engineering* manual clássica por **Deep Learning End-to-End**.

::: callout-important
## Objetivo

Comparar rigorosamente o desempenho de redes **Recorrentes (LSTM)** vs. **Convolucionais (1D-CNN)** no dataset UCI HAR.
:::
::::

:::: {.column width="40%"}
<center>**Figura 1: Eixos do Sensor**</center>

::: r-stack
![](figuras_artigo/fig_smartphone_axis.png){width="90%"}
:::

<center><small>Fonte: Adaptado de Anguita et al. (2013).</small></center>
::::
:::::::

# 2. Dados e Pré-processamento

## Caracterização do Dataset UCI HAR

Antes da modelagem, é fundamental entender a volumetria e a estrutura física dos dados capturados.

::: panel-tabset
### Volumetria
**Divisão do Conjunto de Dados (Hold-out):**

| Conjunto | Quantidade de Amostras ($N$) | Proporção |
|:---|:---:|:---:|
| **Treinamento** | 7.352 janelas | 70% |
| **Teste** | 2.947 janelas | 30% |
| **Total** | **10.299 janelas** | **100%** |

* **Definição de 1 Amostra:** Uma matriz de dimensão $(128 \times 9)$.
* **Total de Pontos de Dados:** $10.299 \times 128 \times 9 \approx 11.8 \text{ Milhões de valores}$.

### As 9 Variáveis ($D$)
Os sinais brutos provêm do Acelerômetro e Giroscópio triaxiais.

| Índice | Sensor | Eixo | Descrição Física |
|:---:|:---|:---:|:---|
| 0-2 | **Body Acc** | $X, Y, Z$ | Aceleração do corpo (sem gravidade). |
| 3-5 | **Gyro** | $X, Y, Z$ | Velocidade angular (giro). |
| 6-8 | **Total Acc** | $X, Y, Z$ | Aceleração bruta (inclui gravidade). |

### Visualização Bruta
Exemplo de estrutura tabular de **uma única janela** ($t=1 \dots 128$):

| Time Step ($t$) | Body_Acc_X | Body_Acc_Y | ... | Gyro_Z | Total_Acc_Z |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $t_1$ | 0.002 | -0.005 | ... | 0.012 | 0.982 |
| $t_2$ | 0.004 | -0.008 | ... | 0.015 | 0.985 |
| $t_3$ | 0.001 | -0.002 | ... | 0.011 | 0.979 |
| $\vdots$ | $\vdots$ | $\vdots$ | $\ddots$ | $\vdots$ | $\vdots$ |
| $t_{128}$ | 0.003 | -0.001 | ... | 0.014 | 0.981 |

:::

# 2. Dados e Pré-processamento

## O Dataset UCI HAR

-   **Amostra:** 30 voluntários (19-48 anos).
-   **Sensor:** Samsung Galaxy S II (Cintura).
-   **Frequência de Amostragem:** 50Hz (50 leituras por segundo).

### Distribuição das Classes (Balanceamento)

------------------------------------------------------------------------

<center>**Figura 2: Contagem de Amostras por Classe**</center>

::: r-stretch
![](figuras_artigo/fig_balanceamento.png){width="100%" fig-align="center"}
:::

<center><small>Fonte: Elaborada pelo autor.</small></center>

------------------------------------------------------------------------

## Análise Temporal dos Sinais ($X_{body}$)

------------------------------------------------------------------------

<center>**Figura 3: Comparativo de Sinais Brutos (Janela 2.56s)**</center>

::: r-stretch
![](figuras_artigo/fig_sinais_temporal_grid.png){width="100%" fig-align="center"}
:::

<center><small>Legenda: Eixos X (Sólido), Y (Tracejado), Z (Pontilhado). Fonte: Elaborada pelo autor.</small></center>

------------------------------------------------------------------------

## Estrutura do Tensor de Entrada ($X$)

Para alimentar as redes neurais, os dados foram estruturados em janelas deslizantes.

$$X \in \mathbb{R}^{N \times 128 \times 9}$$

::::: columns
::: {.column width="50%"}
**1. Dimensão Temporal (Time Steps):** \* Frequência: **50Hz**. \* Tamanho da Janela: **128 passos**. \* Duração Física: $\frac{128}{50} = \mathbf{2.56 \text{ segundos}}$. \* **Overlap (50%):** A janela desliza a cada 64 passos para manter a continuidade do movimento.
:::

::: {.column width="50%"}
**2. Dimensão de Features (Canais):** As 9 variáveis de entrada ($X_t$): 1. **Aceleração Corporal** ($x, y, z$) 2. **Aceleração Total** ($x, y, z$) $\rightarrow$ Inclui Gravidade! 3. **Giroscópio** ($x, y, z$)
:::
:::::

## Tratamento do Target ($Y$)

Utilizamos **One-Hot Encoding** para todas as 6 classes, evitando ordinalidade numérica.

<center>**Tabela 1: Mapeamento de Classes**</center>

| ID  | Atividade            | Vetor One-Hot (Target) |
|:---:|:---------------------|:----------------------:|
|  1  | **Caminhando**       |  `[1, 0, 0, 0, 0, 0]`  |
|  2  | **Subindo Escadas**  |  `[0, 1, 0, 0, 0, 0]`  |
|  3  | **Descendo Escadas** |  `[0, 0, 1, 0, 0, 0]`  |
|  4  | **Sentado**          |  `[0, 0, 0, 1, 0, 0]`  |
|  5  | **Em Pé**            |  `[0, 0, 0, 0, 1, 0]`  |
|  6  | **Deitado**          |  `[0, 0, 0, 0, 0, 1]`  |

<center><small>Fonte: Elaborada pelo autor.</small></center>

# 3. Modelagem Matemática: LSTM

## 3.1. LSTM: Do Tensor ao Vetor ($t$)

A rede processa a matriz de entrada $X \in \mathbb{R}^{128 \times 9}$ passo a passo.

:::: columns
::: {.column width="40%"}
**1. O Fatiamento ($x_t$)**
No instante $t$, extraímos a **linha** correspondente às 9 variáveis e a transpomos.

$$x_t = (X_{t, :})^T \in \mathbb{R}^{9 \times 1}$$



:::

::: {.column width="60%"}
**2. A Fusão de Contexto ($v_{in}$)**
Para processar o "agora" ($x_t$) junto com o "passado" ($h_{t-1}$), concatenamos os vetores verticalmente:

$$v_{in} = \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix}$$

* $h_{t-1} \in \mathbb{R}^{128 \times 1}$ (Memória Anterior)
* $x_t \in \mathbb{R}^{9 \times 1}$ (Entrada Atual - Transposta)
* **Vetor Combinado:** $\mathbf{v_{in} \in \mathbb{R}^{137 \times 1}}$
:::
::::

## 3.2. Micro-dinâmica: Cálculo dos Portões

Como transformamos o vetor $v_{in}$ (137) de volta para o tamanho da memória (128)?

:::: columns
::: {.column width="50%"}
**Matrizes de Pesos ($W$)**
Cada portão possui sua própria matriz para realizar a projeção linear:
$$W_{\{f,i,c,o\}} \in \mathbb{R}^{128 \times 137}$$

**Cálculo Dimensional (Álgebra Linear):**
$$(128 \times 137) \cdot (137 \times 1) = (128 \times 1)$$
:::

::: {.column width="50%"}
**Equações dos 4 Portões:**
Geram vetores de controle $\in \mathbb{R}^{128}$.

1.  **Forget:** $f_t = \sigma(W_f \cdot v_{in} + b_f)$
2.  **Input:** $i_t = \sigma(W_i \cdot v_{in} + b_i)$
3.  **Candidate:** $\tilde{C}_t = \tanh(W_c \cdot v_{in} + b_c)$
4.  **Output:** $o_t = \sigma(W_o \cdot v_{in} + b_o)$
:::
::::



## 3.3. Atualização de Estados e Saída

O fechamento do ciclo no passo $t$ e a propagação do sinal.

:::: columns
::: {.column width="50%"}
**1. Estado da Célula ($C_t$):**
Soma linear (preserva o gradiente).
$$C_t = (f_t \odot C_{t-1}) + (i_t \odot \tilde{C}_t)$$
Dimensão: $\mathbb{R}^{128 \times 1}$

**2. Saída / Estado Oculto ($h_t$):**
$$h_t = o_t \odot \tanh(C_t)$$
Dimensão: $\mathbb{R}^{128 \times 1}$
:::

::: {.column width="50%"}
**Destino do Vetor $h_t$:**

1.  **Recorrência:** Torna-se o $h_{t-1}$ para o passo $t+1$.
2.  **Próxima Camada:** Sobe como entrada para a LSTM 2.

$$x_t^{(Layer2)} = h_t^{(Layer1)}$$
:::
::::

## Fluxo Tensorial: LSTM (Camada a Camada)

Acompanhe a transformação das dimensões do tensor $X$ (batch_size omitido para clareza).

| Etapa | Operação | Dimensão de Entrada | Dimensão de Saída | Explicação |
|:---|:---|:---:|:---:|:---|
| **Input** | Janela Deslizante | $(128, 9)$ | $(128, 9)$ | 128 instantes, 9 sensores. |
| **Layer 1** | LSTM (128 units) | $(128, 9)$ | $(128, 128)$ | `return_sequences=True`. Gera um vetor de 128 features para *cada* instante $t$. |
| **Layer 2** | LSTM (64 units) | $(128, 128)$ | $(64)$ | `return_sequences=False`. Apenas o último estado oculto ($h_{128}$) é retornado. |
| **Classificador** | Dense (Linear) | $(64)$ | $(6)$ | $z = W_{out} \cdot h_{final} + b$. Gera os *Logits*. |
| **Probabilidade** | **Softmax** | $(6)$ | $(6)$ | $\hat{y}_i = \frac{e^{z_i}}{\sum e^{z_j}}$. Soma = 1. |

# 3. Modelagem Matemática: 1D-CNN

## 3.4. Anatomia do Kernel (Filtro)

Na série temporal, o filtro desliza no tempo ($t$), mas cobre **todos** os sensores ($D$) simultaneamente.

**Entrada:** $X \in \mathbb{R}^{128 \times 9}$

:::: columns
::: {.column width="50%"}
**O Kernel Único ($W$):**
Para capturar correlações entre sensores, o kernel tem a profundidade da entrada.
Se $K=3$ (janela de tempo):

$$W \in \mathbb{R}^{3 \times 9}$$
:::

::: {.column width="50%"}
**A Operação de Deslizamento:**
O kernel computa o produto escalar cobrindo a matriz $3 \times 9$ atual, depois desliza 1 passo ($S=1$).

:::
::::

::: callout-note
## Multiplicidade
Como definimos **64 filtros**, existem 64 matrizes $W$ distintas, gerando 64 mapas de características (Feature Maps) independentes.
:::

## 3.5. Micro-dinâmica: O Cálculo Convolucional

Para um único filtro $m$ (dentre os 64), o valor de saída em $t$ é a soma total.

**Equação Detalhada:**
$$z_t^{[m]} = \sigma \left( \sum_{i=0}^{K-1} \sum_{c=0}^{D-1} x_{(t+i), c} \cdot w_{i, c}^{[m]} + b^{[m]} \right)$$

* $\sum_{i}$: Soma no tempo (janela 3).
* $\sum_{c}$: Soma nos canais (9 sensores).
* **Resultado:** Um único escalar por passo $t$.

**Saída da Camada (Feature Maps):**
Cada filtro gera um vetor temporal. Empilhando os 64 filtros:
$$\text{Output} \in \mathbb{R}^{126 \times 64}$$

## 3.6. Redução Dimensional e Hierarquia

Como a rede transita do sinal bruto para a classificação?

**1. Redução Temporal (Pooling):**
Após a convolução, o MaxPool ($P=2$) divide o tempo pela metade.
$$L_{out} = \lfloor \frac{126}{2} \rfloor = 63$$
**Saída do Bloco 1:** $\mathbb{R}^{63 \times 64}$

**2. Adaptação do Próximo Kernel (Bloco 2):**
A entrada agora tem profundidade 64 (não mais 9).
O novo kernel ($K=3$) deve se adaptar:
$$W_{Bloco2} \in \mathbb{R}^{3 \times 64}$$

::: callout-tip
## Final da Rede (GAP)
No último estágio ($28 \times 256$), o **Global Average Pooling** condensa o tempo em uma média simples, gerando o vetor final $\mathbb{R}^{256}$ para a Softmax.
:::

## Fluxo Tensorial: 1D-CNN (Camada a Camada)

Note como a dimensão temporal ($L$) diminui enquanto a profundidade ($D$ - canais) aumenta.

| Camada | Configuração | Entrada $(L \times D)$ | Saída $(L \times D)$ | Parâmetros ($W$) |
|:---|:---|:---:|:---:|:---|
| **Input** | Sinal Bruto | $128 \times 9$ | $128 \times 9$ | - |
| **Conv1D_1** | 64 Filtros, $K=3$ | $128 \times 9$ | $126 \times 64$ | $3 \times 9 \times 64 + 64$ |
| **MaxPool_1** | Pool=2 | $126 \times 64$ | $63 \times 64$ | - |
| **Conv1D_2** | 128 Filtros, $K=3$ | $63 \times 64$ | $61 \times 128$ | $3 \times 64 \times 128 + 128$ |
| **MaxPool_2** | Pool=2 | $61 \times 128$ | $30 \times 128$ | - |
| **GAP** | Global Avg Pooling | $30 \times 128$ | $128$ | Média simples no eixo $t$. |
| **Output** | Dense + Softmax | $128$ | $6$ | Classificação Final. |

## O Cálculo Final: Softmax

A camada final recebe o vetor de features latentes $z$ (seja $h_{128}$ da LSTM ou o GAP da CNN).

$$z = [z_1, z_2, z_3, z_4, z_5, z_6]$$

A probabilidade da classe "Caminhando" (índice 1) é:

$$P(y=1|X) = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4} + e^{z_5} + e^{z_6}}$$

::: callout-tip
## Interpretação
O modelo não diz "É caminhada". Ele diz: "Tenho 98% de certeza que é caminhada, 1% que é subindo escada e 1% ruído". A decisão é o `argmax`.
:::

## Arquiteturas Propostas (Detalhamento)

::: panel-tabset
### A. Stacked LSTM (Sequencial)

**Estratégia:** Empilhamento de camadas recorrentes para abstração progressiva temporal.

:::: columns
::: {.column width="40%"}
**Fluxo de Dados:**
1.  **Entrada:** Sequência crua.
2.  **LSTM 1:** Extrai *features* de baixo nível mantendo a sequência (`return_seq=True`).
3.  **LSTM 2:** Comprime a sequência temporal em um único vetor de contexto (`return_seq=False`).
4.  **Dense:** Classificação não-linear.


:::

::: {.column width="60%"}
<center>**Especificação Camada a Camada**</center>

| Camada | Configuração | Output Shape $(N, T, D)$ |
|:---|:---|:---:|
| **Input** | Janela 2.56s | $(None, 128, 9)$ |
| **LSTM 1** | 128 Units, $\tanh$ | $(None, 128, 128)$ |
| **Dropout** | Rate = 0.3 | $(None, 128, 128)$ |
| **LSTM 2** | 64 Units, $\tanh$ | $(None, \mathbf{64})$ |
| **Dropout** | Rate = 0.3 | $(None, 64)$ |
| **Dense** | 64 Units, ReLU | $(None, 64)$ |
| **Output** | **Softmax (6 classes)** | $(None, 6)$ |

<small>**Total de Parâmetros:** ~124.870</small>
:::
::::

### B. Pure 1D-CNN (Hierárquica)

**Estratégia:** Extração de *features* locais com redução espacial agressiva via Pooling.

:::: columns
::: {.column width="40%"}
**Fluxo de Dados:**
1.  **Blocos Conv:** 3 estágios de (Convolução + ReLU + MaxPool).
2.  **Aprofundamento:** O número de filtros dobra a cada bloco (64 $\to$ 128 $\to$ 256).
3.  **GAP:** Redução drástica de dimensão (média temporal) antes da classificação.


:::

::: {.column width="60%"}
<center>**Especificação Camada a Camada**</center>

| Bloco | Camada | Config (Kernel=3) | Output Shape |
|:---:|:---|:---|:---:|
| **1** | Conv1D | 64 Filtros | $(126, 64)$ |
| | MaxPool | Size=2 | $(63, 64)$ |
| **2** | Conv1D | 128 Filtros | $(61, 128)$ |
| | MaxPool | Size=2 | $(30, 128)$ |
| **3** | Conv1D | 256 Filtros | $(28, 256)$ |
| | **GAP** | Global Avg | $(None, \mathbf{256})$ |
| **Out** | Dense | Softmax | $(None, 6)$ |

<small>**Total de Parâmetros:** ~143.686</small>
:::
::::
:::

## Hiperparâmetros de Treinamento

<center>**Tabela 2: Configuração Experimental**</center>

| Parâmetro | Configuração | Justificativa |
|:-----------------------|:-----------------------|:-----------------------|
| **Otimizador** | Adam | Momento Adaptativo para convergência rápida. |
| **Learning Rate** | 0.001 | Padrão inicial, ajustado dinamicamente. |
| **Inicialização** | He Normal | Ideal para função de ativação ReLU. |
| **Scheduler** | `ReduceLROnPlateau` | Ajuste fino (*fine-tuning*) em mínimos locais. |
| **Critério Parada** | `EarlyStopping` | Monitoramento da *Val Loss* (Patience=12). |

<center><small>Fonte: Elaborada pelo autor.</small></center>

# 4. Resultados Experimentais

## Dinâmica de Treinamento

------------------------------------------------------------------------

<center>**Figura 4: Histórico de Convergência (Acurácia e Perda)**</center>

::: r-stretch
![](figuras_artigo/fig_desempenho_completo.png){width="100%" fig-align="center"}
:::

<center><small>Legenda: (A-B) LSTM; (C-D) CNN. Linha Sólida: Treino; Tracejada: Validação. Fonte: Elaborada pelo autor.</small></center>

------------------------------------------------------------------------

## Métricas Finais (Conjunto de Teste)

Avaliação em **2.947 amostras** independentes.

<center>**Tabela 3: Comparativo de Desempenho**</center>

| Métrica               | Stacked LSTM | Pure 1D-CNN |
|:----------------------|:------------:|:-----------:|
| **Acurácia Global**   |     90%      |   **93%**   |
| **Precision (Média)** |     0.90     |  **0.94**   |
| **F1-Score (Média)**  |     0.90     |  **0.93**   |

:::: {.fragment .fade-up}
::: callout-tip
## Veredito

A **1D-CNN** é superior tanto em acurácia quanto em eficiência computacional para janelas curtas (2.56s).
:::
::::

## Análise de Erros: A Matriz de Confusão

------------------------------------------------------------------------

<center>**Figura 5: Matrizes de Confusão Normalizadas**</center>

::: r-stretch
![](figuras_artigo/fig_matrizes_confusao_lado_a_lado.png){width="90%" fig-align="center"}
:::

<center><small>Legenda: Esquerda (LSTM), Direita (CNN). Fonte: Elaborada pelo autor.</small></center>

# 5. Conclusão

## Considerações Finais

1.  **Eficácia do Deep Learning:** Ambas as arquiteturas eliminaram a necessidade de *feature engineering* manual, superando 90% de acurácia global.
2.  **Superioridade na Classe "Deitado":** A **1D-CNN** atingiu **100% de acurácia** nesta categoria. A capacidade de filtrar a forma da onda (morfologia) mostrou-se mais eficaz para isolar essa postura do que a memória sequencial.
3.  **Generalização (CNN vs. LSTM):** A LSTM apresentou indícios de **superajuste (*overfitting*)**: apesar do rápido decaimento da perda no treinamento, sua generalização nos dados de teste foi inferior, indicando memorização de ruído nas sequências.
4.  **Eficiência para Deploy:** A CNN oferece o melhor equilíbrio, sendo computacionalmente mais leve (paralelizável) e estatisticamente mais robusta.

## Referências Bibliográficas

-   **Anguita, D. et al.** (2013). A Public Domain Dataset for HAR. *ESANN*.
-   **Goodfellow, I. et al.** (2016). *Deep Learning*. MIT Press.
-   **Hochreiter, S. & Schmidhuber, J.** (1997). Long Short-Term Memory.
-   **Kiranyaz, S. et al.** (2021). 1D Convolutional Neural Networks: A Survey.

## 

::: r-fit-text
**Obrigado!**
:::

<center>ramon.tavares\@ufpe.br</center>
