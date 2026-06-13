# modeling_sentiment_analysis

O objetivo desse repositório é a elaboração, condução e execução do trabalho final da Disciplina de Pensamento Analítico de Dados. Aos integrantes (**Felipe Albernaz**, **Gustavo Leite**, **Gustavo Barros** e **João Andrade**), foi endereçado um projeto de análise de sentimentos (denominado FMF). A esse respeito, tendo em vista que todo projeto de ciência de dados segue uma estrutura concisa dentro do modelo AGEMC (**ASK, GET, EXPLORE, MODEL e COMMUNICATE**), foi feita a seguinte proposta: dentre os ciclos de ASK, GET, EXPLORE, MODEL e COMMUNICATE presentes no projeto FMF apresentado, o grupo deve escolher um dos ciclos para ser alterado, desenvolvendo um projeto distinto nos ciclos subsequentes.

Nas seções abaixo, é definido:

1. [Análise do Projeto FMF](#1-análise-do-projeto-fmf)
2. [Definição dos Ciclos AGEMC no Projeto FMF](#2-definição-dos-ciclos-agemc-no-projeto-fmf)
3. [Execução do Projeto FMF](#3-execução-do-projeto-fmf)
4. [Definição do Ciclo de Divergência](#4-definição-do-ciclo-de-divergência)
5. [Execução do Projeto Original](#5-execução-do-projeto-original)
6. [Cronograma de Desenvolvimento](#6-cronograma-de-desenvolvimento)

> **Fase 1** (Semanas 1–2): Seções 1, 2 e 3  
> **Fase 2** (Semanas 3–4): Seções 4 e 5  
> **Contínuo**: Seção 6 (atualizada ao longo do progresso)

---

## 1. Análise do Projeto FMF

O projeto FMF é um guia prático de construção de um modelo de aprendizado de máquina voltado para **análise de sentimentos em críticas de filmes**, desenvolvido com a linguagem Python e a ferramenta **scikit-learn** (sklearn). Seu objetivo é analisar qual modelo de Machine Learning melhor se enquadra na análise de sentimentos (output) dado a crítica de uum filme (input).

### Contexto

A análise de sentimentos é uma das tarefas mais recorrentes em Processamento de Linguagem Natural (PLN) e tem aplicações diretas em sistemas de recomendação, monitoramento de reputação de marcas e pesquisa de opinião. O projeto FMF utiliza esse domínio como pano de fundo para introduzir, de forma didática, os principais conceitos e ferramentas de um pipeline de Machine Learning supervisionado.

### Problema Abordado

O problema tratado é de **classificação binária de texto**: dado o texto de uma crítica cinematográfica, o modelo deve classificá-la como `positive` (positiva) ou `negative` (negativa). Trata-se de um problema supervisionado, pois os rótulos de saída são conhecidos e estão presentes no dataset utilizado.

### Tecnologias e Bibliotecas Utilizadas

O projeto FMF usa as seguintes bibliotecas: 
1. `pandas` para a leitura, manipulação e amostragem do dataset;
2. `imblearn` para o tratamento de desbalancamento de classes via undersampling;
3. `sckit-learn` para a vetorização de texto, treinamento de modelos, avaliação e tuning.

### Estrutura Geral do Pipeline

O projeto segue uma pipeline bem delimitada, composta pelas seguintes etapas:

1. **Aquisição dos dados**: download do dataset IMDB (50.000 críticas) via Kaggle.
2. **Preparação dos dados**: amostragem, balanceamento de classes e divisão em conjuntos de treino e teste.
3. **Vetorização textual**: conversão do texto em representações numéricas com TF-IDF.
4. **Modelagem**: treinamento e comparação de quatro classificadores (SVM, Árvore de Decisão, Naive Bayes e Regressão Logística).
5. **Avaliação**: análise de desempenho com acurácia, F1-score, relatório de classificação e matriz de confusão.
6. **Otimização**: ajuste fino dos hiperparâmetros do melhor modelo via `GridSearchCV`.

### Características Notáveis

- O projeto introduz deliberadamente um **cenário de desbalanceamento de classes** (9.000 positivos × 1.000 negativos) para motivar o uso de técnicas de reamostragem, tornando-o mais representativo de situações reais de ciência de dados.
- A comparação entre **CountVectorizer e TF-IDF** oferece uma discussão conceitual relevante sobre representação textual, justificando a escolha do TF-IDF pela sua capacidade de ponderar a importância relativa das palavras.
- A etapa de **GridSearchCV** demonstra boas práticas de otimização de modelos, evitando ajuste manual de hiperparâmetros.
- O modelo vencedor (**SVM com kernel linear**, C=1) alcança **84% de acurácia** e F1-scores equilibrados entre as classes (~0,84 e ~0,83), indicando boa generalização mesmo com o dataset reduzido.

### Limitações Observadas

- O pipeline não inclui etapas de **pré-processamento textual** mais sofisticadas, como remoção de HTML (visível em algumas entradas do dataset), lematização ou stemming, o que pode limitar a qualidade da vetorização.
- A amostra de 10.000 registros, embora suficiente para fins didáticos, representa apenas 20% do dataset original, podendo subestimar o potencial dos modelos em escala completa.
- Apenas algoritmos clássicos de ML são avaliados; abordagens baseadas em **embeddings** (Word2Vec, BERT, etc.) não são contempladas.

---

## 2. Definição dos Ciclos AGEMC no Projeto FMF

### ASK — Formulação do Problema:

O ponto de partida do projeto FMF é a definição de uma pergunta objetiva:

> **"Qual modelo de Machine Learning melhor se enquadra na análise de sentimentos (output) dado a crítica de uum filme (input)?"**

O problema é classificado como **classificação binária de texto** (*binary text classification*), em que a entrada (*input*) é o texto da crítica e a saída (*output*) é o rótulo de sentimento, sendo positivo ou negativo. A escolha desse problema justifica o uso de algoritmos supervisionados, conforme heurísticas, pois tanto as variáveis de entrada quanto as de saída são bem identificadas e rotuladas.

---

### GET (Obtenção dos Dados):

Os dados utilizados provêm do **dataset IMDB de 50.000 críticas de filmes**, disponibilizado publicamente na plataforma **Kaggle**. O dataset possui duas colunas: 

1. `review`: texto da crítica escrita pelo usuário;
2. `sentimet`: rótulo de sentimento como positivo ou negativo.

Para viabilizar o treinamento em tempo razoável, é extraída uma amostra de **10.000 linhas** — intencionalmente desequilibrada, com 9.000 críticas positivas e 1.000 negativas... para introduzir a problemática de classes desbalanceadas nas etapas seguintes.

---

### EXPLORE (Exploração e Preparação dos Dados)

Esta etapa engloba três atividades principais:

**a) Tratamento de Classes Desbalanceadas**

A amostra gerada apresenta desbalanceamento: 9.000 exemplos positivos contra 1.000 negativos. Para corrigir esse desvio, é aplicada a técnica de **undersampling aleatório** por meio da classe `RandomUnderSampler` da biblioteca `imblearn`, que reduz a classe majoritária ao tamanho da classe minoritária. O resultado é um dataset balanceado (`df_review_bal`) com **1.000 exemplos por classe**.

**b) Divisão em Conjuntos de Treino e Teste**

O dataset balanceado é dividido com `train_test_split` do scikit-learn, reservando **33% para teste** e **67% para treino**:

**c) Representação Textual (Bag of Words)**

Como modelos de aprendizado de máquina operam sobre vetores numéricos, o texto das críticas precisa ser convertido. São apresentadas duas abordagens dentro de Bag of Words:

- **CountVectorizer**: registra a frequência absoluta de cada palavra no documento. Palavras como "love", "hate" e "code" recebem o mesmo peso caso apareçam o mesmo número de vezes.
- **TF-IDF** (*Term Frequency – Inverse Document Frequency*): atribui pesos que refletem a importância relativa de cada palavra no corpus. Palavras comuns a muitos documentos recebem peso menor, enquanto palavras distintivas recebem peso maior.

O projeto opta pelo **TF-IDF**, pois o objetivo é identificar palavras representativas de cada classe de sentimento:

A matriz resultante possui **1.340 críticas × 20.625 palavras** (vocabulário completo do conjunto de treino).

---

### MODEL — Seleção e Treinamento dos Modelos:

Dado o caráter supervisionado e de classificação do problema, são treinados e comparados **quatro algoritmos**:

1. `Support Vector Machine (SVM)`, com acurácia média de 84%;
2. `Logistic Regression`, com acurácia média de 83%;
3. `Decision Tree`, com acurácia média de 64%;
4. `Naive Bayes`, com acurácia média de 63%.

O treinamento segue o mesmo padrão para todos os modelos: fornecer os vetores TF-IDF do conjunto de treino e os rótulos correspondentes.

Após a seleção do SVM como modelo de melhor desempenho, é realizado o ajuste de hiperparâmetros com `GridSearchCV`, que realiza busca sobre combinações de parâmetros via validação cruzada de 5 folds.

---

### COMMUNICATE (Avaliação e Comunicação dos Resultados):

Os resultados são avaliados com quatro métricas:

1. **Acurácia Média**: proporção de predições corretas sobre o total. O SVM alcança **84%**.
2. **F1-Score**: média harmônica entre precisão e revocação, relevante em cenários com classes desbalanceadas. Para o SVM: **0,84** (positivo) e **0,83** (negativo).
3. **Relatório de Classificação**: exibe precisão, revocação e F1-score por classe, além da acurácia geral.
4. **Matriz de Confusão**: tabela que decompõe as predições em verdadeiros positivos (290), falsos positivos (45), falsos negativos (60) e verdadeiros negativos (265), permitindo identificar padrões de erro do modelo.

A comunicação dos resultados evidencia o SVM como a solução mais robusta para o problema de classificação binária de sentimentos no dataset IMDB.

---

## 3. Execução do Projeto FMF

> *A completar.*

---

## 4. Definição do Ciclo de Divergência

> *A completar.*

---

## 5. Execução do Projeto Original

> *A completar.*

---

## 6. Cronograma de Desenvolvimento
 
> *Atualizado continuamente ao longo do progresso do projeto.*
 
**Fase 1**
 
1. _Análise do Projeto FMF_
   a) status: concluída ✅
2. _Definição dos Ciclos AGEMC no Projeto FMF_
   a) status: concluída ✅
3. _Execução do Projeto FMF_
   a) status: em progresso
**Fase 2**
 
4. _Definição do Ciclo de Divergência_
   a) status: em progresso
5. _Execução do Projeto Original_
   a) status: em progresso
---

*Trabalho Final da Disciplina de Pensamento Analítico de Dados*
*Bacharelado em Inteligência Artificial*

