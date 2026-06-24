# modeling_sentiment_analysis

O objetivo desse repositório é a elaboração, condução e execução do trabalho final da Disciplina de Pensamento Analítico de Dados. Aos integrantes (**Felipe Albernaz**, **Gustavo Leite**, **Gustavo Barros** e **João Guilherme**), foi endereçado um projeto de análise de sentimentos (denominado FMF). A esse respeito, tendo em vista que todo projeto de ciência de dados segue uma estrutura concisa dentro do modelo AGEMC (**ASK, GET, EXPLORE, MODEL e COMMUNICATE**), foi feita a seguinte proposta: dentre os ciclos de ASK, GET, EXPLORE, MODEL e COMMUNICATE presentes no projeto FMF apresentado, o grupo deve escolher um dos ciclos para ser alterado, desenvolvendo um projeto distinto nos ciclos subsequentes.

Nas seções abaixo, é definido:

1. [Análise do Projeto FMF](#1-análise-do-projeto-fmf): *Finalizado*
2. [Definição dos Ciclos AGEMC no Projeto FMF](#2-definição-dos-ciclos-agemc-no-projeto-fmf): *Finalizado*
3. [Execução do Projeto FMF](#3-execução-do-projeto-fmf): *Finalizado*
4. [Definição do Ciclo de Divergência](#4-definição-do-ciclo-de-divergência): *Finalizado*
5. [Execução do Projeto Original](#5-execução-do-projeto-original): _Em andamento_
6. [Cronograma de Desenvolvimento](#6-cronograma-de-desenvolvimento): *contínuo*
7. [Desenvolvimento por Semana](#7-desenvolvimento-por-semana): *contínuo*

---

## 1. Análise do Projeto FMF

O projeto FMF é um guia prático de construção de um modelo de aprendizado de máquina voltado para **análise de sentimentos em críticas de filmes**, desenvolvido com a linguagem Python e a ferramenta **scikit-learn** (sklearn). Seu objetivo é analisar qual modelo de Machine Learning melhor se enquadra na análise de sentimentos (output) dado a crítica de um filme (input).

O projeto original, desenvolvido por "thepycoach", pode ser encontrado através do link ao seu repositório: [GitHub do Autor](https://github.com/thepycoach/data-science-projects/blob/main/Sklearn%20Tutorial%20(Binary%20Text%20Classification).ipynb)

### Contexto

A análise de sentimentos é uma das tarefas mais recorrentes em Processamento de Linguagem Natural (PLN) e tem aplicações diretas em sistemas de recomendação, monitoramento de reputação de marcas e pesquisa de opinião. O projeto FMF utiliza esse domínio como pano de fundo para introduzir, de forma didática, os principais conceitos e ferramentas de um pipeline de Machine Learning supervisionado.

### Problema Abordado

O problema tratado é de **classificação binária de texto**: dado o texto de uma crítica cinematográfica, o modelo deve classificá-la como positive (positiva) ou negative (negativa). Trata-se de um problema supervisionado, pois os rótulos de saída são conhecidos e estão presentes no dataset utilizado.

### Tecnologias e Bibliotecas Utilizadas

O projeto FMF usa as seguintes bibliotecas: 
1. `pandas` para a leitura, manipulação e amostragem do dataset;
2. `imblearn` para o tratamento de desbalanceamento de classes via undersampling;
3. `scikit-learn` para a vetorização de texto, treinamento de modelos, avaliação e tuning.

### Estrutura Geral do Pipeline

O projeto segue uma pipeline bem delimitada, composta pelas seguintes etapas:

1. `Aquisição dos dados`: download do dataset IMDB (50.000 críticas) via Kaggle.
2. `Preparação dos dados`: amostragem, balanceamento de classes e divisão em conjuntos de treino e teste.
3. `Vetorização textual`: conversão do texto em representações numéricas com TF-IDF.
4. `Modelagem`: treinamento e comparação de quatro classificadores (SVM, Árvore de Decisão, Naive Bayes e Regressão Logística).
5. `Avaliação`: análise de desempenho com acurácia, F1-score, relatório de classificação e matriz de confusão.
6. `Otimização`: ajuste fino dos hiperparâmetros do melhor modelo via GridSearchCV.

### Características

1. O projeto introduz um **cenário de desbalanceamento de classes** (9.000 positivos × 1.000 negativos);
2. A comparação entre **CountVectorizer e TF-IDF**, com a escolha do TF-IDF pela sua capacidade de ponderar a importância relativa das palavras;
3. Etapa de **GridSearchCV** na otimização de modelos, evitando ajuste manual de hiperparâmetros;
4. O modelo vencedor (**SVM com kernel linear**, C=1) alcança **84% de acurácia** e F1-scores equilibrados entre as classes (~0,84 e ~0,83).

### Limitações Observadas

1. A amostra de 10.000 registros, embora suficiente para fins didáticos, representa apenas 20% do dataset original, podendo subestimar o potencial dos modelos em escala completa;
2. Apenas algoritmos clássicos de ML são avaliados... abordagens baseadas em embeddings (Word2Vec, BERT, etc.) não são analisadas.

---

## 2. Definição dos Ciclos AGEMC no Projeto FMF

### ASK (Formulação do Problema):

O ponto de partida do projeto FMF é a definição da pergunta: **"Qual modelo de Machine Learning melhor se enquadra na análise de sentimentos (output) dado a crítica de um filme (input)?"**

Classificamos o problema como classificação binária de texto (binary text classification), em que a entrada (input) é o texto da crítica e a saída (output) é o rótulo de sentimento, sendo positivo ou negativo. A escolha desse problema justifica o uso de algoritmos supervisionados, conforme heurísticas, pois tanto as variáveis de entrada quanto as de saída são bem identificadas e rotuladas.

---

### GET (Obtenção dos Dados):

Os dados utilizados provêm do dataset IMDB de 50.000 críticas de filmes, disponibilizado publicamente na plataforma Kaggle. O dataset possui duas colunas: 

1. `review`: texto da crítica escrita pelo usuário;
2. `sentiment`: rótulo de sentimento como positivo ou negativo.

Para viabilizar o treinamento em tempo razoável, é extraída uma amostra de 10.000 linhas — intencionalmente desequilibrada, com 9.000 críticas positivas e 1.000 negativas... para introduzir a problemática de classes desbalanceadas nas etapas seguintes.

Os dados estão disponíveis através do link: [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

### EXPLORE (Exploração e Preparação dos Dados):

Esta etapa engloba três atividades principais:

**a) Tratamento de Classes Desbalanceadas**

A amostra gerada apresenta desbalanceamento: 9.000 exemplos positivos contra 1.000 negativos. Para corrigir esse desvio, é aplicada a técnica de undersampling aleatório por meio da classe RandomUnderSampler da biblioteca imblearn, que reduz a classe majoritária ao tamanho da classe minoritária. O resultado é um dataset balanceado (df_review_bal) com 1.000 exemplos por classe.

**b) Divisão em Conjuntos de Treino e Teste**

O dataset balanceado é dividido com train_test_split do scikit-learn, reservando **33% para teste** e **67% para treino**:

**c) Representação Textual (Bag of Words)**

Como modelos de aprendizado de máquina operam sobre vetores numéricos, o texto das críticas precisa ser convertido. São apresentadas duas abordagens dentro de Bag of Words:

1. `CountVectorizer`: registra a frequência absoluta de cada palavra no documento. Palavras como "love", "hate" e "code" recebem o mesmo peso caso apareçam o mesmo número de vezes.
2. `TF-IDF` (Term Frequency – Inverse Document Frequency): atribui pesos que refletem a importância relativa de cada palavra no corpus. Palavras comuns a muitos documentos recebem peso menor, enquanto palavras distintivas recebem peso maior.

O projeto opta pelo **TF-IDF**, pois o objetivo é identificar palavras representativas de cada classe de sentimento:

A matriz resultante possui **1.340 críticas × 20.625 palavras** (vocabulário completo do conjunto de treino).

---

### MODEL — Seleção e Treinamento dos Modelos:

Dado o caráter supervisionado e de classificação do problema, são treinados e comparados quatro algoritmos:

1. `Support Vector Machine (SVM)`, com acurácia média de 84%;
2. `Logistic Regression`, com acurácia média de 83%;
3. `Decision Tree`, com acurácia média de 64%;
4. `Naive Bayes`, com acurácia média de 63%.

O treinamento segue o mesmo padrão para todos os modelos: fornecer os vetores TF-IDF do conjunto de treino e os rótulos correspondentes.

Após a seleção do SVM como modelo de melhor desempenho, é realizado o ajuste de hiperparâmetros com GridSearchCV, que realiza busca sobre combinações de parâmetros via validação cruzada de 5 folds.

---

### COMMUNICATE (Avaliação e Comunicação dos Resultados):

Os resultados são avaliados com quatro métricas:

1. `Acurácia Média`: proporção de predições corretas sobre o total. O SVM alcança 84%.
2. `F1-Score`: média harmônica entre precisão e revocação, relevante em cenários com classes desbalanceadas. Para o SVM: 0,84(positivo) e 0,83 (negativo).
3. `Relatório de Classificação`: exibe precisão, revocação e F1-score por classe, além da acurácia geral.
4. `Matriz de Confusão`: tabela que decompõe as predições em verdadeiros positivos (290), falsos positivos (45), falsos negativos (60) e verdadeiros negativos (265), permitindo identificar padrões de erro do modelo.

A comunicação dos resultados evidencia o SVM como a solução mais robusta para o problema de classificação binária de sentimentos no dataset IMDB.

---

## 3. Execução do Projeto FMF

A partir do repositório oficial do autor, "thepycoach", reproduzimos os métodos, técnicas e resultados obtidos no projeto FMF. O objetivo da ação é estudar, com base numa execução prática, como o projeto foi modelado e como se aplicam de fato todas as ferramentas utilizadas.

Sendo assim, neste repositório, `modeling_sentiment_analysis`, foi criada a pasta "FMF", contendo um notebook Jupyter desenvolvido pelo autor do projeto.

---

## 4. Definição do Ciclo de Divergência

Com base nos estudos realizados no projeto FMF, foi decidido como *primeira ideia* o ciclo `MODEL` como ciclo de divergência. A escolha foi fundamentada na escolha de apenas algoritmos clássicos de Machine Learning no benchmark realizado pelo projeto. A proposta é avaliar como algoritmos de embeddings se comportam, de forma comparativa, na tarefa de análise de sentimento em críticas cinematográficas.

### Observações Importantes:

1. `Limites`: no ciclo GET, apenas uma parte dos dados foi escolhida para a realização da tarefa. Como o projeto original deve ser desenvolvido a partir do ciclo de divergência escolhido, apenas essa parte dos dados será usada. Isso implica, naturalmente, em dúvidas sobre o desempenho de um modelo de embeddings com tão poucos dados;
2. `Compatibilidade com a Ideia`: a pergunta de pesquisa, do projeto FMF, é: **"Qual modelo de Machine Learning melhor se enquadra na análise de sentimentos (output) dado a crítica de um filme (input)?"**. Tendo em vista que em "Modelos de Machine Learning" também estão incluídos Modelos de Embeddings, o time avaliou que a ideia é compatível;
3. `Processo não-Linear`: devido ao limite citado acima, entende-se que o processo pode não ser bem sucedido. Caso isso aconteça, um novo ciclo de divergência será escolhido... o que configura a não-linearidade de um projeto, conforme o modelo AGEMC;

> **Atualização (pós-execução):** o risco de desempenho com poucos dados (item 1) foi investigado na prática. Além da amostra reduzida de 2.000 registros, o projeto Original foi reexecutado em escala, com 20.000 registros (ver Seção 5). Os resultados confirmaram a viabilidade dos modelos de embeddings em ambas as escalas, de modo que **não foi necessário escolher um novo ciclo de divergência** (item 3).

---

## 5. Execução do Projeto Original

A partir do ciclo de divergência definido na Seção 4, o projeto Original mantém **exatamente o mesmo problema, os mesmos dados e os mesmos classificadores finais** do projeto FMF, alterando apenas o ciclo `MODEL` — mais especificamente, a **representação textual**. Em vez de converter as críticas apenas com TF-IDF, comparamos quatro formas de representar o texto, da mais lexical à mais semântica, e medimos o impacto dessa troca no desempenho da classificação de sentimento.

As execuções estão registradas em três notebooks na pasta `Original/`:

1. `Sentiment Analysis with Embeddings (Original).ipynb` — execução **local em GPU**, sobre a amostra reduzida de **2.000 registros** (1000/1000), idêntica em dados ao FMF;
2. `Sentiment_Analysis_with_Embeddings_(Original, 10k dados).ipynb` — execução **em escala no Google Colab**, sobre **20.000 registros** (10.000/10.000);
3. `Sentiment_Analysis_with_RoBERTa_(Fine_tuning).ipynb` — experimento **complementar de fine-tuning** (RoBERTa-base), também em escala (20k) no Google Colab.

### Representações comparadas

1. `TF-IDF` — reproduz a linha de base do FMF (frequência de termos ponderada).
2. `BM25` — baseline lexical mais forte que o TF-IDF (saturação de frequência de termo + normalização por comprimento do documento).
3. `BGE-large-en-v1.5` — embedding denso **especialista em inglês**, com janela de contexto de **512 tokens** (críticas mais longas são truncadas).
4. `BGE-M3` — embedding denso **multilíngue de contexto longo** (até **8192 tokens**), capaz de ler a crítica inteira sem truncar.

### Metodologia

Para isolar a variável "representação textual", todas as quatro representações alimentam **os mesmos quatro classificadores do projeto FMF** — `SVM (linear, C=1)`, `Regressão Logística`, `Naive Bayes` e `Árvore de Decisão` — sobre **o mesmo conjunto de dados balanceado (1000/1000)** e a **mesma divisão treino/teste** (67%/33%, com semente fixa) usados no FMF. Os embeddings densos (BGE) são gerados uma única vez por modelo e usados como vetores de entrada (features) para os classificadores clássicos, sem qualquer ajuste fino (*fine-tuning*) das redes. A métrica principal de comparação é o **F1 macro**, coerente com o ciclo COMMUNICATE do FMF.

### Resultados

A tabela abaixo apresenta o F1 macro de cada representação, por classificador, e o melhor resultado de cada uma (execução local, **2.000 registros**):

| Representação | SVM (linear) | Regressão Logística | Naive Bayes | Árvore de Decisão | Melhor F1 macro |
|---|---|---|---|---|---|
| TF-IDF (baseline FMF) | 0,841 | 0,830 | 0,632 | 0,642 | **0,841** |
| BM25 | 0,845 | 0,851 | 0,635 | 0,636 | **0,851** |
| BGE-large-en-v1.5 | 0,948 | 0,950 | 0,948 | 0,897 | **0,950** |
| BGE-M3 | 0,927 | 0,930 | 0,912 | 0,810 | **0,930** |

### Análise

1. **Os embeddings densos superam os métodos lexicais com folga.** O melhor embedding (BGE-large) alcança **0,950** de F1 macro contra **0,841** do TF-IDF do FMF — um salto de aproximadamente **11 pontos** —, confirmando a hipótese que motivou o ciclo de divergência.
2. **O especialista em inglês vence o multilíngue.** O BGE-large-en-v1.5 (0,950) supera o BGE-M3 (0,930) mesmo truncando em 512 tokens. Como o dataset IMDB é integralmente em inglês, o modelo dedicado à língua leva vantagem sobre o multilíngue, e ler a crítica inteira (contexto longo do BGE-M3) não compensou esse ganho de especialização.
3. **BM25 e TF-IDF ficam próximos e no piso da comparação** (0,851 e 0,841). Por serem ambos lexicais (*bag of words*), não capturam significado, o que evidencia que o ganho dos embeddings vem da **semântica**, e não apenas de uma técnica de ponderação melhor.
4. **O risco de "poucos dados" não se concretizou.** Mesmo com apenas 2.000 registros (a mesma amostra reduzida do FMF), os embeddings pré-treinados — usados como features fixas, sem *fine-tuning* — já entregam ~0,95 de F1 macro. Isso atenua diretamente a preocupação levantada na Seção 4 sobre o desempenho de modelos de embeddings com volume reduzido de dados.
5. **A Regressão Logística foi o melhor classificador em três das quatro representações** (BM25, BGE-M3 e BGE-large), perdendo apenas no TF-IDF, e por margem mínima. Isso reforça que, sobre embeddings, um classificador linear simples já é suficiente para extrair o sinal de sentimento.
6. **O Naive Bayes é ressuscitado pelos embeddings.** Nos métodos lexicais, o Naive Bayes é o pior ou quase-pior classificador (~0,632–0,635). Com o BGE-large, ele atinge **0,948** — praticamente empatado com o SVM e apenas 0,002 abaixo da Regressão Logística. Isso revela que a fraqueza do Naive Bayes no projeto FMF não é intrínseca ao classificador, mas sim uma consequência da representação esparsa do TF-IDF: com vetores semânticos densos, o pressuposto de independência condicional do Naive Bayes deixa de ser tão prejudicial.
7. **A Árvore de Decisão melhora substancialmente, mas permanece a mais fraca.** Ela sai de ~0,642 (TF-IDF) para **0,897** (BGE-large) — um ganho de ~25 pontos —, porém é o único classificador que não alcança 0,90 com o BGE-M3 (0,810). Árvores de decisão têm maior dificuldade com espaços de alta dimensionalidade e contínuos, o que explica a desvantagem residual mesmo sobre embeddings.
8. **Os embeddings comprimem as diferenças entre classificadores.** No TF-IDF, a amplitude entre o melhor (SVM, 0,841) e o pior (Naive Bayes, 0,632) classificador é de **20,9 pontos**. Com o BGE-large, essa amplitude cai para apenas **5,3 pontos** (0,950 vs. 0,897). A qualidade da representação textual se torna, portanto, o principal determinante do desempenho — mais do que a escolha do algoritmo de classificação.

### Execução em escala (10k/10k)

Para investigar diretamente o risco de "poucos dados" levantado na Seção 4, o projeto foi reexecutado em escala maior — **10.000 críticas por classe (20.000 no total)**, contra os 2.000 da execução local —, mantendo o mesmo pipeline (mesmos quatro classificadores, mesma divisão 67%/33%). A tabela abaixo apresenta o F1 macro completo por representação e classificador nessa escala:

| Representação | SVM (linear) | Regressão Logística | Naive Bayes | Árvore de Decisão | Melhor F1 macro |
|---|---|---|---|---|---|
| TF-IDF (baseline FMF) | 0,876 | 0,874 | 0,637 | 0,691 | **0,876** |
| BM25 | 0,861 | 0,877 | 0,628 | 0,693 | **0,877** |
| BGE-large-en-v1.5 | 0,945 | 0,943 | 0,938 | 0,893 | **0,945** |
| BGE-M3 | 0,920 | 0,920 | 0,903 | 0,810 | **0,920** |

A tabela a seguir compara o **melhor F1 macro** de cada representação nas duas escalas:

| Representação | Melhor F1 — 2k (local) | Melhor F1 — 20k (Colab) |
|---|---|---|
| TF-IDF (baseline FMF) | 0,841 | 0,876 |
| BM25 | 0,851 | 0,877 |
| BGE-large-en-v1.5 | 0,950 | 0,945 |
| BGE-M3 | 0,930 | 0,920 |

Quatro observações emergem dessa comparação:

1. **Os métodos lexicais melhoram com mais dados.** TF-IDF e BM25 sobem cerca de 3 pontos de F1 macro (de ~0,84–0,85 para ~0,88), beneficiados pelo vocabulário maior e pelo maior número de exemplos. O Naive Bayes, contudo, permanece estagnado (~0,63) independentemente da escala nos métodos lexicais, confirmando que seu teto é imposto pela representação, não pelo volume de dados.
2. **Os embeddings permanecem estáveis e mais confiáveis.** BGE-large e BGE-M3 variam pouco (queda de ~0,5 a 1 ponto). Essa leve redução não indica piora real: o conjunto de teste de 20k (6.600 exemplos) é dez vezes maior que o de 2k (660), tornando a estimativa de ~0,945 **menos sujeita a variância** que o 0,950 da amostra pequena.
3. **A vantagem dos embeddings diminui, mas se mantém.** A diferença entre o melhor embedding e o melhor método lexical cai de ~0,11 (2k) para ~0,07 (20k) — os embeddings densos continuam à frente, e a hierarquia entre os modelos (BGE-large > BGE-M3 > BM25 ≈ TF-IDF) se mantém em ambas as escalas.
4. **O Naive Bayes sobre embeddings permanece competitivo em escala.** Com BGE-large e 20k registros, o Naive Bayes alcança **0,938** — apenas 0,007 abaixo do SVM (0,945). O padrão observado na execução local se repete: a ressurreição do Naive Bayes pelos embeddings é um fenômeno robusto à escala de dados.

### Fine-tuning (RoBERTa)

As abordagens acima usam os modelos como extratores de *features* **congeladas**. Como experimento complementar, o ciclo `MODEL` foi levado adiante com **fine-tuning** do `RoBERTa-base` sobre os mesmos 20.000 registros — ajustando todos os pesos da rede à tarefa de sentimento. O experimento avalia duas variantes: (a) **fine-tuning E2E** com uma cabeça linear treinada diretamente no topo do RoBERTa (*Linear Head E2E*) e (b) uso das **features extraídas do RoBERTa já ajustado** como entrada para os quatro classificadores clássicos. A execução foi feita no Google Colab (GPU), com truncamento em 512 tokens (limite arquitetural do RoBERTa).

| Abordagem (20k) | F1 macro |
|---|---|
| TF-IDF / BM25 (lexical) | ~0,877 |
| BGE-M3 (congelado) | 0,920 |
| BGE-large (congelado) | 0,945 |
| Fine-tuned features — Árvore de Decisão | 0,943 |
| Fine-tuned features — Regressão Logística | 0,948 |
| Fine-tuned features — SVM (linear) | 0,949 |
| Fine-tuned features — Naive Bayes | 0,950 |
| RoBERTa fine-tuning E2E (Linear Head) | **0,950** |

Três observações se destacam:

1. **O fine-tuning E2E assume a liderança (0,950), mas por margem mínima** sobre o melhor embedding congelado (BGE-large, 0,945), ao custo de tempo de treino significativamente maior — reforçando o bom custo-benefício das *features* congeladas.
2. **O Naive Bayes sobre features fine-tunadas empata com o E2E (0,950).** Após o ajuste fino da rede, as features extraídas tornam-se ainda mais informativas e alinhadas à tarefa, o que beneficia desproporcionalmente o Naive Bayes e elimina completamente a desvantagem que ele apresenta sobre métodos lexicais. Esse resultado estende o padrão observado com embeddings congelados: a qualidade da representação é o fator dominante no desempenho do Naive Bayes.
3. **Todos os classificadores convergem para uma faixa estreita** (0,943–0,950) sobre as features fine-tunadas — amplitude de apenas 0,7 ponto contra os ~21 pontos do TF-IDF. O fine-tuning comprime ainda mais as diferenças entre classificadores do que os embeddings congelados, evidenciando que, com uma representação suficientemente boa, a escolha do algoritmo de classificação se torna secundária.

### Conclusão do ciclo

Respondendo à pergunta de pesquisa do projeto FMF — *"Qual modelo de Machine Learning melhor se enquadra na análise de sentimentos?"* — sob a ótica do ciclo de divergência, a resposta é que a **escolha da representação textual importa muito mais do que a escolha do classificador**: trocar o TF-IDF por um embedding denso especialista (BGE-large-en-v1.5) elevou o F1 macro de 0,841 para 0,950 na amostra reduzida, mantendo todo o restante do pipeline inalterado. Esse resultado se mostrou **robusto à escala**: na execução com 20.000 registros, os embeddings densos seguiram à frente dos métodos lexicais (~0,945 contra ~0,877), e a hierarquia entre representações se manteve. O achado mais surpreendente do projeto é a **ressurreição do Naive Bayes**: o classificador que ocupa o último lugar no projeto FMF (~0,632 com TF-IDF) quase empata com os melhores modelos ao receber vetores semânticos (~0,948–0,950), revelando que sua fraqueza original era da representação, não do algoritmo. Por fim, levar o ciclo `MODEL` ao **fine-tuning E2E** (RoBERTa-base) liderou a comparação (0,950 em 20k), porém por margem estreita sobre os embeddings congelados — situando o trabalho como uma exploração de custo-benefício entre métodos lexicais, embeddings congelados e fine-tuning.

---

## 6. Cronograma de Desenvolvimento
 
A proposta do Líder, João Guilherme, foi separar o desenvolvimento do projeto em duas fases: a primeira corresponde ao mapeamento do projeto FMF e à organização no modelo AGEMC; a segunda fase corresponde à definição do ciclo de divergência e à execução do projeto Original. Cada fase possui o tempo de duas semanas, cada. Na quarta e última semana (18/06 a 25/06), além da finalização da execução do projeto Original, será organizada a apresentação do projeto em sala de aula, no dia 25/06: o que inclui refinamento do ciclo communicate, escolha de membros para a apresentação, elaboração de slides e resolução de eventuais problemas.

Abaixo, está o status de cada tarefa das duas fases principais:
 
**Fase 1**
 
1. _Análise do Projeto FMF_

   a) status: `concluída`
2. _Definição dos Ciclos AGEMC no Projeto FMF_

   a) status: `concluída`
3. _Execução do Projeto FMF_

   a) status: `concluída`

**Fase 2**
 
1. _Definição do Ciclo de Divergência_

   a) status: `concluída`
2. _Execução do Projeto Original_

   a) status: `em andamento`
   
## 7. Desenvolvimento por Semana

### *Semana 1 (28/05 a 04/06)*
- `Finalizada` -
1. Foi atribuído a todos os membros a tarefa de ler e estudar o material disponibilizado, como um primeiro contato com o projeto que será desenvolvido nas próximas semanas.

### *Semana 2 (04/06 a 11/06)*
- `Finalizada` -
1. Foi atribuído aos membros a tarefa de organizar o projeto FMF, com base nos estudos da semana anterior, em um rascunho do que seria o ciclo de desenvolvimento AGEMC;
2. O trabalho individual foi discutido de forma coletiva com o objetivo de alcançar um consenso;
3. A organização baseado no ciclo de desenvolvimento AGEMC foi definida.

### *Semana 3 (11/06 a 18/06)*
- `Finalizada` -
1. Foi atribuído ao líder João Guilherme a criação do repositório e a escrita do documento README. O objetivo é dar início ao desenvolvimento prático do projeto FMF e do projeto Original;
2. Foi atribuído ao membro Gustavo Barros a execução do Projeto FMF a partir do repositório oficial de "thepycoach". O notebook foi adicionado na pasta "FMF";
3. Foi atribuído aos membros a discussão sobre qual seria o ponto de divergência, com base no ciclo de desenvolvimento AGEMC, no projeto FMF que seria o fundamento para a criação do projeto Original;
4. A primeira ideia levantada foi a mudança no ciclo `MODEL`, vinda a partir da segunda limitação observada durante os estudos da primeira semana. O objetivo inicial será adicionar modelos de embeddings, além do algoritmos clássicos de Machine Learning utilizados no projeto FMF;

### *Semana 4 (18/06 a 25/06)* — *última semana*
- `Em andamento` -
1. Finalização da execução do projeto Original (ciclo de divergência `MODEL`);
2. Implementação da mudança no ciclo `MODEL` (substituição do TF-IDF por embeddings: BM25, BGE-large e BGE-M3), por Felipe Albernaz;
3. Reexecução do notebook de embeddings em GPU local (amostra de 2.000 registros), com resultados reproduzíveis salvos no repositório;
4. Execução em escala (10.000/10.000 = 20.000 registros) no Google Colab, por Gustavo Barros;
5. Experimento de fine-tuning do RoBERTa-base no Colab, por Gustavo Leite, estendendo o ciclo MODEL além das representações congeladas;
6. Atualização da Seção 5 do README com a metodologia, os resultados e a comparação entre as duas escalas;
7. Refinamento do ciclo COMMUNICATE com base nos resultados obtidos;
8. Elaboração dos slides e escolha dos membros para a apresentação;
9. Apresentação do projeto em sala de aula no dia 25/06.


*Trabalho Final da Disciplina de Pensamento Analítico de Dados*

*Bacharelado em Inteligência Artificial*
