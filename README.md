# Do Som ao Sucesso: Machine Learning aplicado à popularidade musical no Spotify

Projeto de Machine Learning desenvolvido em grupo no âmbito do Mestrado em Sistemas Integrados em Business Intelligence (ISCTE-IUL), 2026, em Python e Jupyter Notebook. O artigo completo está em [`Do_Som_ao_Sucesso_Spotify_ML.pdf`](Do_Som_ao_Sucesso_Spotify_ML.pdf).

## Pergunta de investigação

Em que medida as características de áudio de uma faixa, em particular a **valência emocional**, influenciam a sua **popularidade** no Spotify, e que padrões surgem ao analisar artistas, músicas e perfis de ouvinte nos géneros **pop, hip-hop e rock**?

O estudo tem quatro objetivos:

1. Testar estatisticamente se a valência difere entre níveis de popularidade (Kruskal-Wallis, com subsampling equilibrado).
2. Desenvolver e comparar modelos de classificação supervisionada para prever a popularidade.
3. Fazer uma análise exploratória dos artistas e músicas mais populares por género.
4. Segmentar as faixas em perfis de ouvinte com K-Means.

## Dados

- Dataset: *Spotify MultiGenre Playlists Data* (Kaggle), com sete ficheiros CSV por género. Foram usados três: `pop_music_data.csv`, `hiphop_music_data.csv` e `rock_music_data.csv`.
- Os três ficheiros foram combinados em `music_data_combined.csv`, com **15.159 faixas** e **23 variáveis**: rock (8.747), pop (3.831) e hip-hop (2.581). Não há valores nulos nem duplicados relevantes.
- Variável-alvo: `popularity` (0–100), discretizada em três níveis: **Baixa** (0–40), **Média** (40–70) e **Alta** (70–100).
- Os dados **não estão incluídos** neste repositório. Podem ser descarregados do Kaggle.

## Metodologia

O projeto segue o **CRISP-DM**:

| Fase | O que foi feito |
| --- | --- |
| Business Understanding | Definição das questões de investigação e das hipóteses H0/H1 |
| Data Understanding | Estatística descritiva, histogramas e popularidade média por género |
| Data Preparation | Limpeza do texto em `Genres`, filtragem de `genre` (pop, hip-hop, rock) e discretização da popularidade |
| Modeling | Shapiro-Wilk, Kruskal-Wallis, correlação de Pearson, classificação supervisionada e K-Means |
| Evaluation | Interpretação dos testes, matriz de confusão, Silhouette Score, Davies-Bouldin e PCA |
| Deployment | Recomendações para editoras e plataformas de streaming |

**Engenharia de atributos:** `intensity` (combinação de `energy` e `loudness`) e `artist_avg_pop` (média de popularidade das faixas de cada artista).

**Classificação:** divisão 80/20 treino/teste, com KNN, Gradient Boosting, Árvore de Decisão, Random Forest e SVC.

**Clustering:** K-Means sobre 9 atributos normalizados com `StandardScaler`, com *k* entre 3 e 9, escolhido pelo Silhouette Score (o método do cotovelo não mostrou um ponto de inflexão claro) e visualizado com PCA.

**Teste de hipóteses:** como a valência não segue uma distribuição normal (Shapiro-Wilk, *p* < 0,05), usou-se o Kruskal-Wallis com subsampling equilibrado de 100 observações por grupo, para evitar o *large-sample bias*. A seed foi fixada (`seed = 42`).

## Principais resultados

| Questão | Resultado |
| --- | --- |
| A valência influencia a popularidade? | Não há evidência estatística: Kruskal-Wallis *p* = 0,397 e correlação de Pearson *r* = 0,043 |
| Melhor modelo de classificação | **SVC com accuracy de 81,13%**, usando `artist_avg_pop` como preditor |
| Segmentação das faixas | K-Means com **k = 6**: Silhouette Score 0,1725 e Davies-Bouldin 1,5837 |
| Género mais popular | O pop tem as medianas de popularidade mais elevadas |

**Classificação.** A notoriedade prévia do artista (`artist_avg_pop`) mostrou-se um preditor muito mais forte do que as propriedades acústicas da faixa, incluindo a valência. O modelo acerta bem na classe "Baixa" e na "Média", mas confunde frequentemente a classe "Alta" com a "Média".

**Artistas e músicas mais populares por género:**

| Género | Artista | Música mais popular |
| --- | --- | --- |
| Pop | Masked Wolf | *Astronaut in the Ocean* (97) |
| Hip-hop | Doja Cat | *Kiss Me More* (98) |
| Rock | Billie Eilish | *Your Power* (93) |

**Perfis de ouvinte (K-Means, k = 6):**

| Perfil | Faixas | % | Género dominante |
| --- | --- | --- | --- |
| Pop | 4.632 | 30,6% | rock |
| Rock | 4.061 | 26,8% | rock (cerca de 87%) |
| Acústica | 2.435 | 16,1% | pop |
| Hip-Hop | 1.804 | 11,9% | hip-hop (cerca de 80%) |
| Energia ao Vivo | 1.302 | 8,6% | rock |
| Instrumental | 925 | 6,1% | rock |

O PCA em duas dimensões explica 38,9% da variância total (PC1: 20,8%; PC2: 18,1%).

## Implicações

- **Editoras:** investir na notoriedade de base do artista parece pesar mais do que otimizar a valência de uma faixa.
- **Plataformas de streaming:** os 6 perfis podem apoiar a criação de playlists temáticas e campanhas segmentadas.
- **Manutenção:** como as tendências musicais mudam, os modelos devem ser reajustados com novos dados.

## Limitações

- A conclusão sobre a valência assenta num subsampling de 100 observações por grupo, e não no dataset completo. Com outra amostra, o resultado poderia variar.
- O dataset só tem *audio features*, sem fatores como redes sociais, exposição mediática ou data de lançamento.
- O Silhouette Score é modesto: os clusters são interpretáveis, mas não bem separados.
- Os dados são estáticos (um único momento), por isso `artist_avg_pop` não capta a evolução da notoriedade ao longo do tempo.
- Só foram analisados três géneros, e o rock está sobrerrepresentado (57,7% das faixas), o que pode influenciar o clustering e as comparações entre géneros.
- A `artist_avg_pop` capta a média histórica do artista, mas não a variabilidade, o que pode prejudicar a classificação de artistas com carreiras irregulares.

## Trabalhos futuros

- Integrar métricas de redes sociais (YouTube, Instagram, TikTok) e de cobertura mediática.
- Testar deep learning sobre espectrogramas de áudio.
- Alargar a análise a outros géneros (jazz, eletrónica, etc.) e a outros mercados.
- Acompanhar as mesmas faixas ao longo do tempo.

## Ferramentas

- **Python** e **Jupyter Notebook**
- **Pandas** e **NumPy** (preparação e engenharia de atributos)
- **SciPy** (Shapiro-Wilk e Kruskal-Wallis)
- **Scikit-learn** (modelos, `StandardScaler`, `train_test_split`, K-Means e PCA)
- **Matplotlib** e **Seaborn** (visualização)
- A fusão dos três ficheiros CSV foi feita com apoio de uma ferramenta de IA (Claude).

