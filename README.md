# Big Data com SUSY Dataset

Projeto final (AC2) da disciplina de **Big Data**. Aplica o ecossistema Apache Spark a um dataset real de física de partículas para classificação binária: identificar colisões que produziram partículas superssimétricas (SUSY) versus colisões do Modelo Padrão convencional.

---

## Contexto Científico

A Supersimetria (SUSY) é uma extensão teórica do Modelo Padrão da física de partículas. Ela propõe que cada partícula conhecida (elétrons, quarks, fótons) teria um "superpar" com spin diferindo em 1/2. Se confirmada, a SUSY explicaria a matéria escura, resolveria o problema da hierarquia e possibilitaria a unificação das forças fundamentais. Até hoje, nenhuma partícula supersimétrica foi detectada no LHC.

O dataset desta análise foi gerado por simulações Monte Carlo replicando colisões próton-próton a 8 TeV no LHC, processadas por **MadGraph** (geração de eventos), **PYTHIA** (showering e hadronização) e **DELPHES** (simulação do detector).

A questão central do paper de referência (Baldi et al., 2014): *é possível substituir as variáveis derivadas manualmente por físicos por uma rede neural que aprende diretamente das medições brutas do detector?* A resposta foi sim, com AUC de 0,876 — referência que buscamos replicar com os modelos da Parte 3.

---

## Dataset

| Propriedade | Valor |
|---|---|
| Fonte | UCI Machine Learning Repository / Kaggle |
| Tamanho | 1,61 GB |
| Linhas | 5.000.000 eventos simulados |
| Colunas | 19 (1 label + 18 features) |
| Tarefa | Classificação binária |
| Balanceamento | Aproximadamente 54% fundo / 46% sinal |

### Estrutura das features

**Label**
- `label`: 0 = evento de fundo (Modelo Padrão), 1 = evento SUSY

**Features low-level (colunas 1-8) — medições diretas do detector**

| Feature | Significado físico |
|---|---|
| `lepton1_pT`, `lepton2_pT` | Momento transverso dos dois léptons (perpendicular ao feixe) |
| `lepton1_eta`, `lepton2_eta` | Pseudorapidez: ângulo em relação ao eixo do feixe |
| `lepton1_phi`, `lepton2_phi` | Ângulo azimutal ao redor do feixe |
| `missing_energy_magnitude` | Energia transversa "perdida" — indica partículas que escaparam do detector (neutrinos ou partículas SUSY) |
| `missing_energy_phi` | Direção azimutal da energia perdida |

**Features high-level (colunas 9-18) — derivadas por físicos de partículas**

`MET_rel`, `axial_MET`, `M_R`, `M_TR_2`, `R`, `MT2`, `S_R`, `M_Delta_R`, `dPhi_r_b`, `cos_theta_r1` — variáveis cinemáticas calculadas sobre as features low-level, projetadas para maximizar a separação entre sinal e fundo (técnicas Razor, massa transversa, ângulos no referencial de repouso).

---

## Stack

| Componente | Versão |
|---|---|
| Apache Spark | 3.5.0 |
| Python | 3.11 |
| Jupyter | via `jupyter/pyspark-notebook` |
| Docker | >= 20.10 (Compose integrado) |

---

## Estrutura do Repositório

```
bigdata_supersymmetry-SUSY-/
├── notebook_parte1_ambiente.ipynb   # Parte 1: configuracao do ambiente Spark
├── notebook_parte2_eda.ipynb        # Parte 2: EDA e pre-processamento
├── notebook_parte3_modelos.ipynb    # Parte 3: modelos de ML e avaliacao
├── docker-compose.yml               # Ambiente Docker reproduzivel
├── data/
│   └── supersymmetry_dataset.csv    # Dataset bruto (1,61 GB, nao versionado)
└── docs/
    ├── plano_ac2.md                 # Plano de desenvolvimento
    └── ncomms5308.pdf               # Paper de referencia (Baldi et al., 2014)
```

> O arquivo `data/supersymmetry_dataset.csv` não está incluído no repositório por conta do tamanho. Faça o download pelo [Kaggle](https://www.kaggle.com/datasets/unsdsn/world-happiness) ou pelo [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SUSY) e coloque na pasta `data/`.

---

## Como Executar

**Pré-requisito**: Docker instalado (versão >= 20.10).

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd bigdata_supersymmetry-SUSY-

# Coloque o dataset em data/supersymmetry_dataset.csv

# Suba o ambiente
docker compose up
```

O Jupyter ficará disponível em `http://localhost:8889`. Execute os notebooks na ordem:

1. `notebook_parte1_ambiente.ipynb`
2. `notebook_parte2_eda.ipynb`
3. `notebook_parte3_modelos.ipynb`

A Spark Web UI (monitoramento de jobs) fica em `http://localhost:4040` enquanto o Spark estiver ativo.

---

## Notebooks

### Parte 1: Ambiente Big Data

Configura e documenta o ambiente de processamento distribuído:

- Inicialização da `SparkSession` e verificação da versão do Spark
- Documentação do container Docker e mapeamento de portas
- Leitura inicial do dataset e validação do schema

### Parte 2: EDA e Pré-processamento

Pipeline completo de análise exploratória e preparação dos dados:

- Distribuição de classes com visualizações (barras e pizza)
- Estatísticas descritivas transpostas por feature (legíveis no Jupyter)
- Verificação e remoção de valores nulos
- Balanceamento de classes com `sampleBy` (frações calculadas dinamicamente)
- Conversão do CSV (1,61 GB) para Parquet — o formato colunar reduz o tempo de leitura nas execuções subsequentes
- `VectorAssembler` (18 features → vetor `"features"`) e `StandardScaler` (normalização)
- Divisão treino/teste 80/20 com `seed=42`

### Parte 3: Modelos de Machine Learning

Treinamento e avaliação comparativa dos três modelos exigidos:

| Modelo | Implementação PySpark |
|---|---|
| Árvore de Decisão | `DecisionTreeClassifier` |
| Regressão Logística | `LogisticRegression` |
| Rede Neural | `MultilayerPerceptronClassifier` (MLP) |

Cada modelo é encapsulado em um `Pipeline` com as etapas de feature engineering. A avaliação usa `MulticlassClassificationEvaluator` (accuracy, F1) e `BinaryClassificationEvaluator` (AUC-ROC), comparando com o benchmark do paper (AUC 0,876 com deep learning).

---

## Referência

Baldi, P., Sadowski, P. & Whiteson, D. **Searching for exotic particles in high-energy physics with deep learning**. *Nature Communications* 5, 4308 (2014). DOI: [10.1038/ncomms5308](https://doi.org/10.1038/ncomms5308)
