# Plano: AC2 — Big Data com SUSY Dataset

## Contexto

Trabalho final (AC2) da disciplina de Big Data. O objetivo é aplicar o ecossistema Apache Spark a um dataset real de física de partículas (SUSY — Supersymmetry) para classificação binária: detectar eventos com partículas superssimétricas. O dataset tem **5.000.000 linhas e 1.61 GB**, extraído de simulações de colisionadores de partículas.

Todos os métodos devem seguir os padrões vistos em aula (notebooks `aula_06_parte_1` e `aula_06_parte_2`).

---

## Estrutura de Arquivos a Criar

```
bigdata_supersymmetry-SUSY-/
├── notebook_parte1_ambiente.ipynb       # Parte 1: setup Docker/Spark
├── notebook_parte2_eda.ipynb            # Parte 2: EDA e pré-processamento
├── notebook_parte3_modelos.ipynb        # Parte 3: modelos ML
└── docker-compose.yml                   # Alternativa ao comando docker run
```

---

## Parte 1: Ambiente Big Data (1 ponto)

**Objetivo**: Documentar e reproduzir o ambiente Spark via Docker.

Conteúdo do `notebook_parte1_ambiente.ipynb`:
- Documentação do comando Docker usado (já definido no guia do professor):
  ```bash
  docker run --name spark-container -p 8888:8888 -p 4040:4040 -p 7077:7077 \
    -v $(pwd):/home/jovyan/work jupyter/pyspark-notebook
  ```
- Célula com `SparkSession.builder.appName("SUSY-AC2").getOrCreate()` e verificação da versão do Spark
- Screenshot/output do Spark UI em localhost:4040 documentado no notebook

---

## Parte 2: EDA e Pré-processamento (5 pontos)

**Referência direta**: `aula_06_parte_2.ipynb` (padrão de carregamento, dropna, filtragem, groupBy)

### 2.1 — Entendimento do Dataset (contextualização para apresentação)

**O que é o SUSY dataset:**
- Origem: paper *"Searching for exotic particles in high-energy physics with deep learning"* (Baldi, Sadowski, Whiteson — Nature Communications, 2014)
- Dados simulados de colisões em colisionadores de partículas (tipo LHC/CERN)
- **Tarefa**: classificação binária — detectar eventos com partículas superssimétricas (SUSY) vs. eventos de fundo (partículas comuns)

**Estrutura das 19 colunas (o CSV possui header — confirmado na Parte 1):**
| Col | Nome proposto | Tipo | Descrição |
|-----|---------------|------|-----------|
| 0 | `label` | int | **Target**: 1=sinal SUSY, 0=fundo |
| 1–8 | `lepton1_pT`, `lepton1_eta`, `lepton1_phi`, `lepton2_pT`, `lepton2_eta`, `lepton2_phi`, `missing_energy_magnitude`, `missing_energy_phi` | float | **Low-level**: medições diretas do detector |
| 9–18 | `MET_rel`, `axial_MET`, `M_R`, `M_TR_2`, `R`, `MT2`, `S_R`, `M_Delta_R`, `dPhi_r_b`, `cos_theta_r1` | float | **High-level**: variáveis derivadas por físicos |

> **Para apresentação**: O dataset testa se redes neurais profundas conseguem aprender sozinhas as features de alto nível que físicos precisaram derivar manualmente. O paper mostra AUC de 0.876 com deep learning vs. ~0.85 com métodos tradicionais.

### 2.2 — Carregamento (padrão professor)

```python
# CSV tem header — confirmado na Parte 1 (show(5) com header=False exibiu
# "SUSY", "lepton 1 pT"... como primeira linha de dados e inferiu tudo como string).
# header=True pula o header e infere os tipos corretos (double).
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("SUSY-AC2").getOrCreate()

column_names = ["label",
    "lepton1_pT", "lepton1_eta", "lepton1_phi",
    "lepton2_pT", "lepton2_eta", "lepton2_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "MET_rel", "axial_MET", "M_R", "M_TR_2", "R",
    "MT2", "S_R", "M_Delta_R", "dPhi_r_b", "cos_theta_r1"
]

df = spark.read.csv("./data/supersymmetry_dataset.csv",
                    header=True, inferSchema=True)
df = df.toDF(*column_names)

print(f"Número de linhas: {df.count()}")  # esperado: 5.000.000
df.printSchema()
df.show(5)
```

### 2.3 — Análise Exploratória

Seguindo padrão do professor (`groupBy`, `describe`, Spark SQL):

```python
# Distribuição de classes
df.groupBy("label").count().show()

# Estatísticas descritivas
df.describe().show()

# Verificação de nulos
from pyspark.sql.functions import isnan, when, count
df.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in df.columns]).show()

# Spark SQL: distribuição e médias por classe
df.createOrReplaceTempView("susy_data")
spark.sql("""
    SELECT label, COUNT(*) as total,
           AVG(lepton1_pT) as avg_pT,
           AVG(missing_energy_magnitude) as avg_missing_E
    FROM susy_data GROUP BY label
""").show()
```

### 2.4 — Pré-processamento

```python
# Remover nulos (padrão professor)
df = df.dropna()

# Cast da label para int (padrão professor)
df = df.withColumn("label", col("label").cast("int"))

# Verificar balanço de classes (problema crítico para o dataset SUSY)
df.groupBy("label").count().show()
# Esperado: desbalanceado (fundo >> sinal)
```

**Tratamento do desbalanceamento** (obrigatório pelo enunciado):
- Undersampling da classe majoritária com `sampleBy`:
  ```python
  # Manter ~50% de equilíbrio
  fractions = {0: 0.5, 1: 1.0}  # ajustar conforme proporção real
  df_balanced = df.sampleBy("label", fractions, seed=42)
  df_balanced.groupBy("label").count().show()
  ```
- Documentar o trade-off: undersampling perde dados; anotar na apresentação

### 2.5 — Conversão CSV → Parquet (ETL checkpoint)

O professor destacou essa conversão como prática fundamental de Big Data. Justificativa para o dataset SUSY:
- Carregar 1.61 GB de CSV com `inferSchema=True` a cada execução é custoso
- Parquet é colunar e comprimido (~3–5x menor), com schema embutido
- Após salvar, as execuções da Parte 3 partem do Parquet limpo, não do CSV bruto

```python
# Salvar dados limpos e balanceados em Parquet
parquet_path = "./data/susy_parquet"
df_balanced.write.mode("overwrite").parquet(parquet_path)
print(f"Dataset salvo em Parquet: {parquet_path}")

# Demonstrar recarga mais rápida (mostrar na apresentação como evidência)
import time
t0 = time.time()
df_parquet = spark.read.parquet(parquet_path)
print(f"Parquet carregado em {time.time() - t0:.2f}s | Linhas: {df_parquet.count()}")
df_parquet.printSchema()
```

A Parte 3 usa `df_parquet` como ponto de entrada, não o CSV original.

### 2.6 — Feature Engineering (padrão VectorAssembler do professor)

```python
from pyspark.ml.feature import VectorAssembler, StandardScaler

feature_cols = [c for c in df_balanced.columns if c != "label"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vec = assembler.transform(df_balanced)

# StandardScaler (boas práticas — validar se professor usou)
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df_vec)
df_scaled = scaler_model.transform(df_vec)
```

### 2.7 — Train/Test Split (padrão professor: 80/20, seed=42)

```python
train_data, test_data = df_scaled.randomSplit([0.8, 0.2], seed=42)
print(f"Treino: {train_data.count()} | Teste: {test_data.count()}")
```

---

## Parte 3: Modelos de ML (4 pontos)

**Referência direta**: `aula_06_parte_2.ipynb` (Pipeline + avaliador)

> **Ponto de entrada**: `notebook_parte3_modelos.ipynb` carrega do Parquet gerado na Parte 2:
> ```python
> df_parquet = spark.read.parquet("./data/susy_parquet")
> ```

Os 3 modelos exigidos pelo professor, usando `Pipeline` como no notebook de aula:

### 3.1 — Árvore de Decisão

```python
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

dt = DecisionTreeClassifier(labelCol="label", featuresCol="scaled_features", maxDepth=10)
pipeline_dt = Pipeline(stages=[dt])
model_dt = pipeline_dt.fit(train_data)
predictions_dt = model_dt.transform(test_data)
```

### 3.2 — Regressão Logística

```python
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="label", featuresCol="scaled_features",
                        maxIter=100, regParam=0.01)
pipeline_lr = Pipeline(stages=[lr])
model_lr = pipeline_lr.fit(train_data)
predictions_lr = model_lr.transform(test_data)
```

### 3.3 — Rede Neural (MultilayerPerceptronClassifier)

Equivalente a neural network no PySpark:

```python
from pyspark.ml.classification import MultilayerPerceptronClassifier

# Arquitetura: [18 inputs, 64 neurônios, 32 neurônios, 2 outputs (classes)]
layers = [18, 64, 32, 2]

mlp = MultilayerPerceptronClassifier(
    labelCol="label", featuresCol="scaled_features",
    layers=layers, maxIter=100, seed=42
)
pipeline_mlp = Pipeline(stages=[mlp])
model_mlp = pipeline_mlp.fit(train_data)
predictions_mlp = model_mlp.transform(test_data)
```

### 3.4 — Avaliação Comparativa (padrão professor)

```python
from pyspark.ml.evaluation import (MulticlassClassificationEvaluator,
                                    BinaryClassificationEvaluator)

def avaliar_modelo(predictions, nome):
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", metricName="f1")
    auc_eval = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC")

    print(f"\n=== {nome} ===")
    print(f"Accuracy: {acc_eval.evaluate(predictions):.4f}")
    print(f"F1-Score: {f1_eval.evaluate(predictions):.4f}")
    print(f"AUC-ROC:  {auc_eval.evaluate(predictions):.4f}")

avaliar_modelo(predictions_dt, "Árvore de Decisão")
avaliar_modelo(predictions_lr, "Regressão Logística")
avaliar_modelo(predictions_mlp, "Rede Neural (MLP)")
```

> **Referência do paper para apresentação**: O paper reporta AUC de 0.876 com deep neural network no dataset completo. Podemos comparar nossos resultados.

### 3.5 — Salvar Resultados (padrão professor)

```python
# Predições dos 3 modelos em Parquet
predictions_dt.select("label", "prediction").write.mode("overwrite").parquet("./output/predicoes_dt")
predictions_lr.select("label", "prediction", "probability").write.mode("overwrite").parquet("./output/predicoes_lr")
predictions_mlp.select("label", "prediction", "probability").write.mode("overwrite").parquet("./output/predicoes_mlp")
```

---

## Pontos de Atenção / A Confirmar

1. ~~**Header do CSV**: verificar se o arquivo em `data/` tem header ou não antes de executar~~ → **CONFIRMADO** (Parte 1): o CSV **tem** header. Usar `header=True`. Com `header=False` + `inferSchema=True` o Spark infere tudo como `string`.
2. **Proporção real de classes**: só saberemos após carregar; ajustar frações do `sampleBy` conforme necessário
3. **Tamanho do dataset vs. MLP**: `MultilayerPerceptronClassifier` pode ser lento com 5M rows — pode ser necessário usar subset ou reduzir `maxIter`
4. **Nomes das features**: o paper usa siglas técnicas; decidir se usamos nomes curtos ou descritivos nos notebooks

---

## Verificação

- [x] `SparkSession` inicializa sem erro no container Docker — Spark 3.5.0, Python 3.11, master local[*]
- [x] `df.count()` retorna 5.000.000 linhas (com `header=True`)
- [ ] `df.groupBy("label").count()` mostra distribuição das duas classes
- [ ] Todos os 3 modelos treinam e geram predições sem erro
- [ ] `avaliar_modelo()` imprime accuracy, F1 e AUC para os 3 modelos
- [ ] AUC da Rede Neural deve ser próxima ao benchmark do paper (~0.87)
- [ ] Parquet intermediário gerado em `./data/susy_parquet` após Parte 2
- [ ] Recarga do Parquet na Parte 3 funciona sem precisar do CSV original
- [ ] Parquets de predições gerados em `./output/`
