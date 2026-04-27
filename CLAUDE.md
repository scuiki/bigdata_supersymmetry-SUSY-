# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Big Data course final project (AC2) applying Apache Spark to the **SUSY particle physics dataset** (1.7 GB, 7M+ rows) for binary classification — predicting supersymmetric particle signatures. Course materials are in Portuguese.

Assignment deliverables (from `class_refs/ac2.txt`):
- **Part 1**: Docker-based Apache ecosystem environment (Spark, HDFS, etc.)
- **Part 2**: EDA and preprocessing on the SUSY dataset (>1GB)
- **Part 3**: ML models — Decision Tree, Logistic/Linear Regression, Neural Networks
- **Presentations**: May 11–12 and May 25–26 (2025)

Academic reference: `docs/ncomms5308.pdf` — the paper describing the SUSY dataset and its feature engineering.

## Environment

The project runs inside a Docker container with Jupyter + PySpark:

```bash
docker run --name spark-container -p 8888:8888 -p 4040:4040 -p 7077:7077 \
  -v $(pwd):/home/jovyan/work jupyter/pyspark-notebook
```

- Jupyter UI: http://localhost:8888
- Spark UI: http://localhost:4040

## Data

- `data/supersymmetry_dataset.csv` — 1.7 GB CSV, 7M+ rows
- First column is the binary label (`0` = background, `1` = SUSY signal)
- Remaining 18 columns are kinematic features; first 8 are low-level detector features, last 10 are derived high-level features (from the paper)

## Architecture

All work is notebook-based. The intended Spark ML pipeline pattern (demonstrated in `class_refs/aula_06_parte_2.ipynb`):

```
SparkSession
  → read.csv(data/) → DataFrame
  → dropna / filter numeric columns
  → VectorAssembler (feature columns → "features" vector)
  → Pipeline([assembler, model])
  → randomSplit([0.8, 0.2]) → train/test
  → model.fit(train) → model.transform(test)
  → MulticlassClassificationEvaluator (accuracy)
  → write.parquet(output/)
```

Key PySpark imports used across notebooks:
```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
```

## Reference Notebooks

- `class_refs/aula_06_parte_1.ipynb` — PySpark basics: SparkSession, DataFrames, Spark SQL
- `class_refs/aula_06_parte_2.ipynb` — Full ML pipeline on US Accidents dataset (template to adapt for SUSY); achieves 96.84% accuracy with Random Forest
- `class_refs/aula_pratica_06_parte_1.md` — PySpark fundamentals guide (lazy evaluation, transformations vs. actions)
