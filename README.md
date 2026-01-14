# Hardware Trojan Detection Toolchain

This repository provides a complete toolchain for detecting hardware Trojans in digital circuits using cycle-level and graph-level feature analysis, machine learning, and automated evaluation.

---

## Overview

The toolchain consists of the following main stages:

1. **Verilog to Graph Conversion**
2. **Cycle Subgraph Extraction**
3. **Feature Extraction & Analysis**
4. **Model Training**
5. **Inference & Evaluation**

Each stage is modular and can be run independently or as part of a pipeline.

---

## Important 

If you want to use the detector without rerunning the whole toolchain you can do so by downloading the files:

1. **release_all**
2. **release_hidden**
3. **output**
4. **batch_detect_and_evaluate.py**
5. **trojan_detector_wrapper.py**
6.  **trojan_detector_cycle_only.py**
7.  **trojan_detector_cycle_only_original.py**

And then running the file batch_detect_and_evaluate.py

---

## 1. Verilog to Graph Conversion

- **Script:** `local_verilog_to_graphs.py`
- **Purpose:** Converts Verilog netlists into graph representations, extracting gate-level connectivity and structural features.
- **Input:** Verilog files (e.g., `design0.v`, ...)
- **Output:** `all_graphs.json`, `all_graphs_metadata.json`
- **How to run:**
  ```bash
  python local_verilog_to_graphs.py
  ```
- **Notes:**
  - Supports batch and single-design processing.
  - Extracts node/edge features, gate types, and basic graph statistics.

---

## 2. Cycle Subgraph Extraction

- **Script:** `kaggle_cycle_extractor.py`
- **Purpose:** Extracts all cycles (strongly connected components) from each design's graph, focusing on feedback structures where Trojans often hide.
- **Input:** `all_graphs.json`, result files for trojan gate labels
- **Output:** Cycle subgraph datasets (JSON, NPZ, or PyTorch Geometric format)
- **How to run:**
  ```bash
  python kaggle_cycle_extractor.py
  ```
- **Notes:**
  - Computes neighborhood and structural features for each node in the cycle subgraph.
  - Designed for both local and Kaggle environments.

---

## 3. Feature Extraction & Analysis

- **Script:** `cycle_feature_analyzer.py`
- **Purpose:** Analyzes extracted cycle subgraphs to compute discriminative features, perform statistical analysis, and select the most informative features for classification.
- **Input:** Cycle subgraph dataset (JSON)
- **Output:**
  - `enhanced_features_dataset.json` (features for ML)
  - `feature_analysis_report.txt` (feature importance, statistics)
- **How to run:**
  ```bash
  python cycle_feature_analyzer.py
  ```
- **Notes:**
  - Prints top features by effect size and correlation with the label.
  - Saves enhanced feature dataset for model training.

---

## 4. Model Training

- **Script:** `train_trojan_detector.py`
- **Purpose:** Trains machine learning models (Random Forest, SVM, XGBoost, etc.) on the extracted features to classify circuits as trojaned or clean.
- **Input:** `enhanced_features_dataset.json` (and optionally full-graph features)
- **Output:** Trained model files (e.g., `cycle_RandomForest.pkl`, `full_SVM_Linear.pkl`)
- **How to run:**
  ```bash
  python train_trojan_detector.py
  ```
- **Notes:**
  - Uses cross-validation and feature selection to avoid overfitting.
  - Saves every trained model with a clear filename (e.g., `cycle_RandomForest.pkl`).
  - Prints training summary and metrics.

---



## 5. Inference & Evaluation

- **Script:** `trojan_detector_cycle_only_original.py`
  - **Purpose:** The core cycle-only model for trojan detection. Parses Verilog, extracts cycles, computes features, and predicts using a trained ML model.
  - **Input:** Verilog netlist, trained model (default: best cycle model)
  - **Output:** Prediction result (Trojan Detected / Clean) and suspicious gate names
  - **How to run:**
    ```bash
    python trojan_detector_cycle_only_original.py -netlist <design.v> -output <result.txt>
    ```

- **Script:** `trojan_detector_cycle_only.py`
  - **Purpose:** Wrapper for the original detector to ensure output format compatibility for evaluation scripts. Calls the original detector and reformats output as required.
  - **Input:** Verilog netlist, output file
  - **How to run:**
    ```bash
    python trojan_detector_cycle_only.py -netlist <design.v> -output <result.txt>
    ```

- **Script:** `trojan_detector_wrapper.py`
  - **Purpose:** Enhanced wrapper that integrates the original detector and provides improved gate-level output and compatibility with batch pipelines.
  - **Input:** Verilog netlist, output file, (optional) model path
  - **How to run:**
    ```bash
    python trojan_detector_wrapper.py -netlist <design.v> -output <result.txt>
    ```

- **Script:** `batch_detect_and_evaluate.py`
  - **Purpose:** Complete pipeline for batch detection and evaluation. Runs the detector on multiple designs, compares predictions to ground truth, and computes detailed metrics (accuracy, F1, gate-level scores).
  - **How to run:**
    ```bash
    python batch_detect_and_evaluate.py
    ```

----


## File/Script Summary

- `local_verilog_to_graphs.py` — Verilog to graph conversion
- `kaggle_cycle_extractor.py` — Cycle subgraph extraction
- `cycle_feature_analyzer.py` — Feature extraction and analysis
- `train_trojan_detector.py` — Model training and selection
- `trojan_detector_cycle_only_original.py` — Core cycle-only detector (ML logic)
- `trojan_detector_cycle_only.py` — Output-format wrapper for cycle-only detector
- `trojan_detector_wrapper.py` — Enhanced wrapper for batch and gate-level output
- `batch_detect_and_evaluate.py` — Complete batch detection and evaluation pipeline

----

## Tips & Best Practices

- Always use the same feature set for both training and inference.
- For best results, retrain models after updating feature extraction or analysis scripts.
- All scripts are designed to be run from the repository root directory.
- Use the provided virtual environment for consistent dependencies.

---

## Requirements

- Python 3.8+
- numpy, scikit-learn, networkx, (optional: xgboost, lightgbm)
- For Kaggle: torch, torch-geometric (for GNN experiments)

---

## Citation
If you use this toolchain in your research or competition submission, please cite appropriately.

---

## License
MIT License
