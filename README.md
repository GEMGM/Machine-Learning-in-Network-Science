# Machine-Learning-in-Network-Science
# Graph Link Prediction using Machine Learning

This repository contains a Jupyter Notebook that implements **link prediction in a graph** using **network analysis and machine learning**. The model is trained to predict whether an edge exists between two nodes based on extracted features.

## 📌 Project Overview
The project involves:
- Loading and preprocessing graph data.
- Extracting node and edge features.
- Training a **logistic regression model** to predict links.
- Evaluating performance using appropriate metrics.

---

## 📂 File Structure
- **`code_cheval.ipynb`** → Jupyter Notebook containing full implementation.
- **`train.txt`** → Training data containing node pairs and labels.
- **`test.txt`** → Test data containing node pairs for prediction.
- **`node_information.csv`** → Node attributes used for feature extraction.
- **`README.md`** → Project documentation (this file).

---

## 📖 Dataset and Preprocessing
### 🔹 **Dataset**
- **`train.txt`**: Contains source and target node pairs with labels (1: edge exists, 0: no edge).
- **`test.txt`**: Contains source and target node pairs without labels (for prediction).
- **`node_information.csv`**: Contains node features used for link prediction.

### 🔹 **Preprocessing Steps**
- **Graph Construction:** Build a graph using `networkx`.
- **Feature Extraction:** Compute node similarity measures (e.g., Euclidean distance, Manhattan distance, vector differences, and products).
- **Dataset Splitting:** Prepare data for training and testing.

---

## 🏗️ Model Architecture
### 🔹 **Features Used**
- Node feature difference (`|f1 - f2|`)
- Node feature product (`f1 * f2`)
- Node feature sum (`f1 + f2`)
- Euclidean distance
- Manhattan distance

### 🔹 **Machine Learning Model**
- **Logistic Regression** is used for classification.
- The dataset is split into training and validation sets.
- Performance is evaluated using the **ROC AUC score**.

---

## 🏋️‍♂️ Training & Evaluation
### 🔹 **Training Procedure**
1. Train a logistic regression model on extracted node-pair features.
2. Use **cross-validation** to assess performance.
3. Tune hyperparameters for optimal results.

### 🔹 **Evaluation Metrics**
- **ROC AUC Score**: Measures the model’s ability to distinguish between linked and non-linked nodes.
- **Accuracy Score**: Assesses prediction correctness.

---

## 🚀 How to Run the Notebook
### 📦 **Dependencies**
Ensure required libraries are installed:
```bash
pip install pandas numpy networkx tqdm gensim scikit-learn
```
### 🔹 **Execution Steps**
1. Place `train.txt`, `test.txt`, and `node_information.csv` in the working directory.
2. Open and run `code_cheval.ipynb` in Jupyter Notebook.
3. Train the model and evaluate link prediction results.

---

## 📌 Future Improvements
- Experiment with **Graph Neural Networks (GNNs)** for improved performance.
- Explore additional feature engineering techniques.
- Optimize model hyperparameters.

---

## 📜 Acknowledgments
- **NetworkX** for graph manipulation.
- **Scikit-learn** for machine learning models.
- **Gensim** for potential embedding techniques.

---

## 📧 Contact
For any queries or contributions, feel free to reach out!

