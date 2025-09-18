# Iris Classification using K-Nearest Neighbors (KNN)

## Project Overview
This project demonstrates how to classify Iris flower species using the K-Nearest Neighbors (KNN) algorithm. The dataset used is the classic Iris dataset, which contains 150 samples of three species: `Iris-setosa`, `Iris-versicolor`, and `Iris-virginica`.

The project includes:
- Data exploration and visualization
- Feature correlation analysis
- KNN model training and evaluation
- Hyperparameter tuning (choosing the best K)
- Cross-validation analysis

---

## Dataset
The Iris dataset contains the following features:
- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`
- `species` (target label)

**Target labels:**  
`['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']`

---

## Steps and Analysis

### 1. Data Exploration
- Displayed the first few rows and descriptive statistics.
- Observed correlations:
  - Petal length & petal width: highly positively correlated.
  - Sepal features: contribute less to classification but improve accuracy in overlapping cases.
- Visualizations:
  - Scatter plots
  - Heatmaps of feature correlations
  - Pairplot to visualize species separability

### 2. Model Training
- Split dataset: 80% training, 20% testing.
- Used K-Nearest Neighbors classifier (`KNeighborsClassifier`).
- Initially trained with `k=1`.
- Predicted new sample:
```python
X_new = pd.DataFrame([[5,2.3,3.3,1]], columns=X.columns)
prediction = knn.predict(X_new)
print("Prediction:", prediction)
Prediction: ['Iris-versicolor']
```
## 3. Model Evaluation

### Accuracy, Precision, and Recall metrics
- **Accuracy:** 1.0  
- **Precision:** 1.0  
- **Recall:** 1.0  

💡 Note: These metrics are perfect because the Iris dataset is small, clean, and well-separated. In real-world datasets with more noise or overlapping classes, metrics are usually lower and more informative.

### Confusion Matrix
- Shows all correctly classified samples:  
  - **Diagonal** = all correct  
  - **Off-diagonal** = no misclassifications

### Training vs Testing Accuracy for different k values
- Small k (e.g., 1) → high training accuracy but risk of overfitting.  
- Large k (e.g., >15) → risk of underfitting.  
- Medium k (around 3–7) → best balance.  
- For Iris dataset, k=5 is usually a good choice.

---

## 4. Cross-Validation
- 5-fold cross-validation to evaluate k from 1 to 25:
- k=1, mean accuracy=0.960
- k=2, mean accuracy=0.947
- ...
- k=10, mean accuracy=0.980
- ...
- k=25, mean accuracy=0.960


💡 Note: In real-world and larger datasets, metrics like F1-score and cross-validation become more important than just accuracy.

---

## Libraries Used
- `pandas`, `numpy` for data handling  
- `matplotlib`, `seaborn`, `plotly` for visualization  
- `scikit-learn` for KNN model and evaluation metrics  

---

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/donyafallah/Iris-Flower-Classification.git
```
### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```
## Run the Project

- Run the Jupyter notebook or Python script.

## Conclusion

- The KNN algorithm successfully classifies Iris species with high accuracy.

- Petal measurements are the most informative features.

- Hyperparameter tuning (choosing the right k) and cross-validation are essential in larger, noisier datasets.

