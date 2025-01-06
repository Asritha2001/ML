# **Machine Learning Capstone Project**

## **Overview**
This project implements various machine learning algorithms to classify data across multiple datasets, including Handwritten Digits Recognition, Titanic, Loan Eligibility Prediction, Oxford Parkinson’s Disease Detection, and UCI’s Adult Income Census. The algorithms used include Decision Trees, Random Forests, k-Nearest Neighbors (k-NN), and Neural Networks. Each model's performance was evaluated and optimized by adjusting relevant hyperparameters for each dataset.

## **Datasets**
- **Handwritten Digits Recognition Dataset:** Classifies handwritten digits based on pixel data.<br>
- **Titanic Dataset:** Predicts survival outcomes based on demographic and social factors.<br>
- **Loan Eligibility Prediction Dataset:** Predicts loan approval based on applicant details.<br>
- **Oxford Parkinson’s Disease Detection Dataset:** Detects Parkinson’s disease using various health metrics.<br>
- **UCI Adult Income Census Dataset:** Predicts income levels based on census data attributes.<br>

## **Algorithms and Implementation**
- **Decision Tree:** Used for handling both numerical and categorical data, tuned by adjusting tree depth.<br>
- **Random Forest:** An ensemble method to reduce variance and improve accuracy by building multiple decision trees.<br>
- **k-Nearest Neighbors (k-NN):** A distance-based algorithm that assigns class labels based on the closest data points.<br>
- **Neural Networks:** Used for complex data patterns, with hyperparameters like hidden layers and neurons optimized for accuracy.

## **Experiments and Results**
Each algorithm was optimized through hyperparameter tuning. Key experiments and results include:

**Handwritten Digits Recognition:** <br>
- **Decision Tree**: Achieved 82% accuracy at a depth of 10.  
- **Random Forest**: Optimal performance with 40 trees, achieving a 96% accuracy.  
- **k-NN**: Highest accuracy at k=1, achieving 95.92%.  

**Titanic Dataset:**  
- **Decision Tree**: Best accuracy at depth 5, with an 83% accuracy.  
- **k-NN**: Optimal performance with k=6, achieving an 81.95% accuracy.  
- **Neural Network**: Achieved 81% accuracy with 12 neurons in the hidden layer.  

**Loan Eligibility Prediction:**  
- **Decision Tree**: Optimal depth at 5, with an 80% accuracy.  
- **k-NN**: Highest accuracy with k=11, achieving 73%.  
- **Neural Network**: Achieved 78.75% accuracy with a three-layer hidden network.  

**Oxford Parkinson’s Disease Detection:**  
- **Decision Tree**: Best performance at depth 7, with 89.71% accuracy.  
- **Random Forest**: Optimal performance with 20 trees, achieving 92.28% accuracy.  
- **k-NN**: Highest accuracy at k=1, achieving 95.92%.  

**UCI Adult Income Census:**  
- **Random Forest**: Optimal performance with 40 trees, achieving 84.95% accuracy.  
- **Neural Network**: Achieved 78.84% accuracy with an 8-neuron hidden layer.  

---

## Installation
To run the project:

1. Clone the repository:
    ```bash
     git clone https://github.com/yourusername/ML.git
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run individual model scripts for each dataset using:

```bash
python <dataset_script.py>
