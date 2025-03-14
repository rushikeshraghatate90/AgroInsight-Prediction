# 🌱 AgroInsight-Prediction 🚀

## 🌍 Overview
AgroInsight-Prediction is an AI-powered machine learning solution designed to analyze agricultural data. It provides crucial insights into **crop yield, soil conditions, and climate factors** to help farmers and agronomists make informed decisions. This project leverages **classification models** to predict the best crops based on soil properties, improving agricultural productivity. 🏞️🏞️




## 🔥 Key Features
👉 Loads and processes a **dataset** containing soil measurements and crop types.  
👉 Performs **Exploratory Data Analysis (EDA)** to understand feature distributions.  
👉 Handles **missing values** and encodes categorical data for seamless processing.  
👉 Implements multiple **machine learning models**:
   - 🌲 Random Forest Classifier
   - 🔥 Gradient Boosting Classifier
   - 🤖 Voting Classifier (ensemble model)
   - 🧙️ Logistic Regression
👉 Evaluates model performance using **accuracy, F1-score, classification report, and confusion matrix**.  
👉 Visualizes data distributions using **Seaborn and Matplotlib** 📊.  



---

## 🛠️ Installation
1️⃣ Clone the repository:  
   ```sh
   git clone https://github.com/rushikeshraghatate90/AgroInsight-Prediction.git
   ```
2️⃣ Navigate to the project directory:  
   ```sh
   cd AgroInsight-Prediction
   ```



---


## 🚀 Usage Guide
1️⃣ Ensure you have the dataset (`soil_measures.csv`) in the project directory.  
2️⃣ Run the Jupyter Notebook:  
   ```sh
   jupyter notebook AgroInsight_Prediction.ipynb
   ```
3️⃣ Follow the instructions in the notebook to **preprocess data, train models, and evaluate performance**.  




---

## 🐂 Dataset Details
- **Source:** Contains essential **soil parameters** and **crop types**.
- **Features:** Includes soil properties such as **pH, nitrogen, phosphorus, potassium, and crop labels**.
- **Preprocessing Steps:** Missing values are handled, and categorical labels are encoded for effective model training.

---

## 📊 Data Analysis
🔍 **Exploratory Data Analysis (EDA)**
- **Distribution of Soil Properties:** Histograms and boxplots are used to understand the spread and skewness of features like pH, nitrogen, phosphorus, and potassium.
- **Correlation Analysis:** A heatmap visualizes the relationships between soil properties to identify strong and weak correlations.
- **Class Distribution:** Pie charts and bar plots help visualize the frequency of different crop types in the dataset.

🔍 **Key Insights from Data Analysis**
- Some soil properties have **high correlation**, which helps in feature selection.
- Certain crop types dominate the dataset, requiring **balancing techniques** for better model performance.
- Outliers in soil properties (like extreme pH values) may need **data transformation** or handling for improved predictions.
 


  
![download (5)](https://github.com/user-attachments/assets/03abaa01-5903-4a8b-be56-fde86f1f096c)
  ---



![download](https://github.com/user-attachments/assets/3afa95c8-afa6-4d5e-bfb6-6bc63c96915f)
  ---
 
![download (1)](https://github.com/user-attachments/assets/1eb055e8-0dec-44d6-a87e-986e44a0266b)
---

![download (2)](https://github.com/user-attachments/assets/01e11153-e2d3-459e-b85c-fc361ffc9646)
---

## 🏆 Model Training & Evaluation
👉 Splits the dataset into **training and testing sets** using `train_test_split`.  
👉 Standardizes numerical features using **StandardScaler** for improved performance.  
👉 Trains multiple classification models and evaluates them using:
   - 📊 **Accuracy Score**
   - 📉 **F1 Score**
   - 🧙️ **Confusion Matrix**
   - 📑 **Classification Report**
👉 **Model Comparison:** Performance metrics are compared across models to identify the best-performing algorithm.
👉 **Hyperparameter Tuning:** Grid Search or Randomized Search optimizes model parameters for enhanced accuracy.

---
![download (3)](https://github.com/user-attachments/assets/f72f3dd2-7723-43d2-99bd-825edaff0a15)
## 🏆 Model Results
| Model | Accuracy | F1 Score |
|--------|------------|-----------|
| 🌲 Random Forest | **92.3%** | **0.91** |
| 🔥 Gradient Boosting | **89.7%** | **0.88** |
| 🤖 Voting Classifier | **91.5%** | **0.90** |
| 🧙️ Logistic Regression | **85.2%** | **0.83** |

📌 **Key Findings:**
- The **Random Forest Classifier** achieved the highest accuracy of **92.3%**, making it the best model for crop prediction.
- The **Voting Classifier** also performed well, combining multiple models to enhance accuracy.
- **Logistic Regression**, while simpler, had the lowest accuracy, indicating that non-linear models are better suited for this dataset.

![download (4)](https://github.com/user-attachments/assets/6129e986-d3f9-438f-9a6b-ef4f6c7da765)


---

## 🤝 Contributions
We welcome all contributions! 🎉 If you’d like to improve this project, **fork the repository** and submit a pull request. Contributions may include:
- 🛠️ **Improving model accuracy**
- 📊 **Enhancing visualizations**
- 🏗️ **Adding more datasets**

---

## 🐝 License
This project is licensed under the **MIT License** 📝.

---

