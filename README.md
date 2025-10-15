🌿 Green vs Root Vegetable Classification Dashboard
🥦 Overview

This project builds a machine learning dashboard using a Decision Tree Classifier to distinguish between green and root vegetables based on their nutritional and color characteristics.
The app is fully interactive, visually rich, and interpretable — perfect for ML learning and demonstration.

🚀 Features

✅ End-to-End ML Workflow:

Data preprocessing, encoding, and feature correlation

Decision Tree training & tuning using GridSearchCV

Comparison with Random Forest model

✅ Visualization & Insights:

Boxplots, violin plots, and heatmaps for feature analysis

Interactive confusion matrix and feature importance

Visualized Decision Tree for interpretability

✅ Interactive Dashboard:

Built with Streamlit

Allows users to input new vegetable characteristics and predict whether it’s a Green or Root vegetable 🌿🍠


🍠

📊 Dataset

Custom dataset vegetable_data.csv (≈250 records) with features:

| Feature            | Description                               |
| ------------------ | ----------------------------------------- |
| **ColorIntensity** | Degree of greenness or color depth (1–10) |
| **WaterContent**   | Percentage of water in vegetable (%)      |
| **Carbs**          | Carbohydrates (g/100g)                    |
| **Sugar**          | Sugar content (g/100g)                    |
| **VitaminC**       | Vitamin C content (mg/100g)               |
| **Category**       | Target label — `Green` or `Root`          |




Model Training

Algorithm: Decision Tree Classifier (Entropy criterion)

Tuning: GridSearchCV on max_depth, min_samples_split, min_samples_leaf

Cross Validation: 5-fold

Comparison: Random Forest benchmark






🧠 Technologies Used

--Python 3.10+

Libraries:

--pandas, numpy, matplotlib, seaborn

--scikit-learn

--streamlit

--joblib

--scipy


🧪 Sample Inputs
| ColorIntensity | WaterContent | Carbs | Sugar | VitaminC |
| -------------- | ------------ | ----- | ----- | -------- |
| 2              | 93           | 4     | 2     | 45       |

🟢 Expected Output: Green Vegetable (Code: 1)



🟤 Expected Output: Root Vegetable (Code: 0)
| ColorIntensity | WaterContent | Carbs | Sugar | VitaminC |
| -------------- | ------------ | ----- | ----- | -------- |
| 8              | 85           | 10    | 5     | 6        |

<img width="1366" height="482" alt="Screenshot (300)" src="https://github.com/user-attachments/assets/b7586ca1-93fb-4fd0-8fbf-2b1166394390" />

<img width="1366" height="548" alt="Screenshot (303)" src="https://github.com/user-attachments/assets/9e302fbf-54e8-48e1-9d6e-71e49e5971bb" />
<img width="1366" height="540" alt="Screenshot (302)" src="https://github.com/user-attachments/assets/0a21308c-1edc-48e3-b1b6-1610f7f7b3fe" />
<img width="1366" height="575" alt="Screenshot (301)" src="https://github.com/user-attachments/assets/77eaa9d2-f04d-4d7e-b3d3-c81b44a1115b" /><img width="1366" height="540" alt="Screenshot (302)" src="https://github.com/user-attachments/assets/5fff870c-916d-4326-840c-5acc9233adf2" />
<img width="1366" height="575" alt="Screenshot (301)" src="https://github.com/user-attachments/assets/84c3e919-36d0-48e3-9ee7-89536d0cd9a9" />
<img width="1366" height="548" alt="Screenshot (303)" src="https://github.com/user-attachments/assets/67a04c69-5bdf-4452-8e65-0b660a027746" />









