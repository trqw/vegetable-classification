# -------------------------------------------------------------
# 🌿 Green vs Root Vegetable Classification Dashboard (Slide Version)
# Author: <Viraj Patel>
# Date: 15/10/2025
# -------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
import io, joblib, os

# --- Streamlit Page Config ---
st.set_page_config(page_title="Vegetable Classifier Slides", page_icon="🥕", layout="wide")

# --- Helper functions ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("vegetable_data.csv")
        df['Target'] = df['Category'].map({'Green': 1, 'Root': 0})
        df = df.fillna(df.mean(numeric_only=True))
        return df
    except FileNotFoundError:
        st.error("❌ 'vegetable_data.csv' not found!")
        return None

@st.cache_resource
def train_and_tune_model(X_train, y_train):
    model_filename = "best_decision_tree.joblib"
    if os.path.exists(model_filename):
        grid_search = joblib.load(model_filename)
    else:
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [1, 2, 3]
        }
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        joblib.dump(grid_search, model_filename)
    best_clf = grid_search.best_estimator_
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    return best_clf, grid_search.best_params_, rf_clf

# --- Main Function ---
def main():
    st.title("🥦 Green vs Root Vegetable Classification Dashboard")
    st.markdown("---")

    df = load_data()
    if df is None:
        return

    selected_feature = ['ColorIntensity', 'Carbs', 'Sugar', 'VitaminC', 'WaterContent']
    X = df[selected_feature]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf, best_params, rf_clf = train_and_tune_model(X_train, y_train)
    y_pred_dt = clf.predict(X_test)
    y_pred_rf = rf_clf.predict(X_test)

    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    # --- Slide Control ---
    if "slide" not in st.session_state:
        st.session_state.slide = 1

    col1, col2 = st.columns([0.85, 0.15])
    with col2:
        if st.button("⏭ Next"):
            st.session_state.slide += 1
        if st.button("⏮ Back") and st.session_state.slide > 1:
            st.session_state.slide -= 1

    slide = st.session_state.slide

    # ---------------- SLIDE 1: Dataset Overview ----------------
    if slide == 1:
        st.header("📊 Slide 1: Dataset Overview")
        st.write("Let's explore the dataset used for Green vs Root Vegetable classification.")
        st.dataframe(df.head())
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        fig_count, ax_count = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Category', data=df, palette=['#2E8B57', '#D2691E'], ax=ax_count)
        ax_count.set_title("Distribution of Vegetable Categories")
        st.pyplot(fig_count)

    # ---------------- SLIDE 2: Exploratory Data Analysis ----------------
    elif slide == 2:
        st.header("🔍 Slide 2: Exploratory Data Analysis (EDA)")
        st.write("We’ll now visualize how Vitamin C differs between Green and Root vegetables.")
        col1, col2 = st.columns(2)

        with col1:
            fig_box, ax_box = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='Category', y='VitaminC', data=df, palette=['#2E8B57', '#D2691E'], ax=ax_box)
            st.pyplot(fig_box)

        with col2:
            fig_violin, ax_violin = plt.subplots(figsize=(6, 4))
            sns.violinplot(x='Category', y='VitaminC', data=df, palette=['#2E8B57', '#D2691E'], ax=ax_violin)
            st.pyplot(fig_violin)

        st.subheader("Feature Correlation Heatmap")
        corr = df.drop(columns=['Target', 'Category']).corr(numeric_only=True)
        fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap='Greens', fmt=".2f", ax=ax_heat)
        st.pyplot(fig_heat)

    # ---------------- SLIDE 3: Model Training & Performance ----------------
    elif slide == 3:
        st.header("🧠 Slide 3: Model Training & Performance")
        st.write("We tuned our Decision Tree using GridSearchCV and compared it with Random Forest.")

        st.subheader("Best Tuned Parameters (Decision Tree)")
        st.json(best_params)

        st.subheader("Model Comparison")
        comp = pd.DataFrame({'Model': ['Decision Tree', 'Random Forest'],
                             'Accuracy': [accuracy_dt, accuracy_rf]})
        fig_comp, ax_comp = plt.subplots()
        sns.barplot(x='Accuracy', y='Model', data=comp, palette='viridis', ax=ax_comp)
        ax_comp.set_xlim(0, 1)
        st.pyplot(fig_comp)

        st.subheader("Classification Report (Decision Tree)")
        st.code(classification_report(y_test, y_pred_dt, target_names=['Root', 'Green']))

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_dt)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Root', 'Green'], yticklabels=['Root', 'Green'], ax=ax_cm)
        st.pyplot(fig_cm)

    # ---------------- SLIDE 4: Visualizing the Decision Tree ----------------
    elif slide == 4:
        st.header("🌳 Slide 4: Visualizing the Decision Tree Structure")
        st.write("Here’s how the tuned Decision Tree model makes classification decisions.")
        fig_tree, ax_tree = plt.subplots(figsize=(15, 8))
        plot_tree(clf, filled=True, feature_names=selected_feature, class_names=['Root', 'Green'], rounded=True, ax=ax_tree)
        st.pyplot(fig_tree)

    # ---------------- SLIDE 5: Interactive Prediction ----------------
    elif slide == 5:
        st.header("🤖 Slide 5: Interactive Prediction Tool")
        st.write("Try entering nutritional values to predict the vegetable category.")

        with st.form("predict_form"):
            colA, colB, colC = st.columns(3)
            color_intensity = colA.number_input("Color Intensity (1–10)", min_value=0.0, max_value=10.0, value=5.0)
            water_content = colB.number_input("Water Content (%)", min_value=60.0, max_value=100.0, value=85.0)
            carbs = colC.number_input("Carbohydrates (g/100g)", min_value=0.0, value=10.0)

            colD, colE = st.columns(2)
            sugar = colD.number_input("Sugar (g/100g)", min_value=0.0, value=3.0)
            vitamin_c = colE.number_input("Vitamin C (mg/100g)", min_value=0.0, value=30.0)

            submit = st.form_submit_button("🔍 Predict Category")

        if submit:
            new_data = pd.DataFrame({
                'ColorIntensity': [color_intensity],
                'Carbs': [carbs],
                'Sugar': [sugar],
                'VitaminC': [vitamin_c],
                'WaterContent': [water_content]
            })[selected_feature]

            pred = clf.predict(new_data)[0]
            result = "🌿 Green Vegetable" if pred == 1 else "🍠 Root Vegetable"
            st.success(f"### Prediction: {result}")

    elif slide > 5:
        st.balloons()
        st.title("🎉 End of Presentation")
        st.write("Thank you for exploring the Green vs Root Vegetable Classifier Dashboard!")

# --- Run App ---
if __name__ == "__main__":
    main()
