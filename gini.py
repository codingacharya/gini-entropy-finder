import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Decision Tree Visualizer", layout="wide")

st.title("🌳 Decision Tree Classifier (with Visualization)")

# 1. Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.write(df.head())

    # 2. Select target
    target_col = st.selectbox("🎯 Select the target column", df.columns)

    # 3. Feature selection (auto-exclude target)
    features = df.drop(columns=[target_col]).columns.tolist()
    st.multiselect("🧬 Features used for prediction", features, default=features, disabled=True)

    # 4. Encode categorical variables
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

    # Split data
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 5. Model Parameters
    st.sidebar.header("🔧 Model Settings")
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)

    # 6. Train model
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                 min_samples_split=min_samples_split, random_state=42)
    clf.fit(X_train, y_train)

    # 7. Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model Accuracy: {accuracy:.2f}")

    # 8. Plot Tree
    st.subheader("🌳 Decision Tree Visualization")

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, feature_names=X.columns, class_names=[str(cls) for cls in clf.classes_], filled=True)
    st.pyplot(fig)
