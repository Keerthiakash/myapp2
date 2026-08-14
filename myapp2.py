import streamlit as st
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load and Train Model
# -----------------------------
wine = load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    wine.data,
    wine.target,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wine Classifier",
    page_icon="🍷",
    layout="centered"
)

st.title("🍷 Wine Cultivar Classifier")

st.write(
    "Enter the chemical analysis values below and click **Predict** to identify the wine's cultivar (class 0, 1, or 2)."
)

st.write(f"### Model Accuracy: **{accuracy:.2%}**")

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
# Use the dataset's own min/mean/max per feature so defaults and ranges make sense
feature_names = wine.feature_names
mins = wine.data.min(axis=0)
maxs = wine.data.max(axis=0)
means = wine.data.mean(axis=0)

st.subheader("Chemical Properties")

# Lay inputs out in two columns to keep the page manageable (13 features)
col1, col2 = st.columns(2)

inputs = []
for i, name in enumerate(feature_names):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        value = st.number_input(
            name.replace("_", " ").title(),
            min_value=float(mins[i]),
            max_value=float(maxs[i]) * 1.2,
            value=float(round(means[i], 2)),
            format="%.2f"
        )
    inputs.append(value)

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("Predict Cultivar"):

    prediction = model.predict([inputs])
    probability = model.predict_proba([inputs])

    cultivar = f"Class {prediction[0]}"

    st.success(f"Predicted Cultivar: **{cultivar}**")

    st.subheader("Prediction Confidence")

    st.write(
        {
            f"Class {i}": f"{probability[0][i]*100:.2f}%"
            for i in range(len(wine.target_names))
        }
    )

    st.progress(float(max(probability[0])))

st.divider()

st.caption("Developed using Streamlit and Scikit-Learn")
