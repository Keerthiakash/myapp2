import streamlit as st
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Wine Cultivar Classifier",
    page_icon="🍷",
    layout="centered"
)

# -----------------------------
# Load Dataset
# -----------------------------
wine = load_wine()

X = wine.data
y = wine.target

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Random Forest Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test)

# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("🍷 Wine Cultivar Classifier")

st.write(
    "Enter the chemical properties of the wine "
    "to predict its cultivar."
)

st.metric(
    label="Model Accuracy",
    value=f"{accuracy:.2%}"
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("🧪 Chemical Properties")

feature_names = wine.feature_names

mins = X.min(axis=0)
maxs = X.max(axis=0)
means = X.mean(axis=0)

col1, col2 = st.columns(2)

inputs = []

for i, feature in enumerate(feature_names):

    column = col1 if i % 2 == 0 else col2

    with column:
        value = st.number_input(
            feature.replace("_", " ").title(),
            min_value=float(mins[i]),
            max_value=float(maxs[i] * 1.2),
            value=float(round(means[i], 2)),
            format="%.2f"
        )

        inputs.append(value)

# -----------------------------
# Prediction
# -----------------------------
st.divider()

if st.button("🔍 Predict Cultivar", use_container_width=True):

    prediction = model.predict([inputs])[0]
    probabilities = model.predict_proba([inputs])[0]

    cultivar_name = wine.target_names[prediction]

    st.success(
        f"🍷 Predicted Cultivar: **{cultivar_name.title()}**"
    )

    st.subheader("📊 Prediction Confidence")

    for i, probability in enumerate(probabilities):
        st.write(
            f"**{wine.target_names[i].title()}**: "
            f"{probability * 100:.2f}%"
        )

    # Highest confidence
    confidence = max(probabilities)

    st.progress(
        float(confidence),
        text=f"Confidence: {confidence * 100:.2f}%"
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Developed using Streamlit and Scikit-Learn"
)
