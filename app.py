import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# ----------------------------
# Load saved files
# ----------------------------
with open('label_encoder_gender.pkl','rb') as file:
    le_gender = pickle.load(file)

with open('label_encoder_churn.pkl','rb') as file:
    le_churn = pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    ohe = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

model = load_model('model.h5')

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Prediction (ANN Model)")
st.write("Enter customer details to predict churn")

st.divider()

# ----------------------------
# Input fields
# ----------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male","Female"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", value=50.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month","One year","Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer","Credit card"]
)

total_charges = st.number_input("Total Charges", value=500.0)

st.divider()

# ----------------------------
# Predict button
# ----------------------------
if st.button("🔮 Predict Churn"):

    # Create dataframe
    input_data = pd.DataFrame({
        'Age':[age],
        'Gender':[gender],
        'Tenure':[tenure],
        'MonthlyCharges':[monthly_charges],
        'Contract':[contract],
        'PaymentMethod':[payment_method],
        'TotalCharges':[total_charges]
    })

    # Label encode gender
    input_data['Gender'] = le_gender.transform(input_data['Gender'])

    # One-hot encode
    encoded = ohe.transform(input_data[['Contract','PaymentMethod']])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(['Contract','PaymentMethod'])
    )

    input_data = input_data.drop(['Contract','PaymentMethod'], axis=1)
    input_data = pd.concat([input_data.reset_index(drop=True), encoded_df], axis=1)

    # Scale input
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)
    prob = prediction[0][0]

    pred_binary = (prob > 0.5).astype(int)
    result = le_churn.inverse_transform([pred_binary])[0]

    st.subheader("Prediction Result")

    if result == "Yes":
        st.error(f"⚠️ Customer likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer not likely to churn (Probability: {prob:.2f})")
