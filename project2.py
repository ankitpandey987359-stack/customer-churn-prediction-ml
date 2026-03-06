import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("placement_model.pkl", "rb"))

# If you also saved columns file, use it (recommended)
# model_columns = pickle.load(open("model_columns.pkl", "rb"))

# If you DON'T have model_columns.pkl, use these columns (based on your dataset)
model_columns = [
    "CGPA",
    "Internships",
    "Projects",
    "Coding_Skills",
    "Communication_Skills",
    "Aptitude_Test_Score",
    "Soft_Skills_Rating",
    "Certifications",
    "Backlogs",
    "Branch_CSE",
    "Branch_Civil",
    "Branch_ECE",
    "Branch_IT",
    "Branch_ME",
]

st.title("🎓 Placement Prediction System")

st.subheader("Enter Student Details")

# Numeric Inputs
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=6.5, step=0.01)
internships = st.number_input("Internships", min_value=0, max_value=20, value=0, step=1)
projects = st.number_input("Projects", min_value=0, max_value=50, value=2, step=1)

coding = st.number_input("Coding Skills (0-10)", min_value=0, max_value=10, value=5, step=1)
comm = st.number_input("Communication Skills (0-10)", min_value=0, max_value=10, value=5, step=1)

aptitude = st.number_input("Aptitude Test Score (0-100)", min_value=0, max_value=100, value=50, step=1)
soft = st.number_input("Soft Skills Rating (0-10)", min_value=0, max_value=10, value=5, step=1)

cert = st.number_input("Certifications", min_value=0, max_value=50, value=0, step=1)
backlogs = st.number_input("Backlogs", min_value=0, max_value=20, value=0, step=1)

# Categorical Input
branch = st.selectbox("Branch", ["CSE", "ECE", "IT", "ME", "Civil"])

def make_input_df():
    # Start all features with 0
    input_dict = {col: 0 for col in model_columns}

    # Fill numeric features
    input_dict["CGPA"] = cgpa
    input_dict["Internships"] = internships
    input_dict["Projects"] = projects
    input_dict["Coding_Skills"] = coding
    input_dict["Communication_Skills"] = comm
    input_dict["Aptitude_Test_Score"] = aptitude
    input_dict["Soft_Skills_Rating"] = soft
    input_dict["Certifications"] = cert
    input_dict["Backlogs"] = backlogs

    # One-hot encode Branch
    branch_col = f"Branch_{branch}"
    if branch_col in input_dict:
        input_dict[branch_col] = 1

    # Create dataframe in exact order
    input_df = pd.DataFrame([input_dict], columns=model_columns)
    return input_df

if st.button("Predict Placement"):
    input_df = make_input_df()

    # Predict
    pred = model.predict(input_df)[0]  # 0 or 1

    if int(pred) == 1:
        st.success("✅ Prediction: PLACED")
    else:
        st.error("❌ Prediction: NOT PLACED")

    st.write("Input used for prediction:")
    st.dataframe(input_df)