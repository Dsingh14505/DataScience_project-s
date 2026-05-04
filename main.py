# Import Dependencies 
import pickle
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import sklearn

url_ = ''
# Load Saved Models 
heart_model = pickle.load(open("D:/Disease_prd_web_app/svd_models/heart.sav", "rb"))
diabetes_model = pickle.load(open("D:/Disease_prd_web_app/svd_models/diabetes.sav", "rb"))
breast_cancer_model = pickle.load(open("D:/Disease_prd_web_app/svd_models/breast_cancer.sav", "rb"))

# Create Sidebar
with st.sidebar:
    selected = option_menu("Multiple Disease Predictor",
    ["Heart", 
    "Breast Cancer", 
    "Diabetes"],
    icons = ["heart-pulse", "person", "activity"],
    default_index  =0)

# If You Want to Write a msg For Users 
if selected == 'Msg':
    with st.expander("Something_that_you_have_to_say"):
        st.write("......")

# For Heart Disease

if selected == "Heart":
    st.markdown("<h1 style='text-align: center;'>Heart Disease Prediction</h1>", unsafe_allow_html=True)
    #st.title("Heart Disease Prediction") # If we use This then Title show's at  bit left side

    # you Can also use Expander 
    # Put all col1,col2 and col3 code in this 
    # with st.expander("Expand it"):
    col1, col2, col3 = st.columns(3)

    with col1:
         Age = st.number_input("Age")
    with col2:
        Sex = st.number_input("Sex")    
    with col3:
        cp  = st.number_input("ChestPain")    
    with col1:
        trestbps = st.number_input("Resting blood pressure.") 
    with col2:
        chol = st.number_input("Serum cholesterol.")
    with col3:
        fbs = st.number_input("Fasting blood sugar")
    with col1:
        restecg = st.number_input("Resting electrocardiographic results.")
    with col2:
        thalach = st.number_input("Maximum heart rate achieved.")
    with col3:
        exang = st.number_input(" Exercise induced angina.")
    with col1:
        oldpeak = st.number_input("ST depression induced by exercise relative to rest")
    with col2:
        slope = st.number_input("Slope")
    with col3:
        ca = st.number_input("Ca")
    with col1:
        thal = st.number_input("Thal")

    heart_result = ""

    if st.button("Heart Report"):

        input_data = pd.DataFrame([[
            float(Age), float(Sex), float(cp), float(trestbps), float(chol),
            float(fbs), float(restecg), float(thalach), float(exang),
            float(oldpeak), float(slope), float(ca), float(thal)]], columns=[
            "age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak",
            "slope","ca","thal"
])
        result = heart_model.predict(input_data)
        if result[0] == 1:
            heart_result = "Defective Heart"
        else:
            heart_result = "Healthy Heart"
        
    st.success(heart_result)


# For Diabetes
if selected == "Diabetes":
    st.markdown("<h1 style='text-align: center;'>Diabetes Prediction</h1>", unsafe_allow_html=True)
    #st.title("Diabetes Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Pregnancies")
        SkinThickness = st.number_input("SkinThickness")
        DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction")
    with col2:
        Glucose =st.number_input("Glucose")
        Insulin = st.number_input("Insulin")
        Age = st.number_input("Age")
    with col3:
        BloodPressure = st.number_input("BloodPressure")
        bmi_diabetes = st.number_input("BMI")

        diabetes_result =  ""

    if st.button("Diabetes"):
        diab_df = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, bmi_diabetes, DiabetesPedigreeFunction, Age]],  
        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

        prd_diabetes = diabetes_model.predict(diab_df)

        if prd_diabetes[0]==0:
            diabetes_result = "Non_Diabetic"
        else:
            diabetes_result = "Diabetic"
    
    st.success(diabetes_result)


# For Breast Cancer 
if selected == "Breast Cancer":
    st.markdown("<h1 style='text-align: center;'>Breast Cancer Prediction</h1>", unsafe_allow_html=True)
    #st.title("Breast Cancer Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.number_input("radius_mean")
        texture_mean = st.number_input("texture_mean")
        perimeter_mean = st.number_input("perimeter_mean")
        area_mean = st.number_input("area_mean")
        smoothness_mean = st.number_input("smoothness_mean")

    with col2:
        compactness_mean = st.number_input("compactness_mean")
        concavity_mean = st.number_input("concavity_mean")
        concave_points_mean = st.number_input("concave points_mean")
        symmetry_mean = st.number_input("symmetry_mean")
        fractal_dimension_mean = st.number_input("fractal_dimension_mean")

    with col3:
        radius_se = st.number_input("radius_se")
        texture_se = st.number_input("texture_se")
        perimeter_se = st.number_input("perimeter_se")
        area_se = st.number_input("area_se")
        smoothness_se = st.number_input("smoothness_se")

    col4, col5, col6 = st.columns(3)

    with col4:
        compactness_se = st.number_input("compactness_se")
        concavity_se = st.number_input("concavity_se")
        concave_points_se = st.number_input("concave points_se")
        symmetry_se = st.number_input("symmetry_se")
        fractal_dimension_se = st.number_input("fractal_dimension_se")


    with col5:
        radius_worst = st.number_input("radius_worst")
        texture_worst = st.number_input("texture_worst")
        perimeter_worst = st.number_input("perimeter_worst")
        area_worst = st.number_input("area_worst")
        smoothness_worst = st.number_input("smoothness_worst")

    with col6:
        compactness_worst = st.number_input("compactness_worst")
        concavity_worst = st.number_input("concavity_worst")
        concave_points_worst = st.number_input("concave points_worst")
        symmetry_worst = st.number_input("symmetry_worst")
        fractal_dimension_worst = st.number_input("fractal_dimension_worst")

    cancer_result = ""

    if st.button("Breast Cancer Report"):

        input_data = pd.DataFrame([[ 
            float(radius_mean), float(texture_mean), float(perimeter_mean), float(area_mean), float(smoothness_mean),
            float(compactness_mean), float(concavity_mean), float(concave_points_mean), float(symmetry_mean), float(fractal_dimension_mean),
            float(radius_se), float(texture_se), float(perimeter_se), float(area_se), float(smoothness_se),
            float(compactness_se), float(concavity_se), float(concave_points_se), float(symmetry_se), float(fractal_dimension_se),
            float(radius_worst), float(texture_worst), float(perimeter_worst), float(area_worst), float(smoothness_worst),
            float(compactness_worst), float(concavity_worst), float(concave_points_worst), float(symmetry_worst), float(fractal_dimension_worst)
        ]], columns=[
            'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
            'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
            'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
            'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
            'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
            'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
        ])

        result = breast_cancer_model.predict(input_data)

        if result[0] == 1:
            cancer_result = "Malignant (Cancer Detected)"
        else:
            cancer_result = "Benign (No Cancer)"

    st.success(cancer_result)
