import streamlit as st
import time
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


# MongoDB connectivity
uri = "mongodb+srv://anu:tiger@cluster0.57jxgvp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new MongoDB client and connect to the Database server
client = MongoClient(uri, server_api = ServerApi('1'))
db = client['diabetes']
collection = db['patient_data']

# Step1: Module for Loading the required model
def load_model(model_name):
    """
    Loads the saved model and its scaler from a pickle file.

        Parametrs:
        ----------
            model_name:str
                Name of the model contains the model and its scaler

        Returns:
        --------
            model:sklearn.linear_model
                The loaded model

            scaler:sklearn.preprocessing
                The loaded Scaler
    """
    with open(model_name, 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler

# Step2 : Custome Module to proces the input data

def preprocessing_input_data(data, scaler):
    """
    Preprocess the input data and prepare it for the model to ingest for processing

    The input data is converted into  DataFrame, The categorical variable
    "SEX", is mapped into numeric value (1 for male and 2 for Female).
    Finally the data is transformed using scaler to standardize the features. 

    Parameters:
    -----------
        data: dict
            A dictionary containing the input data
        scaler: object
            scaler object from sklearn

    Returns:
    --------
        df_transformed: array
            Teh transformed data ready for predection
    """
    
    df = pd.DataFrame([data])
    df["SEX"] = df["SEX"].map({"Male": 1,"Female": 2})
    df_transformed = scaler.transform(df)
    return df_transformed

# Step3: Module for predection of Diabetes progression
def predict_data(data, model_name):
    """
    Predict the output based on the input data given. 

    The input data is prossed through "preprocessing_input_data"
    and pass to "predection" module to predict the output through loaded model. 

    Parameters:
    -----------
        data: dict
            input is a dictionary containing the patient records

        model_name: str
            The name of the model containing the saved model and scaler.

    Returns:
    --------
        perdiction: array
            It contains the predected output.

    """

    model, scaler = load_model(model_name)
    processed_data = preprocessing_input_data(data, scaler)
    prediction = model.predict(processed_data)
    return prediction


#-------------------------------------------------------------------------------------------#
# Define the main function for the streamlit application.
def main():
    st.set_page_config(layout="wide")
    st.title("Diabetes Progression Predection.")

    # Sidebar layout for entering the data
    st.sidebar.header("⚙️ **Model Selection** ")
    model_choice = st.sidebar.radio(" **Chose the model** ",["Ridge Model", "Lasso Model"])
    
    st.sidebar.markdown("---") # Divider line

    # Get the patient data input
    st.sidebar.write(" **The Patient Data** ")
    st.sidebar.write(":spiral_note_pad: Enter the following details to predect the diabetes progression")
    st.sidebar.markdown("---") # Divider line

    age = st.sidebar.slider("Enter the age", 15, 99, 25)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    bmi = st.sidebar.slider("Enter the Body Mass Index",18,45,25)
    bp = st.sidebar.slider("Enter the Blood pressure",80,180,120)
    s1 = st.sidebar.slider("Enter total serun cholesterol(TC)",90,400,200)
    s2 = st.sidebar.slider("Enter Low-Density Lipoproteins(LDL)",50,250,100)
    s3 = st.sidebar.slider("Enter High-Density Lipoproteins(HDL)",20,100,50)
    s4 = st.sidebar.slider("Totel Cholesterol / HDL Ratio(TCH)", 1.5,10.0,4.5)
    s5 = st.sidebar.slider("Serum Triglycerides Level(LTG)", 3.0,6.5,5.2)
    s6 = st.sidebar.slider("Blood Sugar Level", 50,600,99)

    # Data Mapping
    patient_data = {
        "AGE":age,
        "SEX":sex,
        "BMI":bmi,
        "BP":bp,
        "S1":s1,
        "S2":s2,
        "S3":s3,
        "S4":s4,
        "S5":s5,
        "S6":s6,
    }

    # Submit button to predect the result
    if st.sidebar.button("Predict"):
        with st.spinner(":stopwatch: Processing the patient data.... Please wait!.."):
            time.sleep(2) # 

            # Map Model selection
            model_name = "diabetes_ridge_model.pkl" if model_choice == "Ridge Model" else "diabetes_lasso_model.pkl"
            model_ID = "Ridge" if model_choice == "Ridge Model" else "Lasso"

            # Make Prediction
            prediction = predict_data(patient_data, model_name)

            # Store User data in MongoDB
            patient_data["Diabetes_progression_result"] = round(float(prediction[0]),2)
            patient_data["Model Name"] = model_ID
            document = {key: int(value) if isinstance(value,np.integer) else 
                        float(value) if isinstance(value, np.floating) else
                        value for key, value in patient_data.items()}
            
            collection.insert_one(document)

        # Display results
        st.markdown(f"## Prediction Results")
        st.success(f" **Diabetes progression score: {prediction[0]:.2f}**")


    # Footer -- APP OWNER
    st.markdown(
        "<br><hr><center>Created by **Anu L Sasidharan** | Using Streamlit. </center></hr></br>",
        unsafe_allow_html=True
    )
    st.markdown("---") # Divider line

if __name__ == "__main__":
    # Run the streamlit app
    main()