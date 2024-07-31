import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import os
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from dotenv import load_dotenv
from pandasai.llm import OpenAI
from pandasai import SmartDataframe

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

llm = OpenAI(api_token=API_KEY)
pandas_ai = None

st.set_page_config(
    page_title="Kubeerr-AI Engine",
    page_icon="Kubeerr4.png",
    layout="wide"
)

st.title("KUBEERR")
st.write("AI Engine")

with st.sidebar:
    st.image("Kubeerr4.png")
    st.title("KUBEERR")
    choice = st.radio("Select your task", ["Upload", "Profiling", "ML", "Downloads"])
    st.info("This AI Engine can accurately model your data with precise assessment")

    if os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv", index_col=None)
    else:
        df = None

if choice == "Upload":
    st.title("Upload your Dataset for Modelling")
    file = st.file_uploader("Upload your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)
        
        query = st.text_area("Chat with your Data")
        st.write(query)

        if st.button("Ask"):
                if query:
                    llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
                    query_engine = SmartDataframe(df, config={"llm": llm})

                    answer = query_engine.chat(query)
                    st.write(answer)
        

if choice == "Profiling":
    st.title("Automated Data Exploratory Analysis")
    if df is not None:
        profile = ydata_profiling.ProfileReport(df, explorative=True)
        st_profile_report(profile)
    else:
        st.info("Please upload a dataset first.")

if choice == "ML":
    st.title("Custom choose and train ML Model")
    if df is not None:
        chosen_target = st.selectbox("Select the target column", df.columns)
        if st.button("Run Modelling"):
            setup(df, target=chosen_target)
            setup_df = pull()
            st.info("This is the ML Experiment Settings")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is ML Model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, "trained_model")
    else:
        st.info("Please upload a dataset first.")

if choice == "Downloads":
    if os.path.exists("trained_model.pkl"):
        with open("trained_model.pkl", "rb") as f:
            st.download_button("Download your Trained Model", f, "trained_model.pkl")
    else:
        st.info("No trained model found. Please run the ML process first.")