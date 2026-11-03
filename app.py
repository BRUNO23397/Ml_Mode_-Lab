import pandas as pd
import numpy as np
import streamlit as st
import Engine
import time


st.set_page_config(
    page_title="Ml Model Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Ml Model Lab")
st.write("Upload a cleaned data and perform machine learning processes on it")

uploaded_file = st.file_uploader(
    "Upload your csv file",
    type = ["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Your dataset preview")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    st.subheader("Select Prediction Target")
    target = st.selectbox("Which column do u want to predict?", options=df.columns)
    st.write("Selected target:", target)

    if target is not None:
        x_train, x_test, y_train, y_test = Engine.split(df, target)
        st.subheader("Cross Validation")
        opt = ["Yes", "No"]
        if "cross" not in st.session_state:
            st.session_state.cross = False
        if "nocross" not in st.session_state:
            st.session_state.nocross = False

        st.write("Do u want to use cross validation ?")
        col3, col4 = st.columns(2)

        with col3:
            if st.button("Yes"):
                st.session_state.cross = True
        with col4:
            if st.button("No"):
                st.session_state.nocross = True

        if st.session_state.cross:#    This trains the model with cross validation
            st.subheader("Select Features")
            column_opt = x_train.columns
            cat_features = st.multiselect( # Let the user select multiple categories to preprocess
                label= "Select your categorical features:",
                options= column_opt,)
        
            st.write(f"Selected categorical features : {cat_features}")
            num_features = st.multiselect( # let the user select multiple numerical categories to preprocess
                label= "Select your numerical features",
                options = column_opt,
            )
            st.write(f"Selected numerical features: {num_features}")

            st.subheader(":red[Choose a model]")
            model = st.selectbox(
            "What model would you like to use for this problem",
            options= ["RandomForestClassifier", "RandomForestRegressor"], key=3
            )
            if st.button("Done"):
                if model == "RandomForestClassifier":
                    model = Engine.RandomForestClassifier()
                    progress = st.progress(0)
                    status = st.empty()

                    status.write("Preparing data...")
                    progress.progress(20)
                    time.sleep(1)

                    status.write("Running cross-validation....")
                    progress.progress(40)

                    preprocessor = Engine.pipline(cat_features, num_features)
                    main_model = Engine.RFC(model, x_train, y_train, preprocessor)

                    progress.progress(90)
                    status.write("preparing results....")
                    time.sleep(1)

                    # st.subheader("Confusion Matrix")
                    # Engine.conf_matrix(main_model, x_test, y_test, preprocessor) #Print out the confusion matrix

                    progress.progress(100)
                    status.success("Trainng complete")

                    st.write("Your model has been evaluated")
                    st.write("The confusion matrix is:")
                    Engine.conf_matrix(main_model,x_train, y_train, preprocessor)

                if model == "RandomForestRegressor": # Handles random forest regressor models
                    model = Engine.RandomForestRegressor()
                    progress = st.progress(0)
                    status = st.empty()

                    st.write("Preparing data....")
                    progress.progress(20)
                    time.sleep(1)

                    st.write("Running cross validation...")
                    progress.progress(40)

                    preprocessor = Engine.pipline(cat_features, num_features)
                    

                    progress.progress(90)
                    st.write("Preparing results....")
                    time.sleep(1)

                    main_model = Engine.RGR(model, preprocessor, x_train, y_train)
                    progress.progress(100)
                    status.success("Training complete")

                    st.write("Your model has been evaluated")
                    st.write("The confusion matrix : ")
                    fig = Engine.plot_reg_matrix(main_model, x_train, y_train)
                    st.pyplot(fig)
                    

        if st.session_state.nocross: # This trains the model without cross validation
            st.write("Implementing Training...")
            st.subheader("Select Features")
            column_opt2 = x_train.columns
            cat_features2 = st.multiselect( # Let the user select multiple categories to preprocess
                        label= "Select your categorical features:",
                        options= column_opt2)
    
            st.write(f"Selected categorical features : {cat_features2}")
            num_features2 = st.multiselect( # let the user select multiple numerical categories to preprocess
                        label= "Select your numerical features",
                        options = column_opt2,
                        )
            st.write(f"Selected numerical features: {num_features2}")
            
            st.subheader(":red[Choose a model]")
            model2 = st.selectbox(
                        "What model would you like to use for this problem ?",
                        options= ["RandomForestClassifier", "RandomForestRegressor"]
                        )
            
            if st.button("Done"):
                if model2 == "RandomForestClassifier":
                    progress = st.progress(0)
                    status = st.empty()
            
                    status.write("Preparing data...")
                    progress.progress(20)
                    time.sleep(1)
            
                    status.write("Running....")
                    progress.progress(40)
            
                    preprocessor2 = Engine.pipline(cat_features2, num_features2)
                    main_model2 = Engine.RFC_nocross(x_train, y_train, x_test, y_test, preprocessor2)
            
                    progress.progress(90)
                    status.write("preparing results....")
                    time.sleep(1)
            
                    # st.subheader("Confusion Matrix")
                    # Engine.conf_matrix(main_model, x_test, y_test, preprocessor) #Print out the confusion matrix
            
                    progress.progress(100)
                    status.success("Training complete")
            
                    st.write("Your model has been trained")
                    st.subheader("Confusion matrix")
                    st.write("Your models confusion matrix is :")
                    Engine.confuse(x_test, y_test, main_model2)
                    st.write("Save model?")
                    Engine.save_model(main_model2)
                if model2 == "RandomForestRegressor":
                    preprocessor3 = Engine.pipline(cat_features2, num_features2)
                    model3 = Engine.RGR_noocross(preprocessor3, x_train, x_test, y_train, y_test)
                




