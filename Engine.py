import pandas as pd
import numpy as np
import pickle
import streamlit as st
import io
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, root_mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt

def split(data, target=[]): # Splitting the data into training and testing sets
    x = data.drop(columns=target)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    ) 
    return x_train, x_test, y_train, y_test

def pipline(categorical_fet = [], numerical_fet=[]): # Creating a pipeline which helps in one hot encoding and filling missing values
    categorical_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    numerical_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", fill_value="missing"))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_trans, categorical_fet),
            ("num", numerical_trans, numerical_fet)
        ]
    )
    return preprocessor



def RFC(model, x_train:np.array, y_train:np.array, preprocessor):
    main_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )
    scoring = {
        "accuracy" : "accuracy",
        "precision" : "precision_weighted",
        "recall" : "recall_weighted",
        "f1" : "f1_weighted"
    }
    scores = cross_validate(
        main_model,
        x_train,
        y_train,
        scoring=scoring,
        cv=5
    )

    results_df = pd.DataFrame({
        "Fold" : range(1, 6),
        "Accuracy" : scores["test_accuracy"],
        "Precision" : scores["test_precision"],
        "Recall" : scores["test_recall"],
        "f1" : scores["test_f1"]
    })
    
    st.dataframe(results_df)
    return main_model

def conf_matrix(main_model, x_train, y_train, cv=5):# The confusion matrix for a cross validated model
    y_pred = cross_val_predict(
        main_model,
        x_train,
        y_train,
        cv=cv
    )
    conf = confusion_matrix(y_train, y_pred)
    fig2, ax2 = plt.subplots(figsize=(3, 2))
    display = ConfusionMatrixDisplay(confusion_matrix=conf)
    display.plot(ax=ax2, cmap="Blues")
    st.pyplot(fig2)


def RFC_nocross(x_train, y_train, x_test, y_test, preprocessor):
    model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier())
        ])
    model.fit(x_train, y_train)
    st.write(f"Accuracy :", model.score(x_test, y_test))
    return model

def confuse(x_test, y_test, model):# Confusion matrix for a non-cross validated model 
    y_pred = model.predict(x_test)
    conf = confusion_matrix(y_pred, y_test)
    fig, ax = plt.subplots(figsize=(3, 2))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf)
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)
    st.dataframe(conf)
# def evaluate_preds(x_test, y_test):

def RGR(model, preprocessor, x_train, y_train):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    scoring = {
        "mae" : "neg_mean_absolute_error",
        "mse" : "neg_mean_squared_error",
        "r2" : "r2"
    }

    scores = cross_validate(
        model,
        x_train,
        y_train,
        scoring=scoring,
        cv=5
    )

    results = pd.DataFrame({
        "Fold" : range(1, 6),
        "MAE" : -scores["test_mae"],
        "MSE" : -scores["test_mse"],
        "R2 score" : scores["test_r2"]
    })
    st.dataframe(results)
    return model

def plot_reg_matrix(model, x_train, y_train):
    predictions = cross_val_predict(
        model,
        x_train,
        y_train,
        cv=5
    )
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.scatter(y_train, predictions, alpha=0.6)

    min_val = min(y_train.min(), predictions.min())
    max_val = max(y_train.max(),predictions.max())

    ax.plot([min_val, max_val],
            [min_val, max_val],
            color="red",
            label="Perfect predictions")
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.set_title("Actual vs Prediction")

    ax.legend()
    ax.grid(alpha=0.3)

    return fig

def save_model(model):
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    model_bytes = buffer.getvalue()
    st.download_button(label="Download model",
                    data=model_bytes,
                    file_name="My model.pkl",
                    mime="application/octet-stream")

def RGR_noocross(preprocessor, x_train, x_test, y_train, y_test):
    model = Pipeline(
        steps=[
            ("preprocessor" , preprocessor),
            ("model" , RandomForestRegressor())
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    results = pd.DataFrame({'R2' : model.score(x_test, y_test),
                'MAE' : mean_absolute_error(y_test, predictions),
                'MSE' : mean_squared_error(y_test, predictions),
                'RMSE' : np.sqrt(mean_squared_error(y_test, predictions))
                }, index=[0])
    st.dataframe(results)