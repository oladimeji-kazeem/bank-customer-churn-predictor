import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from scipy import stats
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


# Load the dataset
df = pd.read_csv("./data/bank_churn.csv")

# Set page title
st.set_page_config(page_title="Bank Customer Churn Data Understanding", layout="wide")

st.title("Bank Customer Churn :red[Analysis] :bar_chart:")
st.markdown("Solution to assist in determining those customers that would likely churn so that appropriate preventive steps could be taken to avert possible loss to bottomlines.")

st.subheader("Stage 1: Data Understanding")
tab1, tab2, tab3, tab4 = st.tabs([":clipboard: Data", ":chart: Analysis", ":bar_chart: Modelling", ":coffee: Evaluation"])

with tab1:
    st.markdown("#### Data Understanding")
    # Display first 5 rows of the dataset
    st.write("**First 5 rows of the dataset**")
    st.write(df.head())


    # Check for missing values
    st.write("**Missing values in the dataset**")
    st.write(df.isnull().sum())
    
    # Check for duplicates values
    st.write("**Duplicate records in the dataset**")
    duplicates = df[df.duplicated()]
    if not duplicates.empty:
        st.write('**Duplicate Rows**')
        st.write(duplicates)
    else:
        st.write('No duplicate rows found.')

    # Display summary statistics of numerical variables
    st.write("**Summary statistics of numerical variables**")
    st.write(df.describe())

    # Display value counts of categorical variables
    st.write("**Value counts of categorical variables**")
    st.write("Geography")
    st.write(df["Geography"].value_counts())
    st.write("Gender")
    st.write(df["Gender"].value_counts())
    

    st.write("Data Cleaning") 
    def data_cleaning(df):
        # Drop unnecessary columns
        df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
        # Convert categorical variables to numerical using label encoding
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
        # Convert categorical variables to one-hot encoding
        df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
        #Remove duplicate records
        df.drop_duplicates()
        return df
    st.write(data_cleaning(df.head()))

with tab2:
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    # Convert categorical variables to numerical using label encoding
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    # Convert categorical variables to one-hot encoding
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
    # Detect and remove outliers using Z-score
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number)))
    threshold = 3
    df_no_outliers = df[(z_scores < threshold).all(axis=1)]

    # Display outliers and outlier-free data
    if len(df) == len(df_no_outliers):
        st.write('No outliers detected.')
    else:
        st.write('**Outliers Detected**')
        st.write(df[~(z_scores < threshold).all(axis=1)].head())
        st.write(df.shape)
        st.write('**Outlier-Free Data**')
        st.write(df_no_outliers.shape)
        st.write(df_no_outliers.head())
    
    
    # Data Analysis
    st.subheader("Exploratory Data Analysis")

    # Select the variable to visualize
    # Multiselect for selecting variables
    selected_vars = st.multiselect("Select variables", df.columns)

    # Checkbox for selecting plot types
    plot_types = st.multiselect("Select plot type(s)", ["Histogram", "Boxplot"], default=["Histogram"])

    # Checkbox for displaying categorical variable value counts
    display_cat_value_count = st.checkbox("Display Categorical Value Counts", value=True)

    # Checkbox for displaying missing value counts
    display_missing_value_count = st.checkbox("Display Missing Value Counts", value=True)

    # Loop through selected variables and plot data distributions
    for var in selected_vars:
        if df[var].dtype != "O":
            if "Boxplot" in plot_types:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x="Exited", y=var, data=df)
                plt.title(f"Boxplot of {var} by Churn")
                plt.xlabel("Churn")
                plt.ylabel(var)
                st.pyplot(plt)
            if "Histogram" in plot_types:
                plt.figure(figsize=(12, 6))
                sns.histplot(df[var], bins=20, kde=True)
                plt.title(f"Distribution of {var}")
                plt.xlabel(var)
                plt.ylabel("Density")
                st.pyplot(plt)
        else:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=var, hue="Exited", data=df)
            plt.title(f"Count of {var} by Churn")
            plt.xlabel(var)
            plt.ylabel("Count")
            plt.legend(["Not Churned", "Churned"])
            st.pyplot(plt)
               
with tab3:
    # Scale numerical columns using MinMaxScaler
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=np.number).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    # Function for feature selection
    def feature_selection(df):
        # Split data into features and target
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        
        
        # Split data into training and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Perform logistic regression
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        y_pred_logreg = logreg.predict(X_test)

        # Perform decision tree
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)


        st.write(X.head())
        
        # Calculate evaluation metrics
        accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
        precision_logreg = precision_score(y_test, y_pred_logreg)
        recall_logreg = recall_score(y_test, y_pred_logreg)
        f1_score_logreg = f1_score(y_test, y_pred_logreg)
        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        precision_dt = precision_score(y_test, y_pred_dt)
        recall_dt = recall_score(y_test, y_pred_dt)
        f1_score_dt = f1_score(y_test, y_pred_dt)
        

        # Display evaluation metrics
        st.write('**Logistic Regression**')
        st.write('Accuracy:', accuracy_logreg)
        st.write('Precision:', precision_logreg)
        st.write('Recall:', recall_logreg)
        st.write('F1 Score:', f1_score_logreg)
        st.write('Confusion Matrix:', confusion_matrix(y_test, y_pred_logreg))

        st.write('**Decision Tree**')
        st.write('Accuracy:', accuracy_dt)
        st.write('Precision:', precision_dt)
        st.write('Recall:', recall_dt)
        st.write('F1 Score:', f1_score_dt)
        st.write('Confusion Matrix:', confusion_matrix(y_test, y_pred_dt))

        