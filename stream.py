# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 11:45:21 2022

@author: b reign
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline

# Load data
df4 = pd.read_csv("ExportCSV_4.csv")
df2 = pd.read_csv("ExportCSV_2.csv")
df3 = pd.read_csv("ExportCSV_3.csv")

# Concatenate and show head
df = pd.concat([df4, df2, df3])
# df.head(5)

# Check data types
## df.info()
# Check null count
# df.isna().sum()
df= df.fillna(0)
y = df['Shareouts']
X = df.drop(['ID', 'VSLA', 'Division'], axis=1)
X['Year'] = X['Year'].astype(object)

# Select numerical and categorical columns
cat_selector = make_column_selector(dtype_include='object')
num_selector = make_column_selector(dtype_include='number')

cat_selector(X)

# Scaler and onehot encorder objects
scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown='ignore')

# Make pipelines
num_pipeline = make_pipeline(scaler)
cat_pipeline = make_pipeline(ohe)
# num_pipeline = make_pipeline(scaler, num_selector)
# cat_pipeline = make_pipeline(ohe, cat_selector)

# Transform columns
number_tuple = (num_pipeline, num_selector)
cat_tuple = (cat_pipeline, cat_selector)


# Colummn transformer
preprocessor = make_column_transformer(number_tuple, cat_tuple)
# preprocessor = make_column_transformer(num_pipeline, cat_pipeline)

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3453, test_size=0.15)


# Fit preprocessor
preprocessor.fit(X_train)

# Tranform data
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)
X_train

from sklearn.linear_model import LinearRegression

# Instantiate model
lr = LinearRegression()

# Fit the model
lr.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


# Instantiate models
dec_tree = DecisionTreeRegressor(random_state=576)
rf = RandomForestRegressor(random_state=7668)
xg = GradientBoostingRegressor(random_state=87879)


# Fit the models
dec_tree.fit(X_train, y_train)
rf.fit(X_train, y_train)
xg.fit(X_train, y_train)

# MAke predictions
preds = lr.predict(X_test)
preds_dt = dec_tree.predict(X_test)
preds_rf = rf.predict(X_test)
preds_xg = xg.predict(X_test)

df4['Year'] = df4['Year'].astype('object')

X_features = df4.drop(['ID', 'VSLA', 'Division'], axis=1)
y_values = df4['Shareouts']

X_features_on_processing = preprocessor.transform(X_features)
new_preds = rf.predict(X_features_on_processing)

res = pd.DataFrame({"Actual": y_values, "Predicted":new_preds})

res["Difference"] = res["Actual"] - res["Predicted"]

# Saving model
import pickle
pickle.dump(rf, open("my_rand_forest_reg.pickle", 'wb'))



loaded_model = pickle.load(open("my_rand_forest_reg.pickle", 'rb'))

def welcome():
    return "Welcome to CFa"

def predict_shareouts(X_features_on_processing):
    prediction=loaded_model.predict(X_features_on_processing)
    print(prediction)
    return prediction

def main():
    st.title("CFA")
    html_temp ="""
    <div style="background-color: tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Decision Support System</h2>
    </div>
    """

    result=""
    if st.button("Predict"):
        result=predict_shareouts(X_features_on_processing)
        
    st.success('The predicted shareouts by the end of the finacial year are: {}'.format(result))
    if st.button("About"):
        st.text("Community Fund Advisor")
        st.text("Built by BSSE22-2")
        
if __name__=='__main__':
    main()