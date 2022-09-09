# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 14:37:40 2022

@author: b reign
"""
import pandas as pd
import streamlit as st
import plotly_express as px
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor




#configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Welcome to CFA")

# add sidebar
st.sidebar.subheader("Visualization Settings")


#setup file upload
uploaded_file = st.sidebar.file_uploader(
                        key="1",
                        label="Upload your csv or Excel file. (200MB max)",
                         type=['csv', 'xlxs'])

df = None

if uploaded_file is not None:
    print(uploaded_file)
    print("Hello")
    try: 
        df = pd.read_csv(uploaded_file)
        st.write('## VSLA Data')
       # st.dataframe(df,3000,500)
    except Exception as e:
        print(e)
        df = pd.read_csv(uploaded_file)




global numeric_columns
try: 
    if df is not None:
        st.write(df)
        numeric_columns = list(df.select_dtypes(['float','int']).columns)
except Exception as e:
    print(e)
    st.write("Please upload file to the application")
    st.write("THEN")


#add a select widget to the side bar

chart_select = st.sidebar.selectbox(
    label="Slect the chart type",
    options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    )



if chart_select == 'Scatterplots':
    st.sidebar.subheader("Scatterplot Settings")
    try: 
         x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
         y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
         plot = px.scatter(data_frame=df, x=x_values, y=y_values)
        
         # display the chart
         st.plotly_chart(plot)
    except Exception as e:
        print(e)
        
if chart_select == 'Lineplots':
    st.sidebar.subheader("Lineplot Settings")
    try: 
         x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
         y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
         plot = px.line(data_frame=df, x=x_values, y=y_values)
        
         # display the chart
         st.plotly_chart(plot)
    except Exception as e:
        print(e)
        
if chart_select == 'Histogram':
    st.sidebar.subheader("Histogram Settings")
    try: 
         x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
         y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
         plot = px.histogram(data_frame=df, x=x_values, y=y_values)
        
         # display the chart
         st.plotly_chart(plot)
    except Exception as e:
        print(e)
        
if chart_select == 'Boxplot':
    st.sidebar.subheader("Boxplot Settings")
    try: 
         x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
         y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
         plot = px.boxplot(data_frame=df, x=x_values, y=y_values)
        
         # display the chart
         st.plotly_chart(plot)
    except Exception as e:
        print(e)
        
        
        
expander = st.expander("Make prediction")
expander.write("""
    CFA (Community Fund Advisor) is not just a Decision Support System. She is your TRO assistant. Upload the data the VSLA data to predict their shareouts by the end of the financial year.
""")
expander.image("vsla.jpg")



# Load data
#df4 = pd.read_csv("ExportCSV_4.csv")
#df2 = pd.read_csv("ExportCSV_2.csv")
#df3 = pd.read_csv("ExportCSV_3.csv")


# Concatenate and show head

# df.head(5)

#df4 = pd.read_csv(uploaded_file)
#global df

global predict_shareouts 


if uploaded_file is not None:
    print(uploaded_file)
    print("Hello")
    try: 
        df = pd.read_csv(uploaded_file)
        st.write('## Data set')
        # st.dataframe(df,3000,500)           
    except Exception as e:
        print(e)

if df is not None:
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
    # Instantiate model
    lr = LinearRegression()

    # Fit the model
    lr.fit(X_train, y_train)
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
    df['Year'] = df['Year'].astype('object')

    X_features = df.drop(['ID', 'VSLA', 'Division'], axis=1)
    y_values = df['Shareouts']

    X_features_on_processing = preprocessor.transform(X_features)
    new_preds = rf.predict(X_features_on_processing)
    #res["Difference"] = res["Actual"] - res["Predicted"]
    import pickle
    pickle.dump(rf, open("my_rand_forest_reg.pickle", 'wb'))

    loaded_model = pickle.load(open("my_rand_forest_reg.pickle", 'rb'))

    def welcome():
         return "Welcome to CFa Decision Support"

    def predict_shareouts(X_features_on_processing):
         prediction=loaded_model.predict(X_features_on_processing)
         print(prediction)
         return prediction

    def show_res(y_values, new_preds):
         myvslaz = df['VSLA']
         res = pd.DataFrame({"VSLAs": myvslaz, "Actual": y_values, "Predicted":new_preds})
         res["Difference"] = abs(res["Actual"] - res["Predicted"])
         st.table(res)
         #print(res)
         return res
     
       
    

def main():
   st.title("CFA")
   html_temp ="""
   <div style="background-color: tomato;padding:10px">
   <h2 style="color:white;text-align:center;"> Decision Support System</h2>
   </div>
   """
   comparison=""
   result=""
   if st.button("Predict"):
       result=predict_shareouts(X_features_on_processing)
       
   st.success('The predicted shareouts by the end of the finacial year are: {}'.format(result))
   
   if st.button("Compare"):
       comparison=show_res(y_values, new_preds)
      
   #st.success('Compare and contrast {}'.format(comparison))
       
   if st.button("About"):
       st.text("Community Fund Advisor")
       st.text("Built by BSSE2-22")
       
if __name__=='__main__':
    main()

    




# Check data types
## df.info()
# Check null count
# df.isna().sum()
#df= df.fillna(0)
#df = pd.read_csv(uploaded_file)
