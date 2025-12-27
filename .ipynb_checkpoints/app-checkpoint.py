import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import numpy as np
from tensorflow.keras.models import load_model
model=load_model('model.h5')

with open('encoder,'rb') as file:
    ohe=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

    
st.title("Flight Price Prediction")
Source=st.selectbox('Source',encoder.categories_[0])
Destination=st.selectbox('Destination',encoder.categories_[0])
Airline=st.selectbox('Airline',encoder.categories_[0])
Total_Stops=st.selectbox('Total_Stops',[1,2,3,4,5])
Day=st.slider('Day',1,30)
Month=st.slider('Month',1,12)
Year=st.slider('Year',1995,2025)      
Dep_hour=st.slider('Dep_hour',0,25)
Dep_min=st.slider('Dep_min',0,56)
Arrival_hour=st.slider('Arrival_hour',0,25)
Arrival_min=st.slider('Arrival_min',0,56)
Duration_hour=st.slider('Duration_hour',0,25)
Duration_min=st.slider('Duration_min',0,56)
input_data=pd.DataFrame({
    'Source': [Source],
    'Destination': [Destination],
    'Airline': [Airline],
    'Total_Stops': [Total_Stops],
    
    'Day':[Day],
    'Month':[Month],
    'Year':[Year],
    'Dep_hour':Dep_hour,
    'Dep_min':[Dep_min],
    'Arrival_hour':[Arrival_hour],
    'Arrival_min':[Arrival_min],
    'Duration_hour':[Duration_hour],
    'Duration_min':[Duration_min]
})
num_features = ['Total_Stops','Day','Month','Year','Dep_hour','Dep_min',
                'Arrival_hour','Arrival_min','Duration_hour','Duration_min']

# Encode categorical features
encoded = ohe.transform(input_data[['Source','Destination','Airline']])

# Convert to DataFrame
encoder_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(),  # correct usage
    index=input_data.index
)

# Drop original categorical columns and concatenate encoded ones
input_data = pd.concat([
    input_data.drop(['Airline','Source','Destination'], axis=1),
    encoder_df
], axis=1)


numeric_data = input_data[num_features]
input_data[num_features]= scaler.transform(numeric_data)
prediction=XGBR.predict(input_data)
st.write('The Flight Price Prediction is ',prediction)





