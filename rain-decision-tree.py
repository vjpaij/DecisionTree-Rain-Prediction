import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

data = pd.read_csv('weatherAus.csv')
data.info()
data.describe()

#drop any rows of target columns that are empty
data.dropna(subset=['RainTomorrow'], inplace=True)

#Split the data into Training, Validate and Test datasets
year = pd.to_datetime(data['Date']).dt.year
train_df = data[year < 2015]
val_df = data[year == 2015]
test_df = data[year > 2015]
print(f"Train Shape: {train_df.shape}\nValidate Shape: {val_df.shape}\nTest Shape: {test_df.shape}")

#Identify the input and target columns
input_cols = list(train_df.columns)[1:-1]
target_cols = 'RainTomorrow'

train_inputs = train_df[input_cols].copy()
train_target = train_df[target_cols].copy()
val_inputs = val_df[input_cols].copy()
val_target = val_df[target_cols].copy()
test_inputs = test_df[input_cols].copy()
test_target = test_df[target_cols].copy()

#Identify Numeric and Categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
print(numeric_cols)
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()
print(categorical_cols)

#Imputing Numeric Null values
train_inputs[numeric_cols].isna().sum().sort_values(ascending=False)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean').fit(data[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

#Scaling numeric Features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(data[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

print(train_inputs.describe().loc[['min', 'max']])

#Encoding Categorical Columns
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(data[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

#Drop the original Categorical columns so that we are left with only numeric fields
X_train = train_inputs.drop(columns=categorical_cols)
X_val = val_inputs.drop(columns=categorical_cols)
X_test = test_inputs.drop(columns=categorical_cols)
print(X_train.columns.tolist())
'''
X_train = train_inputs[numeric_cols + encoder_cols]
X_val = val_inputs[numeric_cols + encoder_cols]
X_test = test_inputs[numeric_cols + encoder_cols]
'''


