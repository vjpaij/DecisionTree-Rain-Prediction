import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
plt.rcParams['text.color'] = 'white'  # Default text color
plt.rcParams['axes.labelcolor'] = 'white'  # Axis labels
plt.rcParams['xtick.color'] = 'white'  # X-axis tick labels
plt.rcParams['ytick.color'] = 'white'  # Y-axis tick labels
plt.rcParams['axes.titlecolor'] = 'white'  # Axis title
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

#Training the Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
model  = DecisionTreeClassifier(random_state=42)
model.fit(X_train, train_target)

#Evaluating the Model
from sklearn.metrics import accuracy_score, confusion_matrix
train_preds = model.predict(X_train)
print(train_preds)
train_score = accuracy_score(train_preds, train_target)
print(train_score)
train_probs = model.predict_proba(X_train)
print(train_probs)

#Evaluating the model now on the validation set
#We can prediction and compute accuracy in one step by using the function 'score'
val_score = model.score(X_val, val_target)
print(val_score)
'''
We can see the accuracy score for train dataset was 100% but for validation dataset it is 79.3%. This is a clear indication of 
overfitting.
'''

#Visualizing the Decision Tree
from sklearn.tree import plot_tree, export_text
# plt.figure(figsize=(20, 10))
# plot_tree(model, feature_names=X_train.columns, max_depth=2, filled=True)
# plt.show()
print(model.tree_.max_depth)
#Similar tree can be seen as text using export_text
tree_text = export_text(model, max_depth=10, feature_names=list(X_train.columns))
print(tree_text[0:5000])
#Based on gini va;ue, mode assigns an 'importance' value to each feature
importance_df = pd.DataFrame({
    'feature' : X_train.columns,
    'importance' : model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df)
# sns.barplot(data=importance_df.head(10), x='importance', y='feature')
# plt.show()  

#Using hyperparameters to tune the model
#max_depth
model_max = DecisionTreeClassifier(max_depth=5, random_state=42)
model_max.fit(X_train, train_target)
max_train_score = model_max.score(X_train, train_target)
max_val_score = model_max.score(X_val, val_target)
print(max_train_score, max_val_score)
'''
we see accuracy score of 83.9 and 84.1 respectively
'''
#Determining the best value for max_depth
def max_depth_error(md):
    model_max = DecisionTreeClassifier(max_depth=md, random_state=42)
    model_max.fit(X_train, train_target)
    train_acc = 1 - model_max.score(X_train, train_target)
    val_acc = 1 - model_max.score(X_val, val_target)
    return {'Max Depth': md, 'Training Error': train_acc, 'Validation Error': val_acc}

errors_df = pd.DataFrame([max_depth_error(md) for md in range(1, 21)])
plt.figure()
plt.plot(errors_df['Max Depth'], errors_df['Training Error'], label='Training Error')
plt.plot(errors_df['Max Depth'], errors_df['Validation Error'], label='Validation Error')
plt.title('Training vs. Validation Error')
plt.xlabel('Max Depth')
plt.ylabel('Error (1 - Accuracy)')
plt.legend(['Training', 'Validation'])
plt.xticks(range(0, 21, 2))
plt.show()
'''
we see max depth value of 7, the validation error agaain start increasing. so the ideal max depth value is 7.
'''
model_max = DecisionTreeClassifier(max_depth=7, random_state=42).fit(X_train, train_target)
model_max.score(X_val, val_target) # 84.5

#hyperparameter tuning using max_leaf_nodes
model_leaf = DecisionTreeClassifier(max_leaf_nodes=128, random_state=42)
model_leaf.fit(X_train, train_target)
leaf_train_score = model_leaf.score(X_train, train_target)
leaf_val_score = model_leaf.score(X_val, val_target)
print(leaf_train_score, leaf_val_score) # 84.8, 84.4
# We can similarly find the ideal max_leaf_nodes just like we did for max_depth

#More effective strategy is to combine results of several decision trees trained with slightly different parameters. This is
#called Random Forest
from sklearn.ensemble import RandomForestClassifier
#n-jobs allows random forest to use multiple parallel workers to train decision trees 
model_rf = RandomForestClassifier(n_jobs=-1, random_state=42)
model_rf.fit(X_train, train_target)
rf_train_score = model_rf.score(X_train, train_target)
rf_val_score = model_rf.score(X_val, val_target)
print(rf_train_score, rf_val_score) # 100, 85.7
rf_train_probs = model_rf.predict_proba(X_train)
print(rf_train_probs)
#we can access individual decision trees using estimators_
print(len(model_rf.estimators_)) #100 (by default but configurable)
model_rf.estimators_[0]
#random forest also assign importance to each feature by combining importance values from individual decision trees
rf_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model_rf.feature_importances_}).sort_values('importance', ascending=False)
rf_importance_df.head(10)
sns.barplot(data=rf_importance_df.head(10), x='importance', y='feature')
plt.show()
'''
we can notice distribution is lot less skewed than single decision tree
'''

#Random Forest Hyperparameter tuning
'''
from the basic setup of Random Forest model above, let us set the accuracy obtained as our benchmark
rf_train_score = 100
rf_val_score = 85.7
'''
#n_estimators -> default is 100
'''
model_rf = RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=200)
keep checking the score by doubling the value until the change in accuracy is negligible
we can also draw a graph between training error and validator error and arrive at the right value
'''
#max_depth and max_leaf_nodes
'''
Similar to decision tree model. the value set would be applicable for all decision trees in the forest
'''
#Let us build a helper function to make hyperparamter testing a little simple
def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params).fit(X_train, train_target)
    train_score = model.score(X_train, train_target)
    val_score = model.score(X_val, val_target)
    return train_score, val_score
'''
test_params(max_depth=5)
test_params(max_depth=26)
test_params(max_leaf_nodes=2**7)
test_params(max_leaf_nodes=2**5, max_depth=15, n_estimators=600)
#verify the accuracy with the base accuracy
'''













