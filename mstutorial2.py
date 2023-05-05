import sklearn
from sklearn import datasets
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Imports for Building the model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
#Evaluation
from sklearn.metrics import classification_report

#LOADING DATA
wine_data = load_wine()

# Convert data to pandas datafram
wine_df = pd.DataFrame(wine_data.data, columns = wine_data.feature_names)

# Add the target label
wine_df["target"] = wine_data.target

# Take a preview
print("First 5 rows of data:\n")
print(wine_df.head())

#data exploration
print("\nData summary: \n")
wine_df.info()
#print(wine_df.describe())
#print(wine_df.tail())

#PREPROCESSING
#Split data into features and label
X = wine_df[wine_data.feature_names].copy()
y = wine_df["target"].copy()

# Instantiante scaler and fit on features
scaler = StandardScaler()
scaler.fit(X.values) #when just scaler.fit(X) throws warning: "X does not have valid feature names"

# Transform features
X_scaled = scaler.transform(X.values)

# View first instance
print("\nScaled Data Values: ")
print(X_scaled[0])

# Split Data intro train and test sets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, train_size=.7, random_state=25)

# Check the splits are correct
print("\nData Splits:")
print(f"Train size: {round(len(X_train_scaled) / len(X) * 100)}% \n\
Test size: {round(len(X_test_scaled) / len(X) * 100)}%\n")

#BUILDING THE MODEL
#Instantiating the models
logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()

#Training the models
logistic_regression.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
tree.fit(X_train_scaled, y_train)

#Making Predictions with each model
log_reg_preds = logistic_regression.predict(X_test_scaled)
svm_preds = svm.predict(X_test_scaled)
tree_preds = tree.predict(X_test_scaled)

#MODEL EVALUATION
# Store model predictions in a dictionary
# this makes it easier to iterate through each model
# and print the results.
model_preds = {
	"Logistic Regression": log_reg_preds,
	"Support Vector Machine": svm_preds,
	"Decision Tree": tree_preds
}

for model, preds in model_preds.items():
	print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")