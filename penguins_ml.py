import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

penguin_df = pd.read_csv('penguins.csv')
penguin_df.dropna(inplace=True) # Removes any rows with missing 
## Separating Features and Target:
output = penguin_df['species']  # Target variable (what we want to predict)

output, uniques = pd.factorize(output)  # Converts species names into numbers

print(f"Output: {output}")
print(f"Uniques: {uniques}")

features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 
                      'flipper_length_mm', 'body_mass_g', 'sex']]  # Input features

features = pd.get_dummies(features)  # Converts categorical variables into numeric
output, uniques = pd.factorize(output) # # Converts species names into numbers

print(f"features: {features}")

print(f"Output converted: {output}")
print(f"Uniques converted: {uniques}")

## Splits data into training (20%) and testing (80%) sets
## stratify=output ensures balanced representation of each species
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8, stratify=output)
rfc = RandomForestClassifier(random_state=15)  # Creates a Random Forest model
rfc = rfc.fit(x_train.values, y_train) # Trains it on the training data
print(f"rfc: {rfc}")
y_pred = rfc.predict(x_test.values) # Makes predictions on the test data
score = accuracy_score(y_pred, y_test) # alculates how accurate the model's predictions are
print('Accuracy of our model is: {}'.format(score))

rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()

fig, ax = plt.subplots()

ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
plt.title("Which features are the most important for species prediction?")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
fig.savefig("feature_importance.png")

# print("-----------------------------------")
# print("RFC properties: ", help(rfc.fit))

