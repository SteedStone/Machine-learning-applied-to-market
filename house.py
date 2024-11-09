import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error



data = pd.read_csv('melb_data.csv')

# print(data)
#Combien de colonne 
house0 = data.columns
# print(data.describe())

# Prediction target denoted by y
y = data.Price 
# print(y)

# Features are denoted by X (En gros ce qui nous intéresse) 

Data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = data[Data_features]


# Creation du modele de nouveau avec un arbre de décision 
model = DecisionTreeRegressor()

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)

# Fit the model
model.fit(X_train, y_train) 

# Tests 
Predictions_based_on_model = model.predict(X_test)

# ATTENTION : Ici on a pas de classification donc on ne peut pas utiliser accuracy_score
# On utilise mean_absolute_error
score = mean_absolute_error(y_test,Predictions_based_on_model) 
print(score)

# Fonction pour répéter le processus mais avec une profondeur d'arbre différente
def number_of_leaf(max_leaf, X_train1, X_test1, y_train1, y_test1):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf,random_state=0)
    model.fit(X_train1, y_train1)
    predictions = model.predict(X_test1) 
    return mean_absolute_error(y_test1,predictions) 

# Itérations pour trouver le bon rapport entre underfitting et overfitting
for max_leaf in [5 ,50 ,500, 5000] : 
    my_mae = number_of_leaf(max_leaf, X_train, X_test, y_train, y_test)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf, my_mae))

# On peut tester d'autres algorithmes de machine learning pour voir si on peut améliorer le score 
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train,y_train)
melb_preds = forest_model.predict(X_test)
print(mean_absolute_error(y_test,melb_preds))




