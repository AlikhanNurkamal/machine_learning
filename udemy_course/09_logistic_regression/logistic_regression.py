import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv( "social_network_ads.csv" )
X = np.array( df.iloc[:, :-1].values )
y = np.array( df.iloc[:, -1].values )

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 0 )

# print( "BEFORE SCALING\n", X_test )
scaler = StandardScaler( )
X_train = scaler.fit_transform( X_train )
X_test = scaler.transform( X_test )
# print( "AFTER SCALING\n", X_test )

model = LogisticRegression( n_jobs = -1 )
model.fit( X_train, y_train )
accuracy = model.score( X_test, y_test )

print( accuracy )

y_pred = model.predict( X_test )
# test_vs_pred = pd.DataFrame( { "True value" : y_test,
#                                "Pred value" : y_pred } )

accuracy1 = accuracy_score( y_test, y_pred )
print( accuracy1 )

# print( test_vs_pred )

print( model.predict( scaler.transform( [[30, 87000]] ) ) )

cf_matrix = confusion_matrix( y_test, y_pred )
labels = ["Not buy", "Buy"]
sns.heatmap( cf_matrix, annot = True, xticklabels = labels, yticklabels = labels )
plt.xlabel( "Predicted" )
plt.ylabel( "True" )
plt.savefig( "heatmap.png", dpi = 300 )
plt.show( )
