import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv( "social_network_ads.csv" )

X = np.array( df.drop( "Purchased", axis = 1 ) )
y = np.array( df["Purchased"] )

# print( X )
# print( y )

scaler = StandardScaler( )
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 0 )
X_train = scaler.fit_transform( X_train )
X_test = scaler.transform( X_test )

# print( X_train )
# print( y_train )

# In order to identify the best value for k:
# accuracies = []
# for k in range( 1, 21 ):
#     model = KNeighborsClassifier( n_neighbors = k, p = 2, n_jobs = -1 )
#     model.fit( X_train, y_train )
#     accuracies.append( model.score( X_test, y_test ) )
#
# plt.plot( range( 1, 21), accuracies, c = 'b' )
# plt.show( )

model = KNeighborsClassifier( n_neighbors = 5, p = 2, n_jobs = -1 )
model.fit( X_train, y_train )
print( model.score( X_test, y_test ) )

y_pred = model.predict( X_test )
cf_matrix = confusion_matrix( y_test, y_pred )

labels = ["Not buy", "Buy"]
sns.heatmap( cf_matrix, annot = True, xticklabels = labels, yticklabels = labels )
plt.xlabel( "Predicted" )
plt.ylabel( "True" )
plt.savefig( "heatmap.png", dpi = 300 )
plt.show( )
