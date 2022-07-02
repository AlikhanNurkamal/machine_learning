import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv( "breast-cancer-wisconsin.data" )
df.drop( columns = "id", inplace = True )
df.replace( '?', np.nan, inplace = True)
df.dropna( inplace = True )

X = np.array( df.iloc[:, :-1].values )
y = np.array( df.iloc[:, -1].values )

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 1 )

model = LogisticRegression( )
model.fit( X_train, y_train )
accuracy = model.score( X_test, y_test )
print( accuracy )

y_pred = model.predict( X_test )
labels = ["Benign", "Malignant"]
cf_matrix = confusion_matrix( y_test, y_pred )
sns.heatmap( cf_matrix, annot = True, xticklabels = labels, yticklabels = labels )
plt.title( "Breast cancer" )
plt.xlabel( "Predicted diagnosis" )
plt.ylabel( "True diagnosis" )
plt.savefig( "cancer_diagnosis_heatmap.png", dpi = 300 )
plt.show( )

# computing the accuracy with k-fold cross validation
accuracies = cross_val_score( estimator = model, X = X, y = y, cv = 10 )
print( "Average accuracy:", np.mean( accuracies) )

