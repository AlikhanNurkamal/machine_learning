import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv( "breast-cancer-wisconsin.data" )
df.replace( '?', -99999, inplace = True )
df.drop( ["id"], axis = 1, inplace = True )

X = np.array( df.iloc[:, :-1].values )
y = np.array( df.iloc[:, -1].values )

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2 )

model = SVC( gamma = "auto" )
model.fit( X_train, y_train )

accuracy = model.score( X_test, y_test )
print( accuracy )

example_measures = np.array( [[4,2,1,1,1,2,3,2,1],
                              [4,2,1,2,2,2,3,2,1]] )
# example_measures.reshape( len( example_measures ), -1 )

prediction = model.predict( example_measures )
print( prediction )
