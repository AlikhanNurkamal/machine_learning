import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

df = pd.read_csv( "social_network_ads.csv" )

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# print( X )
# print( y )

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2 )

scaler = StandardScaler( )
X_train = scaler.fit_transform( X_train )
X_test = scaler.transform( X_test )

model = SVC( )
model.fit( X_train, y_train )
accuracy = model.score( X_test, y_test )

print( accuracy )

y_pred = model.predict( X_test )
cf_matrix = confusion_matrix( y_test, y_pred )
print( cf_matrix )

