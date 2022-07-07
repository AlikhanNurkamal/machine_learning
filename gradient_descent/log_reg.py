import numpy as np
import matplotlib.pyplot as plt


def get_function( x, theta, theta0 ):
    f_x = 1 / ( 1 + np.exp( -( np.dot( theta, x ) + theta0 ) ) )
    return f_x


def calculate_cost( x, y, theta, theta0 ):
    m = x.shape[0]
    loss = 0
    for i in range( m ):
        f_x = get_function( x[i], theta, theta0 )
        loss += -y[i] * np.log( f_x ) - ( 1 - y[i] ) * np.log( 1 - f_x )

    cost = loss / m
    return cost


def calculate_derivative( x, y, theta, theta0 ):
    m, n = x.shape
    der_theta = np.zeros( n )
    der_theta0 = 0
    for i in range( m ):
        f_x = get_function( x[i], theta, theta0 )
        err = f_x - y[i]
        for j in range( n ):
            der_theta[j] += err * x[i][j]
        der_theta0 += err

    for j in range( n ):
        der_theta[j] = der_theta[j] / m
    der_theta0 = der_theta0 / m

    return der_theta, der_theta0


def gradient_descent( x, y, theta, theta0 ):
    m, n = x.shape
    learning_rate = 0.01
    iterations = 100000
    for i in range( iterations ):
        der_theta, der_theta0 = calculate_derivative( x, y, theta, theta0 )
        theta0 = theta0 - learning_rate * der_theta0
        for j in range( n ):
            theta[j] = theta[j] - learning_rate * der_theta[j]

        if i % 1000 == 0:
            print(f"Iteration {i}: cost", calculate_cost(x, y, theta, theta0))

    return theta0, theta


def predict( x, theta, theta0 ):
    m = x.shape[0]
    f_x = np.zeros( m, dtype = "int" )
    for i in range( m ):
        val = get_function( x[i], theta, theta0 )
        if val >= 0.5:
            f_x[i] = 1
        else:
            f_x[i] = 0

    return f_x


def predict_proba( x, theta, theta0 ):
    m = x.shape[0]
    f_x = np.zeros( m )
    for i in range( m ):
        f_x[i] = get_function( x[i], theta, theta0 )

    return f_x


# First, I will be making a simple logistic regression model with one feature.
# The feature will be the age of a patient and the target is whether they bought an insurance.
X = np.array( [[18], [25], [66], [49], [30], [53], [22], [87], [70],
               [27], [45], [69], [20], [88], [91], [51], [19], [32]] )
y = np.array( [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0] )

theta_weights = np.ones( X.shape[1] )
theta_0 = 1

plt.scatter( X, y, c = y )

print( "Initial cost:", calculate_cost( X, y, theta_weights, theta_0 ) )
theta_0, theta_weights = gradient_descent( X, y, theta_weights, theta_0 )

print( "\nParameters:" )
print( "theta:", theta_weights )
print( "theta0:", theta_0 )

print( "\nTrue:", y )
print( "Pred:", predict( X, theta_weights, theta_0 ) )

plt.plot( np.arange( 91 ), predict_proba( np.arange( 91 ), theta_weights, theta_0 ), c = 'b' )
plt.show( )

# The same functions, defined above, will work for logistic regression with multiple features.
