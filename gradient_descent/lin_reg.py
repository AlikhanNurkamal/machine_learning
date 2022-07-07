import numpy as np
import matplotlib.pyplot as plt


def calculate_cost( x, y, theta, theta0 ):
    m, n = x.shape
    cost = 0
    for i in range( m ):
        f_x = np.dot( theta, x[i] ) + theta0    # calculating the hypothesis function
        err = f_x - y[i]
        cost += err ** 2

    cost = cost / ( 2 * m )
    return cost


def calculate_derivative( x, y, theta, theta0 ):
    m, n = x.shape
    der_theta = np.zeros( n )
    der_theta0 = 0
    for i in range( m ):
        f_x = np.dot( theta, x[i] ) + theta0
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
    learning_rate = 0.001
    iterations = 100001
    for i in range( iterations ):
        der_theta, der_theta0 = calculate_derivative( x, y, theta, theta0 )
        theta0 = theta0 - learning_rate * der_theta0
        for j in range( n ):
            theta[j] = theta[j] - learning_rate * der_theta[j]

        if i % 1000 == 0:
            print( f"Iteration {i}: cost", calculate_cost( x, y, theta, theta0 ) )

    return theta0, theta


def predict( x, theta, theta0 ):
    m = x.shape[0]
    f_x = np.zeros( m )
    for i in range( m ):
        f_x[i] = np.dot( theta, x[i] ) + theta0

    return f_x


# # First, I will be making a simple linear regression model with one feature.
# # The feature will be the number of rooms in the house, and the target is the price.
# X = np.array( [[1], [2], [1], [4], [3], [2], [3], [3], [1], [5], [4], [2], [4], [5]] )
# y = np.array( [5000, 11000, 6000, 23000, 19000, 14000, 18000,
#                17000, 5500, 28000, 24000, 12000, 23000, 29000] )
# theta_weights = np.zeros( X.shape[1] )
# theta_0 = 0
#
# plt.scatter( X, y, marker = 'x', c = 'r')
#
# print( calculate_cost( X, y, theta_weights, theta_0 ) )
# theta_0, theta_weights = gradient_descent( X, y, theta_weights, theta_0 )
#
# print( "\nParameters:" )
# print( "theta:", theta_weights )
# print( "theta0:", theta_0 )
#
# plt.plot( X, get_f_x( X, theta_weights, theta_0 ), c = 'b' )
# plt.show( )


# Now, I will try the same functions with multiple linear regression.
# X = np.array( [[2104, 5, 1, 45],
#                [1416, 3, 2, 40],
#                [1534, 3, 2, 30],
#                [852, 2, 1, 36],
#                [1829, 4, 4, 15],
#                [594, 1, 5, 33],
#                [1113, 2, 3, 27],
#                [668, 1, 3, 44]] )
X = np.array( [[5, 1, 45],
               [3, 2, 40],
               [3, 2, 30],
               [2, 1, 36],
               [4, 4, 15],
               [1, 5, 33],
               [2, 3, 27],
               [1, 3, 44]] )
y = np.array( [460,
               232,
               315,
               178,
               428,
               155,
               219,
               160] )

theta_weights = np.ones( X.shape[1] )
theta_0 = 1
print( "Initial cost:", calculate_cost( X, y, theta_weights, theta_0 ) )
theta_0, theta_weights = gradient_descent( X, y, theta_weights, theta_0 )

print( "\nParameters:" )
print( "theta:", theta_weights )
print( "theta0:", theta_0 )

examples = np.array( [[4, 2, 10],
                      [2, 5, 15]] )
print( "\nWhat I want to predict:" )
print( examples )
print( "Predictions:" )
print( predict( examples, theta_weights, theta_0 ) )