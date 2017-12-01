import numpy as np
import data_preprocessing as dp


# THe activation function sigmoid
def sigmoid(z):
    """
    Calculating and outputting n the sigmoid value from the given value z.
    
    Args:
        z: The variable z of the sigmoid funtion
        
    Returns:
        The calculation result 
    """
    
    return 1/(1+np.exp(-z))


# The initialisation function
def initialise(dim):
    """
    Initialising weights and bias with 0.
    
    Args:
        dim: The dimension of vector W
    
    Returns:
        W: A vector of 0, the shape of W is (dim, 1)
        b: A scalar of 0
    """
    
    W = np.zeros(shape=(dim, 1))
    b = 0
    
    return W, b


# The flattening function 
def flatten(m):
    """
    Given a matrix whose shape is (x, a, b, c), outputting a new flattened matrix by converting the shape from 
    (x, a, b, c) to (a * b *c, x).
    
    Args:
        m: A matrix whose shape is (x, a, b, c)
        
    Returns:
        A new matrix whose shape is (a * b * c, x)
    """
    
    return m.reshape(m.shape[0], -1).T


# The propagation function (forward propagation and backward propagation)
def propagate(W, b, X, Y):
    """
    Calculating the cost value, gradients dw and db which are used for updating the parameters weights and bias.
    
    Args:
        W: The vector of weights  
        b:The parameter scalar of bias
        X: The images used for training whose shape is (x, a, b, c)
        Y: The ground truth labels of the corresponding images used for training whose shape is (x, 1)
    
    Returns:
        cost: the cost value based on the current parameters
        gradients: a dictionary composed of gradients dw and db based on the current parameters
    """
    
    m = X.shape[1]
    
    # Forward propagation
    A = sigmoid(np.dot(W.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
                
    # Backward propagation
    db = (1 / m) * np.sum(A - Y)
    dW = (1 / m) * np.dot(X, (A - Y).T)
    
    gradients = {
        "dW": dW,
        "db": db
    }
    
    return gradients, cost


# The optimisation function
def optimise(W, b, X, Y, lr, iterations):
    """
    Given training sampeles X and Y, updating the parameters weights and bias.
    
    Args:
        W: The vector of weights whose shape is (sample_num, 1).
        b: The scalar of bias, an int number.
        X: Images used for training whose shape is (sample_num, width, height, colour_chanels).
        Y: Labels used for training whose shape is (sample_num, 1).
        lr: Learning rate, an int number.
        iterations: The number of training iterations, an int number.
    
    Returns:
        cost: A array of cost values whose length is sample_num.
        params: A dictionary of updated weights and bias.
        gradients: a dictionary of final gradients dw and db.
    """

    # the array of costs
    costs = []
    updated_W = W
    updated_b = b
    
    for i in range(iterations):
        gradients, cost = propagate(W, b, X, Y)
        dW = gradients["dW"]
        db = gradients["db"]

        # Updating W and b by subtracting dW and db from them
        updated_W -= lr * dW
        updated_b -= lr * db
        
        
        # Saving and printing the cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print("The cost after iteration %i: %f" % (i, cost))
        
    params = {
        "W": updated_W,
        "b": updated_b
    }
    
    gradients = {
        "dW": dW,
        "db": db
    }
    
    return params, gradients, costs


# The prediction function
def predict(params, X):
    """
    Given test set X, prediect the corresponding label Y.
    
    Args:
        params: The dictionary comprising the updated parameters weights and bias.
        X: The images for test whose shape is (sample_num, width, height, colour_chanels).
    
    Returns:
        prediction_Y: The array of predicted labels.
    """
    W = params["W"]
    b = params["b"]
    m = X.shape[1]
    prediction_Y = np.zeros(shape=(1,m))
    
    # Calculating the prediction value A by using updated parameters W and b based on training samples X
    A = sigmoid(np.dot(W.T, X) + b) 
    print("A shape:")
    print(A.shape)
    
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            prediction_Y[0, i] = 1
        else:
            prediction_Y[0, i] = 0
    
    return prediction_Y


# The model function
def model(X_train, X_test, Y_train, Y_test, lr, iterations):
    """
    Integrating all the functions together, calculating prediction_Y on training sets and tests and the 
    accuracy rate.
    
    Args:
        X_train: The images for training whose shape is (sample_num, width, height, colour_chanels).
        X_test: The images for test whose shape is (sample_num, width, height, colour_chanels).
        Y_train: The ground truth label for training whose shape is (sample_num, 1).
        Y_test: The ground truth label for test whose shape is (sample_num, 1).
        lr: Learning rate, an int number. 
        iterations: The number of training iterations, an int number.
        
    Returns:
        costs: An array of costs
        Y_predictions: An dictionary composed of the prediction of training set and the prediction of test test
        params: An dictionary composed of the updated parameters weights and bias
    """
    # m is the number of features
    X_train_flatten = flatten(X_train)
    X_test_flatten = flatten(X_test)
    m = X_train_flatten.shape[0]

    
    # Initialisation of weights and bias
    W, b = initialise(m)
    
    # propagation and optimisation
    params, gradients, costs = optimise(W, b, X_train_flatten, Y_train, lr, iterations)
    
    # prediction
    prediction_Y_train = predict(params, X_train_flatten)
    prediction_Y_test = predict(params, X_test_flatten)
    predictions = {
        "prediction_Y_train": prediction_Y_train,
        "prediction_Y_test": prediction_Y_test
    }
    
    # Calculating accuracy rate
    print("train accuracy: {} %".format(100 - np.mean(np.abs(prediction_Y_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction_Y_test - Y_test)) * 100))
    
    return params, gradients, costs, predictions



# Testing models
X_train, X_test, Y_train, Y_test = dp.get_dataset()
d = model(X_train, X_test, Y_train, Y_test, 0.005, 1000)

