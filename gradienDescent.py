def gradientDescent(x, y, alpha, num_iters):
    """
    GRADIENTDESCENT Performs gradient descent to learn theta

    :param X: input variables
    :param y: output variables
    :param theta: hypothesis
    :param alpha: learning rate
    :param num_iters: number of cg iterations
    :return:
    """
    import numpy as np
    from computeCost import computeCost

    theta = np.zeros(shape=(2, 1))  # initialize fitting parameters

    m = len(y) # number of training examples
    J_history = np.zeros([num_iters, 1])
    print (theta.shape)

    for iter in range(num_iters):
        txmy = theta.T * x.T -y
        d1 = - alpha/m * np.sum(txmy)
        d2 = - alpha/m * np.sum(np.multiply(txmy, x[:,1].T))
        #print ("changes", d1, d2)
        theta[0,0] = theta[0,0] + d1
        # print(txmy.shape, x[:,1].T.shape)
        # print(np.multiply(txmy, x[:,1].T))
        #print(iter, np.sum(np.multiply(txmy, x[:,1].T)))
        theta[1,0] = theta[1,0] + d2

        # Save the cost
        J_history[iter] = computeCost(x, y, theta);
        # print("iter, cost: ", iter, J_history[iter], theta)

    return theta, J_history
