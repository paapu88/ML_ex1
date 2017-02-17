def computeCost(x, y, theta):
    """

    :param x: input data
    :param y: output data
    :param theta: hypohesis function theta[0]+theta[1]*x
    :return: squared and normalised difference between hypothesis and y
    """
    tx = theta.T * x.T
    # print(tx.shape, y.shape)
    cost = 1/2*len(y) * (tx - y) * (tx - y).T
    # print(cost.shape)
    return cost