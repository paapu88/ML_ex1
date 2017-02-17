def gradientDescent(x, y, theta, alpha, num_iters):
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


m = len(y) # number of training examples
J_history = np.zeros([num_iters, 1])

for iter in range(len(num_iters)):





    # Save the cost

    J_history[iter] = computeCost(x, y, theta);

end

end