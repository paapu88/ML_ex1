from warmUpExercise import warmUpExercise
import matplotlib.pyplot as plt
import numpy as np
from computeCost import computeCost
from gradienDescent import gradientDescent


# part 1 warm up
print(warmUpExercise())


# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
x, y = [], []
with open('ex1data1.txt','r') as f:
    for l in f:
        row = l.split(',')
        x.append(float(row[0]))
        y.append(float(row[1]))

plt.plot(x, y, 'ro')
plt.ylabel('Profit in $10,000s');
plt.xlabel('Population of City in 10,000s');
plt.show()

myones = np.ones([len(x)])

data = np.asmatrix([myones, x]).transpose()
print(data.shape)
print (data)

theta = np.zeros(shape=(2,1))  # initialize fitting parameters
# theta = np.matrix([[0], [0]])

# Some gradient descent settings
iterations = 1500
alpha = 0.0001

x=np.asmatrix(x)
y=np.asmatrix(y)

print(x.shape, y.shape, theta.shape, data.shape)

# compute and display initial cost
print("cost: ",computeCost(data, y, theta) )

# run gradient descent
theta = gradientDescent(data, y, theta, alpha, iterations);
