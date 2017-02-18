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


# theta = np.matrix([[0], [0]])

# Some gradient descent settings
iterations = 1500
alpha = 0.0001

x=np.asmatrix(x)
y=np.asmatrix(y)

print(x.shape, y.shape, data.shape)

# compute and display initial cost
theta1 = np.zeros(shape=(2, 1))  # initialize fitting parameters
print("cost: ",computeCost(data, y, theta1) )

# run gradient descent
theta, j_history = gradientDescent(data, y,  alpha, iterations);


print('Theta found by gradient descent: ');
print(theta[0,0], theta[1,0]);

# Plot data
plt.plot(x, y, 'ro', label='Training Data')
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City inn 10,000s')
# plot linear fit found by cg
plt.plot(x, theta.T*data.T, 'g+',label='Linear regression' )
# plt.legend(loc='upper left')
plt.show()


# Predict values for population sizes of 35,000 and 70,000
predict1 = np.asmatrix([[1, 3.5]]) *theta
print('For population = 35,000, we predict a profit of ')
print(predict1*10000)
predict2 = np.asmatrix([[1, 7]]) * theta
print('For population = 70,000, we predict a profit of ')
print(predict2*10000)
