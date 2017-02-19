from warmUpExercise import warmUpExercise
import matplotlib.pyplot as plt
import numpy as np
from computeCost import computeCost
from gradienDescent import gradientDescent
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros(shape=(len(theta0_vals), len(theta1_vals)))
# Fill out J_vals
t = np.zeros(shape=(2, 1))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
          t[0,0] = theta0_vals[i]
          t[1,0] = theta1_vals[j]
          J_vals[i,j] = computeCost(data, y, t)
          print(i,j,J_vals[i,j])

# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T;
# Surface plot

# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

#surf(theta0_vals, theta1_vals, J_vals)
#xlabel('\theta_0'); ylabel('\theta_1');
plt.show()

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
#contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
#xlabel('\theta_0'); ylabel('\theta_1');
#hold on;
#plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);

CS = plt.contour(theta0_vals, theta1_vals, J_vals)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.show()
