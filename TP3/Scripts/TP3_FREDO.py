from numpy import linspace, array, zeros, size, cos
from numpy.linalg import norm, svd
from numpy.random import normal, randint
from matplotlib.pyplot import plot, grid, show, xlabel, ylabel, title, legend
from scipy.optimize import least_squares

x = linspace(0, 1, 100)

e = normal(0.0, 1.0, 100)

y = 10 + 2*x + e

plot(x, y, "o", color="blue")

def inliers(x, y, model, threshold):
    x_inl = list()
    y_inl = list()
    for i in range(len(x)):
        if abs(model[0] + model[1]*x[i] - y[i]) <= threshold:
            x_inl.append(x[i])
            y_inl.append(y[i])
    return x_inl, y_inl

def compareModel(x, y, model1, model2, threshold):
    """
0: model1 is better than model2
1: model2 is better than model1
    """
    n = size(x)
    beta11 = model1[0]
    beta21 = model1[1]
    beta12 = model2[0]
    beta22 = model2[1]

    inliers1 = 0
    inliers2 = 0

    for i in range(n):
        k1 = abs(beta11+beta21*x[i] - y[i])
        k2 = abs(beta12+beta22*x[i] - y[i])

        if k1 < threshold:
            inliers1 += 1
        if k2 < threshold:
            inliers2 += 1

    return inliers1 < inliers2

def findModel(x1, y1, x2, y2):
    """
Finds linear model for 2 samples (x1,y1) and (x2,y2)
    """
    beta2 = (y2-y1) / (x2-x1)
    beta1 = y1-beta2*x1
    return beta1, beta2

s1 = randint(0, len(x)-1)
s2 = randint(0, len(x)-1)
while s2 == s1:
    s2 = randint(0, len(x)-1)
beta1, beta2 = findModel(x[s1], y[s1], x[s2], y[s2])
model = array([beta1, beta2], dtype="float64")

for i in range(1, 100):
    s1 = randint(0, len(x)-1)
    s2 = randint(0, len(x)-1)
    while s2 == s1:
        s2 = randint(0, len(x)-1)

    beta1, beta2 = findModel(x[s1], y[s1], x[s2], y[s2])
    model2 = array([beta1, beta2], dtype="float64")
    x_in2, y_in2 = inliers(x, y, model2, threshold=1)
    better = compareModel(x, y, model, model2, threshold=1)

    if better:
        n = len(x_in2)
        m_x = sum(x_in2) / n
        m_y = sum(y_in2) / n
        Cxy = 1/n * sum([x_in2[i]*y_in2[i] - m_y*x_in2[i] - y_in2[i]*m_x + m_x*m_y for i in range(0, n)])
        sig2_x = 1/n * sum([x_in2[i]**2 - 2*m_x*x_in2[i] + m_x**2 for i in range(0, n)])
        beta2 = Cxy / sig2_x
        beta1 = m_y - beta2 * m_x
        model = array([beta1, beta2], dtype="float64")

beta1 = model[0]
beta2 = model[1]

xf = linspace(0, 1, 100)
yf = [beta1 + beta2*k for k in xf]

plot(xf, yf, "x", color="red")
plot(xf, [10 + 2*k for k in xf], color="black")
grid()
legend()
show()

parameters = [0,0] #[p,e]
def outputs(parameters, theta, f_theta):
    return parameters[0] / (1-parameters[1]*cos(theta))-f_theta

theta = array([43, 45, 52, 93, 108, 126])
f_theta = array([4.7126, 4.5542, 4.0419, 2.2187, 1.8910, 1.7599])

solution = least_squares(outputs, parameters, args=(theta, f_theta))
print("blabla : ",solution.x)

A = array([[1, 2], [3, 4], [4, 6]], dtype="float64")
u, s, vh = svd(A)
sigma = array([[s[0], 0], [0, s[1]], [0, 0]])
print(u@sigma@vh)
