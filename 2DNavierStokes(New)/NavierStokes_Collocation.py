from numpy import*
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mytools import pbf, pbw
from NumericalTools import*
from Collocation import*

x0 = -1; x1 = 1;								# Domain dimentions
y0 = -1; y1 = 1;
boundaries = [x0, x1, y0, y1];
n = 20;											# Domain division
x = linspace(x0, x1, n);
y = linspace(y0, y1, n);
x_mesh, y_mesh = meshgrid(x, y);
mu = 0.8;										# Viscosity
rho = 2;										# Density
nodes = array([x_mesh.reshape(1, -1)[0], y_mesh.reshape(1, -1)[0]]).T;
a = zeros((n**2, 1));
b = zeros((n**2, 1));

def Jacobnian(A, B, C, S, mu, a, b):
	N = len(A);
	J = zeros((2*N, 2*N));
	J[0:N, 0:N] = 2*A*a.reshape(1, -1)[0] + B*b.reshape(1, -1)[0] - mu*C;
	J[0:N, N:2*N] = B*a.reshape(1, -1)[0];
	J[N:2*N, 0:N] = A*b.reshape(1, -1)[0];
	J[N:2*N, N:2*N] = 2*B*b.reshape(1, -1)[0] + A*a.reshape(1, -1)[0] - mu*C;
	return J;

def func(A, B, C, S, mu, a, b):
	F = concatenate((A.dot(a**2) + B.dot(a*b) - mu*C.dot(a), B.dot(b**2) + A.dot(a*b) - mu*C.dot(b))) + concatenate((S, S));
	return F;

def Newton(x0, f, df, fargs, iter_max):
	x = x0;
	iter = 1;
	while iter <= iter_max:
		x -= inv(df(*fargs, x[0: int(len(x)/2)], x[int(len(x)/2): len(x)])).dot(f(*fargs, x[0: int(len(x)/2)], x[int(len(x)/2): len(x)]));
		iter += 1;
	return x;

k1 = RBF_Matrix(nodes, ["basis"]);
k2_x = RBF_Matrix(nodes, ["diff", x_mesh.reshape(n**2, 1)]);
k2_y = RBF_Matrix(nodes, ["diff", y_mesh.reshape(n**2, 1)]);
k3 = RBF_Matrix(nodes, ["2diff"]);

A = k1.dot(k2_x.T);
B = k1.dot(k2_y.T);
C = -mu*k3;
S = zeros_like(a);
C, S, bc_idx = Boundary_Conditions(C, S, nodes, [x0, x1, y0, y1], "dir", 10);
A[bc_idx, :] = 0; B[bc_idx, :] = 0;
X = Newton(concatenate((a, b)), func, Jacobnian, (A, B, C, S, mu), 50);
a = X[0: int(len(X)/2)].reshape(1, -1)[0]; b = X[int(len(X)/2): len(X)].reshape(1, -1)[0];
print("Done");
soln_u = Inter(a, x, y, nodes, ["basis"]);
soln_v = Inter(b, x, y, nodes, ["basis"]);

fig = plt.figure();
ax = plt.axes(projection = '3d');
ax.plot_surface(x_mesh, y_mesh, soln_u, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('u');

fig = plt.figure();
ax = plt.axes(projection = '3d');
ax.plot_surface(x_mesh, y_mesh, soln_v, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('v');
plt.show();