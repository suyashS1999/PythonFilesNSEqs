from numpy import*
from Collocation import*
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv, det, norm
from NumericalTools import plot_stability_region, EulerImplicit
from matplotlib import animation

def IC(x, y): return 40*exp(-100*((x - 0.5)**2 + (y - 0.5)**2));

#%% Inputs
x0 = 0;		x1 = 1;
y0 = 0;		y1 = 1;
N = 30;
X = linspace(x0, x1, N);
Y = linspace(y0, y1, N);
x_mesh, y_mesh = meshgrid(X, Y);
c = 1;																	# Advection Speed
mu = 0.95;																# Viscosity
t_max = 10;																# Maximum time
dt = 0.0001;																# Time step

nodes = array([x_mesh.reshape(1, -1)[0], y_mesh.reshape(1, -1)[0]]).T;
K1 = -c*(RBF_Matrix(nodes, ["diff", nodes[:, 0]]) + RBF_Matrix(nodes, ["diff", nodes[:, 1]]));
K2 = mu*(RBF_Matrix(nodes, ["2diff"]) + RBF_Matrix(nodes, ["2diff"]));
M = RBF_Matrix(nodes, ["basis"]);
A = inv(M).dot(K1 + K2);
S = zeros(len(A));

A, _, _ = Boundary_Conditions(A, S, nodes, [x0, x1, y0, y1], "dir", 0);

stab_fig = plt.figure(figsize = (8, 8));
s = lambda z: 1/(1 - z);
plot_stability_region(s, A, dt, stab_fig);

F = IC(x_mesh, y_mesh).reshape(1, -1)[0];
a0 = inv(M).dot(F);
f = Inter(a0, X, Y, nodes, ["basis"]);

fig = plt.figure();
ax = plt.axes(projection = '3d');
ax.plot_surface(x_mesh, y_mesh, f, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
ax.set_xlabel('x');
ax.set_ylabel('y');
plt.show();


a_i = EulerImplicit(A, dt, a0, t_max, 0, None);

#%% Solution Plots
fig = plt.figure(figsize = (10, 10));
ax = plt.axes(projection = '3d');
ax.plot_surface(x_mesh, y_mesh, f, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
ax.set_xlabel('x');
ax.set_ylabel('y');
k = 1;
def updatefig(i):
	global k;
	f = Inter(a_i[:, k], X, Y, nodes, ["basis"]);
	ax = plt.axes(projection = '3d');
	ax.plot_surface(x_mesh, y_mesh, f, rstride = 1, cstride = 1,
					cmap = 'viridis', edgecolor = 'none');
	ax.set_xlabel('x');
	ax.set_ylabel('y');
	k += 1;
	if k == shape(a_i)[1]:
		k = 0;
	return 0;
anim = animation.FuncAnimation(fig, updatefig, interval = 1, blit = False);
plt.show();




