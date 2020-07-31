from numpy import*
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from NumericalTools import*
from TriMeshFEM import TriMesh, MeshElementFEM
from mytools import pbf
import time
from FEM2D import WaveEquation
from scipy.sparse import csr_matrix as sparse

#%% Functions
def InitialCondition(x, y): return 40*exp(-100*((x - 2)**2 + (y - 2)**2));

#%% Input data
x1 = 0.;		x2 = 4.;												# Domain dimentions
y1 = 0.;		y2 = 4.;
c = 5;																	# Wave Speed
t_max = 10;																# Maximum time
dt = 0.01;																# Time step
DOP = 4;																# Degree of precision for integration
w_int_stdtri, x_int_stdtri = Quadrature_weights(DOP, 0, 1, "lin");		# Quadrature weights and nodes for intgration
w_fact = 0.5;															# Area ratio between square and triangle
w_int_stdtri *= w_fact;
x_int_mesh, y_int_mesh = meshgrid(x_int_stdtri, x_int_stdtri);
quadrature_parm = (w_int_stdtri, x_int_stdtri, w_int_stdtri, x_int_stdtri);

mesh = TriMesh(x1, x2, y1, y2);
mesh.loadMesh(10, 0, 0, 0);
mesh.plotMesh();

#%%
v0, w0, _, _, A_, _ = WaveEquation(mesh, c**2, x_int_mesh, y_int_mesh, w_int_stdtri, InitialCondition, InitialCondition, "Periodic", 0);
zero = zeros_like(A_);
one = identity(shape(A_)[0]);
A = vstack((concatenate((zero, A_), axis = 1), concatenate((one, zero), axis = 1)));
a = concatenate((w0, v0));

s = lambda z: 1/(1 - z);
stab_fig = plt.figure(figsize = (8, 8));
plot_stability_region(s, A_, dt, stab_fig);
X = EulerImplicit(A, dt, a, t_max, 0, 0);
plt.show();

#%% Solution Plots
fig = plt.figure(figsize = (18, 8));
mesh.plotSoln(v0, fig, "Solution");
k = 1;
def updatefig(i):
	global k;
	mesh.plotSoln(X[int(shape(X)[0]/2):, k], fig, "Solution");
	k += 1;
	if k == shape(X)[1]:
		k = 0;
	return 0;
anim = animation.FuncAnimation(fig, updatefig, interval = 1, blit = False);
plt.show();


