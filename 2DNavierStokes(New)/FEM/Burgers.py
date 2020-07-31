from numpy import*
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from NumericalTools import*
from TriMeshFEM import TriMesh, MeshElementFEM
from mytools import pbf
import time
from FEM2D import BurgersEquationFEM_Matrix, NonLinearStiff_Matrix

#%% Functions
def InitialCondition(x, y): return 40*exp(-100*((x - 0.8)**2 + (y - 0.8)**2));

#%% Input data
x1 = 0.;		x2 = 2.;												# Domain dimentions
y1 = 0.;		y2 = 2.;
mu = 0.8;																# Viscosity
t_max = 0.05;															# Maximum time
dt = 0.001;																# Time step
DOP = 4;																# Degree of precision for integration
w_int_stdtri, x_int_stdtri = Quadrature_weights(DOP, 0, 1, "lin");		# Quadrature weights and nodes for intgration
w_fact = 0.5;															# Area ratio between square and triangle
w_int_stdtri *= w_fact;
x_int_mesh, y_int_mesh = meshgrid(x_int_stdtri, x_int_stdtri);
quadrature_parm = (w_int_stdtri, x_int_stdtri, w_int_stdtri, x_int_stdtri);

mesh = TriMesh(x1, x2, y1, y2);
mesh.loadMesh(6, 0, 0, 0);
mesh.plotMesh();
BC = "Periodic";



#%% Generate Matrix Burgers Equation
a, b, mass_M_inv, diffusion_M = BurgersEquationFEM_Matrix(mesh, mu, x_int_mesh, y_int_mesh, w_int_stdtri, InitialCondition, BC);
stab_fig = plt.figure(figsize = (8, 8));
X, Y = EulerExplicit2D_nonLin(NonLinearStiff_Matrix, (mesh, mass_M_inv, x_int_mesh, y_int_mesh, w_int_stdtri, BC), diffusion_M, dt, a, b, t_max, stab_fig);
plt.show();

#%% Solution Plots
fig = plt.figure(figsize = (18, 8));
mesh.plotSoln(sqrt(a**2 + b**2), fig, "Solution");
k = 1;
def updatefig(i):
	global k;
	mesh.plotSoln(sqrt(X[:, k]**2 + Y[:, k]**2), fig, "Solution");
	k += 1;
	if k == shape(X)[1]:
		k = 0;
	return 0;
anim = animation.FuncAnimation(fig, updatefig, interval = 1, blit = False);
plt.show();
