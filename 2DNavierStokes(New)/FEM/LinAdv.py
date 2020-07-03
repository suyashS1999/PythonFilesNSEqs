from numpy import*
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from NumericalTools import*
from TriMeshFEM import TriMesh, MeshElementFEM
from mytools import pbf
import time
from FEM2D import LinearAdvectionFEM_Matrix
from scipy.sparse import csr_matrix as sparse

#%% Functions
def InitialCondition(x, y): return 40*exp(-100*((x - 0.8)**2 + (y - 0.8)**2));

def Verif_f(x, y): return sin(0.5*pi*x)*cos(0.5*pi*(y - 1));

def Verif_Source(x, y): return -sin(0.5*pi*x)*cos(0.5*pi*(y - 1)) + c*(0.5*pi*cos(0.5*pi*x)*cos(0.5*pi*(y - 1)) -
								0.5*pi*sin(0.5*pi*x)*sin(0.5*pi*(y - 1))) + mu*((0.5*pi)**2*sin(0.5*pi*x)*cos(0.5*pi*(y - 1)) + 
								(0.5*pi)**2*sin(0.5*pi*x)*cos(0.5*pi*(y - 1)));

#%% Input data
x1 = 0.;		x2 = 4.;												# Domain dimentions
y1 = 0.;		y2 = 4.;
c = 5;																	# Advection Speed
mu = 0.5;																# Viscosity
t_max = 10;																# Maximum time
dt = 0.005;																# Time step
DOP = 4;																# Degree of precision for integration
w_int_stdtri, x_int_stdtri = Quadrature_weights(DOP, 0, 1, "lin");		# Quadrature weights and nodes for intgration
w_fact = 0.5;															# Area ratio between square and triangle
w_int_stdtri *= w_fact;
x_int_mesh, y_int_mesh = meshgrid(x_int_stdtri, x_int_stdtri);
quadrature_parm = (w_int_stdtri, x_int_stdtri, w_int_stdtri, x_int_stdtri);

mesh = TriMesh(x1, x2, y1, y2);
mesh.loadMesh(8, 0, 0, 0);
mesh.plotMesh();

a, mass_M, stiff_M, A, S  = LinearAdvectionFEM_Matrix(mesh, c, mu, x_int_mesh, y_int_mesh, w_int_stdtri, InitialCondition, "Periodic", Verif_Source);

#mesh.plotSoln(a, None, title = "Initial Condition");
#plt.figure();
#plt.imshow(mass_M);
#plt.colorbar();

#plt.figure();
#plt.imshow(stiff_M);
#plt.colorbar();
S_t = lambda t: S*exp(-t);
Manifactured_Soln = lambda t: Verif_f(mesh.vertices[:, 0], mesh.vertices[:, 1])*exp(-t);
s = lambda z: 1/(1 - z);
stab_fig = plt.figure(figsize = (8, 8));
plot_stability_region(s, sparse.todense(A), dt, stab_fig);
X = EulerImplicit(A, dt, a, t_max, 0, 0);
X_verif = EulerImplicit(A, dt, Manifactured_Soln(0), t_max, S_t, Manifactured_Soln);
plt.show();

#%% Solution Plots
fig = plt.figure(figsize = (18, 8));
mesh.plotSoln(a, fig, "Solution");
k = 1;
def updatefig(i):
	global k;
	mesh.plotSoln(X[:, k], fig, "Solution");
	k += 1;
	if k == shape(X)[1]:
		k = 0;
	return 0;
anim = animation.FuncAnimation(fig, updatefig, interval = 1, blit = False);
plt.show();

#%% Solution Verification Plots
fig = plt.figure(figsize = (18, 8));
mesh.plotSoln(Manifactured_Soln(0), fig, "Solution");
k = 1;
def updatefig(i):
	global k;
	mesh.plotSoln(X_verif[:, k], fig, "Solution");
	#mesh.plotSoln(Manifactured_Soln(k*dt), fig, "Solution");
	k += 1;
	if k == shape(X)[1]:
		k = 0;
	return 0;
anim = animation.FuncAnimation(fig, updatefig, interval = 1, blit = False);
plt.show();
