from numpy import*
import sympy as syp
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from NumericalTools import*
from matplotlib.widgets import Slider
from matplotlib import animation


#%% Functions
def GenMesh1D(x0, x1, N, xi):
	N += 1;
	mesh = linspace(x0, x1, N);
	#mesh = Cosine_Sampler(x0, x1, N);
	element_div = 4;
	element_nodes = zeros((2*N - 2, element_div));
	phi = syp.MutableDenseNDimArray(zeros((1, 2*N - 2))[0]);
	lin_func = lambda X, Y: ((Y[1] - Y[0])/(X[1] - X[0]), Y[1] - (Y[1] - Y[0])/(X[1] - X[0])*X[1]);
	f_max = 1; f = [f_max, 0];
	c = 0;
	for i in range(N - 1):
		a, b = lin_func([mesh[i], mesh[i + 1]], f);
		element_nodes[c, :] = linspace(mesh[i], mesh[i + 1], element_div);
		phi[c] = a*xi + b;
		c += 1;
		element_nodes[c, :] = linspace(mesh[i], mesh[i + 1], element_div);
		phi[c] = -phi[c - 1] + f_max;
		c += 1;
	return mesh, element_nodes, phi;

def FiniteElementMatrix(phi, c, N, element_nodes, x_int_std, w_int_std, xi):
	mass_M = zeros((N + 1, N + 1));
	stiff_M = zeros((N + 1, N + 1));
	element_nodes = element_nodes[::2];
	last_element = N;
	element = 0;
	while element != last_element:
		w_int, x_int = Quadrature_weightTransform(x_int_std, w_int_std, element_nodes[element, 0], element_nodes[element, -1]);
		globals()["m%s" % element] = zeros((2, 2));
		globals()["k%s" % element] = zeros((2, 2));
		for i in range(2):
			for j in range(2):
				idx_i = i + 2*element;
				idx_j = j + 2*element;
				globals()["m%s" % element][i, j] = integrate1D(phi[idx_i]*phi[idx_j], w_int, x_int);
				globals()["k%s" % element][i, j] = c*integrate1D(phi[idx_i]*syp.diff(phi[idx_j], xi), w_int, x_int);
				
		element += 1;
	column = 0;
	for row in range(N):
		mass_M[row:row + 2, column:column + 2] += globals()["m%s" % row];
		stiff_M[row:row + 2, column:column + 2] += globals()["k%s" % row];
		column += 1;
	return mass_M, stiff_M;

def ConstructFunction(phi, element_nodes, xi, a, fig):
	r, c = shape(element_nodes);
	a = repeat(a, 2)[1: -1];
	a = repeat(repeat(a, c).reshape(len(a), c), r, axis = 0);
	phi_func = syp.lambdify(xi, phi, "numpy");
	F = array(phi_func(element_nodes)).reshape(shape(a))*a;
	idx = arange(0, shape(a)[0], r) + range(0, r);
	soln = F[idx, :];
	even_idx = arange(0, shape(soln)[0], 2);	odd_idx = arange(1, shape(soln)[0], 2);
	soln = soln[even_idx, :] + soln[odd_idx, :];
	plt.figure(fig.number);
	plt.axes();
	plt.cla();
	plt.plot(element_nodes[::2].reshape(1, -1)[0], soln.reshape(1, -1)[0]);
	plt.grid(True);
	#plt.ylim([-0.1, 5.0]);
	return soln;

def plot_stability_region(stabfn, A, dt):
	x = linspace(-4, 4, 100);
	X = meshgrid(x, x);
	z = X[0] + 1j*X[1];
	Rlevel = abs(stabfn(z));
	plt.figure(figsize = (8, 8));
	plt.contourf(x, x, Rlevel, [1, 1000]);
	plt.contour(x, x, Rlevel, [1, 1000]);
	plt.xlabel(r'Re'); plt.ylabel(r'Im');
	plt.plot([0, 0], [-4, 4], '-k');
	plt.plot([-4, 4], [0, 0], '-k');
	plt.axes().set_aspect('equal');
	evals = eigvals(A*dt);
	Re = [i.real for i in evals];
	Im = [i.imag for i in evals];
	plt.scatter(Re, Im, color = 'yellow', marker = 'x');
	return 0;

def ApplyInitialCondition(mesh, f):
	a = f(mesh);
	return a;

def f(x): return sin(0.5*pi*x);





#%% Input data
xi = syp.symbols("xi");								# Symbolic variable
x0 = 0;		x1 = 10;								# Domain dimentions
c = 5;												# Advection speed
t_max = 10;											# Maximum time
dt = 0.01;											# Time step
N = 40;												# Number of elements = N, number of basis functions = N + 1
DOP = 4;											# Degree of precision for integration
w_int_std, x_int_std = Quadrature_weights(DOP, -1, 1);
x_mesh, element_nodes, phi = GenMesh1D(x0, x1, N, xi);	# Generate mesh

mass_M, stiff_M = FiniteElementMatrix(phi, c, N, element_nodes, x_int_std, w_int_std, xi);
A = inv(mass_M).dot(-stiff_M);


#a = zeros((1, N + 1))[0];
#a[0: 4] = array([0, 2, 5, 3]);
#a = zeros((1, N + 1))[0];
a = ApplyInitialCondition(x_mesh, f);
fig = plt.figure(figsize = (9, 8));
_ = ConstructFunction(phi, element_nodes, xi, a, fig);
plt.show();
s = lambda z: 1/(1 - z);
plot_stability_region(s, A, dt);
A[0, :] = 0;
X = EulerImplicit(A, dt, a, t_max);
plt.show();

#%% Solution Plots
fig = plt.figure(figsize = (9, 8));
soln = ConstructFunction(phi, element_nodes, xi, a, fig);
k = 1;
def updatefig(i):
	global k;
	_ = ConstructFunction(phi, element_nodes, xi, X[:, k], fig);
	k += 1;
	if k == shape(X)[1]:
		k = 0;
	return 0;
anim = animation.FuncAnimation(fig, updatefig, interval = 1, blit = False);
plt.show();