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

def LinAdvFiniteElementMatrix(phi, c, mu, N, element_nodes, x_int_std, w_int_std, xi):
	print("Assembeling Matrices");
	mass_M = zeros((N + 1, N + 1));
	stiff_M = zeros((N + 1, N + 1));
	element_nodes = element_nodes[::2];
	last_element = N;
	element = 0;
	while element != last_element:
		w_int, x_int = Quadrature_weightTransform(x_int_std, w_int_std, element_nodes[element, 0], element_nodes[element, -1]);
		globals()["m%s" % element] = zeros((2, 2));
		globals()["k%s" % element] = zeros((2, 2));
		globals()["d%s" % element] = zeros((2, 2));
		for i in range(2):
			for j in range(2):
				idx_i = i + 2*element;
				idx_j = j + 2*element;
				globals()["m%s" % element][i, j] = integrate1D(phi[idx_i]*phi[idx_j], w_int, x_int);
				globals()["k%s" % element][i, j] = c*integrate1D(phi[idx_i]*syp.diff(phi[idx_j], xi), w_int, x_int);
				phi_dxphi = phi[idx_i]*syp.diff(phi[idx_j], xi);
				globals()["d%s" % element][i, j] = mu*((phi_dxphi.subs(xi, x_int[-1]) - phi_dxphi.subs(xi, x_int[0])) -
													integrate1D(syp.diff(phi[idx_i], xi)*syp.diff(phi[idx_j], xi), w_int, x_int));
		element += 1;
	column = 0;
	for row in range(N):
		mass_M[row:row + 2, column:column + 2] += globals()["m%s" % row];
		stiff_M[row:row + 2, column:column + 2] += globals()["k%s" % row];
		stiff_M[row:row + 2, column:column + 2] -= globals()["d%s" % row];
		column += 1;
	print("Done");
	return mass_M, stiff_M;

def BurgersFiniteElementMatrix(phi, mu, N, element_nodes, x_int_std, w_int_std, xi):
	print("Assembeling Matrices");
	mass_M = zeros((N + 1, N + 1));
	stiff_M1 = syp.MutableDenseNDimArray(zeros((N + 1, N + 1)));
	stiff_M2 = zeros((N + 1, N + 1));
	a_sym = syp.symbols("a_sym0:%d" %(2*N));
	element_nodes = element_nodes[::2];
	last_element = N;
	element = 0;
	while element != last_element:
		w_int, x_int = Quadrature_weightTransform(x_int_std, w_int_std, element_nodes[element, 0], element_nodes[element, -1]);
		globals()["m%s" % element] = zeros((2, 2));
		globals()["k%s" % element] = syp.MutableDenseNDimArray(zeros((2, 2)));
		globals()["d%s" % element] = zeros((2, 2));
		for i in range(2):
			for j in range(2):
				idx_i = i + 2*element;
				idx_j = j + 2*element;
				globals()["m%s" % element][i, j] = integrate1D(phi[idx_i]*phi[idx_j], w_int, x_int);
				phi_dxphi = phi[idx_i]*syp.diff(phi[idx_j], xi);
				globals()["d%s" % element][i, j] = mu*((phi_dxphi.subs(xi, x_int[-1]) - phi_dxphi.subs(xi, x_int[0])) -
													integrate1D(syp.diff(phi[idx_i], xi)*syp.diff(phi[idx_j], xi), w_int, x_int));
				for k in range(2):
					idx_k = k + 2*element;
					globals()["k%s" % element][i, j] += a_sym[idx_k]*integrate1D(phi[idx_i]*phi[idx_j]*syp.diff(phi[idx_k], xi), w_int, x_int);
		element += 1;
	column = 0;
	for row in range(N):
		mass_M[row:row + 2, column:column + 2] += globals()["m%s" % row];
		stiff_M1[row:row + 2, column:column + 2] += globals()["k%s" % row];
		stiff_M2[row:row + 2, column:column + 2] += globals()["d%s" % row];
		column += 1;
	stiff_M1_func = syp.lambdify([a_sym], stiff_M1, "numpy");
	
	print("Done");
	return mass_M, stiff_M1_func, stiff_M2;

def Update_stiff_M1(a, stiff_M1_func, N):
	a = repeat(a, 2)[1: -1];
	stiff_M1 = stiff_M1_func(a);
	stiff_M1 = array(stiff_M1, dtype = float).reshape(N + 1, N + 1);
	return stiff_M1;

def ConstructFunction(phi, element_nodes, xi, a, y_lim, fig):
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
	plt.ylim([y_lim[0], y_lim[1]]);
	return soln;

def ApplyInitialCondition(mesh, f):
	a = f(mesh);
	return a, (amin(a), amax(a));

#def f(x): return 10*exp(-1*(x - 3)**2);
def f(x): return 10*sin(pi/5*x);

#%% Input data
xi = syp.symbols("xi");								# Symbolic variable
x0 = 0;		x1 = 10;								# Domain dimentions
c = 5;												# Advection speed
mu = 0.1;											# Viscosity
t_max = 20;											# Maximum time
dt = 0.01;											# Time step
N = 50;												# Number of elements = N, number of basis functions = N + 1
DOP = 4;											# Degree of precision for integration
w_int_std, x_int_std = Quadrature_weights(DOP, -1, 1, "cos");
x_mesh, element_nodes, phi = GenMesh1D(x0, x1, N, xi);	# Generate mesh

#%% Linear Advection
#mass_M, stiff_M = LinAdvFiniteElementMatrix(phi, c, mu, N, element_nodes, x_int_std, w_int_std, xi);
#A = inv(mass_M).dot(-stiff_M);

#a, y_lim = ApplyInitialCondition(x_mesh, f);
#fig = plt.figure(figsize = (9, 8));
#_ = ConstructFunction(phi, element_nodes, xi, a, y_lim, fig);
#plt.show();
#s = lambda z: 1/(1 - z);
#stab_fig = plt.figure(figsize = (8, 8));
#plot_stability_region(s, A, dt, stab_fig);
#P_BC = A[0, :];
#A[0, :] = A[-1, :];
#A[-1, :] = P_BC;
#X = EulerImplicit(A, dt, a, t_max);
#plt.show();

#%% Burgers Equation
mass_M, stiff_M1_func, stiff_M2 = BurgersFiniteElementMatrix(phi, mu, N, element_nodes, x_int_std, w_int_std, xi);
a, y_lim = ApplyInitialCondition(x_mesh, f);
fig = plt.figure(figsize = (9, 8));
_ = ConstructFunction(phi, element_nodes, xi, a, y_lim, fig);
plt.show();
stiff_M = Update_stiff_M1(a, stiff_M1_func, N) - stiff_M2;
A = inv(mass_M).dot(-stiff_M);
stab_fig = plt.figure(figsize = (8, 8));

P_BC = A[0, :];
A[0, :] = A[-1, :];
A[-1, :] = P_BC;
X = RK4_call((dt, inv(mass_M), Update_stiff_M1, (stiff_M1_func, N), stiff_M2, 0), dt, a, t_max, stab_fig);
plt.show();

#%% Solution Plots
fig = plt.figure(figsize = (9, 8));
soln = ConstructFunction(phi, element_nodes, xi, a, y_lim, fig);
k = 1;
def updatefig(i):
	global k;
	_ = ConstructFunction(phi, element_nodes, xi, X[:, k], y_lim, fig);
	k += 1;
	if k == shape(X)[1]:
		k = 0;
	return 0;
anim = animation.FuncAnimation(fig, updatefig, interval = 1, blit = False);
plt.show();