from numpy import*
import sympy as syp
from numpy.linalg import inv, solve, eigvals, det, norm
from mytools import pbf
from matplotlib import pyplot as plt
import time
from scipy.sparse import csr_matrix as sparse

def Cosine_Sampler(a, b, n):
	ksi = zeros((1, n));
	for i in range(1, n + 1):
		ksi[0][i - 1] = cos((2*i - 1)*pi/(2*n));
	xi = (a + b)/2 + (b - a)/2*ksi;
	return xi[0];

## Quadrature
def GenMatrix(N, x):
	A = ones((N, N));
	for i in range(1, N):
		A[i, :] = x**i;
	return inv(A);

def Quadrature_weights(N, a, b, spacing):
	if spacing == "lin":	x = linspace(a, b, N);
	elif spacing == "cos":	x = Cosine_Sampler(a, b, N);
	A_i = GenMatrix(N, x);
	B = zeros((N, 1));
	for i in range(1, N + 1):
		B[i - 1] = (b**i - a**i)/i;
	w = A_i.dot(B);
	return w.T[0], x;

def Quadrature_weightTransform(nodes, weights, a, b):
	""" Function to transform Quadrature weights and nodes from one domain [c, d] to another [a, b]
		while maintaining Degree of Percision
	Input Arguments:
		nodes = array with nodes in [c, d] (numpy array)
		weights = array with weights calculated for nodes in [c, d] (numpy array)
		[a, b] = new interval (float, float)
	Output:
		x_int = new Quadrature nodes (numpy array)
		w = new Quadrature weights (numpy array)
	"""
	x_int = (b - a)/2*(nodes + 1) + a;
	w = (b - a)/2*weights;
	return w, x_int;

def integrate(f, wx, x, wy, y):
	if len(f.free_symbols) <= 2:
		xi, yi = f.free_symbols;
		X, Y = meshgrid(x, y);
		F = syp.lambdify([xi, yi], f, "numpy");
		I = F(X, Y).dot(wx).dot(wy);
	else:
		xi, yi = syp.symbols("xi yi");
		Ix = 0;
		I = 0;
		for i in range(len(x)):
			Ix += wx[i]*f.subs(xi, x[i]);
		for j in range(len(y)):
			I += wy[j]*Ix.subs(yi, y[j]);
	
	#print("numerical integral   :", syp.N(I, n = 10, chop = True));
	#print("syp integral         :", syp.integrate(f, (xi, x0, x1), (yi, y0, y1)));
	return I;

def integrate1D(f, wx, x):
	xi = f.free_symbols;
	F = syp.lambdify(xi, f, "numpy");
	try:
		I = F(x).dot(wx);
	except:
		I = f*ones_like(x).dot(wx);
	return I;

def RK4(dt, UpdateFunc, Func_args, Constant_M, Sx, Sy, a, b, Pu, Pv):
	N = len(a);
	w = array([1/6, 1/3, 1/3, 1/6]);
	Ka = zeros((len(a), 4)); Kb = zeros((len(b), 4));
	a0 = a; b0 = b;
	
	for i in range(4):
		A_non, B_non, C_non, D_non = UpdateFunc(a0, b0, *Func_args);
		Au = -A_non + Constant_M;
		Bu = -B_non;
		Av = -C_non;
		Bv = -D_non + Constant_M;
		if i == 0: st = time.time();
		Ka[:, i] = dt*(Au.dot(a0) + Bu.dot(b0) + Sx);
		Kb[:, i] = dt*(Av.dot(a0) + Bv.dot(b0) + Sy);
		if i != 3:
			if i < 2:
				j = 2;
			else:
				j = 1;
			a0 = a + Ka[:, i]/j;
			b0 = b + Kb[:, i]/j;
	print(time.time() - st);
	bigMatrix = zeros((3*N, 3*N));
	bigMatrix[0: N, 0: N] = Au; bigMatrix[N: 2*N, 0: N] = Av; bigMatrix[0: N, N: 2*N] = Bu;
	bigMatrix[N: 2*N, N: 2*N] = Bv; bigMatrix[0: N, 2*N: 3*N] = Pu; bigMatrix[N: 2*N, 2*N: 3*N] = Pv;
	Ma = array([sum(x) for x in (w*Ka)]); Mb = array([sum(x) for x in (w*Kb)]);
	return a + Ma, b + Mb, bigMatrix;

def RK42D_call(RK4_args, dt, x0, y0, t_max, fig):
	no_t_steps = int(t_max/dt);
	X = zeros((len(x0), no_t_steps));
	Y = zeros((len(x0), no_t_steps));
	X[:, 0] = x0;	Y[:, 0] = y0;
	s = lambda z: (1 + z + z**2/2 + z**3/6 + z**4/24);
	for t in range(no_t_steps - 1):
		pbf(t, no_t_steps - 2, "Time Marching");
		X[:, t + 1], Y[:, t + 1], A = RK4(*RK4_args, X[:, t], Y[:, t], 0, 0);
		if t == 0 or t == no_t_steps - 2:
			plot_stability_region(s, A, dt, fig);
	return X, Y;


def RK41D(dt, inv_mass_M, UpdateFunc, Func_args, Constant_M, Sx, a):
	N = len(a);
	w = array([1/6, 1/3, 1/3, 1/6]);
	Ka = zeros((len(a), 4));
	a0 = a;
	for i in range(4):
		stiff_M = UpdateFunc(a0, *Func_args) - Constant_M;
		A = inv_mass_M.dot(-stiff_M); A_ = A;
		P_BC = A[0, :];
		A[0, :] = A[-1, :];
		A[-1, :] = P_BC;
		Ka[:, i] = dt*(A.dot(a0) + Sx);
		if i != 3:
			if i < 2:
				j = 2;
			else:
				j = 1;
			a0 = a + Ka[:, i]/j;
	Ma = array([sum(x) for x in (w*Ka)]);
	return a + dt*A.dot(a), A_;

def RK4_call(RK4_args, dt, x0, t_max, fig):
	no_t_steps = int(t_max/dt);
	X = zeros((len(x0), no_t_steps));
	X[:, 0] = x0;
	s = lambda z: (1 + z + z**2/2 + z**3/6 + z**4/24);
	for t in range(no_t_steps - 1):
		pbf(t, no_t_steps - 2, "Time Marching");
		X[:, t + 1], A = RK41D(*RK4_args, X[:, t]);
		if t == 0 or t == no_t_steps - 2:
			plot_stability_region(s, A, dt, fig)
	return X;

def EulerExplicit(A, dt, x0, t_max):
	no_t_steps = int(t_max/dt);
	try:
		X = zeros((len(x0), no_t_steps));
		X[:, 0] = x0;
	except:
		X = zeros((1, no_t_steps));
		X[0] = x0;
	for t in range(no_t_steps - 1):
		pbf(t, no_t_steps - 2, "Time Marching");
		X[:, t + 1] = X[:, t] + dt*A.dot(X[:, t]);
	return X;

def EulerExplicit2D_nonLin(Update_func, Func_args, Constant_M, dt, x0, y0, t_max, fig):
	no_t_steps = int(t_max/dt);
	X = zeros((len(x0), no_t_steps));
	Y = zeros((len(x0), no_t_steps));
	X[:, 0] = x0;		Y[:, 0] = y0;
	s = lambda z: 1 + z;
	for t in range(no_t_steps - 1):
		pbf(t, no_t_steps - 2, "Time Marching");
		A_non, B_non, C_non, D_non = Update_func(X[:, t], Y[:, t], *Func_args);
		Au = -A_non + Constant_M;
		Bu = -B_non;
		Av = -C_non;
		Bv = -D_non + Constant_M;
		X[:, t + 1] = X[:, t] + dt*(Au.dot(X[:, t]) + Bu.dot(Y[:, t]));
		Y[:, t + 1] = Y[:, t] + dt*(Av.dot(X[:, t]) + Bv.dot(Y[:, t]));
		if t == 0 or t == no_t_steps - 2:
			A = vstack((concatenate((sparse.todense(Au), sparse.todense(Bu)), axis = 1), concatenate((sparse.todense(Av), sparse.todense(Bv)), axis = 1)));
			plot_stability_region(s, A, dt, fig);
	return X, Y;

def EulerImplicit(A, dt, x0, t_max, S_t, Manifactured_Soln):
	no_t_steps = int(t_max/dt);
	try:
		X = zeros((len(x0), no_t_steps));
		X[:, 0] = x0;
	except:
		X = zeros((1, no_t_steps));
		X[0] = x0;
	M = inv(eye(shape(A)[0]) - dt*A);
	if S_t == 0:
		for t in range(no_t_steps - 1):
			pbf(t, no_t_steps - 2, "Time Marching");
			X[:, t + 1] = M.dot(X[:, t]);
	else:
		error = zeros(no_t_steps);
		for t in range(no_t_steps - 1):
			pbf(t, no_t_steps - 2, "Verifying Time March");
			X[:, t + 1] = M.dot(X[:, t] + dt*S_t((t + 0)*dt));
			error[t + 1] = mean(absolute(X[:, t + 1]) - absolute(Manifactured_Soln((t + 1)*dt)));
		plt.figure();
		plt.plot(linspace(0, t_max, no_t_steps), error);
		plt.grid(True);
		plt.show();
	return X;

def plot_stability_region(stabfn, A, dt, fig):
	x = linspace(-4, 4, 100);
	X = meshgrid(x, x);
	z = X[0] + 1j*X[1];
	Rlevel = abs(stabfn(z));
	plt.figure(fig.number);
	plt.contourf(x, x, Rlevel, [1, 1000]);
	plt.contour(x, x, Rlevel, [1, 1000]);
	plt.xlabel(r'Re'); plt.ylabel(r'Im');
	plt.plot([0, 0], [-4, 4], '-k');
	plt.plot([-4, 4], [0, 0], '-k');
	#plt.axes().set_aspect('equal');
	evals = eigvals(A*dt);
	Re = [i.real for i in evals];
	Im = [i.imag for i in evals];
	plt.scatter(Re, Im, color = 'yellow', marker = 'x');
	return 0;

#x = syp.symbols("x");
#f1 = 5e-9*x**4 - 1e-5*x**3 + 0.0029*x**2 - 0.1794*x - 0.4279;
#f2 = 1e-8*x**5 - 3e-6*x**4 - 0.0197*x**2 + 0.0003*x**3 + 0.5455*x + 0.6581;
#w, x = Quadrature_weights(10, 0, 1, "lin");
#I1 = integrate1D(f1, w, x);
#I2 = integrate1D(f2, w, x);
#print(abs(I1) + abs(I2));