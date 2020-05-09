from numpy import*
import sympy as syp
from numpy.linalg import inv, solve, eigvals, det, norm
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

def Quadrature_weights(N, a, b):
	x = Cosine_Sampler(a, b, N);
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
	I = F(x).dot(wx);
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
		Ka[:, i] = dt*(Au.dot(a0) + Bu.dot(b0) + Sx);
		Kb[:, i] = dt*(Av.dot(a0) + Bv.dot(b0) + Sy);
		if i != 3:
			if i < 2:
				j = 2;
			else:
				j = 1;
			a0 = a + Ka[:, i]/j;
			b0 = b + Kb[:, i]/j;
	bigMatrix = zeros((3*N, 3*N));
	bigMatrix[0: N, 0: N] = Au; bigMatrix[N: 2*N, 0: N] = Av; bigMatrix[0: N, N: 2*N] = Bu;
	bigMatrix[N: 2*N, N: 2*N] = Bv; bigMatrix[0: N, 2*N: 3*N] = Pu; bigMatrix[N: 2*N, 2*N: 3*N] = Pv;
	Ma = array([sum(x) for x in (w*Ka)]); Mb = array([sum(x) for x in (w*Kb)]);
	return a + Ma, b + Mb, bigMatrix;

def EulerExplicit(A, dt, x0, t_max):
	no_t_steps = int(t_max/dt);
	try:
		X = zeros((len(x0), no_t_steps));
		X[:, 0] = x0;
	except:
		X = zeros((1, no_t_steps));
		X[0] = x0;
	for t in range(no_t_steps - 1):
		X[:, t + 1] = X[:, t] + dt*A.dot(X[:, t]);
	return X;

def EulerImplicit(A, dt, x0, t_max):
	no_t_steps = int(t_max/dt);
	try:
		X = zeros((len(x0), no_t_steps));
		X[:, 0] = x0;
	except:
		X = zeros((1, no_t_steps));
		X[0] = x0;
	M = inv(eye(len(A)) - dt*A);
	for t in range(no_t_steps - 1):
		X[:, t + 1] = M.dot(X[:, t]);
	return X;