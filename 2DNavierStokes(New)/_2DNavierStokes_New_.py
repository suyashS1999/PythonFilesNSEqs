#%% Imports
import sympy as syp
from numpy import*
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time as ti
from mytools import pbf, pbw
from matplotlib.widgets import Slider
# Add to test
# Generic Functions
def plot_stability_region(stabfn, A, dt, bool):
	if bool == True:
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
	if bool == True:
		plt.scatter(Re, Im, color = 'red', marker = '^');
	else:
		plt.scatter(Re, Im, color = 'yellow', marker = 'x');

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

def integrate(f, wx, x, wy, y):
	if len(f.free_symbols) <= 2:
		X, Y = meshgrid(x, y);
		F = syp.lambdify([xi, yi], f, 'numpy');
		I = F(X, Y).dot(wx).dot(wy);
	else:
		Ix = 0;
		I = 0;
		for i in range(len(x)):
			Ix += wx[i]*f.subs(xi, x[i]);
		for j in range(len(y)):
			I += wy[j]*Ix.subs(yi, y[j]);
	
	#print("numerical integral   :", syp.N(I, n = 10, chop = True));
	#print("syp integral         :", syp.integrate(f, (xi, x0, x1), (yi, y0, y1)));
	return syp.N(I, n = 10, chop = True);

def RK4(UpdateFunc, Func_args, Constant_M, Sx, Sy, a, b, Pu, Pv):
	w = array([1/6, 1/3, 1/3, 1/6]);
	Ka = zeros((len(a), 4)); Kb = zeros((len(b), 4));
	a0 = a; b0 = b;
	for i in range(4):
		A_non, B_non, C_non, D_non = UpdateFunc(a0, b0, *Func_args);
		Au = -A_non + mu*Constant_M;
		Bu = -B_non;
		Av = -C_non;
		Bv = -D_non + mu*Constant_M;
		Ka[:, i] = dt*(Au.dot(a0) + Bu.dot(b0) + Sx);
		Kb[:, i] = dt*(Av.dot(a0) + Bv.dot(b0) + Sy);
		if i != 3:
			if i < 2:
				j = 2;
			else:
				j = 1;
			a0 = a + Ka[:, i]/j;
			b0 = b + Kb[:, i]/j;
	bigMatrix = zeros((3*NS.N, 3*NS.N));
	bigMatrix[0: NS.N, 0: NS.N] = Au; bigMatrix[NS.N: 2*NS.N, 0: NS.N] = Av; bigMatrix[0: NS.N, NS.N: 2*NS.N] = Bu;
	bigMatrix[NS.N: 2*NS.N, NS.N: 2*NS.N] = Bv; bigMatrix[0: NS.N, 2*NS.N: 3*NS.N] = -1/rho*Pu; bigMatrix[NS.N: 2*NS.N, 2*NS.N: 3*NS.N] = -1/rho*Pv;
	Ma = array([sum(x) for x in (w*Ka)]); Mb = array([sum(x) for x in (w*Kb)]);
	return a + Ma, b + Mb, bigMatrix;


#%% Inputs
x0 = -1; x1 = 1;				# Domain dimentions
y0 = -1; y1 = 1;
t_max = 1;						# Max time
dt = 0.001;						# Time step
tn = int(t_max/dt);				# Number of steps
time = linspace(0, t_max, tn);
n = 30;							# Domain division
x = linspace(x0, x1, n);
y = linspace(y0, y1, n);
dx = x[1] - x[0]; dy = dx;
x_mesh, y_mesh = meshgrid(x, y);
mu = 0.8;						# Viscosity
rho = 0.5;						# Density
xi, yi = syp.symbols("xi yi");	# Symbolic variables
c = 1;
Writer = animation.writers['ffmpeg'];
writer = Writer(fps = 100, metadata = dict(artist = 'Me'), bitrate = 1800);

## Objects Navier_Stokes and Basis Functions
class NavierStokes():
	# Object to generate matrix for NS equations
	def __init__(self, basis_funcs, basis_funcs_p, weighting_funcs, x, y):
		self.phi = basis_funcs;
		self.phi_p = basis_funcs_p;
		self.w = weighting_funcs;
		self.orthW = 1/(syp.sqrt(1 - x**2)*syp.sqrt(1 - y**2));
		self.N = len(basis_funcs);			# Number of basis functions
		self.a_sym = syp.symbols("a_sym0:%d" %self.N);
		self.b_sym = syp.symbols("b_sym0:%d" %self.N);
		self.UV = syp.lambdify([self.a_sym, xi, yi], array(self.phi).dot(self.a_sym), 'numpy');
		self.P = syp.lambdify([self.a_sym, xi, yi], array(self.phi_p).dot(self.a_sym), 'numpy');
		self.a = zeros((self.N, tn));		# Weights for u
		self.b = zeros((self.N, tn));		# Weights for v
		self.p = zeros((self.N, tn));		# Weights for p
		self.S_m = syp.MutableDenseNDimArray(zeros((1, self.N))); # Source Term for pressure
		self.S_mf = 0;												# Source Term function for pressure
		self.dxphi = []*len(self.phi);		# Derivatives of basis functions
		self.dxphi_p = []*len(self.phi);
		self.dyphi = []*len(self.phi);
		self.dyphi_p = []*len(self.phi);
		print("Computing derivatives of basis functions");
		for i in range(len(self.phi)):
			self.dxphi.append(syp.diff(self.phi[i], x));
			self.dxphi_p.append(syp.diff(self.phi_p[i], x));
			self.dyphi.append(syp.diff(self.phi[i], y));
			self.dyphi_p.append(syp.diff(self.phi_p[i], y));
		print("Done");

	def test(self):
		print("test");
		for i in range(self.N):
			f = syp.lambdify([xi, yi], self.phi_p[i], "numpy");
			plt.figure();
			ax = plt.axes(projection = '3d');
			ax.plot_surface(x_mesh, y_mesh, f(x_mesh, y_mesh), rstride = 1, cstride = 1,
							cmap = 'viridis', edgecolor = 'none');
			ax.set_xlabel('x');
			ax.set_ylabel('y');
		plt.show();

	def AssembleLinearTerms(self):
		print("Assembling Linear terms...");
		L1 = zeros((self.N, self.N));
		L2 = zeros((self.N, self.N));
		L3 = zeros((self.N, self.N));
		L4 = zeros((self.N, self.N));
		L5 = zeros((self.N, self.N));
		L = zeros((self.N, self.N));
		starttime = ti.time();
		for i in range(self.N):
			phi1 = self.w[i];
			for j in range(self.N):
				phi2 = self.phi[j];
				dxphi2 = self.dxphi[j];
				dyphi2 = self.dyphi[j];
				# x momentum:
				L1[i, j] = integrate(phi1*phi2*self.orthW, wx, x_int, wy, y_int);									# Left side of equation   phi_i*phi_j
				L2[i, j] = integrate(phi1*self.dxphi_p[j]*self.orthW, wx, x_int, wy, y_int);						# Grad P   dphi_i/dx*phi_j
				L3[i, j] = integrate(phi1*syp.diff(dxphi2, xi)*self.orthW, wx, x_int, wy, y_int);					# Visc terms   d2phi_i/dx*phi_j
				L4[i, j] = integrate(phi1*syp.diff(dyphi2, yi)*self.orthW, wx, x_int, wy, y_int);					# Visc terms   d2phi_i/dy*phi_j
				L[i, j] = integrate(self.phi_p[i]*syp.diff(self.dxphi_p[j], xi)*self.orthW, wx, x_int, wy, y_int) + integrate(self.phi_p[i]*syp.diff(self.dyphi_p[j], yi), wx, x_int, wy, y_int);			# Test with pressure basis function
				# y momentum:
				L5[i, j] = integrate(phi1*self.dyphi_p[j]*self.orthW, wx, x_int, wy, y_int);						# Grad P   dphi_i/dy*phi_j
		Pu = inv(L1).dot(L2);
		Pv = inv(L1).dot(L5);
		Al = inv(L1).dot(L3 + L4);
		print("Done");
		print("that took:", ti.time() - starttime, "s");
		# Returns phi_i*phi_j, dphi_i/dx*phi_j, dphi_i/dy*phi_j
		return L1, Pu, Pv, Al, L;

	def AssembleNonLinearTerms(self):
		print("Assembling Non-Linear terms...");
		starttime = ti.time();
		i = 0;
		w = self.w;
		phi = self.phi;
		dxphi = self.dxphi;
		dyphi = self.dyphi;
		A = syp.MutableDenseNDimArray(zeros((self.N, self.N)));
		B = syp.MutableDenseNDimArray(zeros((self.N, self.N)));
		while i < self.N:
			weight_func = w[i];
			for j in range(self.N):
				phi1 = phi[j];
				for k in range(self.N):
					A[i, j] += self.a_sym[k]*integrate(weight_func*phi1*dxphi[k]*self.orthW, wx, x_int, wy, y_int);
					B[i, j] += self.a_sym[k]*integrate(weight_func*phi1*dyphi[k]*self.orthW, wx, x_int, wy, y_int);
			i += 1;
		print("Done");
		print("that took:", ti.time() - starttime, "s");
		return syp.lambdify([self.a_sym], A*c), syp.lambdify([self.a_sym], B*c);

	def AssembleNonLinearMatrix(self, a, b, L1, A_s, B_s):
		#print("Updating Non Linear terms");
		Ai = A_s(a);
		Bi = B_s(a);
		Ci = A_s(b);
		Di = B_s(b);

		Ai = array(Ai, dtype = float).reshape(self.N, self.N);
		Bi = array(Bi, dtype = float).reshape(self.N, self.N);
		Ci = array(Ci, dtype = float).reshape(self.N, self.N);
		Di = array(Di, dtype = float).reshape(self.N, self.N);
		return inv(L1).dot(Ai), inv(L1).dot(Bi), inv(L1).dot(Ci), inv(L1).dot(Di);

	def PressureSouce(self):
		print("Assembling Pressure Matrix");
		starttime = ti.time();
		source = -rho*((array(self.dxphi).dot(self.a_sym))**2 + (array(self.dyphi).dot(self.b_sym))**2 + 2*array(self.dyphi).dot(self.a_sym)*array(self.dxphi).dot(self.b_sym));
		for i in range(self.N):
			phi1 = self.phi_p[i];
			self.S_m[0, i] = integrate(phi1*source*self.orthW, wx, x_int, wy, y_int);
		self.S_mf = syp.lambdify([self.a_sym, self.b_sym], self.S_m);
		print("Done");
		print("that took:", ti.time() - starttime, "s");

	def PressureUpdate(self, L, t):
		if t == 0:
			self.PressureSouce();
		S = self.S_mf(self.a[:, t], self.b[:, t]);
		S = array(S, dtype = float).reshape(1, self.N);
		self.p[:, t] = inv(L).dot(S[0]);

	def SourceTerm(self, t):
		u = self.UV(self.a[:, t], x_mesh, y_mesh);
		v = self.UV(self.b[:, t], x_mesh, y_mesh);
		b = zeros_like(u);
		b[1: -1, 1: -1] = (rho*(1/dt*((u[1: -1, 2:] - u[1: -1, 0: -2])/(2*dx) + (v[2:, 1: -1] - v[0: -2, 1: -1])/(2*dy)) - ((u[1: -1, 2:] - u[1: -1, 0: -2])/(2*dx))**2 - 
							2*((u[2:, 1: -1] - u[0: -2, 1: -1])/(2*dy)*(v[1: -1, 2:] - v[1: -1, 0: -2])/(2*dx)) - ((v[2:, 1: -1] - v[0: -2, 1: -1])/(2*dy))**2));
		return b;

	def Pressure_Update(p, t, iter_max = 50):
		iter = 1;
		b = self.SourceTerm(t);
		while iter <= iter_max:
			pn = p.copy();
			p[1: -1, 1: -1] = (((pn[1: -1, 2:] + pn[1: -1, 0: -2])*dy**2 + (pn[2:, 1: -1] + pn[0: -2, 1: -1])*dx**2)/(2*(dx**2 + dy**2)) -
							dx**2*dy**2/(2*(dx**2 + dy**2))*b[1: -1, 1: -1]);

			# Wall BC
			p[:, -1] = p[:, -2];				# @ x = 1
			p[0, :] = p[1, :];					# @ y = -1
			p[:, 0] = p[:, 1];					# @ x = -1
			p[-1, :] = p[-2, :];				# @ y = 1
			iter += 1
		return p;

class BasisFunctions():
	def __init__(self, N, boundary_conditions):
		self.N = N;
		self.BC = boundary_conditions;
		self.basis_funcsx = [];
		self.basis_funcsy = [];
		self.basis_funcs_px = [];
		self.basis_funcs_py = [];
		self.basis_funcs_p = [];
		self.basis_funcs_f = [];

	def Factorial(self, n):
		if n == 0:
			return 1;
		else:
			return n*self.Factorial(n - 1);

	def LagrangePol(self, N):
		x = linspace(x0, x1, N + 2);
		y = linspace(y0, y1, N + 2);
		for i in range(1, len(x) - 1):
			P = 1;
			for j in range(len(x)):
				if i != j:
					P *= (xi - x[j])/(x[i] - x[j])*(yi - y[j])/(y[i] - y[j]);
			self.basis_funcs_w.append(P);
		return self.basis_funcs_w;

	def LegendrePol(self):
		print("Generating Basis Functions");
		for i in range(3, self.N + 3):
			f = (xi**2 - 1)**i;
			for n in range(1, i + 1):
				f = syp.diff(f, xi);
			P = 1/(2**i*self.Factorial(i))*f;
			self.basis_funcs_px.append(P);
			self.basis_funcs_py.append(P.subs(xi, yi));

	def ChebyshevPol(self):
		print("Generating Basis Functions");
		T0 = 1;
		T1 = xi;
		self.basis_funcsx.append(T0);
		self.basis_funcsx.append(T1);
		for i in range(2, self.N + 2):
			P = 2*xi*self.basis_funcsx[i - 1] - self.basis_funcsx[i - 2];
			self.basis_funcsx.append(P);
			self.basis_funcsy.append(P.subs(xi, yi));
		self.basis_funcsx = self.basis_funcsx[2:];
	
	def ChebyshevPolU(self):
		print("Generating Basis Functions");
		U0 = 1;
		U1 = 2*xi;
		self.basis_funcs_px.append(U0);
		self.basis_funcs_px.append(U1);
		for i in range(2, self.N + 3):
			P = 2*xi*self.basis_funcs_px[i - 1] - self.basis_funcs_px[i - 2];
			self.basis_funcs_px.append(P);
			self.basis_funcs_py.append(P.subs(xi, yi));
		self.basis_funcs_px = self.basis_funcs_px[3:];
		self.basis_funcs_py = self.basis_funcs_py[1:];

	def Newton(self, f, df, k, guess, iter_max):
		print("Newton FPI...");
		x0 = guess;
		for iter in range(iter_max):
			#print("iter:", iter);
			x0 -= float(f.subs(k, x0)/df.subs(k, x0));
		return x0;

	def GenTaylored_basis_funcs(self):
		self.ChebyshevPol();
		print("Tayloring basis function to match BCs");
		bf_copy = self.basis_funcsx.copy();
		iter_max = 20;
		for i in range(len(self.basis_funcsx)):
			K = self.Newton(bf_copy[i], syp.diff(bf_copy[i], xi), xi, 1, iter_max);
			self.basis_funcsx[i] = self.basis_funcsx[i].subs(xi, xi*K);
			self.basis_funcsy[i] = self.basis_funcsy[i].subs(yi, yi*K);
		for i in range(len(self.basis_funcsx)):
			for j in range(len(self.basis_funcsy)):
				self.basis_funcs_f.append(self.basis_funcsx[i]*self.basis_funcsy[j]);
		print("Basis Functions sucessfully altered.");
		return self.basis_funcs_f;
	
	def Chebyshev(self):
		print("Generating Basis Functions");
		chebyshevU = [];
		chebyshevT = [];
		U0 = 1;
		U1 = xi;
		chebyshevU.append(U0);
		chebyshevU.append(2*U1);
		chebyshevT.append(U0);
		chebyshevT.append(U1);
		for i in range(2, self.N + 4):
			PU = 2*xi*chebyshevU[i - 1] - chebyshevU[i - 2];
			chebyshevU.append(PU);
			PT = 2*xi*chebyshevT[i - 1] - chebyshevT[i - 2];
			chebyshevT.append(PT);
		for i in range(self.N):
			j = 1 + i;
			Cheb_neumann = chebyshevT[j] - j**2/(j + 2)**2*chebyshevT[j + 2];
			Cheb_dir = chebyshevU[i] - (i + 1)/(i + 3)*chebyshevU[i + 2];
			self.basis_funcsx.append(Cheb_dir);
			self.basis_funcsy.append(Cheb_dir.subs(xi, yi));
			self.basis_funcs_px.append(Cheb_neumann);
			self.basis_funcs_py.append(Cheb_neumann.subs(xi, yi));

		for i in range(self.N):
			for j in range(self.N):
				self.basis_funcs_p.append(self.basis_funcs_px[i]*self.basis_funcs_py[j]);
				self.basis_funcs_f.append(self.basis_funcsx[i]*self.basis_funcsy[j]);
		return self.basis_funcs_p, self.basis_funcs_f;

N = 3;							# Number of basis functions = N^2
nodes = N + 4;
wx, x_int = Quadrature_weights(nodes, x0, x1);
wy, y_int = Quadrature_weights(nodes, y0, y1);
BF = BasisFunctions(N, 0);
basis_funcs_p, basis_funcs = BF.Chebyshev();
NS = NavierStokes(basis_funcs, basis_funcs_p, basis_funcs, xi, yi);

#%% Initial Conditions
fig = plt.figure();
ax = plt.axes(projection = '3d');
plot = ax.plot_surface(x_mesh, y_mesh, NS.UV(NS.a[:, 0], x_mesh, y_mesh), rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
i = 0;
m = len(basis_funcs);
for i in range(m):
	globals()["sld%s" % i] = plt.axes([0.2, 0.1*(i + 1), 0.65, 0.03]);
	globals()["slider%s" % i] = Slider(globals()["sld%s" % i], 'i', -10.0, 10.0, valinit = 0);

def update(val): 
	global plot;
	for i in range(m):
		NS.a[i, 0] = globals()["slider%s" % i].val;
	plot.remove();
	plot = ax.plot_surface(x_mesh, y_mesh, NS.UV(NS.a[:, 0], x_mesh, y_mesh), rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
for i in range(m):
	globals()["slider%s" % i].on_changed(update);
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('u');
plt.show();

fig = plt.figure();
ax = plt.axes(projection = '3d');
plot = ax.plot_surface(x_mesh, y_mesh, NS.UV(NS.b[:, 0], x_mesh, y_mesh), rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
i = 0;
m = len(basis_funcs);
for i in range(m):
	globals()["sld%s" % i] = plt.axes([0.2, 0.1*(i + 1), 0.65, 0.03]);
	globals()["slider%s" % i] = Slider(globals()["sld%s" % i], 'i', -10.0, 10.0, valinit = 0);

def update(val): 
	global plot;
	for i in range(m):
		NS.b[i, 0] = globals()["slider%s" % i].val;
	plot.remove();
	plot = ax.plot_surface(x_mesh, y_mesh, NS.UV(NS.b[:, 0], x_mesh, y_mesh), rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
for i in range(m):
	globals()["slider%s" % i].on_changed(update);
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('v');
plt.show();

#%% Init
L1, Pu, Pv, Al, L = NS.AssembleLinearTerms();
A_sym, B_sym = NS.AssembleNonLinearTerms();
NS.PressureUpdate(L, 0);


#%% Final Assembly
k = 0;
soln_u = NS.UV(NS.a[:, k], x_mesh, y_mesh);
soln_v = NS.UV(NS.b[:, k], x_mesh, y_mesh);
soln_p = NS.P(NS.p[:, k], x_mesh, y_mesh);
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

fig = plt.figure();
ax = plt.axes(projection = '3d');
ax.plot_surface(x_mesh, y_mesh, soln_p, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('p');

#%% Time March
s = lambda z: (1 + z + z**2/2 + z**3/6 + z**4/24);
print("Beginning time march");
print("max number of time steps =", tn, "dt =", dt);
for t in range(tn - 1):
	NS.a[:, t + 1], NS.b[:, t + 1], BM = RK4(NS.AssembleNonLinearMatrix, (L1, A_sym, B_sym), Al, -1/rho*Pu.dot(NS.p[:, t]), -1/rho*Pv.dot(NS.p[:, t]), NS.a[:, t], NS.b[:, t], Pu, Pv);
	NS.PressureUpdate(L, t + 1);
	if t == 0: plot_stability_region(s, BM, dt, True);
	pbf(t, tn - 1, "Time Marching");
	if t != tn - 2:
		plot_stability_region(s, BM, dt, False);
plt.show();

#%% Solution Plots 1
fig = plt.figure(figsize = (8, 8));
k = 0;
U = NS.UV(NS.a[:, k], x_mesh, y_mesh);
V = NS.UV(NS.b[:, k], x_mesh, y_mesh);
P = NS.P(NS.p[:, k], x_mesh, y_mesh);
CS = plt.contourf(x_mesh, y_mesh, P, alpha = 0.8, cmap = cm.viridis);
cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.8]);
cbar = plt.colorbar(CS, cax = cbaxes);
Q = plt.quiver(x_mesh, y_mesh, U, V, angles = 'xy', scale_units = 'xy', pivot = 'mid', scale = 1);
text = plt.text(-10, 1., 'time: {}'.format(k*dt), fontsize = 10);
def updatefig(i):
	global CS, Q, text, k, cbar;
	for c in CS.collections: c.remove();
	text.set_visible(False);
	Q.remove();
	cbar.remove();
	U = NS.UV(NS.a[:, k], x_mesh, y_mesh);
	V = NS.UV(NS.b[:, k], x_mesh, y_mesh);
	P = NS.P(NS.p[:, k], x_mesh, y_mesh);
	CS = plt.contourf(x_mesh, y_mesh, P, alpha = 0.8, cmap = cm.viridis);
	Q = plt.quiver(x_mesh, y_mesh, U, V, angles = 'xy', scale_units = 'xy', pivot = 'mid', scale = 1);
	cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.8]);
	cbar = plt.colorbar(CS, cax = cbaxes)
	text = plt.text(-10, 1., 'time: {}'.format(round(k*dt, 4)) + ' s', fontsize = 10);

	k += 1;
	if k == tn:
		k = 0;

anim = animation.FuncAnimation(fig, updatefig, interval = 1, blit = False);
#anim.save('NavierStokes1.mp4', writer = writer);
plt.show();

#%% Solution Plots 2
fig, ax = plt.subplots(1, 2, figsize = (16, 8));
ax = ax.flatten();
k = 0;
U = NS.UV(NS.a[:, k], x_mesh, y_mesh);
V = NS.UV(NS.b[:, k], x_mesh, y_mesh);
P = NS.P(NS.p[:, k], x_mesh, y_mesh);
Cuv = ax[0].contourf(x_mesh, y_mesh, sqrt(U**2 + V**2), alpha = 0.8, cmap = cm.viridis);
mesh = ax[0].pcolormesh(x_mesh, y_mesh, sqrt(U**2 + V**2));
plt.colorbar(mesh, ax = ax[0]);
Cp = ax[1].contourf(x_mesh, y_mesh, P, alpha = 0.8, cmap = cm.viridis);
mesh1 = ax[1].pcolormesh(x_mesh, y_mesh, P);
plt.colorbar(mesh1, ax = ax[1]);
text = plt.text(-1, 1.05, 'time: {}'.format(k*dt), fontsize = 10);
def updatefig(i):
	global Cuv, Cp, text, k;
	for c in Cuv.collections: c.remove();
	for c in Cp.collections: c.remove();
	text.set_visible(False);
	U = NS.UV(NS.a[:, k], x_mesh, y_mesh);
	V = NS.UV(NS.b[:, k], x_mesh, y_mesh);
	P = NS.P(NS.p[:, k], x_mesh, y_mesh);	
	Cuv = ax[0].contourf(x_mesh, y_mesh, sqrt(U**2 + V**2), alpha = 0.8, cmap = cm.viridis);
	Cp = ax[1].contourf(x_mesh, y_mesh, P, alpha = 0.8, cmap = cm.viridis);
	text = plt.text(-1, 1.05, 'time: {}'.format(round(k*dt, 4)) + ' s', fontsize = 10);
	k += 1;
	if k == tn:
		k = 0;

anim = animation.FuncAnimation(fig, updatefig, interval = 10, blit = False);
#anim.save('NavierStokes1.mp4', writer = writer);
plt.show();