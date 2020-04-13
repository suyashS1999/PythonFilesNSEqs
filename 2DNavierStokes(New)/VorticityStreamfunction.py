#%% Imports
import sympy as syp
from numpy import*
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time as ti
from matplotlib.widgets import Slider
from mytools import pbf
from basis import BasisFunctions

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
		print("#####################");
		print("Computing EigenValues");
		print("#####################");
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


#%% Inputs
x0 = -1; x1 = 1;				# Domain dimentions
y0 = -1; y1 = 1;
t_max = 10;					# Max time
dt = 0.001;						# Time step
tn = int(t_max/dt);				# Number of steps
time = linspace(0, t_max, tn);
n = 30;							# Domain division
x = linspace(x0, x1, n);
y = linspace(y0, y1, n);
x_mesh, y_mesh = meshgrid(x, y);
mu = 0.5;						# Viscosity
rho = 0.5;						# Density
xi, yi = syp.symbols("xi yi");	# Symbolic variables
c = 1;
Writer = animation.writers['ffmpeg'];
writer = Writer(fps = 100, metadata = dict(artist = 'Me'), bitrate = 1800);

## Objects Navier_Stokes and Basis Functions
class NavierStokes():
	# Object to generate matrix for NS equations
	def __init__(self, basis_funcs, weighting_funcs, x, y):
		self.phi = basis_funcs;
		self.w = weighting_funcs;
		self.N = len(basis_funcs);			# Number of basis functions
		self.a_sym = syp.symbols("a_sym0:%d" %self.N);
		self.a = zeros((self.N, tn));		# Weights for w
		self.b = zeros((self.N, tn));		# Weights for psi
		self.S_m = syp.MutableDenseNDimArray(zeros((1, self.N))); # Source Term for Stream Function
		self.S_mf = 0;												# Source Term function for Stream Function
		self.dxphi = []*len(self.phi);		# Derivatives of basis functions
		self.dyphi = []*len(self.phi);
		self.orthW = 1/(syp.sqrt(1 - x**2)*syp.sqrt(1 - y**2));
		print("Computing derivatives of basis functions");
		for i in range(len(self.phi)):
			self.dxphi.append(syp.diff(self.phi[i], x));
			self.dyphi.append(syp.diff(self.phi[i], y));
		print("Done");

	def test(self):
		print("test");
		for i in range(self.N):
			f = syp.lambdify([xi, yi], self.phi[i], "numpy");
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
		L = zeros((self.N, self.N));
		starttime = ti.time();
		for i in range(self.N):
			phi1 = self.w[i];
			for j in range(self.N):
				phi2 = self.phi[j];
				dxphi2 = self.dxphi[j];
				dyphi2 = self.dyphi[j];
				L1[i, j] = integrate(phi1*phi2*self.orthW, wx, x_int, wy, y_int);									# Left side of equation   phi_i*phi_j
				L2[i, j] = -integrate(syp.diff(phi1, xi)*dxphi2*self.orthW, wx, x_int, wy, y_int);					# Visc terms   d2phi_i/dx*phi_j
				L3[i, j] = -integrate(syp.diff(phi1, yi)*dyphi2*self.orthW, wx, x_int, wy, y_int);					# Visc terms   d2phi_i/dy*phi_j
				L[i, j] = integrate(self.phi[i]*syp.diff(self.dxphi[j], xi)*self.orthW, wx, x_int, wy, y_int) + integrate(self.phi[i]*syp.diff(self.dyphi[j], yi)*self.orthW, wx, x_int, wy, y_int);			# Test with pressure basis function

		Al = inv(L1).dot(L2 + L3);
		print("Done");
		print("that took:", ti.time() - starttime, "s");
		# Returns phi_i*phi_j, dphi_i/dx*phi_j, dphi_i/dy*phi_j
		return L1, Al, L;

	def AssembleNonLinearTerms(self):
		print("Assembling Non-Linear terms...");
		starttime = ti.time();
		i = 0;
		w = array(self.w);
		phi = array(self.phi);
		dxphi = array(self.dxphi);
		dyphi = array(self.dyphi);
		A = syp.MutableDenseNDimArray(zeros((self.N, self.N)));
		B = syp.MutableDenseNDimArray(zeros((self.N, self.N)));
		while i < self.N:
			weight_func = w[i];
			for j in range(self.N):
				for k in range(self.N):
					A[i, j] += self.a_sym[k]*integrate(weight_func*dyphi[j]*dxphi[k]*self.orthW, wx, x_int, wy, y_int);
					B[i, j] += self.a_sym[k]*integrate(weight_func*dxphi[j]*dyphi[k]*self.orthW, wx, x_int, wy, y_int);
		#while i < self.N:
		#	weight_func = w[i];
		#	for j in range(self.N):
		#		A[i, j] = integrate(weight_func*syp.diff(self.phi[j], xi)*comm1, wx, x_int, wy, y_int);
		#		B[i, j] = integrate(weight_func*syp.diff(self.phi[j], yi)*comm2, wx, x_int, wy, y_int);
			i += 1;
		print("Done");
		print("that took:", ti.time() - starttime, "s");
		return syp.lambdify([self.a_sym], A*c), syp.lambdify([self.a_sym], B*c);

	def AssembleNonLinearMatrix(self, L1, t, A_s, B_s):
		Ai = A_s(self.b[:, t]);
		Bi = B_s(self.a[:, t]);
		Ai = array(Ai, dtype = float).reshape(self.N, self.N);
		Bi = array(Bi, dtype = float).reshape(self.N, self.N);
		return inv(L1).dot(Ai), inv(L1).dot(Bi);

	def PressureSouce(self):
		source = -(array(self.phi).dot(self.a_sym));
		for i in range(self.N):
			phi1 = self.phi[i];
			self.S_m[0, i] = integrate(phi1*source*self.orthW, wx, x_int, wy, y_int);
		self.S_mf = syp.lambdify([self.a_sym], self.S_m);

	def PressureUpdate(self, L, t):
		if t == 0:
			self.PressureSouce();
		S = self.S_mf(self.a[:, t]);
		S = array(S, dtype = float).reshape(1, self.N);
		self.b[:, t] = inv(L).dot(S[0]);


	def u_v_p(self, t, p_only, w_only, u_only, v_only):
		basis = array(self.phi);
		dxbasis = array(self.dxphi);
		dybasis = array(self.dyphi);
		if p_only == True:
			pf = basis.dot(self.b[:, t]);
			P = syp.lambdify([xi, yi], pf, "numpy");
			p = P(x_mesh, y_mesh);
			if isinstance(p, int):
				p = p*ones_like(x_mesh);
			return p;
		elif w_only == True:
			wf = basis.dot(self.a[:, t]);
			W = syp.lambdify([xi, yi], wf, "numpy");
			w = W(x_mesh, y_mesh);
			if isinstance(w, int):
				w = w*ones_like(x_mesh);
			return w;
		elif v_only == True:
			vf = -dxbasis.dot(self.b[:, t]);
			V = syp.lambdify([xi, yi], vf, "numpy");
			v = V(x_mesh, y_mesh);
			if isinstance(v, int):
				v = v*ones_like(x_mesh);
			return v;
		elif u_only == True:
			uf = dybasis.dot(self.b[:, t]);
			U = syp.lambdify([xi, yi], uf, "numpy");
			u = U(x_mesh, y_mesh);
			if isinstance(u, int):
				u = u*ones_like(x_mesh);
			return u;

		else:
			uf = dybasis.dot(self.b[:, t]);
			vf = -dxbasis.dot(self.b[:, t]);
			U = syp.lambdify([xi, yi], uf, "numpy");
			V = syp.lambdify([xi, yi], vf, "numpy");
			u = U(x_mesh, y_mesh);
			if isinstance(u, int):
				u = u*ones_like(x_mesh);
			v = V(x_mesh, y_mesh);
			if isinstance(v, int):
				v = v*ones_like(x_mesh);
			return u, v;


N = 2;							# Number of basis functions = N^2
nodes = N + 4;
wx, x_int = Quadrature_weights(nodes, x0, x1);
wy, y_int = Quadrature_weights(nodes, y0, y1);
BF = BasisFunctions(N, x0, x1, y0, x1, xi, yi);
_, basis_funcs = BF.Chebyshev();
NS = NavierStokes(basis_funcs, basis_funcs, xi, yi);

#%% Init (Only run if N or basis functions are changed)
L1, Al, L = NS.AssembleLinearTerms();
A_sym, B_sym = NS.AssembleNonLinearTerms();

#%% Initial Conditions
fig = plt.figure();
ax = plt.axes(projection = '3d');
plot = ax.plot_surface(x_mesh, y_mesh, NS.u_v_p(0, False, True, False, False), rstride = 1, cstride = 1,
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
	plot = ax.plot_surface(x_mesh, y_mesh, NS.u_v_p(0, False, True, False, False), rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
for i in range(m):
	globals()["slider%s" % i].on_changed(update);
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('u');
plt.show();

NS.PressureUpdate(L, 0);


#%% Final Assembly
A_non, B_non = NS.AssembleNonLinearMatrix(L1, 0, A_sym, B_sym);
Au = -A_non + B_non + mu*Al;
print(A_non);
soln_w = NS.u_v_p(0, False, True, False, False);
soln_p = NS.u_v_p(0, True, False, False, False);

fig = plt.figure();
ax = plt.axes(projection = '3d');
ax.plot_surface(x_mesh, y_mesh, soln_w, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('w');

fig = plt.figure();
ax = plt.axes(projection = '3d');
ax.plot_surface(x_mesh, y_mesh, soln_p, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
ax.set_xlabel('x');
ax.set_ylabel('y');
ax.set_zlabel('psi');


#%% Time March
s = lambda z: (1 + z + z**2/2);
plot_stability_region(s, Au, dt, True);
print("Beginning time march");
print("max number of time steps =", tn, "dt =", dt);
for t in range(tn - 1):
	NS.a[:, t + 1] = NS.a[:, t] + dt*Au.dot(NS.a[:, t]);
	NS.PressureUpdate(L, t + 1);
	A_non, B_non = NS.AssembleNonLinearMatrix(L1, t + 1, A_sym, B_sym);
	Au = -A_non + B_non + mu*Al;
	NS.a[:, t + 1] = NS.a[:, t] + dt/2*((NS.a[:, t + 1] - NS.a[:, t])/dt + Au.dot(NS.a[:, t + 1]));
	NS.PressureUpdate(L, t + 1);
	pbf(t, tn - 1, "Time Marching");
	if t != tn - 2:
		A_non, B_non = NS.AssembleNonLinearMatrix(L1, t + 1, A_sym, B_sym);
		Au = -A_non + B_non + mu*Al;
		plot_stability_region(s, Au, dt, False);
plt.show();

#%% Solution Plots 1
fig = plt.figure(figsize = (8, 8));
k = 1;
U, V = NS.u_v_p(0, False, False, False, False);
Q = plt.quiver(x_mesh, y_mesh, U, V, angles = 'xy', scale_units = 'xy');
def updatefig(i):
	global Q, k;
	Q.remove();
	U, V = NS.u_v_p(k, False, False, False, False);
	Q = plt.quiver(x_mesh, y_mesh, U, V, angles = 'xy', scale_units = 'xy');

	k += 1;
	if k == tn:
		k = 0;

anim = animation.FuncAnimation(fig, updatefig, interval = 10, blit = False);
#anim.save('NavierStokes1.mp4', writer = writer);
plt.show();

