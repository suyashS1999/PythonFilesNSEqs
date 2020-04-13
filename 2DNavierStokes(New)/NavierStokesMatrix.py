import sympy as syp
from numpy import*
import time as ti
from NumericalTools import integrate
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class NavierStokes():
	# Object to generate matrix for NS equations
	def __init__(self, basis_funcs, basis_funcs_p, weighting_funcs, x, y, int_vals, tn):
		self.phi = basis_funcs;
		self.phi_p = basis_funcs_p;
		self.w = weighting_funcs;
		self.wx, self.x_int, self.wy, self.y_int = int_vals; 
		self.x = x;
		self.y = y;
		self.orthW = 1/(syp.sqrt(1 - x**2)*syp.sqrt(1 - y**2));
		self.N = len(basis_funcs);							# Number of basis functions
		self.a_sym = syp.symbols("a_sym0:%d" %self.N);
		self.b_sym = syp.symbols("b_sym0:%d" %self.N);
		self.UV = syp.lambdify([self.a_sym, self.x, self.y], array(self.phi).dot(self.a_sym), 'numpy');
		P = array(self.phi_p).dot(self.a_sym);
		self.P = syp.lambdify([self.a_sym, self.x, self.y], P, 'numpy');
		self.a = zeros((self.N, tn));						# Weights for u
		self.b = zeros((self.N, tn));						# Weights for v	
		self.a_error = zeros((self.N, tn));					# Weights for u_error
		self.b_error = zeros((self.N, tn));					# Weights for v_error
		self.p = zeros((self.N, tn));						# Weights for p
		self.p_error = zeros((self.N, tn));					# Weights for p_error
		self.S_m = syp.MutableDenseNDimArray(zeros((1, self.N))[0]);			# Source Term for pressure
		self.S_mf = 0;														# Source Term function for pressure
		self.dxphi = []*len(self.phi);										# Derivatives of basis functions
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
		self.d2P = syp.lambdify([self.a_sym, self.x, self.y], array(syp.diff(syp.diff(P, x), x)).dot(self.a_sym) + array(syp.diff(syp.diff(P, y), y)).dot(self.a_sym), "numpy");

	def test(self, x_mesh, y_mesh):
		print("test");
		for i in range(self.N):
			f = syp.lambdify([self.x, self.y], self.dxphi[i], "numpy");
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
		Lp1 = zeros((self.N, self.N));
		Lp2 = zeros((self.N, self.N));
		Lp3 = zeros((self.N, self.N));
		starttime = ti.time();
		for i in range(self.N):
			phi1 = self.w[i];
			for j in range(self.N):
				phi2 = self.phi[j];
				dxphi2 = self.dxphi[j];
				dyphi2 = self.dyphi[j];
				# x & y momentum:
				L1[i, j] = integrate(phi1*phi2*self.orthW, self.wx, self.x_int, self.wy, self.y_int);									# Left side of equation   phi_i*phi_j
				L2[i, j] = integrate(phi1*self.dxphi_p[j]*self.orthW, self.wx, self.x_int, self.wy, self.y_int);						# Grad P   dphi_i/dx*phi_j
				L3[i, j] = integrate(phi1*syp.diff(dxphi2, self.x)*self.orthW, self.wx, self.x_int, self.wy, self.y_int);				# Visc terms   d2phi_i/dx*phi_j
				L4[i, j] = integrate(phi1*syp.diff(dyphi2, self.y)*self.orthW, self.wx, self.x_int, self.wy, self.y_int);				# Visc terms   d2phi_i/dy*phi_j
				L5[i, j] = integrate(phi1*self.dyphi_p[j]*self.orthW, self.wx, self.x_int, self.wy, self.y_int);						# Grad P   dphi_i/dy*phi_j
				# Pressure
				Lp1[i, j] = integrate(self.phi_p[i]*self.phi_p[j]*self.orthW, self.wx, self.x_int, self.wy, self.y_int);
				Lp2[i, j] = integrate(self.phi_p[i]*dxphi2*self.orthW, self.wx, self.x_int, self.wy, self.y_int);
				Lp3[i, j] = integrate(self.phi_p[i]*dyphi2*self.orthW, self.wx, self.x_int, self.wy, self.y_int);
		Pu = inv(L1).dot(L2);
		Pv = inv(L1).dot(L5);
		Al = inv(L1).dot(L3 + L4);
		Lup = inv(Lp1).dot(Lp2);
		Lvp = inv(Lp1).dot(Lp3);
		print("Done");
		print("that took:", ti.time() - starttime, "s");
		# Returns phi_i*phi_j, dphi_i/dx*phi_j, dphi_i/dy*phi_j
		return L1, Pu, Pv, Al, Lp1, Lup, Lvp;

	#def AssembleLinearTerms(self):
	#	print("Assembling Linear terms...");
	#	L1 = zeros((self.N, self.N));
	#	L2 = zeros((self.N, self.N));
	#	L3 = zeros((self.N, self.N));
	#	L4 = zeros((self.N, self.N));
	#	L5 = zeros((self.N, self.N));
	#	Lp1 = zeros((self.N, self.N));
	#	Lp2 = zeros((self.N, self.N));
	#	starttime = ti.time();
	#	for i in range(self.N):
	#		phi1 = self.w[i];
	#		for j in range(self.N):
	#			phi2 = self.phi[j];
	#			dxphi2 = self.dxphi[j];
	#			dyphi2 = self.dyphi[j];
	#			# x momentum:
	#			L1[i, j] = integrate(phi1*phi2*self.orthW, self.wx, self.x_int, self.wy, self.y_int);									# Left side of equation   phi_i*phi_j
	#			L2[i, j] = integrate(phi1*self.dxphi_p[j]*self.orthW, self.wx, self.x_int, self.wy, self.y_int);						# Grad P   dphi_i/dx*phi_j
	#			L3[i, j] = integrate(phi1*syp.diff(dxphi2, self.x)*self.orthW, self.wx, self.x_int, self.wy, self.y_int);					# Visc terms   d2phi_i/dx*phi_j
	#			L4[i, j] = integrate(phi1*syp.diff(dyphi2, self.y)*self.orthW, self.wx, self.x_int, self.wy, self.y_int);					# Visc terms   d2phi_i/dy*phi_j
	#			Lp1[i, j] = integrate(self.phi_p[i]*syp.diff(self.dxphi_p[j], self.x)*self.orthW, self.wx, self.x_int, self.wy, self.y_int);
	#			Lp2[i, j] = integrate(self.phi_p[i]*syp.diff(self.dyphi_p[j], self.y)*self.orthW, self.wx, self.x_int, self.wy, self.y_int);			# Test with pressure basis function
	#			# y momentum:
	#			L5[i, j] = integrate(phi1*self.dyphi_p[j]*self.orthW, self.wx, self.x_int, self.wy, self.y_int);						# Grad P   dphi_i/dy*phi_j
	#	Pu = inv(L1).dot(L2);
	#	Pv = inv(L1).dot(L5);
	#	Al = inv(L1).dot(L3 + L4);
	#	L = Lp1 + Lp2;
	#	print("Done");
	#	print("that took:", ti.time() - starttime, "s");
	#	# Returns phi_i*phi_j, dphi_i/dx*phi_j, dphi_i/dy*phi_j
	#	return L1, Pu, Pv, Al, L;

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
					A[i, j] += self.a_sym[k]*integrate(weight_func*phi1*dxphi[k]*self.orthW, self.wx, self.x_int, self.wy, self.y_int);
					B[i, j] += self.a_sym[k]*integrate(weight_func*phi1*dyphi[k]*self.orthW, self.wx, self.x_int, self.wy, self.y_int);
			i += 1;
		print("Done");
		print("that took:", ti.time() - starttime, "s");
		return syp.lambdify([self.a_sym], A), syp.lambdify([self.a_sym], B);

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

	#def PressureSouce(self, rho):
	#	print("Assembling Pressure Matrix");
	#	starttime = ti.time();
	#	source = -rho*(((array(self.dxphi).dot(self.a_sym))**2) + ((array(self.dyphi).dot(self.b_sym))**2) + (2*(array(self.dyphi).dot(self.a_sym))*(array(self.dxphi).dot(self.b_sym))));
	#	for i in range(self.N):
	#		self.S_m[i] = integrate(self.phi_p[i]*source*self.orthW, self.wx, self.x_int, self.wy, self.y_int);
	#	self.S_mf = syp.lambdify([self.a_sym, self.b_sym], self.S_m);
	#	print("Done");
	#	print("that took:", ti.time() - starttime, "s");
	#	return syp.lambdify([self.a_sym, self.b_sym, self.x, self.y], source, "numpy");

	#def PressureUpdate(self, L, t, rho):
	#	if t == 0:
	#		source = self.PressureSouce(rho);
	#	S = self.S_mf(self.a[:, t], self.b[:, t]);
	#	S = array(S, dtype = float).reshape(1, self.N);
	#	self.p[:, t] = inv(L).dot(S[0]);
	#	if t == 0:
	#		return source;

	def Pressure_Update(self, delta, t, dt, Lup, Lvp):
		self.p[:, t + 1] = self.p[:, t] - dt/delta*(Lup.dot(self.a[:, t]) + Lvp.dot(self.b[:, t]));

	def Error_test_Momentum(self, mu):
		t_sym = syp.symbols("t_sym");
		fakeSolnx = syp.exp(-t_sym)*self.phi[0];
		fakeSolny = syp.exp(-t_sym)*self.phi[0];
		fx = syp.lambdify([self.x, self.y, t_sym], fakeSolnx, "numpy");
		fy = syp.lambdify([self.x, self.y, t_sym], fakeSolny, "numpy");
		SourceTermx = syp.diff(fakeSolnx, t_sym) + fakeSolnx*syp.diff(fakeSolnx, self.x) + fakeSolny*syp.diff(fakeSolnx, self.y) - mu*(syp.diff(syp.diff(fakeSolnx, self.x), self.x) + syp.diff(syp.diff(fakeSolnx, self.y), self.y));
		SourceTermy = syp.diff(fakeSolny, t_sym) + fakeSolnx*syp.diff(fakeSolny, self.x) + fakeSolny*syp.diff(fakeSolny, self.y) - mu*(syp.diff(syp.diff(fakeSolny, self.x), self.x) + syp.diff(syp.diff(fakeSolny, self.y), self.y));
		source_x = syp.MutableDenseNDimArray(zeros((1, self.N))[0]);
		source_y = syp.MutableDenseNDimArray(zeros((1, self.N))[0]);
		self.a_error[0, 0] = 1.0; self.b_error[0, 0] = 1.0;
		for i in range(self.N):
			source_x[i] = integrate(SourceTermx*self.phi[i]*self.orthW, self.wx, self.x_int, self.wy, self.y_int);
			source_y[i] = integrate(SourceTermy*self.phi[i]*self.orthW, self.wx, self.x_int, self.wy, self.y_int);
		return fx, fy, syp.lambdify([t_sym], source_x), syp.lambdify([t_sym], source_y);


	#def SourceTerm(self, t):
	#	u = self.UV(self.a[:, t], x_mesh, y_mesh);
	#	v = self.UV(self.b[:, t], x_mesh, y_mesh);
	#	b = zeros_like(u);
	#	b[1: -1, 1: -1] = (rho*(1/dt*((u[1: -1, 2:] - u[1: -1, 0: -2])/(2*dx) + (v[2:, 1: -1] - v[0: -2, 1: -1])/(2*dy)) - ((u[1: -1, 2:] - u[1: -1, 0: -2])/(2*dx))**2 - 
	#						2*((u[2:, 1: -1] - u[0: -2, 1: -1])/(2*dy)*(v[1: -1, 2:] - v[1: -1, 0: -2])/(2*dx)) - ((v[2:, 1: -1] - v[0: -2, 1: -1])/(2*dy))**2));
	#	return b;

	#def Pressure_Update(p, t, iter_max = 50):
	#	iter = 1;
	#	b = self.SourceTerm(t);
	#	while iter <= iter_max:
	#		pn = p.copy();
	#		p[1: -1, 1: -1] = (((pn[1: -1, 2:] + pn[1: -1, 0: -2])*dy**2 + (pn[2:, 1: -1] + pn[0: -2, 1: -1])*dx**2)/(2*(dx**2 + dy**2)) -
	#						dx**2*dy**2/(2*(dx**2 + dy**2))*b[1: -1, 1: -1]);

	#		# Wall BC
	#		p[:, -1] = p[:, -2];				# @ x = 1
	#		p[0, :] = p[1, :];					# @ y = -1
	#		p[:, 0] = p[:, 1];					# @ x = -1
	#		p[-1, :] = p[-2, :];				# @ y = 1
	#		iter += 1
	#	return p;