class BasisFunctions():
	def __init__(self, N, x0, x1, y0, y1, xi, yi):
		self.N = N;
		self.basis_funcsx = [];
		self.basis_funcsy = [];
		self.basis_funcs_px = [];
		self.basis_funcs_py = [];
		self.basis_funcs_p = [];
		self.basis_funcs_f = [];
		self.x0 = x0;
		self.x1 = x1;
		self.y0 = y0;
		self.y1 = y1;
		self.xi = xi;
		self.yi = yi;

	def Factorial(self, n):
		if n == 0:
			return 1;
		else:
			return n*self.Factorial(n - 1);

	def LagrangePol(self, N):
		x = linspace(self.self.x0, x1, N + 2);
		y = linspace(self.y0, self.y1, N + 2);
		for i in range(1, len(x) - 1):
			P = 1;
			for j in range(len(x)):
				if i != j:
					P *= (self.xi - x[j])/(x[i] - x[j])*(self.yi - y[j])/(y[i] - y[j]);
			self.basis_funcs_w.append(P);
		return self.basis_funcs_w;

	def LegendrePol(self):
		print("Generating Basis Functions");
		for i in range(3, self.N + 3):
			f = (self.xi**2 - 1)**i;
			for n in range(1, i + 1):
				f = syp.diff(f, self.xi);
			P = 1/(2**i*self.Factorial(i))*f;
			self.basis_funcs_px.append(P);
			self.basis_funcs_py.append(P.subs(self.xi, self.yi));

	def ChebyshevPol(self):
		print("Generating Basis Functions");
		T0 = 1;
		T1 = self.xi;
		self.basis_funcsx.append(T0);
		self.basis_funcsx.append(T1);
		for i in range(2, self.N + 2):
			P = 2*self.xi*self.basis_funcsx[i - 1] - self.basis_funcsx[i - 2];
			self.basis_funcsx.append(P);
			self.basis_funcsy.append(P.subs(self.xi, self.yi));
		self.basis_funcsx = self.basis_funcsx[2:];
	
	def ChebyshevPolU(self):
		print("Generating Basis Functions");
		U0 = 1;
		U1 = 2*self.xi;
		self.basis_funcs_px.append(U0);
		self.basis_funcs_px.append(U1);
		for i in range(2, self.N + 3):
			P = 2*self.xi*self.basis_funcs_px[i - 1] - self.basis_funcs_px[i - 2];
			self.basis_funcs_px.append(P);
			self.basis_funcs_py.append(P.subs(self.xi, self.yi));
		self.basis_funcs_px = self.basis_funcs_px[3:];
		self.basis_funcs_py = self.basis_funcs_py[1:];

	def Newton(self, f, df, k, guess, iter_max):
		print("Newton FPI...");
		self.x0 = guess;
		for iter in range(iter_max):
			#print("iter:", iter);
			self.x0 -= float(f.subs(k, self.x0)/df.subs(k, self.x0));
		return self.x0;

	def GenTaylored_basis_funcs(self):
		self.ChebyshevPol();
		print("Tayloring basis function to match BCs");
		bf_copy = self.basis_funcsx.copy();
		iter_max = 20;
		for i in range(len(self.basis_funcsx)):
			K = self.Newton(bf_copy[i], syp.diff(bf_copy[i], self.xi), self.xi, 1, iter_max);
			self.basis_funcsx[i] = self.basis_funcsx[i].subs(self.xi, self.xi*K);
			self.basis_funcsy[i] = self.basis_funcsy[i].subs(self.yi, self.yi*K);
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
		U1 = self.xi;
		chebyshevU.append(U0);
		chebyshevU.append(2*U1);
		chebyshevT.append(U0);
		chebyshevT.append(U1);
		for i in range(2, self.N + 4):
			PU = 2*self.xi*chebyshevU[i - 1] - chebyshevU[i - 2];
			chebyshevU.append(PU);
			PT = 2*self.xi*chebyshevT[i - 1] - chebyshevT[i - 2];
			chebyshevT.append(PT);
		for i in range(self.N):
			j = 1 + i;
			Cheb_neumann = chebyshevT[j] - j**2/(j + 2)**2*chebyshevT[j + 2];
			Cheb_dir = chebyshevU[i] - (i + 1)/(i + 3)*chebyshevU[i + 2];
			self.basis_funcsx.append(Cheb_dir);
			self.basis_funcsy.append(Cheb_dir.subs(self.xi, self.yi));
			self.basis_funcs_px.append(Cheb_neumann);
			self.basis_funcs_py.append(Cheb_neumann.subs(self.xi, self.yi));

		for i in range(self.N):
			for j in range(self.N):
				self.basis_funcs_p.append(self.basis_funcs_px[i]*self.basis_funcs_py[j]);
				self.basis_funcs_f.append(self.basis_funcsx[i]*self.basis_funcsy[j]);
		return self.basis_funcs_p, self.basis_funcs_f;