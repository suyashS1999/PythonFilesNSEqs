#%% Imports
import sympy as syp
from numpy import*
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mytools import pbf, pbw
from matplotlib.widgets import Slider
from basis import BasisFunctions
from NavierStokesMatrix import NavierStokes
from NumericalTools import*

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

#%% Inputs
x0 = -1; x1 = 1;								# Domain dimentions
y0 = -1; y1 = 1;
t_max = 1;										# Max time
dt = 0.001;										# Time step
tn = int(t_max/dt);								# Number of steps
time = linspace(0, t_max, tn);
n = 30;											# Domain division
x = linspace(x0, x1, n);
y = linspace(y0, y1, n);
dx = x[1] - x[0]; dy = dx;
x_mesh, y_mesh = meshgrid(x, y);
mu = 0.8;										# Viscosity
rho = 2;										# Density
delta = 0.5;
xi, yi = syp.symbols("xi yi");					# Symbolic variables
N = 3;											# Number of basis functions = N^2
nodes = N + 4;									# Quadrature DOP
wx, x_int = Quadrature_weights(nodes, x0, x1);	# Quadrature weights
wy, y_int = Quadrature_weights(nodes, y0, y1);
 
BF = BasisFunctions(N, x0, x1, y0, x1, xi, yi);		# Generate Basis Functions
basis_funcs_p, basis_funcs = BF.Chebyshev();
NS = NavierStokes(basis_funcs, basis_funcs_p, basis_funcs, xi, yi, (wx, x_int, wy, y_int), tn);
#NS.test(x_mesh, y_mesh);
Writer = animation.writers['ffmpeg'];
writer = Writer(fps = 100, metadata = dict(artist = 'Me'), bitrate = 1800);

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
L1, Pu, Pv, Al, Lp1, Lup, Lvp = NS.AssembleLinearTerms();
A_sym, B_sym = NS.AssembleNonLinearTerms();
#P_source = NS.PressureUpdate(L, 0, rho);
#NS.Pressure_Update(delta, 0, dt, Lup, Lvp);
#plt.figure();
#plt.imshow(NS.d2P(NS.p[:, 0], x_mesh, y_mesh));
#plt.colorbar();
#plt.show();
#print(NS.d2P(NS.p[:, 0], x_mesh, y_mesh))
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
fakeSolnx, fakeSolny, ErrSourceX, ErrSourceY = NS.Error_test_Momentum(mu);			# Manifactured Solution for Code Verification
error_u = [0.];
error_v = [0.];
#error_p = [amax(absolute(NS.d2P(NS.p[:, 0], x_mesh, y_mesh)) - absolute(P_source(NS.a[:, 0], NS.b[:, 0], x_mesh, y_mesh)))];
s = lambda z: (1 + z + z**2/2 + z**3/6 + z**4/24);
print("Beginning time march");
print("max number of time steps =", tn, "dt =", dt);
for t in range(tn - 1):
	NS.a[:, t + 1], NS.b[:, t + 1], BM = RK4(dt, NS.AssembleNonLinearMatrix, (L1, A_sym, B_sym), mu*Al, 
										-1/rho*Pu.dot(NS.p[:, t]), -1/rho*Pv.dot(NS.p[:, t]), NS.a[:, t], NS.b[:, t], -1/rho*Pu, -1/rho*Pv);
	NS.a_error[:, t + 1], NS.b_error[:, t + 1], _ = RK4(dt, NS.AssembleNonLinearMatrix, (L1, A_sym, B_sym), mu*Al, 
													inv(L1).dot(ErrSourceX(t*dt)), inv(L1).dot(ErrSourceY(t*dt)), NS.a_error[:, t], NS.b_error[:, t], 0, 0);
	#NS.PressureUpdate(L, t + 1, rho);
	NS.Pressure_Update(delta, t, dt, Lup, Lvp);
	if t == 0: plot_stability_region(s, BM, dt, True);
	pbf(t, tn - 1, "Time Marching");
	error_u.append(amax(absolute(fakeSolnx(x_mesh, y_mesh, (t + 1)*dt)) - absolute(NS.UV(NS.a_error[:, t + 1], x_mesh, y_mesh))));
	error_v.append(amax(absolute(fakeSolny(x_mesh, y_mesh, (t + 1)*dt)) - absolute(NS.UV(NS.b_error[:, t + 1], x_mesh, y_mesh))));
	#error_p.append(amax(absolute(NS.d2P(NS.p[:, t + 1], x_mesh, y_mesh)) - absolute(P_source(NS.a[:, t + 1], NS.b[:, t + 1], x_mesh, y_mesh))))
	#fig = plt.figure();
	#ax = plt.axes(projection = '3d');
	#ax.plot_surface(x_mesh, y_mesh, NS.d2P(NS.p[:, t + 1], x_mesh, y_mesh)[0], rstride = 1, cstride = 1,
	#				cmap = 'viridis', edgecolor = 'none');
	#ax.set_xlabel('x');
	#ax.set_ylabel('y');
	#ax.set_zlabel('d2p');

	#fig = plt.figure();
	#ax = plt.axes(projection = '3d');
	#ax.plot_surface(x_mesh, y_mesh, P_source(NS.a[:, t + 1], NS.b[:, t + 1], x_mesh, y_mesh), rstride = 1, cstride = 1,
	#				cmap = 'viridis', edgecolor = 'none');
	#ax.set_xlabel('x');
	#ax.set_ylabel('y');
	#ax.set_zlabel('S');
	#plt.show();
	if t != tn - 2:
		plot_stability_region(s, BM, dt, False);
plt.figure();
plt.plot(time, error_u, label = "Error x momentum");
plt.plot(time, error_v, label = "Error y momentum");
#plt.plot(time, error_p, label = "Error pressure");
plt.grid(True);
plt.legend();
plt.xlabel("time");
plt.ylabel("error");
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
Q = plt.quiver(x_mesh, y_mesh, U, V, angles = 'xy', scale_units = 'xy', pivot = 'mid');
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
	Q = plt.quiver(x_mesh, y_mesh, U, V, angles = 'xy', scale_units = 'xy', pivot = 'mid');
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
cbar = plt.colorbar(mesh, ax = ax[0]);
Cp = ax[1].contourf(x_mesh, y_mesh, P, alpha = 0.8, cmap = cm.viridis);
mesh1 = ax[1].pcolormesh(x_mesh, y_mesh, P);
cbar1 = plt.colorbar(mesh1, ax = ax[1]);
text = plt.text(-1, 1.05, 'time: {}'.format(k*dt), fontsize = 10);
def updatefig(i):
	global Cuv, Cp, text, k, cbar, cbar1;
	for c in Cuv.collections: c.remove();
	for c in Cp.collections: c.remove();
	cbar.remove();
	cbar1.remove();
	text.set_visible(False);
	U = NS.UV(NS.a[:, k], x_mesh, y_mesh);
	V = NS.UV(NS.b[:, k], x_mesh, y_mesh);
	P = NS.P(NS.p[:, k], x_mesh, y_mesh);	
	Cuv = ax[0].contourf(x_mesh, y_mesh, sqrt(U**2 + V**2), alpha = 0.8, cmap = cm.viridis);
	Cp = ax[1].contourf(x_mesh, y_mesh, P, alpha = 0.8, cmap = cm.viridis);
	text = plt.text(-1, 1.05, 'time: {}'.format(round(k*dt, 4)) + ' s', fontsize = 10);
	cbaxes1 = fig.add_axes([0.9, 0.1, 0.03, 0.8]);
	cbar1 = plt.colorbar(Cp, cax = cbaxes1)
	cbaxes = fig.add_axes([0.45, 0.1, 0.03, 0.8]);
	cbar = plt.colorbar(Cuv, cax = cbaxes)
	k += 1;
	if k == tn:
		k = 0;

anim = animation.FuncAnimation(fig, updatefig, interval = 10, blit = False);
#anim.save('NavierStokes1.mp4', writer = writer);
plt.show();