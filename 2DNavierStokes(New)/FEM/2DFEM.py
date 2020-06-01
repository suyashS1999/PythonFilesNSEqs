from numpy import*
import sympy as syp
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from NumericalTools import*
from TriMeshFEM import TriMesh, MeshElementFEM
from matplotlib import animation
from mytools import pbf
import time

#%% Input data
x1 = 0.;		x2 = 4.;												# Domain dimentions
y1 = 0.;		y2 = 4.;
c = 5;																	# Advection Speed
mu = 0.5;																# Viscosity
t_max = 1;																# Maximum time
dt = 0.01;																# Time step
DOP = 4;																# Degree of precision for integration
w_int_stdtri, x_int_stdtri = Quadrature_weights(DOP, 0, 1, "lin");		# Quadrature weights and nodes for intgration
w_fact = 0.5;															# Area ratio between square and triangle
w_int_stdtri *= w_fact;
x_int_mesh, y_int_mesh = meshgrid(x_int_stdtri, x_int_stdtri);
quadrature_parm = (w_int_stdtri, x_int_stdtri, w_int_stdtri, x_int_stdtri);

mesh = TriMesh(x1, x2, y1, y2);
mesh.loadMesh(6, 0, 0, 0);
mesh.plotMesh();

#%% Functions
def InitialCondition(x, y): return 40*exp(-100*((x - 0.8)**2 + (y - 0.8)**2));

def Verif_f(x, y): return sin(0.5*pi*x)*cos(0.5*pi*(y - 1));
def Verif_Source(x, y): return -sin(0.5*pi*x)*cos(0.5*pi*(y - 1)) + c*(0.5*pi*cos(0.5*pi*x)*cos(0.5*pi*(y - 1)) -
								0.5*pi*sin(0.5*pi*x)*sin(0.5*pi*(y - 1))) + mu*((0.5*pi)**2*sin(0.5*pi*x)*cos(0.5*pi*(y - 1)) + 
								(0.5*pi)**2*sin(0.5*pi*x)*cos(0.5*pi*(y - 1)));

def LinearAdvectionFEM_Matrix(mesh, c, mu):
	mass_M = zeros((mesh.nVert, mesh.nVert));
	stiff_M = zeros((mesh.nVert, mesh.nVert));
	S_Vect = zeros((mesh.nVert, 1));
	for element in range(mesh.nElem):
		pbf(element, mesh.nElem, "Assembeling Matrix");
		elem = MeshElementFEM(mesh, element);
		mass_elemMatrix = zeros((elem.N, elem.N));
		stiff_elemMatrix1 = zeros((elem.N, elem.N));
		stiff_elemMatrix2 = zeros((elem.N, elem.N));
		S_elemVect = zeros((elem.N, 1));
		if element == 0:
			IC = elem.ApplyInitialCondition(InitialCondition);
	
		elem.Transform_to_stdtri();
		J = elem.Jacobian;
		J_inv = elem.Jacobian_inv;
		j_c = elem.Jacobian_c;
		det_J = det(J);
		ksi = J[0][0]*x_int_mesh + J[0][1]*y_int_mesh + j_c[0];			eta = J[1][0]*x_int_mesh + J[1][1]*y_int_mesh + j_c[1];
		phi, dphi = elem.ShapeFunctions(ksi, eta);
		for i in range(elem.N):
			S_elemVect[i] = (phi[i]*Verif_Source(ksi, eta)*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
			for j in range(elem.N):
				mass_elemMatrix[i, j] = (phi[i]*phi[j]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
				stiff_elemMatrix1[i, j] = c*(phi[i]*(dphi[j][0] + dphi[j][1])*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
				phi_dxphi = (phi[i]*dphi[j][0]);		phi_dyphi = (phi[i]*dphi[j][1]);
				intx_part1 = (phi_dxphi[:, -1] - phi_dxphi[:, 0]);
				inty_part2 = (phi_dyphi[:, -1] - phi_dyphi[:, 0]);
				stiff_elemMatrix2[i, j] = mu*((intx_part1*det_J).dot(w_int_stdtri) - (dphi[i][0]*dphi[j][0]*det_J).dot(w_int_stdtri).dot(w_int_stdtri)	
											+ (inty_part2*det_J).dot(w_int_stdtri) - (dphi[i][1]*dphi[j][1]*det_J).dot(w_int_stdtri).dot(w_int_stdtri));
		elem.Assemble_FEM_Matrix(mass_elemMatrix, mass_M);
		elem.Assemble_FEM_Matrix(stiff_elemMatrix1 - stiff_elemMatrix2, stiff_M);
		elem.Assemble_FEM_Vector(S_elemVect, S_Vect);
	print("\n Inverting Matrix to compute A matrix ...");
	mass_M_inv = inv(mass_M);
	A = mass_M_inv.dot(-stiff_M);
	S = mass_M_inv.dot(S_Vect);
	print("Done");
	print("\n Applying Boundary Conditions ...\n");
	mesh.ApplyBoundaryConditions(A);
	print("Done");
	return IC, mass_M, stiff_M, A, S.reshape(1, len(S))[0];

def BurgersEquationFEM_Matrix(mesh, mu):
	mass_M = zeros((mesh.nVert, mesh.nVert));
	stiff_M1_non_lin = zeros((mesh.nElem, 3, 3, 3));
	stiff_M2_non_lin = zeros((mesh.nElem, 3, 3, 3));
	stiff_M3 = zeros((mesh.nVert, mesh.nVert));
	for element in range(mesh.nElem):
		pbf(element, mesh.nElem, "Assembeling Matrix");
		elem = MeshElementFEM(mesh, element);
		mass_elemMatrix = zeros((elem.N, elem.N));
		stiff_elemMatrix3 = zeros((elem.N, elem.N));
		if element == 0:
			ICu = elem.ApplyInitialCondition(InitialCondition);
			ICv = elem.ApplyInitialCondition(InitialCondition);

		elem.Transform_to_stdtri();
		J = elem.Jacobian;
		j_c = elem.Jacobian_c;
		det_J = det(J);
		phi, dphi = elem.ShapeFunctions(J[0][0]*x_int_mesh + J[0][1]*y_int_mesh + j_c[0], J[1][0]*x_int_mesh + J[1][1]*y_int_mesh + j_c[1]);
		for i in range(elem.N):
			for j in range(elem.N):
				mass_elemMatrix[i, j] = (phi[i]*phi[j]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
				phi_dxphi = (phi[i]*dphi[j][0]);		phi_dyphi = (phi[i]*dphi[j][1]);
				intx_part1 = (phi_dxphi[:, -1] - phi_dxphi[:, 0]);
				inty_part2 = (phi_dyphi[:, -1] - phi_dyphi[:, 0]);
				stiff_elemMatrix3[i, j] = mu*(intx_part1.dot(w_int_stdtri)*det_J - (dphi[i][0]*dphi[j][0]*det_J).dot(w_int_stdtri).dot(w_int_stdtri)	
											+ (inty_part2.dot(w_int_stdtri)*det_J - (dphi[i][1]*dphi[j][1]*det_J).dot(w_int_stdtri).dot(w_int_stdtri)));
				for k in range(elem.N):
					stiff_M1_non_lin[element, k] = (phi[i]*phi[j]*dphi[k][0]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
					stiff_M2_non_lin[element, k] = (phi[i]*phi[j]*dphi[k][1]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);

		elem.Assemble_FEM_Matrix(mass_elemMatrix, mass_M);
		elem.Assemble_FEM_Matrix(stiff_elemMatrix3, stiff_M3);
	

	def Update_stiff_M(u, v, stiff_M1_non_lin, stiff_M2_non_lin, mass_M_inv, mesh):
		stiff_M1 = zeros((mesh.nVert, mesh.nVert));
		stiff_M2 = zeros((mesh.nVert, mesh.nVert));
		stiff_M3 = zeros((mesh.nVert, mesh.nVert));
		stiff_M4 = zeros((mesh.nVert, mesh.nVert));
		#st = time.time();
		for element in range(mesh.nElem):
			elem = MeshElementFEM(mesh, element);
			stiff_elemMatrix1 = u[elem.vertices_idx[0]]*stiff_M1_non_lin[element][0] + u[elem.vertices_idx[1]]*stiff_M1_non_lin[element][1] + u[elem.vertices_idx[2]]*stiff_M1_non_lin[element][2];
			stiff_elemMatrix2 = u[elem.vertices_idx[0]]*stiff_M2_non_lin[element][0] + u[elem.vertices_idx[1]]*stiff_M2_non_lin[element][1] + u[elem.vertices_idx[2]]*stiff_M2_non_lin[element][2];
			stiff_elemMatrix3 = v[elem.vertices_idx[0]]*stiff_M1_non_lin[element][0] + v[elem.vertices_idx[1]]*stiff_M1_non_lin[element][1] + v[elem.vertices_idx[2]]*stiff_M1_non_lin[element][2];
			stiff_elemMatrix4 = v[elem.vertices_idx[0]]*stiff_M2_non_lin[element][0] + v[elem.vertices_idx[1]]*stiff_M2_non_lin[element][1] + v[elem.vertices_idx[2]]*stiff_M2_non_lin[element][2];
			elem.Assemble_FEM_Matrix(stiff_elemMatrix1, stiff_M1);
			elem.Assemble_FEM_Matrix(stiff_elemMatrix2, stiff_M2);
			elem.Assemble_FEM_Matrix(stiff_elemMatrix3, stiff_M3);
			elem.Assemble_FEM_Matrix(stiff_elemMatrix4, stiff_M4);
			#map(elem.Assemble_FEM_Matrix, [stiff_elemMatrix1, stiff_elemMatrix2, stiff_elemMatrix3, stiff_elemMatrix4], [stiff_M1, stiff_M2, stiff_M3, stiff_M4]);
		#print(time.time() - st);
		convection_ux = mass_M_inv.dot(stiff_M1);
		convection_uy = mass_M_inv.dot(stiff_M2);
		convection_vx = mass_M_inv.dot(stiff_M3);
		convection_vy = mass_M_inv.dot(stiff_M4);
		mesh.ApplyBoundaryConditions(convection_ux);
		mesh.ApplyBoundaryConditions(convection_uy);
		mesh.ApplyBoundaryConditions(convection_vx);
		mesh.ApplyBoundaryConditions(convection_vy);
		
		return convection_ux, convection_uy, convection_vx, convection_vy;

	print("\n Inverting Matrices ...");
	mass_M_inv = inv(mass_M);
	diffusion_M = mass_M_inv.dot(stiff_M3);
	print("Done");
	print("\n Applying Boundary Conditions ...");
	mesh.ApplyBoundaryConditions(diffusion_M);
	print("Done");
	print("\n Final assembly of matrices ...");
	print("Done");
	return ICu, ICv, (mass_M_inv), stiff_M1_non_lin, stiff_M2_non_lin, stiff_M3, (diffusion_M), Update_stiff_M;

#def BurgersEquationFEM_Matrix(mesh, mu):
#	mass_M = zeros((mesh.nVert, mesh.nVert));
#	stiff_M1 = zeros((mesh.nVert, mesh.nVert), dtype = object);
#	stiff_M2 = zeros((mesh.nVert, mesh.nVert), dtype = object);
#	stiff_M3 = zeros((mesh.nVert, mesh.nVert));
#	a_sym = syp.symbols("a_sym0:%d" % mesh.nVert);
#	for element in range(mesh.nElem):
#		pbf(element, mesh.nElem, "Assembeling Matrix");
#		elem = MeshElementFEM(mesh, element);
#		mass_elemMatrix = zeros((elem.N, elem.N));
#		stiff_elemMatrix1 = zeros((elem.N, elem.N), dtype = object);
#		stiff_elemMatrix2 = zeros((elem.N, elem.N), dtype = object);
#		stiff_elemMatrix3 = zeros((elem.N, elem.N));
#		if element == 0:
#			ICu = elem.ApplyInitialCondition(InitialCondition);
#			ICv = elem.ApplyInitialCondition(InitialCondition);

#		elem.Transform_to_stdtri();
#		J = elem.Jacobian;
#		j_c = elem.Jacobian_c;
#		det_J = det(J);
#		phi, dphi = elem.ShapeFunctions(J[0][0]*x_int_mesh + J[0][1]*y_int_mesh + j_c[0], J[1][0]*x_int_mesh + J[1][1]*y_int_mesh + j_c[1]);
#		for i in range(elem.N):
#			for j in range(elem.N):
#				mass_elemMatrix[i, j] = (phi[i]*phi[j]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
#				phi_dxphi = (phi[i]*dphi[j][0]);		phi_dyphi = (phi[i]*dphi[j][1]);
#				intx_part1 = (phi_dxphi[:, -1] - phi_dxphi[:, 0]);
#				inty_part2 = (phi_dyphi[:, -1] - phi_dyphi[:, 0]);
#				stiff_elemMatrix3[i, j] = mu*(intx_part1.dot(w_int_stdtri)*det_J - (dphi[i][0]*dphi[j][0]*det_J).dot(w_int_stdtri).dot(w_int_stdtri)	
#											+ (inty_part2.dot(w_int_stdtri)*det_J - (dphi[i][1]*dphi[j][1]*det_J).dot(w_int_stdtri).dot(w_int_stdtri)));
#				for k in range(elem.N):
#					stiff_elemMatrix1[i, j] += a_sym[elem.vertices_idx[k]]*(phi[i]*phi[j]*dphi[k][0]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
#					stiff_elemMatrix2[i, j] += a_sym[elem.vertices_idx[k]]*(phi[i]*phi[j]*dphi[k][1]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);

#		elem.Assemble_FEM_Matrix(mass_elemMatrix, mass_M);
#		elem.Assemble_FEM_Matrix(stiff_elemMatrix1, stiff_M1);
#		elem.Assemble_FEM_Matrix(stiff_elemMatrix2, stiff_M2);
#		elem.Assemble_FEM_Matrix(stiff_elemMatrix3, stiff_M3);
	
#	print("\n Inverting Matrices ...");
#	mass_M_inv = inv(mass_M);
#	diffusion_M = mass_M_inv.dot(stiff_M3);
#	print("Done");
#	print("\n Applying Boundary Conditions ...");
#	mesh.ApplyBoundaryConditions(diffusion_M);
#	print("Done");
#	print("\n Final assembly of matrices ...");
#	convection_u_M_func = syp.lambdify([a_sym], (stiff_M1), "numpy");
#	convection_v_M_func = syp.lambdify([a_sym], (stiff_M2), "numpy");
#	print("Done");
#	return ICu, ICv, (mass_M_inv), stiff_M1, stiff_M2, stiff_M3, (diffusion_M), convection_u_M_func, convection_v_M_func;


#%% Generate Matrix Lin Adv
a, mass_M, stiff_M, A, S  = LinearAdvectionFEM_Matrix(mesh, c, mu);

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
plot_stability_region(s, A, dt, stab_fig);
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

#%% Generate Matrix Burgers Equation
#a, b, mass_M_inv, _, _, _, diffusion_M, convection_u_M_func, convection_v_M_func = BurgersEquationFEM_Matrix(mesh, mu);
#def Update_convection_M(a, b): 
#	#A = (mass_M_inv @ ((asarray(convection_u_M_func(a), dtype = float))));
#	#B = (mass_M_inv @ ((asarray(convection_v_M_func(a), dtype = float))));
#	#C = (mass_M_inv @ ((asarray(convection_u_M_func(b), dtype = float))));
#	#D = (mass_M_inv @ ((asarray(convection_v_M_func(b), dtype = float))));
#	A = (mass_M_inv.dot((asarray(convection_u_M_func(a), dtype = float))));
#	B = (mass_M_inv.dot((asarray(convection_v_M_func(a), dtype = float))));
#	C = (mass_M_inv.dot((asarray(convection_u_M_func(b), dtype = float))));
#	D = (mass_M_inv.dot((asarray(convection_v_M_func(b), dtype = float))));
#	mesh.ApplyBoundaryConditions(A);
#	mesh.ApplyBoundaryConditions(B);
#	mesh.ApplyBoundaryConditions(C);
#	mesh.ApplyBoundaryConditions(D);
#	return A, B, C, D

a, b, mass_M_inv, stiff_M1_non_lin, stiff_M2_non_lin, _, diffusion_M, Update_stiff_M = BurgersEquationFEM_Matrix(mesh, mu);
stab_fig = plt.figure(figsize = (8, 8));
#X, Y = RK42D_call((dt, Update_stiff_M, (stiff_M1_non_lin, stiff_M2_non_lin, mass_M_inv, mesh), diffusion_M, 0, 0), dt, a, b, t_max, stab_fig);
#X, Y = RK42D_call((dt, Update_convection_M, 0, diffusion_M, 0, 0), dt, a, b, t_max, stab_fig);
X, Y = EulerExplicit2D_nonLin(Update_stiff_M, (stiff_M1_non_lin, stiff_M2_non_lin, mass_M_inv, mesh), diffusion_M, dt, a, b, t_max, stab_fig);
plt.show();

#%% Solution Plots
fig = plt.figure(figsize = (18, 8));
mesh.plotSoln(sqrt(a**2 + b**2), fig, "Solution");
k = 1;
def updatefig(i):
	global k;
	mesh.plotSoln(sqrt(X[:, k]**2 + Y[:, k]**2), fig, "Solution");
	k += 1;
	if k == shape(X)[1]:
		k = 0;
	return 0;
anim = animation.FuncAnimation(fig, updatefig, interval = 1, blit = False);
plt.show();