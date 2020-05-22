from numpy import*
import sympy as syp
from numpy.linalg import inv, solve, eigvals, det, norm
from matplotlib import pyplot as plt
from matplotlib import animation
from NumericalTools import*
from TriMeshFEM import TriMesh, MeshElementFEM
from matplotlib import animation
from mytools import pbf

def InitialCondition(x, y): return 40*exp(-100*((x - 0.8)**2 + (y - 0.8)**2)); # 3*x**3 + y**2*x + 5*y;

#%% Input data
x1 = 0.;		x2 = 4.;									# Domain dimentions
y1 = 0.;		y2 = 4.;
c = 1;														# Advection Speed
mu = 0.01;													# Viscosity
t_max = 20;													# Maximum time
dt = 0.01;													# Time step
DOP = 3;													# Degree of precision for integration
w_int_stdtri, x_int_stdtri = Quadrature_weights(DOP, 0, 1);	# Quadrature weights and nodes for intgration
w_fact = 0.5;												# Area ratio between square and triangle
w_int_stdtri *= w_fact;
x_int_mesh, y_int_mesh = meshgrid(x_int_stdtri, x_int_stdtri);
quadrature_parm = (w_int_stdtri, x_int_stdtri, w_int_stdtri, x_int_stdtri);

mesh = TriMesh(x1, x2, y1, y2);
mesh.loadMesh(5, 0, 0, 0);
mesh.plotMesh();


#%% Generate Matrix
mass_M = zeros((mesh.nVert, mesh.nVert));
stiff_M = zeros((mesh.nVert, mesh.nVert));
for element in range(mesh.nElem):
	pbf(element, mesh.nElem, "Assembeling Matrix");
	elem = MeshElementFEM(mesh, element);
	mass_elemMatrix = zeros((elem.N, elem.N));
	stiff_elemMatrix1 = zeros((elem.N, elem.N));
	stiff_elemMatrix2 = zeros((elem.N, elem.N));
	if element == 0:
		a = elem.ApplyInitialCondition(InitialCondition);
	
	elem.Transform_to_stdtri();
	J = elem.Jacobian;
	j_c = elem.Jacobian_c;
	det_J = det(J);
	phi, dphi = elem.ShapeFunctions(J[0][0]*x_int_mesh + J[0][1]*y_int_mesh + j_c[0], J[1][0]*x_int_mesh + J[1][1]*y_int_mesh + j_c[1]);
	for i in range(elem.N):
		for j in range(elem.N):
			mass_elemMatrix[i, j] = (phi[i]*phi[j]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
			stiff_elemMatrix1[i, j] = c*(phi[i]*(dphi[j][0] + dphi[j][1])*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
			phi_dxphi = (phi[i]*dphi[j][0]);		phi_dyphi = (phi[i]*dphi[j][1]);
			intx_part1 = (phi_dxphi[:, -1] - phi_dxphi[:, 0]);
			inty_part1 = (phi_dyphi[:, -1] - phi_dyphi[:, 0]);
			stiff_elemMatrix2[i, j] = mu*(intx_part1.dot(w_int_stdtri)*det_J - (dphi[i][0]*dphi[j][0]*det_J).dot(w_int_stdtri).dot(w_int_stdtri)	
										+ (inty_part1.dot(w_int_stdtri)*det_J - (dphi[i][1]*dphi[j][1]*det_J).dot(w_int_stdtri).dot(w_int_stdtri)));
			
				 

	elem.Assemble_FEM_Matrix(mass_elemMatrix, mass_M);
	elem.Assemble_FEM_Matrix(stiff_elemMatrix1 - stiff_elemMatrix2, stiff_M);

mesh.plotSoln(a, None, title = "Initial Condition");
plt.figure();
plt.imshow(mass_M);
plt.colorbar();

plt.figure();
plt.imshow(stiff_M);
plt.colorbar();


print("\n Inverting Matrix A");
A = inv(mass_M).dot(-stiff_M);


s = lambda z: 1/(1 - z);
stab_fig = plt.figure(figsize = (8, 8));
plot_stability_region(s, A, dt, stab_fig);
print("\n Applying Boundary Conditions \n");
A = elem.ApplyBoundaryConditions(A);
X = EulerImplicit(A, dt, a, t_max);
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