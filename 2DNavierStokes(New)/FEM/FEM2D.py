from numpy import*
import sympy as syp
from numpy.linalg import det, inv
from mytools import pbf
import time
from TriMeshFEM import TriMesh, MeshElementFEM
from scipy.sparse import csr_matrix as sparse

#%% Functions
def LinearAdvectionFEM_Matrix(mesh, c, mu, x_int_mesh, y_int_mesh, w_int_stdtri, IC, BC, Verif_Source):
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
			IC = elem.ApplyInitialCondition(IC);
	
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
	mass_M_inv = sparse(inv(mass_M));
	A = sparse(mass_M_inv.dot(-stiff_M));
	S = mass_M_inv.dot(S_Vect);
	print("Done");
	print("\n Applying Boundary Conditions ...\n");
	mesh.ApplyBoundaryConditions(A, BC);
	print("Done");
	return IC, mass_M, stiff_M, A, S.reshape(1, len(S))[0];


def BurgersEquationFEM_Matrix(mesh, mu, x_int_mesh, y_int_mesh, w_int_stdtri, IC, BC):
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
			ICu = elem.ApplyInitialCondition(IC);
			ICv = elem.ApplyInitialCondition(IC);

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

	print("\n Inverting Matrices ...");
	mass_M_inv = sparse(inv(mass_M));
	diffusion_M = sparse(mass_M_inv.dot(stiff_M3));
	print("Done");
	print("\n Applying Boundary Conditions ...");
	mesh.ApplyBoundaryConditions(diffusion_M, BC);
	print("Done");
	return ICu, ICv, mass_M_inv, stiff_M3, diffusion_M;

def NonLinearStiff_Matrix(u, v, mesh, mass_M_inv, x_int_mesh, y_int_mesh, w_int_stdtri, BC):
	stiff_M1_non_lin = zeros((mesh.nVert, mesh.nVert));
	stiff_M2_non_lin = zeros((mesh.nVert, mesh.nVert));

	for element in range(mesh.nElem):
		elem = MeshElementFEM(mesh, element);
		stiff_elemMatrix1 = zeros((elem.N, elem.N));
		stiff_elemMatrix2 = zeros((elem.N, elem.N));
		elem.Transform_to_stdtri();
		J = elem.Jacobian;
		j_c = elem.Jacobian_c;
		det_J = det(J);
		phi, dphi = elem.ShapeFunctions(J[0][0]*x_int_mesh + J[0][1]*y_int_mesh + j_c[0], J[1][0]*x_int_mesh + J[1][1]*y_int_mesh + j_c[1]);
		for i in range(elem.N):
			for j in range(elem.N):
				for k in range(elem.N):
					stiff_elemMatrix1[i, j] += u[elem.vertices_idx[k]]*(phi[i]*phi[j]*dphi[k][0]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
					stiff_elemMatrix2[i, j] += v[elem.vertices_idx[k]]*(phi[i]*phi[j]*dphi[k][1]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);

		elem.Assemble_FEM_Matrix(stiff_elemMatrix1, stiff_M1_non_lin);
		elem.Assemble_FEM_Matrix(stiff_elemMatrix2, stiff_M2_non_lin);
	convection_x = sparse(mass_M_inv.dot(stiff_M1_non_lin));
	convection_y = sparse(mass_M_inv.dot(stiff_M2_non_lin));
	mesh.ApplyBoundaryConditions(convection_x, BC);
	mesh.ApplyBoundaryConditions(convection_y, BC);
	return convection_x, convection_y, convection_x, convection_y;


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
#	return A, B, C, D;
#X, Y = RK42D_call((dt, Update_convection_M, 0, diffusion_M, 0, 0), dt, a, b, t_max, stab_fig);