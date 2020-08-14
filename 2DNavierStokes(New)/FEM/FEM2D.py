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
			if Verif_Source != False:
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
	print("\nInverting Matrix to compute A matrix ...");
	mass_M_inv = sparse(inv(mass_M));
	A = sparse(mass_M_inv.dot(-stiff_M));
	S = mass_M_inv.dot(S_Vect);
	print("Done");
	print("\nApplying Boundary Conditions ...");
	mesh.ApplyBoundaryConditions(A, BC);
	print("Done");
	return IC, mass_M, stiff_M, A, S.reshape(1, len(S))[0];


def BurgersEquationFEM_Matrix(mesh, mu, x_int_mesh, y_int_mesh, w_int_stdtri, IC, BC):
	mass_M = zeros((mesh.nVert, mesh.nVert));
	stiff_M1_non_lin = zeros((mesh.nElem, 3, 3, 3));
	stiff_M2_non_lin = zeros((mesh.nElem, 3, 3, 3));
	stiff_M3 = zeros((mesh.nVert, mesh.nVert));
	map2glob = zeros((mesh.nElem, 3), dtype = int);
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
					stiff_M1_non_lin[element, k, i, j] = (phi[i]*phi[j]*dphi[k][0]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
					stiff_M2_non_lin[element, k, i, j] = (phi[i]*phi[j]*dphi[k][1]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);

		map2glob[element, :] = array([elem.vertices_idx[0], elem.vertices_idx[1], elem.vertices_idx[2]]);
		elem.Assemble_FEM_Matrix(mass_elemMatrix, mass_M);
		elem.Assemble_FEM_Matrix(stiff_elemMatrix3, stiff_M3);

	map2glob = map2glob.reshape(mesh.nElem, 3, 1);
	sum_lst = Glob_Map(map2glob);
	print("\nInverting Matrices ...");
	mass_M_inv = sparse(inv(mass_M));
	diffusion_M = sparse(mass_M_inv.dot(stiff_M3));
	print("Done");
	print("\nApplying Boundary Conditions ...");
	mesh.ApplyBoundaryConditions(diffusion_M, BC);
	print("Done");
	return ICu, ICv, mass_M_inv, diffusion_M, stiff_M1_non_lin, stiff_M2_non_lin, map2glob, sum_lst;

def Glob_Map(map2glob):
	sp = shape(map2glob);
	map = map2glob.reshape(sp[0]*sp[1]);
	glob_idx = arange(0, amax(map) + 1, 1);
	sum_lst = [];
	for i in glob_idx:
		sum_lst.append(where(map == i)[0].tolist());
	return sum_lst;


def NonLinearStiff_Matrix(u, v, mass_M_inv, stiff_M1_non_lin, stiff_M2_non_lin, map2glob, sum_lst, mesh, BC):
	u_hat = u[map2glob];
	v_hat = v[map2glob];
	#print(amax(u_hat), amin(u_hat));
	shp = shape(u_hat);
	U1 = zeros((shp[0], shp[1], shp[1], shp[2]));
	U1[:, 0] = u_hat*repeat(u_hat[:, 0], 3).reshape(shape(u_hat));
	U1[:, 1] = u_hat*repeat(u_hat[:, 1], 3).reshape(shape(u_hat));
	U1[:, 2] = u_hat*repeat(u_hat[:, 2], 3).reshape(shape(u_hat));

	V1 = zeros(shape(U1));
	V1[:, 0] = v_hat*repeat(u_hat[:, 0], 3).reshape(shape(u_hat));
	V1[:, 1] = v_hat*repeat(u_hat[:, 1], 3).reshape(shape(u_hat));
	V1[:, 2] = v_hat*repeat(u_hat[:, 2], 3).reshape(shape(u_hat));

	U2 = zeros(shape(U1));
	U2[:, 0] = u_hat*repeat(v_hat[:, 0], 3).reshape(shape(u_hat));
	U2[:, 1] = u_hat*repeat(v_hat[:, 1], 3).reshape(shape(u_hat));
	U2[:, 2] = u_hat*repeat(v_hat[:, 2], 3).reshape(shape(u_hat));

	V2 = zeros(shape(U1));
	V2[:, 0] = v_hat*repeat(v_hat[:, 0], 3).reshape(shape(u_hat));
	V2[:, 1] = v_hat*repeat(v_hat[:, 1], 3).reshape(shape(u_hat));
	V2[:, 2] = v_hat*repeat(v_hat[:, 2], 3).reshape(shape(u_hat));
	

	A = matmul(stiff_M1_non_lin, U1);
	A = sum(A, axis = 1);
	B = matmul(stiff_M2_non_lin, V1);
	B = sum(B, axis = 1);
	C = matmul(stiff_M1_non_lin, U2);
	C = sum(C, axis = 1);
	D = matmul(stiff_M2_non_lin, V2);
	D = sum(D, axis = 1);

	convection_ux = AssembleFEM_nonLinVect(A, sum_lst);
	convection_uy = AssembleFEM_nonLinVect(B, sum_lst);
	convection_vx = AssembleFEM_nonLinVect(C, sum_lst);
	convection_vy = AssembleFEM_nonLinVect(D, sum_lst);

	convection_ux = mass_M_inv.dot(convection_ux);
	convection_uy = mass_M_inv.dot(convection_uy);
	convection_vx = mass_M_inv.dot(convection_vx);
	convection_vy = mass_M_inv.dot(convection_vy);

	mesh.ApplyBoundaryConditionsVect(convection_ux, BC);
	mesh.ApplyBoundaryConditionsVect(convection_uy, BC);
	mesh.ApplyBoundaryConditionsVect(convection_vx, BC);
	mesh.ApplyBoundaryConditionsVect(convection_vy, BC);

	return convection_ux, convection_uy, convection_vx, convection_vy;

def AssembleFEM_nonLinVect(elemVect, sum_lst):
	vect = elemVect.reshape(shape(elemVect)[0]*shape(elemVect)[1]);
	globVect = asarray([sum(vect[i] for i in indices) for indices in sum_lst]);
	return globVect;

def WaveEquation(mesh, c2, x_int_mesh, y_int_mesh, w_int_stdtri, IC_v, IC_w, BC, Verif_Source):
	mass_M = zeros((mesh.nVert, mesh.nVert));
	stiff_M = zeros((mesh.nVert, mesh.nVert));
	S_Vect = zeros((mesh.nVert, 1));
	for element in range(mesh.nElem):
		pbf(element, mesh.nElem, "Assembeling Matrix");
		elem = MeshElementFEM(mesh, element);
		mass_elemMatrix = zeros((elem.N, elem.N));
		stiff_elemMatrix1 = zeros((elem.N, elem.N));
		S_elemVect = zeros((elem.N, 1));
		if element == 0:
			IC_v = elem.ApplyInitialCondition(IC_v);
			IC_w = elem.ApplyInitialCondition(IC_w);
	
		elem.Transform_to_stdtri();
		J = elem.Jacobian;
		J_inv = elem.Jacobian_inv;
		j_c = elem.Jacobian_c;
		det_J = det(J);
		ksi = J[0][0]*x_int_mesh + J[0][1]*y_int_mesh + j_c[0];			eta = J[1][0]*x_int_mesh + J[1][1]*y_int_mesh + j_c[1];
		phi, dphi = elem.ShapeFunctions(ksi, eta);
		for i in range(elem.N):
			if Verif_Source != False:
				S_elemVect[i] = (phi[i]*Verif_Source(ksi, eta)*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
			for j in range(elem.N):
				mass_elemMatrix[i, j] = (phi[i]*phi[j]*det_J).dot(w_int_stdtri).dot(w_int_stdtri);
				phi_dxphi = (phi[i]*dphi[j][0]);		phi_dyphi = (phi[i]*dphi[j][1]);
				intx_part1 = (phi_dxphi[:, -1] - phi_dxphi[:, 0]);
				inty_part2 = (phi_dyphi[:, -1] - phi_dyphi[:, 0]);
				stiff_elemMatrix1[i, j] = c2*((intx_part1*det_J).dot(w_int_stdtri) - (dphi[i][0]*dphi[j][0]*det_J).dot(w_int_stdtri).dot(w_int_stdtri)	
											+ (inty_part2*det_J).dot(w_int_stdtri) - (dphi[i][1]*dphi[j][1]*det_J).dot(w_int_stdtri).dot(w_int_stdtri));
		elem.Assemble_FEM_Matrix(mass_elemMatrix, mass_M);
		elem.Assemble_FEM_Matrix(stiff_elemMatrix1, stiff_M);
		elem.Assemble_FEM_Vector(S_elemVect, S_Vect);
	print("\nInverting Matrix to compute A matrix ...");
	mass_M_inv = (inv(mass_M));
	A = (mass_M_inv.dot(stiff_M));
	S = mass_M_inv.dot(S_Vect);
	print("Done");
	print("\nApplying Boundary Conditions ...");
	mesh.ApplyBoundaryConditions(A, BC);
	print("Done");
	return IC_v, IC_w, mass_M, stiff_M, A, S.reshape(1, len(S))[0];


#def BurgersEquationFEM_Matrix_SYM(mesh, mu, x_int_mesh, y_int_mesh, w_int_stdtri, IC, BC):
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
#			ICu = elem.ApplyInitialCondition(IC);
#			ICv = elem.ApplyInitialCondition(IC);

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
	
#	print("\nInverting Matrices ...");
#	mass_M_inv = sparse(inv(mass_M));
#	diffusion_M = mass_M_inv.dot(stiff_M3);
#	print("Done");
#	print("\nApplying Boundary Conditions ...");
#	mesh.ApplyBoundaryConditions(diffusion_M, BC);
#	print("Done");
#	print("\nFinal assembly of matrices ...");
#	convection_u_M_func = syp.lambdify([a_sym], (stiff_M1), "numpy");
#	convection_v_M_func = syp.lambdify([a_sym], (stiff_M2), "numpy");
#	print("Done");
#	return ICu, ICv, (mass_M_inv), sparse(diffusion_M), convection_u_M_func, convection_v_M_func;

#def Update_convection_M(a, b, mesh, mass_M_inv, BC, convection_u_M_func, convection_v_M_func):
#	#A = (mass_M_inv @ ((asarray(convection_u_M_func(a), dtype = float))));
#	#B = (mass_M_inv @ ((asarray(convection_v_M_func(a), dtype = float))));
#	#C = (mass_M_inv @ ((asarray(convection_u_M_func(b), dtype = float))));
#	#D = (mass_M_inv @ ((asarray(convection_v_M_func(b), dtype = float))));
#	A = sparse(mass_M_inv.dot((asarray(convection_u_M_func(a), dtype = float))));
#	B = sparse(mass_M_inv.dot((asarray(convection_v_M_func(a), dtype = float))));
#	C = sparse(mass_M_inv.dot((asarray(convection_u_M_func(b), dtype = float))));
#	D = sparse(mass_M_inv.dot((asarray(convection_v_M_func(b), dtype = float))));
#	mesh.ApplyBoundaryConditions(A, BC);
#	mesh.ApplyBoundaryConditions(B, BC);
#	mesh.ApplyBoundaryConditions(C, BC);
#	mesh.ApplyBoundaryConditions(D, BC);
#	return A, B, C, D;
