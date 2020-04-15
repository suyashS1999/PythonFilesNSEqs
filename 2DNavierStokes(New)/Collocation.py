from numpy import*
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv, det, norm
from basis import BasisFunctions
import time

#x0 = -1; x1 = 1;								# Domain dimentions
#y0 = -1; y1 = 1;
#n = 40;											# Domain division
#x = linspace(x0, x1, n);
#y = linspace(y0, y1, n);
#x_mesh, y_mesh = meshgrid(x, y);

def RBF(r, bool):
	""" Function to define Radial Basis Function
	Input Arguments:
		r = distance between two nodes (float or numpy array)
	Output:
		basis = Value of basis function
	"""
	e = 260;
	if bool[0] == "basis":
		basis = sqrt(1 + (e*r)**2);
	elif bool[0] == "diff":
		basis = (e**2*bool[1])/sqrt(1 + (e*r)**2);
	elif bool[0] == "diff^2":
		basis = (e**4*bool[1]**2 + e**2)/(e**2*r**2 + 1)**(3/2);
	elif bool[0] == "2diff":
		basis = (e**4*r**2 + e**2)/(e**2*r**2 + 1)**(3/2);
	return basis;

def RBF_Matrix(nodes, bool):
	""" Function to Generate Matrix for RBF interpolation
	Input Arguments:
		nodes = N by 2 array containing interpolation nodes, in the form [x, y] in each row (numpy array)
	Output:
		A = Matrix A
	"""
	print("Generating Matrix for RBF interpolation");
	ts = time.time();
	Norm = lambda x, y: sqrt(x**2 + y**2);
	X, Y = meshgrid(nodes[:, 0], nodes[:, 1]);
	dX = X.T - X;
	dY = Y - Y.T;
	A = RBF(Norm(dX, dY), bool);
	print("That took: ", time.time() - ts);
	return A;

def Boundary_Conditions(A, S, nodes, boundaries, BC_type, BC_val):
	idx_b1 = where(nodes[:, 0] == boundaries[0])[0];
	idx_b2 = where(nodes[:, 0] == boundaries[1])[0];
	idx_b3 = where(nodes[:, 1] == boundaries[2])[0];
	idx_b4 = where(nodes[:, 1] == boundaries[3])[0];
	boundary_idx = concatenate((idx_b1, idx_b2, idx_b3, idx_b4));
	boundary_nodes = nodes[boundary_idx];
	Norm = lambda x, y: sqrt(x**2 + y**2);
	X, Y = meshgrid(nodes[:, 0], nodes[:, 1]);
	dX = X.T - X;
	dY = Y - Y.T;
	if BC_type == "dir":
		A[boundary_idx, :] = RBF(Norm(dX[boundary_idx, :], dY[boundary_idx, :]), ["basis"]);
	elif BC_type == "neumm":
		A[idx_b1, :] = RBF(Norm(dX[idx_b1, :], dY[idx_b1, :]), ["diff", nodes[idx_b1, 0].reshape(len(idx_b1), 1)]);
		A[idx_b2, :] = RBF(Norm(dX[idx_b2, :], dY[idx_b2, :]), ["diff", nodes[idx_b2, 0].reshape(len(idx_b2), 1)]);
		A[idx_b3, :] = RBF(Norm(dX[idx_b3, :], dY[idx_b3, :]), ["diff", nodes[idx_b3, 1].reshape(len(idx_b3), 1)]);
		A[idx_b4, :] = RBF(Norm(dX[idx_b4, :], dY[idx_b4, :]), ["diff", nodes[idx_b4, 1].reshape(len(idx_b4), 1)]);
	S[boundary_idx] = BC_val;
	return A, S, boundary_idx;


def Inter(a, x, y, nodes, bool):
	""" Function to evaluate RBF at desired location
	Input Arguments:
		a = Coefficient of each basis function (numpy array)
		x = x coordinate (numpy array)
		y = y coordinate (numpy array)
		nodes = Interpolation nodes
	Output:
		fi = Value of RBF at all x, y 's
	"""
	#print("Reconstructing Function from RBF");
	#ts = time.time();
	Norm = lambda x, y: sqrt(x**2 + y**2);
	ts = time.time();
	fi = zeros((len(x), len(y)));
	for i in range(len(x)):
		for j in range(len(y)):
			r = norm(array([x[i], y[j]]) - nodes, axis = 1);
			fi[i, j] = a.dot(RBF(r, bool));
	#print("That took: ", time.time() - ts);
	return fi.T;

#def f(x, y): return cos(2*pi*x)*cos(2*pi*y);
#S = f(x_mesh, y_mesh);
##S = -2*ones((1, n**2))[0];
#nodes = array([x_mesh.reshape(1, -1)[0], y_mesh.reshape(1, -1)[0]]).T;

#A = RBF_Matrix(nodes, ["2diff"]);
#A, S = Boundary_Conditions(A, S.reshape(1, -1)[0], nodes, [x0, x1, y0, y1], "neumm", 0);
#plt.imshow(A);
#plt.colorbar();
#p = inv(A).dot(S.reshape(1, -1)[0]);
#F = Inter(p, x, y, nodes, ["basis"]);

#fig = plt.figure();
#ax = plt.axes(projection = '3d');
#ax.plot_surface(x_mesh, y_mesh, F, rstride = 1, cstride = 1,
#				cmap = 'viridis', edgecolor = 'none');
#ax.set_xlabel('x');
#ax.set_ylabel('y');
#ax.set_zlabel('p');

#dF = Inter(p, x[1: -2], y[1: -2], nodes, ["2diff"]);
#fig = plt.figure(figsize = (10, 5));
#ax = fig.add_subplot(1, 2, 1, projection = '3d');
#ax.plot_surface(x_mesh[1: -2, 1: -2], y_mesh[1: -2, 1: -2], dF, rstride = 1, cstride = 1,
#				cmap = 'viridis', edgecolor = 'none');
##plt.plot(nodes[:, 0], nodes[:, 1], "x");
#ax.set_xlabel('x');
#ax.set_ylabel('y');
#ax.set_zlabel('d2pdx');
#ax = fig.add_subplot(1, 2, 2, projection = '3d');
#ax.plot_surface(x_mesh, y_mesh, f(x_mesh, y_mesh), rstride = 1, cstride = 1,
#				cmap = 'viridis', edgecolor = 'none');
#ax.set_xlabel('x');
#ax.set_ylabel('y');
#ax.set_zlabel('S(x, y)');
#plt.show();
