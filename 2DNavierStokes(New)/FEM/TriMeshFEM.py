from numpy import*
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from NumericalTools import integrate
import sympy as syp
import time

class TriMesh():
	def __init__(self, x1, x2, y1, y2):
		self.x1 = x1;											# Left boundary position
		self.x2 = x2;											# Right boundary position
		self.y1 = y1;											# lower boundary position
		self.y2 = y2;											# Upper boundary position
		self.dlx = 1.5;											# Delamination x position
		self.dly = 0.5;											# Delamination x position
		self.dlr = 0.4;											# Delamination radius
		self.bnx = 5;											# Background elements in x
		self.bny = 5;											# Background elements in y
		self.dx = (self.x2 - self.x1)/float(self.bnx - 1);		# Mesh spacing in x
		self.dy = (self.y2 - self.y1)/float(self.bny - 1);		# Mesh spacing in y
		self.minTriArea = (self.dx + self.dy)/10000.;			# Minimum triangle size
		self.refine = 1;										# Mesh refinement factor

		self.vertices = None;									# Mesh Vertices as list
		self.vertArray = None;									# Mesh Vertices as array
		self.elements = None;									# Mesh Elements
		self.elemArray = None;									# Mesh Elements as array
		self.nVert = None;										# Number of vertArray
		self.nElem = None;										# Number of elements
		self.dtri = None;										# Delaunay triangulation
		self.mask = None;										# Triangle mask
		self.leftVI = None;										# Left boundary vertex list
		self.rightVI = None;									# Right boundary vertex list
		self.lowerVI = None;									# Lower boundary vertex list
		self.upperVI = None;									# Upper boundary vertex list

	#=========================================================
	# Defines triangles removed from Delaunay triangulation
	#=========================================================
	def triMask(self, triangles):
		out = [];
		self.elements = [];
		for points in triangles:
			a, b, c = points;
			va = self.vertices[a];
			vb = self.vertices[b];
			vc = self.vertices[c];
			x1 = float(va[0]);	y1 = float(va[1]);
			x2 = float(vb[0]);	y2 = float(vb[1]);
			x3 = float(vc[0]);	y3 = float(vc[1]);
			Ae = 0.5*(x2*y3 + x1*y2 + x3*y1 - x3*y2 - x1*y3 - x2*y1);
			#if (Ae<self.minTriArea):
			if (Ae == 0):
				out.append(True);
			else: 
				out.append(False);
				self.elements.append(points);

		return out;

	#=========================================================
	# Loads the vertices and elements (call after setting
	# the desired parameters)
	#=========================================================
	def loadMesh(self, n, dlx, dly, dlr): 
		self.bnx *= n;
		self.bny *= n;
		self.dlx = dlx;
		self.dly = dly;
		self.dlr = dlr;
		self.refine = n;

		# Load background mesh
		xb = zeros(self.bnx + 1);
		yb = zeros(self.bnx + 1);
		nb = zeros(self.bny + 1);
		xb[:] = linspace(self.x1, self.x2, self.bnx + 1);
		#yb[:] = linspace(self.y1, self.y2, self.bnx+1)
		nb[:] = linspace(0., self.y2, self.bny + 1)
		yb[:] = 1;
		#yref = 0.1*sin(1.5*pi*2/(self.x2 - self.x1));
		#for i in range(self.bnx + 1): 
		#	if (xb[i] < 2):
		#		yb[i] = 1. + 0.1*sin(1.5*pi*xb[i]/(self.x2 - self.x1));
		#	else:
		#		yb[i] = 1. + yref - 0.2*(xb[i] - 2.)/(self.x2 - 2.);

		self.vertices = [];
		self.leftVI = [];
		self.rightVI = [];
		self.lowerVI = [];
		self.upperVI = [];
		vi = 0;
		for j in range(self.bny + 1):
			for i in range(self.bnx + 1):
				if (i == 0):			self.leftVI.append(vi);
				if (i == self.bnx):		self.rightVI.append(vi);
				if (j == 0):			self.lowerVI.append(vi);
				if (j == self.bny):		self.upperVI.append(vi);
				self.vertices.append((xb[i], nb[j]*yb[i]));
				vi += 1;
		self.vertices = asarray(self.vertices);

		# Enrich near delamination
		if (dlx > 1.4):
			self.addDelam(5*n, 10*n);

		self.nVert = len(self.vertices);
		self.vertArray = asarray(self.vertices);
		#   self.smoothVert(2,0.01);

		# Use Delaunay triangulation and mask bnd-only elements
		self.dtri = tri.Triangulation(self.vertArray[:, 0], self.vertArray[:, 1]);
		self.mask = self.triMask(self.dtri.triangles);
		self.dtri.set_mask(self.mask);


		self.nElem = len(self.elements);
		self.elemArray = asarray(self.elements);

	def ApplyBoundaryConditions(self, A, BC):
		if BC == "Periodic":
			A[self.leftVI, :] = A[self.rightVI, :];
			A[self.lowerVI, :] = A[self.upperVI, :];
	
		elif BC == "Dir":
			A[self.leftVI, :] = 0;
			A[self.rightVI, :] = 0;
			A[self.lowerVI, :] = 0;
			A[self.upperVI, :] = 0;

	#=========================================================
	# Plots the mesh
	#=========================================================
	def plotMesh(self):
		fig = plt.figure(figsize = (18, 8));
		ax = fig.add_subplot(111);
		ax.set_title("Mesh");
		ax.set_xlabel('x', size = 14, weight = 'bold');
		ax.set_ylabel('y', size = 14, weight = 'bold');
		plt.axes().set_aspect('equal', 'datalim');
		xy = asarray(self.vertices);
		plt.triplot(xy[:, 0], xy[:, 1], self.elements, 'bo-');
		plt.show();

	def plotSoln(self, solVec, fig, title=""):
		try:
			figure = plt.figure(fig.number);
		except:
			figure = plt.figure(figsize = (18, 8));
		plt.clf();
		ax = figure.add_subplot(111);
		plt.axes().set_aspect('equal', 'datalim');
		xy = asarray(self.vertices);
		if xy.size < 10000:
			plt.triplot(xy[:, 0], xy[:, 1],self.elements, 'b-', linewidth = 0.5);
		vals = plt.tricontourf(self.dtri, solVec, cmap = "jet");
		plt.colorbar();
		return vals

class MeshElementFEM():
	def __init__(self, mesh, element_idx):
		self.N = 3;
		self.mesh = mesh;
		self.vertices_idx = mesh.elements[element_idx];
		self.v1 = mesh.vertices[self.vertices_idx[0]];
		self.v2 = mesh.vertices[self.vertices_idx[1]];
		self.v3 = mesh.vertices[self.vertices_idx[2]];
		self.idx_i, self.idx_j = meshgrid(self.vertices_idx, self.vertices_idx);

		x1 = self.v1[0];		y1 = self.v1[1];
		x2 = self.v2[0];		y2 = self.v2[1];
		x3 = self.v3[0];		y3 = self.v3[1];

		self.v_array = vstack((self.v1, self.v2, self.v3));

		self.Area = 0.5*(x2*y3 + x1*y2 + x3*y1 - x3*y2 - x1*y3 - x2*y1);
		self.phi1_c = x2*y3 - x3*y2;		self.phi1_x = y2 - y3;			self.phi1_y = x3 - x2;
		self.phi2_c = x3*y1 - x1*y3;		self.phi2_x = y3 - y1;			self.phi2_y = x1 - x3;
		self.phi3_c = x1*y2 - x2*y1;		self.phi3_x = y1 - y2;			self.phi3_y = x2 - x1;
		
		self.Jacobian = None;
		self.Jacobian_inv = None;
		self.Jacobian_c = None;

	def Transform_to_stdtri(self):
		origin_idx = 0;
		Ori = self.v_array[origin_idx, :];
		A = array([[self.v_array[1][0] - Ori[0], self.v_array[2][0] - Ori[0]],
				   [self.v_array[1][1] - Ori[1], self.v_array[2][1] - Ori[1]]]);
		self.Jacobian = A;
		#self.Jacobian_inv = inv(A);
		self.Jacobian_c = Ori;
		#A_inv = inv(A);

		#local_coord = zeros_like(self.v_array);
		#row = 0;
		#for coord in self.v_array:
		#	if where(self.v_array == coord) != origin_idx:
		#		local_coord[row] = A_inv.dot(coord - Ori);
		#		row += 1;
		#return local_coord;


	def ShapeFunctions(self, x, y):
		phi = zeros((3, x.shape[0], x.shape[1]));
		phi[0] = 1/(2*self.Area)*(self.phi1_c + self.phi1_x*x + self.phi1_y*y);
		phi[1] = 1/(2*self.Area)*(self.phi2_c + self.phi2_x*x + self.phi2_y*y);
		phi[2] = 1/(2*self.Area)*(self.phi3_c + self.phi3_x*x + self.phi3_y*y);

		dphi = zeros((3, 2, x.shape[0], x.shape[1]));
		m = ones_like(x);
		dphi[0, 0] = 1/(2*self.Area)*self.phi1_x*m;		dphi[0, 1] = 1/(2*self.Area)*self.phi1_y*m;
		dphi[1, 0] = 1/(2*self.Area)*self.phi2_x*m;		dphi[1, 1] = 1/(2*self.Area)*self.phi2_y*m;
		dphi[2, 0] = 1/(2*self.Area)*self.phi3_x*m;		dphi[2, 1] = 1/(2*self.Area)*self.phi3_y*m;
		return phi, dphi;

	def Assemble_FEM_Matrix(self, elemMat, gloMat):
		idx = array([[self.vertices_idx[0], self.vertices_idx[0], self.vertices_idx[0], 
					  self.vertices_idx[1], self.vertices_idx[1], self.vertices_idx[1], 
					  self.vertices_idx[2], self.vertices_idx[2], self.vertices_idx[2]], 
					 [self.vertices_idx[0], self.vertices_idx[1], self.vertices_idx[2], 
					  self.vertices_idx[0], self.vertices_idx[1], self.vertices_idx[2],
					  self.vertices_idx[0], self.vertices_idx[1], self.vertices_idx[2]]]);
		gloMat[idx[0], idx[1]] += elemMat.reshape(1, len(elemMat)**2)[0];

	def Assemble_FEM_Vector(self, elemVect, gloVect):
		idx = array([self.vertices_idx[0], self.vertices_idx[1], self.vertices_idx[2]]);
		gloVect[idx] += elemVect;

	def ApplyInitialCondition(self, f):
		a = f(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1]);
		return a;