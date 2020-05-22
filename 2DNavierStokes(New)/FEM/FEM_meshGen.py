from numpy import*
from matplotlib import pyplot as plt
import matplotlib.tri as mtri
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D

def Transformation(global_coord):
	origin_idx = 0;
	Ori = global_coord[origin_idx, :];
	A = array([[global_coord[1][0] - Ori[0], global_coord[2][0] - Ori[0]], 
				[global_coord[1][1] - Ori[1], global_coord[2][1] - Ori[1]]]);

	A_inv = inv(A);

	local_coord = zeros_like(global_coord);
	row = 0;
	for coord in global_coord:
		if where(global_coord == coord) != origin_idx:
			local_coord[row] = A_inv.dot(coord - Ori);
			row += 1;
	return local_coord;

def LocalBasis(x, y):
	f1 = 1 - x - y;
	f2 = x;
	f3 = y;
	return f1, f2, f3;

def meshgrid_to_tri_mesh(x_mesh, y_mesh):
	f = lambda x: 1 - x;
	bool_idx = y_mesh > f(x_mesh);
	y_mesh[bool_idx] = 0;
	x_mesh[bool_idx] = 0;
	return x_mesh, y_mesh;

nodes = array([[1., 2.001],
			   [5.9, 6.2],
			   [8., 1.2]]);
local_nodes = Transformation(nodes);
triangles = array([[0, 1, 2]]);



trig = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles = triangles);
trig_local = mtri.Triangulation(local_nodes[:, 0], local_nodes[:, 1], triangles = triangles);
plt.triplot(trig, marker = "o");
plt.triplot(trig_local, marker = "o");
plt.grid(True);

z0 = zeros((1, 3))[0];
z = array([1., 0., 4.]);
fig, ax = plt.subplots(subplot_kw = dict(projection = "3d"));
#ax.plot_trisurf(trig, z);
#ax.plot_trisurf(trig, z0);
#ax.plot_trisurf(trig_local, z0);
x = linspace(0, 1, 10);
x_mesh, y_mesh = meshgrid(x, x);
x_mesh, y_mesh = meshgrid_to_tri_mesh(x_mesh, y_mesh);
f1, f2, f3 = LocalBasis(x_mesh, y_mesh);
ax.plot_surface(x_mesh, y_mesh, f1, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
ax.plot_surface(x_mesh, y_mesh, f2, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
ax.plot_surface(x_mesh, y_mesh, f3, rstride = 1, cstride = 1,
				cmap = 'viridis', edgecolor = 'none');
plt.show();
