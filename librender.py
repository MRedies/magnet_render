#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import argparse
from scipy.interpolate import griddata
from vapory import *
import mcubes

def spin(center, theta, phi, l=1, cmap=plt.get_cmap("viridis"), norm_scalar=None):
    vec = np.array([l * np.sin(theta) * np.cos(phi),
                    l * np.sin(theta) * np.sin(phi),
                    l * np.cos(theta)])

    top    = center + 0.5*vec
    bottom = center - 0.5*vec

    if(norm_scalar is None):
        r, g, b, _ = cmap((theta / np.pi))
    else:
        r, g, b, _ = cmap(norm_scalar)
    
    return Cone(bottom, 0.25 * l, top, 0.0, Texture( Pigment( 'color', [r,g,b])))

def arrow(center, theta, phi, l=1, color=[1.0,0.65, 0.0]):
    vec = np.array([l * np.sin(theta) * np.cos(phi),
                    l * np.sin(theta) * np.sin(phi),
                    l * np.cos(theta)])

    top    = center + vec
    bottom = center - vec

    return Union( Cone(center, 0.25*l, top,    0.0),
                 Cone(bottom, 0.12*l, center, 0.12),
                 Texture( Pigment( 'color', color)),
                 Finish( 'phong', 0.5)
                 )

def axis(c):
    return Union(arrow([c[0]+1, c[1],   c[2]],   np.pi/2, 0,       color=[1, 0, 0]),
            arrow([c[0],   c[1]+1, c[2]],   np.pi/2, np.pi/2, color=[0, 1, 0]),
            arrow([c[0],   c[1],   c[2]+1], 0.0,     0.0,     color=[0, 0, 1])
            )

def norm(x):
    x -= np.min(x)
    x /= np.max(x)
    return x



def get_pos_angle(folder):
    X = np.load(folder + "pos_x.npy")
    Y = np.load(folder + "pos_y.npy")
    Z = np.load(folder + "pos_z.npy")
    PHI = np.load(folder + "m_phi.npy")
    THETA = np.load(folder + "m_theta.npy")

    return X,Y,Z, PHI, THETA

def calc_middle(X,Y,Z):
    middle_x = 0.5 * (np.max(X) + np.min(X))
    middle_y = 0.5 * (np.max(Y) + np.min(Y))
    middle_z = 0.5 * (np.max(Z) + np.min(Z))
    return middle_x, middle_y, middle_z

def show_img(folder, file='render.png'):
    _, ax = plt.subplots(1,1, figsize=(15,15))
    img=mpimg.imread(folder + file)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def get_isosurf(folder):
    X,Y,Z, phi, theta = get_pos_angle(folder)
    dim_x = np.unique(X).shape[0]
    dim_y = np.unique(Y).shape[0]
    dim_z = np.unique(Z).shape[0]
    m_z = np.cos(theta)

    m_z = m_z.reshape((dim_x, dim_y, dim_z), order="F")

    vert, tri = mcubes.marching_cubes(m_z, 0)
    vert[:,0] = translate_grids(vert[:,0], X, dim_x)
    vert[:,1] = translate_grids(vert[:,1], Y, dim_y)
    vert[:,2] = translate_grids(vert[:,2], Z, dim_z)
    
    data_points = np.zeros((X.shape[0], 3))
    data_points[:,0] = X
    data_points[:,1] = Y
    data_points[:,2] = Z
    phi_vert = griddata(data_points, phi, vert, method='nearest')
    phi_tri  = triangle_phi(tri, phi_vert)
    
    return vert, tri, phi_tri, phi_vert

def triangle_phi(tri, phi_vert):
    phi_tri = np.zeros(tri.shape[0])
    for i in range(tri.shape[0]):
        phi_tri[i] = np.mean([phi_vert[tri[i,0]], 
                              phi_vert[tri[i,1]],
                              phi_vert[tri[i,2]]])
    return phi_tri

def translate_grids(pts, X, dim_x):
    u_X  = np.unique(X)
    dx = 1.0
    dy = u_X[1] - u_X[0]
    m  = dy/dx
    b  = np.min(u_X)

    return m * pts + b

def get_norm_tri(vert, tri, idx):
    x1 = vert[tri[idx,0],:]
    x2 = vert[tri[idx,1],:]
    x3 = vert[tri[idx,2],:]
    
    v1 = x1 - x2
    v2 = x1 - x3
    n  = np.cross(v1,v2)
    
    n /= np.linalg.norm(n)

    return n

def get_norm_vert(vert, tri):
    vert_norm = np.zeros((vert.shape[0],3))

    for i in range(tri.shape[0]):
        n = get_norm_tri(vert, tri, i)
        for j in range(3):
            vert_norm[tri[i,j],:] += n
    norm = np.linalg.norm(vert_norm, axis=1)

    for i in range(3):
        vert_norm[:,i] /= norm
    
    return vert_norm

def get_normal_vectors(vert, tri):
    norm_vec = get_norm_vert(vert, tri)

    vec_list = []

    for i in range(norm_vec.shape[0]):
        vec_list.append(norm_vec[i,:].tolist())
    
    return NormalVectors(len(vec_list), *vec_list)

def create_isomesh(folder, vis_area=Box([0,0,0], [0, 0, 0], "inverse")):
    vert, tri,_, phi = get_isosurf(folder)
    nv               = get_normal_vectors(vert, tri)

    norm = matplotlib.colors.Normalize(
    vmin=np.min(np.min(phi)),
    vmax=np.max(np.max(phi)))
    c_m = matplotlib.cm.hsv

    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    vertis = []
    for i in range(vert.shape[0]):
        vertis.append(vert[i,:].tolist())


    tris = []
    for i in range(tri.shape[0]):
        tris.append(tri[i,:].tolist())
        tris.append(tri[i,0])
        tris.append(tri[i,1])
        tris.append(tri[i,2])

    textures = []
    for p in phi:
        r, g, b,_ = s_m.to_rgba(p) 
        textures.append( Texture( Pigment( 'color', [r,g,b])))

    f_ind = FaceIndices(tri.shape[0], *tris)
    vv    = VertexVectors(len(vertis), *vertis)
    tl    = TextureList(len(textures), *textures)



    objects = Mesh2(vv, nv, tl, f_ind, ClippedBy(vis_area))
    return objects

def plot_3d(points, color=[0.16,0.48,0.14], radius=1):
    cylinder_list = []

    for i in range(points.shape[0]-1):
        cylinder_list.append(Cylinder(points[i,:], points[i+1,:], radius, Pigment( 'color', color)))

    return cylinder_list

def plot_ring(center, radius, z=1.0, steps=300, 
              color=[13./255.,142./255.,7./255.], tube_radius=1):
    x = np.linspace(0, 2*np.pi, steps)

    points      = np.zeros((steps, 3))
    points[:,0] = radius          * np.cos(x) - center[0]
    points[:,1] = radius          * np.sin(x) - center[1]
    points[:,2] = z



    return plot_3d(points, color=color , radius=tube_radius)


def plot_border(X,Y, Theta, color=np.array([226.0, 178.0, 18.0])/256.0):
    R = np.sqrt(X**2 + Y**2)
    z = np.cos(Theta)

    less = np.where(z < 0.0)
    more = np.where(z> 0.0)
    r = 0.5 * (np.max(R[less]) + np.min(R[more]))

    return Cone([0,0,-1], r, [0,0,1], r, "open", Texture( Pigment( 'color', color)))


def get_LDOS(fol, E_range):
    E = np.load (fol + "DOS_E.npy")
    PDOS = np.load(fol + "DOS_partial.npy")
    
    sel = np.where(np.logical_and(E >= E_range[0], E <= E_range[1]))
    PDOS = PDOS[:,sel][:,0,:]
    PDOS = np.sum(PDOS, axis=1)

    N = PDOS.shape[0]/6
    LDOS = np.zeros(N)

    for n,i in enumerate(range(0, PDOS.shape[0]/2, 3)):
        LDOS[n] = np.sum(PDOS[i:i+3]) + np.sum(PDOS[i+N/2:i+N/2+3])
    return LDOS

def dbl_arrow(base, top, width=1.0, color=[0,0.3,0]):
    conn    = top - base
    n_conn  = conn     / np.linalg.norm(conn)
    meet1   = top - 2 * width * n_conn
    meet2   = base     + 2 * width * n_conn

    obj = []
    obj.append(Cone(meet2, 1.3*width, base, 0.0, Texture( Pigment( 'color', color))))
    obj.append(Cylinder(meet2, meet1, 0.8*width, Texture( Pigment( 'color', color))))
    obj.append(Cone(meet1, 1.3*width, top, 0.0, Texture( Pigment( 'color', color))))

    return obj

def measure(base, top, shift, width=0.2, color=[0,0.3,0]):
    obj = []

    conn   = top - base
    n_conn = conn / np.linalg.norm(conn) 

    arrow_base = base + shift + (0.8 * width * n_conn) 
    arrow_top  = top  + shift - (0.8 * width * n_conn)

    obj.append(Cylinder(top,  top  + 1.1*shift, 0.8*width, Texture( Pigment( 'color', color))))
    obj.append(Cylinder(base, base + 1.1*shift, 0.8*width, Texture( Pigment( 'color', color))))
    obj.extend(dbl_arrow(arrow_base, arrow_top, width=width, color=color))

    return obj