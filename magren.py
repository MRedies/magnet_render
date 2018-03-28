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
        r, g, b, _ = cmap(theta / np.pi)
    else:
        r, g, b, _ = cmap(norm_scalar)
    # r *= 1 - theta / np.pi
    # b *= 1 - theta / np.pi
    # g *= 1 - theta / np.pi

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

def get_args():
    parser = argparse.ArgumentParser(description='POVray everything!')
    parser.add_argument("-f", "--folder",
                        dest="folder", type=str, default='./',
                        help="Where do look for and save data")
    parser.add_argument("-c", "--colormap",
                        dest="cmap", type=str, default='viridis',
                        help="What colormap shall I use")
    parser.add_argument("-d", "--dist",
                        dest="dist", type=float, default=20.0,
                        help="Camera distance")
    parser.add_argument("-b", "--brightness",
                        dest="brightness", type=float, default=1.0,
                        help="Camera distance")
    parser.add_argument("-x", "--x_shift",
                            dest="x_shift", type=float, default=0.0,
                            help="Camera position shift in x")
    parser.add_argument("-y", "--y_shift",
                            dest="y_shift", type=float, default=0.5,
                            help="Camera position shift in y")
    parser.add_argument("-z", "--z_shift",
                            dest="z_shift", type=float, default=0.7,
                            help="Camera position shift in z")
    parser.add_argument("--noshow", 
                        help="Show rendering at the end",action="store_true")
    parser.add_argument("-W", "--width",
                            dest="width", type=int, default=800,
                            help="width of rendered image")
    parser.add_argument("-H", "--height",
                            dest="height", type=int, default=800,
                            help="height of rendered image")

    return parser.parse_args()

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

def show_img(folder):
    _, ax = plt.subplots(1,1, figsize=(15,15))
    img=mpimg.imread(folder + 'render.png')
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

    centeroid  = (x1 + x2 + x3)/3
    
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

def create_isosurf(folder):
    vert, tri, phi, _ = get_isosurf(folder)
    objects = []
    cmap = plt.get_cmap("plasma")
    phi += np.pi
    phi /= 2 * np.pi 

    
    for i in range(tri.shape[0]):
        r, g, b, _ = cmap(phi[i])  
        objects.append(Triangle(vert[tri[i,0],:],
                                vert[tri[i,1],:],
                                vert[tri[i,2],:],
                                Texture( Pigment( 'color', [r,g,b]))  ))
    return objects
        


def create_isomesh(folder, vis_area=Box([0,0,0], [0, 0, 0], "inverse")):
    vert, tri,_, phi = get_isosurf(folder)
    nv              = get_normal_vectors(vert, tri)

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


args     = get_args()
folder   = args.folder
dist     = args.dist
cmap     = args.cmap


shift    = []
shift.append(args.x_shift)
shift.append(args.y_shift)
shift.append(args.z_shift)


X, Y, Z, PHI, THETA = get_pos_angle(folder)

middle_x, middle_y, middle_z = calc_middle(X,Y,Z)

objects = [
    # SUN
    LightSource([0,0,500], 'color', args.brightness),
    LightSource([1500,2500,2500], 'color', args.brightness),
    LightSource([-1500,2500,2500], 'color', args.brightness),
    LightSource([1500,-2500,2500], 'color', args.brightness),
    LightSource([-1500,-2500,2500], 'color', args.brightness),
    Background("color", [1, 1,1]),
]

box   = 100.3
lower = [0,   0,   0]
upper = [100, 100, 100]

for x,y,z,p,t in zip(X, Y, Z, PHI, THETA):
    vec = np.array([x,y,z])
    if(not(np.all(np.logical_and(vec >= lower, vec <= upper)))):
        objects.append(spin(vec, t, p, l=0.6, cmap=plt.get_cmap(cmap)))


vis_area = Box(lower, upper, "inverse")
objects.append(create_isomesh(folder, vis_area=vis_area))


scene = Scene( Camera( 'perspective',
                       'angle',       45,
                       'location',    [middle_x + shift[0] * dist,
                                       middle_y + shift[1] * dist,
                                       middle_z + shift[2] * dist],
                       'look_at',     [middle_x,        middle_y, middle_z],
                       'sky',         [0,             0  ,        1],
                     ),

               objects= objects,
               included=['colors.inc']
              )

scene.render(folder + 'render.png', remove_temp=False, width=args.width,
             height=args.height, antialiasing=0.00001,quality=10)


if(not args.noshow):
    show_img(folder)


