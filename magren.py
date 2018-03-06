#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import argparse
from vapory import *

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
    parser.add_argument("-s", "--show",
                            dest="img_show", type=bool, default=True,
                            help="Show rendering at the end")
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
    f, ax = plt.subplots(1,1, figsize=(15,15))
    img=mpimg.imread(folder + 'render.png')
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()



args     = get_args()
folder   = args.folder
dist     = args.dist
cmap     = args.cmap
img_show = args.img_show
shift    = []
shift.append(args.x_shift)
shift.append(args.y_shift)
shift.append(args.z_shift)


X, Y, Z, PHI, THETA = get_pos_angle(folder)

middle_x, middle_y, middle_z = calc_middle(X,Y,Z)

objects = [
    # SUN
    LightSource([1500,2500,2500], 'color', args.brightness),
    LightSource([-1500,2500,2500], 'color', args.brightness),
    LightSource([1500,-2500,2500], 'color', args.brightness),
    LightSource([-1500,-2500,2500], 'color', args.brightness),
    Background("color", [1, 1,1]),
]

#objects.append(axis([0,0,24]))
box = 100.3
for x,y,z,p,t in zip(X, Y, Z, PHI, THETA):
    objects.append(spin([x,y,z], t, p, cmap=plt.get_cmap(cmap)))

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
             height=args.height, antialiasing=0.001)
if(img_show):
    show_img(folder)
