#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import argparse
from scipy.interpolate import griddata
from vapory import *

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

brightness = 0.3
width      = 800
height     = 600

objects = [
    # SUN
    LightSource([0,0,500], 'color', brightness),
    LightSource([0,0,-500], 'color', brightness),
    LightSource([1500,2500,2500], 'color', brightness),
    LightSource([-1500,2500,2500], 'color', brightness),
    LightSource([1500,-2500,2500], 'color', brightness),
    LightSource([-1500,-2500,2500], 'color', brightness),
    Background("color", [1, 1,1]),
]

objects.append(Sphere([0,0,0], 3, Texture( Pigment( 'color', [102/255,204/255,255/255]))))
objects.append(Sphere([15,0,0], 1, Texture( Pigment( 'color', [255/255,0/255,0/255]))))
# objects.append(arrow([17,0,0], np.pi*0.5, np.pi*1.5))
#objects.extend(plot_ring([0,0,0], 15, tube_radius=0.2, z=0.0))

objects.append(arrow([15,0,2], 0.0, 0.0, l=3, color=[0,0,1]))

d = 25
scene = Scene( Camera( 'perspective',
                       'angle',       45,
                       'location',    [d, d, d],
                       'look_at',     [0.0, 0.0, 0.0],
                       'sky',         [0,             0  ,        1],
                     ),

               objects= objects,
               included=['colors.inc']
              )


filename = "spin_mag.png"
scene.render(filename, remove_temp=False, width=width,
             height=height, antialiasing=0.00001,quality=10)