#! /usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import argparse
from scipy.interpolate import griddata
from vapory import *
import mcubes

def spin(center, vec):
    top    = center + 0.5*vec
    bottom = center - 0.5*vec

    r = 255.0
    g = 0.0
    b = 0.0

    l = np.linalg.norm(vec)
    
    return Cone(bottom, 0.25 * l, top, 0.0, Texture( Pigment( 'color', [r,g,b])))

def arrow(center, vec, color=[1.0,0.65, 0.0]):
   
    top    = center + vec
    bottom = center - vec

    l = np.linalg.norm(vec)

    return Union( Cone(center, 0.25*l, top,    0.0),
                 Cone(bottom, 0.12*l, center, 0.12*l),
                 Texture( Pigment( 'color', color)),
                 Finish( 'phong', 0.5)
                 )

def show_img(folder, file='render.png'):
    _, ax = plt.subplots(1,1, figsize=(15,15))
    img=mpimg.imread(folder + file)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


brightness = 1

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

alpha = np.load("/home/matthias/Thesis/figs/mixers/mix_dbl_tube/alpha.npy")
q     = np.load("/home/matthias/Thesis/figs/mixers/mix_dbl_tube/q.npy")
print(q.shape)

for i in range(alpha.shape[0]):
    vec    = q[i,:]
    center = np.array([0.0, -i*300.0, 0.0])

    objects.append(arrow(center, vec))


scene = Scene( Camera( 'perspective',
                       'angle',       45,
                       'location',    [4500,1000,1000],
                       "look_at", [0,-5*300,0],
                       'sky',         [0,             0  ,        1],
                     ),

               objects= objects,
               included=['colors.inc']
              )

scene.render("vis.png", remove_temp=False, width=1000,
             height=1000, antialiasing=0.00001,quality=10)


show_img("", file="vis.png")
