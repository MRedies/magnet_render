#!/usr/local/bin/python
from librender import *

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
    parser.add_argument("-s", "--spinsize",
                        dest="spin_sz", type=float, default=1.0,
                        help="Size of spins")
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
    parser.add_argument("--nomesh", 
                        help="show no mesh", action="store_true")
    parser.add_argument("--bounds", 
                        help="show ring around z=0 area", action="store_true")
    parser.add_argument("-W", "--width",
                            dest="width", type=int, default=800,
                            help="width of rendered image")
    parser.add_argument("-H", "--height",
                            dest="height", type=int, default=800,
                            help="height of rendered image")
    parser.add_argument("-o", "--outfile",
                        dest="outfile", type=str, default='render.png',
                        help="What's is the output filename")
                

    return parser.parse_args()



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
    LightSource([0,0,-500], 'color', args.brightness),
    LightSource([1500,500,0], 'color', args.brightness),
    LightSource([-1500,500,0], 'color', args.brightness),
    LightSource([1500,-500,0], 'color', args.brightness),
    LightSource([-1500,-500,0], 'color', args.brightness),
    Background("color", [1, 1, 1]),
]

if(args.bounds):
    objects.append(plot_border(X,Y, THETA))


offset =  np.array([1.0,-1.0,0.0])
offset *= 10.0 / np.linalg.norm(offset)
#objects.extend(measure(np.array([0,0,4.5]), np.array([0,0,24.5]), offset   , color=[0.0, 0.4, 0.0]))
#objects.extend(measure(np.array([6.5,-6.5,0]), np.array([6.5,-6.5,29]), 0.1*offset   , color=[74.0/255.0, 19.0/255.0, 86.0/255.0]))

box   = 100.3
upper = [100, 100, 100]
lower = [0, 0, 0]

for x,y,z,p,t in zip(X, Y, Z, PHI, THETA):
    vec = np.array([x,y,z])

    if(not(np.all(np.logical_and(vec >= lower, vec <= upper)))):
        objects.append(spin(vec,       t, p, l=args.spin_sz, cmap=plt.get_cmap(args.cmap)))
        



if(not args.nomesh):
    vis_area = Box([0,0,0], [100,100,100], "inverse")
    objects.append(create_isomesh(folder, vis_area=vis_area))


z_sh = 0

scene = Scene( Camera( 'perspective',
                       'angle',       45,
                       'location',    [middle_x + shift[0] * dist,
                                       middle_y + shift[1] * dist,
                                       middle_z + shift[2] * dist + z_sh],
                       'look_at',     [middle_x, middle_y, middle_z  + z_sh],
                       'sky',         [0,             0  ,        1],
                     ),

               objects= objects,
               included=['colors.inc']
              )

scene.render(folder + args.outfile, remove_temp=False, width=args.width,
             height=args.height, antialiasing=0.00001,quality=10)


if(not args.noshow):
    show_img(folder, file=args.outfile)


