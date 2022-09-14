"""
Generate test disks in z-stack centered at x = n*a, y = midway, z = n
if diameter is 30 micron, and 20x oil has 0.538 micron/pixel ratio, then make
the disks with radius 27.88 pixels.
"""
import numpy as np
import tifffile
import matplotlib.pyplot as plt

def generate_z_disk_test():
    #make shape 1536 * 1536 * 27.
    diameter = 56
    zstep = 18.6 #10 microns = 18.6 pixels
    outDisk = np.zeros(shape=(1536, 1536, 27),dtype=np.bool)
    outSphere = np.zeros(shape=(1536, 1536, 27),dtype=np.bool)
    for ch in range(27):
        #generate disk
        ctr = [((ch+1) * diameter) - int(diameter/2), ((ch+1) * diameter) - int(diameter/2), ch]
        for x in range(ctr[0]-int(diameter/2), ctr[0]+int(diameter/2)+1):
            for y in range(ctr[1]-int(diameter/2), ctr[1]+int(diameter/2)+1):
                if np.sqrt(((ctr[0]-x)**2)+((ctr[1]-y)**2))<=diameter/2 :
                    outDisk[x,y,ch] = True
                for z in range(max(0,ctr[2] - 1), min(27,ctr[2]+2)):
                    if np.sqrt(((ctr[0]-x)**2)+((ctr[1]-y)**2) + (((ctr[2]-z)*zstep)**2))<=diameter/2:
                        outSphere[x,y,z] = True


    #now generate an image for outImage.
    sig_obj = 0.2
    mu_obj = 0.6
    sig_bg = 0.01
    mu_bg = 0.05
    #draw from normal distributions.
    IM = np.zeros(shape=outDisk.shape, dtype=np.float32)
    obj = np.random.normal(mu_obj, sig_obj, outDisk.shape)
    bg = np.random.normal(mu_bg, sig_bg, outDisk.shape)
    IM = np.where(outDisk, obj, bg)
    IM = np.clip(IM, 0., 1.)
    #return IM
    IM = IM*255.
    IM = IM.astype(np.uint8)
    tifffile.imwrite("/users/czeddy/documents/workingfolder/disk_file.ome.tif",np.rollaxis(np.expand_dims(IM,axis=-1),2,0))

def generate_z_sphere_test():
    #make shape 1536 * 1536 * 27.
    diameter = 56
    zstep = 18.6 #10 microns = 18.6 pixels
    outSphere = np.zeros(shape=(1536, 1536, 27),dtype=np.bool)
    for ch in range(27):
        #generate disk
        ctr = [((ch+1) * diameter) - int(diameter/2), ((ch+1) * diameter) - int(diameter/2), ch]
        for x in range(ctr[0]-int(diameter/2), ctr[0]+int(diameter/2)+1):
            for y in range(ctr[1]-int(diameter/2), ctr[1]+int(diameter/2)+1):
                for z in range(max(0,ctr[2] - 1), min(27,ctr[2]+2)):
                    if np.sqrt(((ctr[0]-x)**2)+((ctr[1]-y)**2) + (((ctr[2]-z)*zstep)**2))<=diameter/2:
                        outSphere[x,y,z] = True
    #draw from normal distributions.
    #now generate an image for outImage.
    sig_obj = 0.2
    mu_obj = 0.6
    sig_bg = 0.01
    mu_bg = 0.05
    IM = np.zeros(shape=outSphere.shape, dtype=np.float32)
    obj = np.random.normal(mu_obj, sig_obj, outSphere.shape)
    bg = np.random.normal(mu_bg, sig_bg, outSphere.shape)
    IM = np.where(outSphere, obj, bg)
    IM = np.clip(IM, 0., 1.)
    #return IM
    IM = IM*255.
    IM = IM.astype(np.uint8)
    tifffile.imwrite("/users/czeddy/documents/workingfolder/sphere_file.ome.tif",np.rollaxis(np.expand_dims(IM,axis=-1),2,0))



def centers_gridded(shape, distance, start=None):
    if start is None:
        xs = np.random.rand()*distance/2
        ys = np.random.rand()*distance/2
    else:
        xs = start
        ys = start
    x = np.arange(xs, shape[0], distance)
    y = np.arange(ys, shape[1], distance)
    return x,y

def Gridded_3D_Spheres(shape, distance=60, z_centers = [0], z_xy_ratio = 10/0.538, r_mu = 20., r_std = 3., start = None):
    output = np.zeros(shape=shape, dtype=np.bool)
    for zc in z_centers:
        x, y = centers_gridded(shape[:-1], distance, start=start)
        for xc in x:
            for yc in y:
                # find indices within radius r from center xc, yc, zc
                r = r_std * np.random.normal() + r_mu
                #find indices of the containing cube of the sphere ^^
                #do this for speed optimization purposes.
                x_inds = np.arange(max(0, int(np.floor(xc - r))), min(shape[0], int(np.ceil(xc + r))),1)
                y_inds = np.arange(max(0, int(np.floor(yc - r))), min(shape[1], int(np.ceil(yc + r))),1)
                z_inds = np.arange(max(0, int(zc - np.ceil(r / z_xy_ratio))), min(shape[2], int(zc + np.ceil(r / z_xy_ratio))),1)
                #import pdb;pdb.set_trace()

                x_dist = x_inds - xc
                y_dist = y_inds - yc
                z_dist = (z_inds - zc) * z_xy_ratio#(r / z_xy_ratio)
                #import pdb;pdb.set_trace()

                for i,xd in enumerate(x_dist):
                    for j,yd in enumerate(y_dist):
                        for k,zd in enumerate(z_dist):
                            if np.sqrt((xd**2)+(yd**2)+(zd**2)) <= r:
                                output[x_inds[i], y_inds[j], z_inds[k]] = True
                #now for the object we would want to probably pull the annotation
                #at this step.
    return output

def generate_realistic_image(gt_array, mu_obj=0.6, sig_obj=0.2, mu_bg=0.05, sig_bg=0.01):
    obj = np.random.normal(mu_obj, sig_obj, gt_array.shape)
    bg = np.random.normal(mu_bg, sig_bg, gt_array.shape)
    IM = np.where(gt_array, obj, bg)
    IM = IM*255./np.max(IM)
    return IM.astype(np.uint8)
