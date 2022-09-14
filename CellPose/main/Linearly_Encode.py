import numpy as np
import matplotlib.pyplot as plt
import skimage.io

input_z = 27
out_channels = 3
space = input_z/(out_channels+1)
centers = [(i+1) * space for i in range(int(out_channels))]
midway_overlap_percentage = 0.5
sigma = np.sqrt(-0.25*(space**2) / (2*np.log(midway_overlap_percentage))) #not sure what this should ideally be. I think between the peaks, we would like there to be 50% of each color.
#so evaluate that. exp(-(x-mu)^2 / (2*sigma^2))
#what about on the bottom and on the top then? Eh, should be fine.
#at 0.5*space, we want peak to be 50%.
#plot it out.
xr = np.linspace(0,input_z-1,input_z)
G = np.exp(-((xr - np.expand_dims(centers,axis=-1))**2)/(2*(sigma**2)))

cs = ['r', 'g', 'b']
for i in range(int(out_channels)):
    plt.plot(xr, G[i,:], c=cs[i])

plt.show()

I = skimage.io.imread("/users/czeddy/documents/workingfolder/disk_file.ome.tif")
I = np.rollaxis(I,0,3)
I = np.expand_dims(I,axis=-1)
I = I.astype(np.float32)
I = I / 255.

G = np.expand_dims(np.expand_dims(np.rollaxis(G,0,2),axis=0),axis=0)
plt.imshow(np.max(I*G,axis=2))


#and so what we will do is for every pixel in each slice, we will multiply by the
#rgb values of that slice, leaving a 27 z stack 3 channel image. Then, for each channel,
#take the maximum value.
