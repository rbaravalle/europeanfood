#from pylab import plot, title, show , legend
import matplotlib
from matplotlib import pyplot as plt
import white
import Image
import numpy as np
from scipy import ndimage

def mca(filename,extra):
    I = Image.open(filename)
    Nx, Ny = I.size
    gray = I.convert('L') # rgb 2 gray
    gray = white.white(gray,Nx,Ny,extra[1],extra[2]) # local thresholding algorithm
    label_im, nb_labels = ndimage.label(gray)
    #plt.imshow(label_im, cmap=matplotlib.cm.gray)
    #plt.show()
    sizes = ndimage.sum(gray, label_im, range(nb_labels + 1))
    return [np.float32(0.00527)*np.mean(sizes), np.float32(0.00527)*np.std(sizes)] # size in mm2


