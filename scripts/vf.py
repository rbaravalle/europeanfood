#from pylab import plot, title, show , legend
import matplotlib
from matplotlib import pyplot as plt
import white
import Image
import numpy as np

def VF(filename,extra):
    I = Image.open(filename)
    Nx, Ny = I.size
    gray = I.convert('L') # rgb 2 gray
    gray = white.white(gray,Nx,Ny,extra[1],extra[2]) # local thresholding algorithm
    #plt.imshow(gray, cmap=matplotlib.cm.gray)
    #plt.show()
    return np.float32(np.float32(Nx*Ny-np.histogram(gray)[0][0])/(Nx*Ny)) # void fraction


