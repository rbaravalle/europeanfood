import numpy as np
import mahotas
import mahotas.features
import pylab
import Image

def lbp(filename):
    I = np.array(Image.open(filename).getdata())
    # 1: radius, 8: eight neighboorhood
    return mahotas.features.lbp(I,1,8)

def haralick(filename):
    img = mahotas.imread(filename)
    return mahotas.features.haralick(img).mean(0)

