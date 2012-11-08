import numpy as np
import mahotas
import mahotas.features
import pylab
import Image

def lbp(filename):
    I = Image.open(filename)
    I = I.convert('L')
    I = np.array(I.getdata()).reshape(I.size[0],I.size[1])
    # 1: radius, 8: eight neighboorhood
    return mahotas.features.lbp(I,1,8)

def haralick(filename):
    img = mahotas.imread(filename)
    return mahotas.features.haralick(img).mean(0)


def tas(filename):
    img = mahotas.imread(filename)
    return mahotas.features.pftas(img)

def zernike(filename):
    I = Image.open(filename)
    I = I.convert('L')
    I = np.array(I.getdata()).reshape(I.size[0],I.size[1])
    # im, degree, radius
    return mahotas.features.zernike(I,16,200)

