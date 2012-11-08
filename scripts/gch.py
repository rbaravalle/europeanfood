import Image
import ImageFilter
import colortransforms as ct
import numpy as np
import time
import math

def rgb_to_cielab_i(I):
    I2 = map(lambda i: i[0:3],I.getdata()) # rgb tuples
    return map(lambda i: ct.rgb_to_cielab(i[0],i[1],i[2]),I2)
# returns global color histogram of data
# 64 df's (four intervals per channel)
def colorHistogram(filename):
    t = time.clock()
    info = (0,100,-127,128,-127,128) # max's and min's in the CIELab space
    I = Image.open(filename)
    I = I.filter(ImageFilter.BLUR)
    data = rgb_to_cielab_i(I)
    cant = 4
    features = np.zeros(cant*cant*cant).astype(np.int32)
    minX = info[0]
    maxX = info[1]
    pasoX = (maxX-minX)/cant
    minY = info[2]
    maxY = info[3]
    pasoY = (maxY-minY)/cant
    minZ = info[4]
    maxZ = info[5]
    pasoZ = (maxZ-minZ)/cant

    # is x in the bucket (i,j,k)?
    def f(x,i,j,k):
        baseX = minX+i*pasoX
        baseY = minY+j*pasoY
        baseZ = minZ+k*pasoZ
        x0 = math.trunc(x[0])
        x1 = math.trunc(x[1])
        x2 = math.trunc(x[2])
        a = x0 >= baseX and x0 <= baseX + pasoX
        b = x1 >= baseY and x1 <= baseY + pasoY
        c = x2 >= baseZ and x2 <= baseZ + pasoZ
        return (a and b and c)

    for i in range(cant):
        for j in range(cant):
            for k in range(cant):
                features[i+j*cant+k*cant*cant] = len(filter(lambda x: f(x,i,j,k),data))

    print time.clock()-t
    return features

