import Image
import colortransforms as ct
import numpy as np


# I: image
# returns data in the cielab space from the image (tuple)
def rgb_to_cielab_i(I):
    I2 = map(lambda i: i[0:3],I.getdata()) # rgb tuples
    return map(lambda i: ct.rgb_to_cielab(i[0],i[1],i[2]),I2)

# returns histogram of data
# using info max's and min's
# 64 df's (four intervals per channel)
def colorHistogram(data,info):
    features = np.zeros(64).astype(np.int32)
    
    minX = info[0]
    maxX = info[1]
    pasoX = (maxX-minX)/4
    minY = info[2]
    maxY = info[3]
    pasoY = (maxY-minY)/4
    minZ = info[4]
    maxZ = info[5]
    pasoZ = (maxZ-minZ)/4

    # is x in the bucket (i,j,k)?
    def f(x,i,j,k):
        a = x[0] >= minX + i*pasoX and x[0] < minX + (i+1)*pasoX
        b = x[1] >= minY + j*pasoY and x[1] < minY + (j+1)*pasoY
        c = x[2] >= minZ + k*pasoZ and x[2] < minZ + (k+1)*pasoZ
        return a and b and c

    for i in range(4):
        for j in range(4):
            for k in range(4):
                features[i+j*4+k*16] = sum(filter(lambda x: f(x,i,j,k),data))

    return features

I = Image.open('../images/scanner/baguette/baguette1.tif')
print colorHistogram(rgb_to_cielab_i(I), (0,100,-127,128,-127,128))
