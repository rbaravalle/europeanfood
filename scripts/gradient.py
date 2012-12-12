import Image
import numpy as np
import scipy.signal
import mfs

def main(filename,extra):    
    #Compute Gradient
    #filename = '../images/nonbread/brodatz/D1.gif'
    IM = Image.open(filename)
    IM = IM.convert("L")
    Nx, Ny = IM.size
    IM = np.array(IM.getdata()).reshape(IM.size)
    fx = np.float32(0.5)*np.array([[-1, 0, 1],[0, 0, 0],[0, 0, 0]])
    fy = fx.T
    fxy = np.float32(0.5)*np.array([[-1, 0, 0],[0, 0, 0],[0, 0, 1]])
    fyx = np.float32(0.5)*np.array([[0, 0, -1],[0, 0, 0],[1, 0, 0]])

    IMG = IM
    a = scipy.signal.convolve2d(IMG, fx,mode="full")
    Nx, Ny = a.shape
    a = a[0:Nx-2,1:Ny-1]

    b = scipy.signal.convolve2d(IMG, fy,mode="full")
    Nx, Ny = b.shape
    b = b[1:Nx-1,0:Ny-2]

    c = scipy.signal.convolve2d(IMG, fxy,mode="full")
    Nx, Ny = c.shape
    c = c[1:Nx-1,1:Ny-1]


    d = scipy.signal.convolve2d(IMG, fyx,mode="full")
    Nx, Ny = d.shape
    d = d[1:Nx-1,1:Ny-1]

    IMG = a**2 + b**2+c**2 +d**2
    IMG = np.sqrt(IMG)
    IMG = np.floor(IMG) 

    #WG = sum(double(IMG(:)));
    #WG = np.sum(IMG)
    #print WG
    extra[3] = False
    return mfs.mfs(IMG,extra)

#main()
