import numpy as np
import singularityCL as sg
import localmfracV as localMF

def featuresMF(filename):
    extra = [20,False,False,True,0,0,0]
    cantDF = extra[0]
    # how many subdivisions
    subdiv = 2
    # Not CIELab
    # MAX
    # extra: cant DF(0), convert to greyscale? (1), , ,
    # measure to use (4), cielab? (5), channel to use (6)
    extra = [cantDF,extra[1],extra[2],extra[3],0,False,0]
    a = localMF.localMF(filename,subdiv, extra)
    # MIN
    extra = [cantDF,extra[1],extra[2],extra[3],1,False,0]
    b = localMF.localMF(filename,subdiv,extra)
    # SUM
    extra = [cantDF,extra[1],extra[2],extra[3],2,False,0]
    c = localMF.localMF(filename,subdiv,extra)
    #return np.vstack((a,b,c))

    extra[3] = True
    # MEASURE MAX
    # convert to L from CIELab
    extra = [cantDF,extra[1],extra[2],extra[3],0,True,0]
    d = localMF.localMF(filename,subdiv,extra)
    # convert to L from CIELab
    extra = [cantDF,extra[1],extra[2],extra[3],0,True,1]
    e = localMF.localMF(filename,subdiv,extra)
    # convert to a from CIELab
    extra = [cantDF,extra[1],extra[2],extra[3],0,True,2]
    f = localMF.localMF(filename,subdiv,extra)

    # MEASURE MIN
    # convert to L from CIELab
    extra = [cantDF,extra[1],extra[2],extra[3],1,True,0]
    g = localMF.localMF(filename,subdiv,extra)
    # convert to a from CIELab
    extra = [cantDF,extra[1],extra[2],extra[3],1,True,1]
    h = localMF.localMF(filename,subdiv,extra)
    # convert to b from CIELab
    extra = [cantDF,extra[1],extra[2],extra[3],1,True,2]
    i = localMF.localMF(filename,subdiv,extra)

    # MEASURE SUM
    # convert to L from CIELab
    extra = [cantDF,extra[1],extra[2],extra[3],2,True,0]
    j = localMF.localMF(filename,subdiv,extra)
    # convert to a from CIELab
    extra = [cantDF,extra[1],extra[2],extra[3],2,True,1]
    k = localMF.localMF(filename,subdiv,extra)
    # convert to b from CIELab
    extra = [cantDF,extra[1],extra[2],extra[3],2,True,2]
    l = localMF.localMF(filename,subdiv,extra)

    return np.vstack((a,b,c,d,e,f,g,h,i,j,k,l))
