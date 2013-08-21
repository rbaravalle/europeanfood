import vf
import csv
import white
import matplotlib
from matplotlib import pyplot as plt
import Image
import numpy as np
import mca
import mfs

FD = 20
cant = 3

maxx = 20
which = 2
cant = 20
dDFs  = 20
fsize = 14

vfBaguette = np.zeros(cant).astype(np.float32)
nonbread = np.zeros((40, dDFs)).astype(np.float32)
baguette = np.zeros((cant, dDFs)).astype(np.float32)
salvado   = np.zeros((cant, dDFs)).astype(np.float32)
lactal   = np.zeros((cant, dDFs)).astype(np.float32)
sandwich = np.zeros((cant, dDFs)).astype(np.float32)

baguetteC = np.zeros((cant, dDFs)).astype(np.float32)
salvadoC   = np.zeros((cant, dDFs)).astype(np.float32)
lactalC   = np.zeros((cant, dDFs)).astype(np.float32)
sandwichC = np.zeros((cant, dDFs)).astype(np.float32)

allied = np.zeros((cant, dDFs)).astype(np.float32)

alliedC = np.zeros((cant, dDFs)).astype(np.float32)

doves = np.zeros((cant, dDFs)).astype(np.float32)

dovesC = np.zeros((cant, dDFs)).astype(np.float32)

vfBaguette = np.zeros(cant).astype(np.float32)
vfBaguetteC = np.zeros(cant).astype(np.float32)
mcaBaguette = np.zeros(cant).astype(np.float32)
mcaBaguetteC = np.zeros(cant).astype(np.float32)
mcadvBaguette = np.zeros(cant).astype(np.float32)
mcadvBaguetteC = np.zeros(cant).astype(np.float32)

vfLactal = np.zeros(cant).astype(np.float32)
vfLactalC = np.zeros(cant).astype(np.float32)
mcaLactal = np.zeros(cant).astype(np.float32)
mcaLactalC = np.zeros(cant).astype(np.float32)
mcadvLactal = np.zeros(cant).astype(np.float32)
mcadvLactalC = np.zeros(cant).astype(np.float32)

vfSalvado = np.zeros(cant).astype(np.float32)
vfSalvadoC = np.zeros(cant).astype(np.float32)
mcaSalvado = np.zeros(cant).astype(np.float32)
mcaSalvadoC = np.zeros(cant).astype(np.float32)
mcadvSalvado = np.zeros(cant).astype(np.float32)
mcadvSalvadoC = np.zeros(cant).astype(np.float32)

vfSandwich = np.zeros(cant).astype(np.float32)
vfSandwichC = np.zeros(cant).astype(np.float32)
mcaSandwich = np.zeros(cant).astype(np.float32)
mcaSandwichC = np.zeros(cant).astype(np.float32)
mcadvSandwich = np.zeros(cant).astype(np.float32)
mcadvSandwichC = np.zeros(cant).astype(np.float32)

vfAllied = np.zeros(cant).astype(np.float32)
vfAlliedC = np.zeros(cant).astype(np.float32)
mcaAllied = np.zeros(cant).astype(np.float32)
mcaAlliedC = np.zeros(cant).astype(np.float32)
mcadvAllied = np.zeros(cant).astype(np.float32)
mcadvAlliedC = np.zeros(cant).astype(np.float32)

vfDoves = np.zeros(cant).astype(np.float32)
vfDovesC = np.zeros(cant).astype(np.float32)
mcaDoves = np.zeros(cant).astype(np.float32)
mcaDovesC = np.zeros(cant).astype(np.float32)
mcadvDoves = np.zeros(cant).astype(np.float32)
mcadvDovesC = np.zeros(cant).astype(np.float32)

with open('../exps/vfs.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    vfBaguette = spamreader.next()
    vfBaguetteC = spamreader.next()
    vfLactal = spamreader.next()
    vfLactalC = spamreader.next()
    vfSalvado = spamreader.next()
    vfSalvadoC = spamreader.next()
    vfSandwich = spamreader.next()
    vfSandwichC = spamreader.next()
with open('../exps/mcas.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    mcaBaguette = spamreader.next()
    mcaBaguetteC = spamreader.next()
    mcaLactal = spamreader.next()
    mcaLactalC = spamreader.next()
    mcaSalvado = spamreader.next()
    mcaSalvadoC = spamreader.next()
    mcaSandwich = spamreader.next()
    mcaSandwichC = spamreader.next()
with open('../exps/mcadvs.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    mcadvBaguette = spamreader.next()
    mcadvBaguetteC = spamreader.next()
    mcadvLactal = spamreader.next()
    mcadvLactalC = spamreader.next()
    mcadvSalvado = spamreader.next()
    mcadvSalvadoC = spamreader.next()
    mcadvSandwich = spamreader.next()
    mcadvSandwichC = spamreader.next()

with open('../exps/baguetteS.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        baguette[i] = row
        i = i+1

with open('../exps/baguetteC.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        baguetteC[i] = row
        i = i+1

with open('../exps/lactalS.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        lactal[i] = row
        i = i+1

with open('../exps/lactalC.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        lactalC[i] = row
        i = i+1

with open('../exps/salvadoS.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        salvado[i] = row
        i = i+1

with open('../exps/salvadoC.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        salvadoC[i] = row
        i = i+1

with open('../exps/sandwichS.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        sandwich[i] = row
        i = i+1

with open('../exps/sandwichC.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        sandwichC[i] = row
        i = i+1


with open('../exps/allied.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        allied[i] = row
        i = i+1

with open('../exps/alliedData.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    vfAllied = spamreader.next()
    mcaAllied = spamreader.next()
    mcadvAllied = spamreader.next()

with open('../exps/alliedDataC.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    vfAlliedC = spamreader.next()
    mcaAlliedC = spamreader.next()
    mcadvAlliedC = spamreader.next()

with open('../exps/alliedC.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        alliedC[i] = row
        i = i+1

with open('../exps/doves.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        doves[i] = row
        i = i+1

with open('../exps/dovesC.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    for row in spamreader:
        dovesC[i] = row
        i = i+1

with open('../exps/dovesData.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    vfDoves = spamreader.next()
    mcaDoves = spamreader.next()
    mcadvDoves = spamreader.next()

with open('../exps/dovesDataC.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    i = 0
    vfDovesC = spamreader.next()
    mcaDovesC = spamreader.next()
    mcadvDovesC = spamreader.next()


alliedCT = np.hstack((alliedC[:,4:20],np.zeros((20,4)).astype(np.float32)))

#plt.plot(np.arange(len(sandwich[10])), alliedC[10], 'b+--', label='Allied S!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[9])), alliedC[9], 'b+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[8])), alliedC[8], 'b+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[7])), alliedC[7], 'b+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[10])), allied[10], 'r+--', label='Allied S!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[9])), allied[9], 'r+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[8])), allied[8], 'r+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[7])), allied[7], 'r+--', label='Allied C!',linewidth=2.0)
#plt.show()
#alliedBoxp = plt.boxplot(np.vstack((allied,alliedCT)))
#plt.show()



#doves = np.hstack((doves[:,4:20],np.zeros((20,4)).astype(np.float32)))
dovesCT = np.hstack((dovesC[:,6:20],np.zeros((20,6)).astype(np.float32)))

#plt.plot(np.arange(len(sandwich[10])), doves[10], 'y+--', label='Allied S!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[9])), doves[9], 'y+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[8])), doves[8], 'y+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[7])), doves[7], 'y+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[10])), dovesC[10], 'k+--', label='Allied S!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[9])), dovesC[9], 'k+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[8])), dovesC[8], 'k+--', label='Allied C!',linewidth=2.0)
#plt.plot(np.arange(len(sandwich[7])), dovesC[7], 'k+--', label='Allied C!',linewidth=2.0)
#plt.show()
#dovesBoxp = plt.boxplot(np.vstack((dovesCT,alliedCT)))
#plt.show()
#exit()


cBs = np.zeros(dDFs).astype(np.float32)
cmcaBs = np.zeros(dDFs).astype(np.float32)
cmcadvBs = np.zeros(dDFs).astype(np.float32)
cLs = np.zeros(dDFs).astype(np.float32)
cmcaLs = np.zeros(dDFs).astype(np.float32)
cmcadvLs = np.zeros(dDFs).astype(np.float32)
cSals = np.zeros(dDFs).astype(np.float32)
cmcaSals = np.zeros(dDFs).astype(np.float32)
cmcadvSals = np.zeros(dDFs).astype(np.float32)
cSans = np.zeros(dDFs).astype(np.float32)
cmcaSans = np.zeros(dDFs).astype(np.float32)
cmcadvSans = np.zeros(dDFs).astype(np.float32)

cAs = np.zeros(dDFs).astype(np.float32)
cmcaAs = np.zeros(dDFs).astype(np.float32)
cmcadvAs = np.zeros(dDFs).astype(np.float32)

cBc = np.zeros(dDFs).astype(np.float32)
cmcaBc = np.zeros(dDFs).astype(np.float32)
cmcadvBc = np.zeros(dDFs).astype(np.float32)
cLc = np.zeros(dDFs).astype(np.float32)
cmcaLc = np.zeros(dDFs).astype(np.float32)
cmcadvLc = np.zeros(dDFs).astype(np.float32)
cSalc = np.zeros(dDFs).astype(np.float32)
cmcaSalc = np.zeros(dDFs).astype(np.float32)
cmcadvSalc = np.zeros(dDFs).astype(np.float32)
cSanc = np.zeros(dDFs).astype(np.float32)
cmcaSanc = np.zeros(dDFs).astype(np.float32)
cmcadvSanc = np.zeros(dDFs).astype(np.float32)

cAc = np.zeros(dDFs).astype(np.float32)
cmcaAc = np.zeros(dDFs).astype(np.float32)
cmcadvAc = np.zeros(dDFs).astype(np.float32)


print "All: ", allied

for i in range(dDFs):
    #print i
    cBs[i] = np.corrcoef(baguette[:,i],vfBaguette)[0,1]
    cmcaBs[i] = np.corrcoef(baguette[:,i],mcaBaguette)[0,1]
    cmcadvBs[i] = np.corrcoef(baguette[:,i],mcadvBaguette)[0,1]
    cLs[i] = np.corrcoef(lactal[:,i],vfLactal)[0,1]
    cmcaLs[i] = np.corrcoef(lactal[:,i],mcaLactal)[0,1]
    cmcadvLs[i] = np.corrcoef(lactal[:,i],mcadvLactal)[0,1]
    cSals[i] = np.corrcoef(salvado[:,i],vfSalvado)[0,1]
    cmcaSals[i] = np.corrcoef(salvado[:,i],mcaSalvado)[0,1]
    cmcadvSals[i] = np.corrcoef(salvado[:,i],mcadvSalvado)[0,1]
    cSans[i] = np.corrcoef(sandwich[:,i],vfSandwich)[0,1]
    cmcaSans[i] = np.corrcoef(sandwich[:,i],mcaSandwich)[0,1]
    cmcadvSans[i] = np.corrcoef(sandwich[:,i],mcadvSandwich)[0,1]
    cAs[i] = np.corrcoef(allied[:,i],vfAllied)[0,1]
    cmcaAs[i] = np.corrcoef(allied[:,i],mcaAllied)[0,1]
    cmcadvAs[i] = np.corrcoef(allied[:,i],mcadvAllied)[0,1]
    #camera
    cBc[i] = np.corrcoef(baguetteC[:,i],vfBaguetteC)[0,1]
    cmcaBc[i] = np.corrcoef(baguetteC[:,i],mcaBaguetteC)[0,1]
    cmcadvBc[i] = np.corrcoef(baguetteC[:,i],mcadvBaguetteC)[0,1]
    cLc[i] = np.corrcoef(lactalC[:,i],vfLactalC)[0,1]
    cmcaLc[i] = np.corrcoef(lactalC[:,i],mcaLactalC)[0,1]
    cmcadvLc[i] = np.corrcoef(lactalC[:,i],mcadvLactalC)[0,1]
    cSalc[i] = np.corrcoef(salvadoC[:,i],vfSalvadoC)[0,1]
    cmcaSalc[i] = np.corrcoef(salvadoC[:,i],mcaSalvadoC)[0,1]
    cmcadvSalc[i] = np.corrcoef(salvadoC[:,i],mcadvSalvadoC)[0,1]
    cSanc[i] = np.corrcoef(sandwichC[:,i],vfSandwichC)[0,1]
    cmcaSanc[i] = np.corrcoef(sandwichC[:,i],mcaSandwichC)[0,1]
    cmcadvSanc[i] = np.corrcoef(sandwichC[:,i],mcadvSandwichC)[0,1]

    cAc[i] = np.corrcoef(alliedC[:,i],vfAlliedC)[0,1]
    cmcaAc[i] = np.corrcoef(alliedC[:,i],mcaAlliedC)[0,1]
    cmcadvAc[i] = np.corrcoef(alliedC[:,i],mcadvAlliedC)[0,1]

print cBs
print cmcaBs
print cmcadvBs
#plt.plot(np.arange(cAc.shape[0]), cLc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cLs, 'g*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cBc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cBs, 'r*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cSalc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cSals, 'b*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cSanc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cSans, 'y*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cAc, 'y*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cAs, 'y*-', linewidth=2.0)
plt.show()

#plt.plot(np.arange(cAc.shape[0]), cLc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcaLs, 'g*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cBc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcaBs, 'r*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cSalc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcaSals, 'b*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cSanc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcaSans, 'y*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cAc, 'y*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcaAs, 'y*-', linewidth=2.0)
plt.show()

#plt.plot(np.arange(cAc.shape[0]), cLc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcadvLs, 'g*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cBc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcadvBs, 'r*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cSalc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcadvSals, 'b*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cSanc, 'b*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcadvSans, 'y*-', linewidth=2.0)
#plt.plot(np.arange(cAc.shape[0]), cAc, 'y*-', linewidth=2.0)
plt.plot(np.arange(cAs.shape[0]), cmcadvAs, 'y*-', linewidth=2.0)
plt.show()

plt.ylabel(r'$R$',fontsize=fsize)
plt.xlabel('FD',fontsize=fsize)
#plt.plot(np.arange(dDFs), cBs, 'b*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcaBs, 'y*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcadvBs, 'r*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cLs, 'b*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcaLs, 'y*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcadvLs, 'r*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cSals, 'b*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcaSals, 'y*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcadvSals, 'r*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cSans, 'b*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcaSans, 'y*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcadvSans, 'r*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cAs, 'b*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcaAs, 'y*--', linewidth=2.0)
#plt.plot(np.arange(dDFs), cmcadvAs, 'r*--', linewidth=2.0)
plt.ylim((-1, 1))
#plt.show()

exit()

# camera

plt.ylabel(r'$R$',fontsize=fsize)
plt.xlabel('FD',fontsize=fsize)
plt.plot(np.arange(dDFs), cBc, 'b*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcaBc, 'y*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcadvBc, 'r*--', linewidth=2.0)
plt.ylim((-1, 1))
plt.show()
plt.plot(np.arange(dDFs), cLc, 'b*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcaLc, 'y*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcadvLc, 'r*--', linewidth=2.0)
plt.ylim((-1, 1))
plt.show()
plt.plot(np.arange(dDFs), cSalc, 'b*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcaSalc, 'y*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcadvSalc, 'r*--', linewidth=2.0)
plt.ylim((-1, 1))
plt.show()
plt.plot(np.arange(dDFs), cSanc, 'b*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcaSanc, 'y*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcadvSanc, 'r*--', linewidth=2.0)
plt.ylim((-1, 1))
plt.show()
plt.plot(np.arange(dDFs), cAc, 'b*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcaAc, 'y*--', linewidth=2.0)
plt.plot(np.arange(dDFs), cmcadvAc, 'r*--', linewidth=2.0)
plt.ylim((-1, 1))
plt.show()
exit()















################
x = np.arange(FD)
plt.ylabel(r'$f(\alpha)$',fontsize=12)
plt.xlabel('FD',fontsize=12)

filen = '../images/camera/sandwich/s5.tif'
san = mfs.mfs(filen,[1,FD,3,True])
plt.plot(x, san, 'b*--', linewidth=2.0)
plt.show()

exit()

sal = np.zeros((cant,FD))
san = np.zeros((cant,FD))
l = np.zeros((cant,FD))
b = np.zeros((cant,FD))
salC = np.zeros((cant,FD))
sanC = np.zeros((cant,FD))
lC = np.zeros((cant,FD))
bC = np.zeros((cant,FD))

for i in range(1,cant):
    filen = '../images/camera/sandwich/s{}.tif'.format(i)
    san[i] = mfs.mfs(filen,[1,FD,3,True])
    plt.plot(x, san[i], 'b+--', linewidth=2.0)
    filen = '../images/camera/salvado/s{}.tif'.format(i)
    sal[i] = mfs.mfs(filen,[1,FD,3,True])
    plt.plot(x, sal[i], 'r+--', linewidth=2.0)
    filen = '../images/camera/baguette/slicer/b{}.tif'.format(i)
    b[i] = mfs.mfs(filen,[1,FD,3,True])
    plt.plot(x, b[i], 'm+--', linewidth=2.0)
    filen = '../images/camera/lactal/l{}.tif'.format(i)
    l[i] = mfs.mfs(filen,[1,FD,3,True])
    plt.plot(x, l[i], 'k+--', linewidth=2.0)

    filen = '../images/scanner/sandwich/sandwich{}.tif'.format(i)
    sanC[i] = mfs.mfs(filen,[1,FD,3,True])
    plt.plot(x, sanC[i], 'b+--', linewidth=2.0)
    filen = '../images/scanner/salvado/salvado{}.tif'.format(i)
    salC[i] = mfs.mfs(filen,[1,FD,3,True])
    plt.plot(x, salC[i], 'r+--', linewidth=2.0)
    filen = '../images/scanner/baguette/baguette{}.tif'.format(i)
    bC[i] = mfs.mfs(filen,[1,FD,3,True])
    plt.plot(x, bC[i], 'm+--', linewidth=2.0)
    filen = '../images/scanner/lactal/lactal{}.tif'.format(i)
    lC[i] = mfs.mfs(filen,[1,FD,3,True])
    plt.plot(x, lC[i], 'k+--', linewidth=2.0)


i = 1
filen = '../images/scanner/baguette/baguette{}.tif'.format(i)

extra = [1,40,1]
vfrac = vf.VF(filen,extra)
arr = mca.mca(filen,extra)
extra = [1,FD,3,True]
#y = mfs.mfs(filen,extra)

print "VF, MCA, STMCA"
print vfrac, arr
#print "MFS: ", y

#plt.plot(x, y, 'k+--', label='MFS 2',linewidth=2.0)
plt.legend(loc = 2) # loc 4: bottom, right
plt.show()

exit()

x = np.arange(10)
y = -np.arange(10)
#plt.plot([x, y, 'b-', label = 'line 1')
#plt.plot([x, y, 'k-', label = 'line 2')
#plt.show()
plt.ylabel(r'$f(\alpha)$')
plt.xlabel(r'$\alpha$')
plt.plot(x, y, 'ko--', label='r${\em }$', linewidth=2)
plt.plot(x, y+1, 'ro--',  label='line 2')
plt.plot(x, np.log(y+1), 'bo--',  label='line 3')
plt.legend()
plt.show()

