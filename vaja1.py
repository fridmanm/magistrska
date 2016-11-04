# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:52:48 2016

@author: uporabnik
"""

import random as rn
import numpy as np
import healpy as hp
import matplotlib.cm as cm
import matplotlib.pyplot as plt

n=100000
Nside=64                 #NSIDE
Npix=12*Nside**2
ra=[]
dec=[]

for i in range(n):
    ra.append(rn.uniform(0,2*np.pi))
    
for j in range(n):
    dec.append(rn.uniform(0,np.pi))
    
ra=np.array(ra)
dec=np.array(dec)

#print(ra)
#print(dec)
pixi=hp.ang2pix(Nside,dec,ra)            #določitev pikslov, katerim pripadajo koordinate
mapa=np.bincount(pixi,minlength=Npix)

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa, cmap=b, title='Naključna porazdelitev', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

nbar=np.mean(mapa)

print('nbar:',nbar)

delta=((mapa.T)/nbar)-1

#print(delta, len(delta))

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(delta, cmap=b, title='Delta polje naključne porazdelitve', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

alm=hp.map2alm(delta, lmax=3*Nside-1)

#print(len(alm))

S=0
k=0
p=0
Cl=[]

for l in range(3*Nside):
    p=p+l+1
    for m in range(k,p):
        S=S+(np.abs(alm[m]))**2
    Cl.append((1/(l+1))*S)
    k=p
    S=0
    
plt.figure()
plt.plot(np.log10(Cl))
#plt.ylim([0,0.002])
plt.xlabel('$\ell$')
plt.ylabel('$log_{10}C_\ell$')
plt.show()

print('Cl povprečni, 1/nbar :',np.mean(Cl),',', 1/nbar)
