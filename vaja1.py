# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 10:52:48 2016

@author: uporabnik
"""

#import random as rn
import numpy as np
import healpy as hp
import matplotlib.cm as cm
import matplotlib.pyplot as plt

n=100000
Nside=64                 #NSIDE
Npix=12*Nside**2

ra=np.random.uniform(0,2*np.pi,n)
u=np.random.uniform(0,1,n)

theta=np.arccos(2*u-1)

#print(ra)
#print(dec)
pixi=hp.ang2pix(Nside,theta,ra)            #določitev pikslov, katerim pripadajo koordinate
mapa=np.bincount(pixi,minlength=Npix)

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa, cmap=b, title='Naključna porazdelitev', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

nbar=n/(4*np.pi)

print('nbar=',nbar)

delta=((mapa.T)/np.mean(mapa))-1

#print(delta, len(delta))

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(delta, cmap=b, title='Delta polje naključne porazdelitve', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

alm=hp.map2alm(delta)

Cl1=hp.alm2cl(alm)

S=0
Cl=[]

for l in range(3*Nside):
    for m in range(l+1):
        #if m<193:
            #print(alm[m])
        w=l+(m/2)*(383-m)
        S=S+(np.abs(alm[w]))**2
    Cl.append((1/(l+1))*S)
    S=0

Cl=np.array(Cl)
fit=[1/nbar]*192
#Cl=hp.anafast(delta)
    
plt.figure()
plt.plot(np.log10(Cl), label='Power spectrum (zanka)')
plt.plot(np.log10(Cl1), label='Power spectrum (alm2cl)')
plt.plot(np.log10(fit), label='1/nbar (logaritem)')
plt.legend(loc=4)
#plt.ylim([-6,-3])
plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$log_{10}C_\ell$')
plt.show()

#print('Cl povprečni (1,2), 1/nbar :',np.mean(Cl),',',np.mean(Cl1),',', 1/nbar)
