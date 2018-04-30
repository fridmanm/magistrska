#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:34:31 2018

@author: mitja
"""

import numpy as np
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
import cltools as ct
import matplotlib.cm as cm
import claw
from numpy import *

n=500000
Nside=64
Npix=12*Nside**2
Cls_in_bin=12
lmax=(3*Nside)-1

#mapa=ct.randmap(n,Nside)
clth=np.loadtxt('theory_fit.txt')
mapa=hp.synfast(clth[:,1][0:lmax+1],Nside)


plt.figure()
plt.plot(clth[:,0],clth[:,1], label=' Power spectrum icecube\n podatkov brez shot noise')
plt.xlim(0,lmax)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()

print ("max,min,mean=",mapa.max(), mapa.min(),mapa.mean())
b=cm.BuPu                                                          #      
b.set_under("w")                                                        #
hp.mollview(mapa, cmap=b, title='Celotno nebo', cbar=True, xsize=1400, unit='Gostota objektov [objektov/piksel]')     #izris mape
ct.mollaxes()
hp.graticule(coord=('E'))                                               #
plt.show()

maska1=np.ones(Npix)
maska1[int(3*Npix/5):Npix]=0

b=cm.BuPu                                                          #      
b.set_under("w")                                                        #
hp.mollview(maska1, cmap=b, title='Maska 1', cbar=True, xsize=1400, unit='Gostota objektov [objektov/piksel]')     #izris mape
ct.mollaxes()
hp.graticule(coord=('E'))                                               #
plt.show()

theta,phi=hp.pix2ang(Nside,np.arange(Npix))

idx=[]

for i in range(Npix):
    if phi[i]>0.3 and phi[i]<=np.pi:
        idx.append(i)
idx=np.array(idx)

theta=np.delete(theta,idx)
phi=np.delete(phi,idx)

pixi=hp.ang2pix(Nside,theta,phi)
maska2=np.float64(np.bincount(pixi,minlength=Npix))

b=cm.BuPu                                                          #      
b.set_under("w")                                                        #
hp.mollview(maska2, cmap=b, title='Maska 2', cbar=True, xsize=1400, unit='Gostota objektov [objektov/piksel]')     #izris mape
ct.mollaxes()
hp.graticule(coord=('E'))                                               #
plt.show()

N1=0.1
N2=0.08

noise1=np.random.normal(0,N1,size=Npix)*maska1
noise2=np.random.normal(0,N2,size=Npix)*maska2


mapa1=(mapa+noise1)*maska1
mapa2=(mapa+noise2)*maska2
f1=np.sum(maska1)/len(maska1)
f2=np.sum(maska2)/len(maska2)

noise1=maska1*np.ones(Npix)*N1
noise2=maska2*np.ones(Npix)*N2

weight1=maska1*(1/noise1**2)
weight2=maska2*(1/noise2**2)

weight1[int(3*Npix/5):Npix]=0

NaNs2 = isnan(weight2)
weight2[NaNs2] = 0

b=cm.BuPu                                                          #      
b.set_under("w")                                                        #
hp.mollview(mapa1, cmap=b, title='Delno nebo 1', cbar=True, xsize=1400, unit='Gostota objektov [objektov/piksel]')     #izris mape
ct.mollaxes()
hp.graticule(coord=('E'))                                               #
plt.show()

b=cm.BuPu                                                          #      
b.set_under("w")                                                        #
hp.mollview(mapa2, cmap=b, title='Delno nebo 2', cbar=True, xsize=1400, unit='Gostota objektov [objektov/piksel]')     #izris mape
ct.mollaxes()
hp.graticule(coord=('E'))                                               #
plt.show()

p=claw.Cl(lmaxdl=(lmax+1,Cls_in_bin))

M1=claw.MeasureCl(p,weight1,noise1,Ng=1000,ignorem0=True)

M1.getEstimate(mapa1)

M1.getCovMat(M1.Cl)

Clbt,error=ct.clavg(clth[:,1][0:lmax+1],lmax,Cls_in_bin,5)
TCl=claw.Cl(lmaxdl=(lmax+1,Cls_in_bin),vals=Clbt)

ell=np.arange(lmax+1)

plt.figure()
plt.plot(ell,M1.Cl.Cl)
plt.errorbar(p.ells,M1.Cl.vals,yerr=np.sqrt(M1.Cl.cov.diagonal()),fmt='o',capsize=5, ecolor='black',label='Bin avereged power spectrum')
plt.plot(ell,clth[:,1][0:lmax+1])
#plt.plot(p.ells,TCl.vals)
plt.xlabel('$\ell$ ')
plt.ylabel('$C_\ell$')
plt.xlim(0,np.max(p.ells)+np.min(p.ells))
#plt.annotate('$N_{side} = %i$ \n $N_b = %i$' %(Nside,3*Nside/Cls_in_bin),xy=(168,0.000055),ha='center',va='top',fontsize=12)
plt.show()

print('chi2 =',M1.Cl.chi2(TCl))
print('chi2diag =',M1.Cl.chi2diag(TCl))

M2=claw.MeasureCl(p,weight2,noise2,Ng=1000,ignorem0=True)

M2.getEstimate(mapa2)

M2.getCovMat(M2.Cl)

TCl=claw.Cl(lmaxdl=(lmax+1,Cls_in_bin),vals=Clbt)

plt.figure()
plt.plot(ell,M2.Cl.Cl)
plt.errorbar(p.ells,M2.Cl.vals,yerr=np.sqrt(M2.Cl.cov.diagonal()),fmt='o',capsize=5, ecolor='black',label='Bin avereged power spectrum')
plt.plot(ell,clth[:,1][0:lmax+1])
plt.xlabel('$\ell$ ')
plt.ylabel('$C_\ell$')
plt.xlim(0,np.max(p.ells)+np.min(p.ells))
#plt.annotate('$N_{side} = %i$ \n $N_b = %i$' %(Nside,3*Nside/Cls_in_bin),xy=(168,0.000055),ha='center',va='top',fontsize=12)
plt.show()

print('chi2 =',M2.Cl.chi2(TCl))
print('chi2diag =',M2.Cl.chi2diag(TCl))

M3=claw.MeasureCl(p,weight1,noise1,Ng=1000,Noise2=noise2,Weight2=weight2,ignorem0=True)

M3.getcrossEstimate(mapa1,mapa2)

M3.getcrossCovMat(M3.Cl)

TCl=claw.Cl(lmaxdl=(lmax+1,Cls_in_bin),vals=Clbt)

plt.figure()
plt.plot(ell,M3.Cl.Cl,'r')
plt.errorbar(p.ells,M3.Cl.vals,yerr=np.sqrt(M3.Cl.cov.diagonal()),fmt='ro',capsize=5, ecolor='black',label='Bin avereged power spectrum')
plt.plot(ell,clth[:,1][0:lmax+1])
plt.xlabel('$\ell$ ')
plt.ylabel('$C_\ell$')
plt.xlim(0,np.max(p.ells)+np.min(p.ells))
#plt.annotate('$N_{side} = %i$ \n $N_b = %i$' %(Nside,3*Nside/Cls_in_bin),xy=(168,0.000055),ha='center',va='top',fontsize=12)
plt.show()

print('chi2 =',M3.Cl.chi2(TCl))
print('chi2diag =',M3.Cl.chi2diag(TCl))


plt.figure()
plt.plot(ell,M1.Cl.Cl,'b')
plt.errorbar(p.ells,M1.Cl.vals,yerr=np.sqrt(M1.Cl.cov.diagonal()),fmt='bo',capsize=5, ecolor='black',label='Bin avereged power spectrum')
plt.plot(ell,M2.Cl.Cl,'g')
plt.errorbar(p.ells,M2.Cl.vals,yerr=np.sqrt(M2.Cl.cov.diagonal()),fmt='go',capsize=5, ecolor='black',label='Bin avereged power spectrum')
plt.plot(ell,M3.Cl.Cl,'r')
plt.errorbar(p.ells,M3.Cl.vals,yerr=np.sqrt(M3.Cl.cov.diagonal()),fmt='ro',capsize=5, ecolor='black',label='Bin avereged power spectrum')
plt.plot(ell,clth[:,1][0:lmax+1])
plt.xlabel('$\ell$ ')
plt.ylabel('$C_\ell$')
plt.xlim(0,np.max(p.ells)+np.min(p.ells))

print('chi2(auto1) =',M1.Cl.chi2(TCl))
print('chi2diag(auto1) =',M1.Cl.chi2diag(TCl))

print('chi2(auto2) =',M2.Cl.chi2(TCl))
print('chi2diag(auto2) =',M2.Cl.chi2diag(TCl))

print('chi2(cross) =',M3.Cl.chi2(TCl))
print('chi2diag(cross) =',M3.Cl.chi2diag(TCl))
