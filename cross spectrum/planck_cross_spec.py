#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:08:28 2018

@author: mitja
"""

import numpy as np
import healpy as hp
from astropy.io import fits
import matplotlib.pyplot as plt
import cltools as ct
import matplotlib.cm as cm

planck_kappa = fits.open('dat_klm.fits')    #odprti planck kappa fits datoteko
planck_kappa.info()

Nside=64
Npix=12*Nside**2
lmax=2048           #določen v opisu podatkov
ic_lmax=3*Nside-1

pdata = planck_kappa[1].data

#print(pdata)

print('data lenght =',len(pdata.INDEX))

alm_data=pdata.REAL+1j*pdata.IMAG           #pretvoriti podatke v 1D tabelo

galm=ct.rearange_alm(lmax,alm_data)     #preoblikovanje tabele podatkov v obliko, ki jo vrne map2alm
                                        #g v imenu definiran za galaktične koordinate
#print(galm)

gmapa=hp.alm2map(galm,nside=64)          #kreacija mape iz galm koeficientov

#gmapa=gmapa*10**-2                        #renormalizacija (še ne pravilna)

b=cm.seismic                                                         #      
b.set_under("w")                                                        #
hp.mollview(gmapa, cmap=b, title='Planck Kappa mapa (galaktični sistem)', cbar=True, xsize=1400)     #izris mape
#ct.mollaxes()
hp.graticule(coord=('E'))                                               #
plt.show()

#alm_check=hp.map2alm(gmapa,lmax=lmax)       #preveritev, če vrne map2alm vsaj podobne koeficiente kot v originalnih podatkih (vrne renormalizirane, ker vrnejo isto obliko mape z renormaliziranimi vrednostmi)

b,l,aomega=ct.eq2gal(0,90)

hp.rotate_alm(galm,psi=l,theta=b,phi=aomega,lmax=lmax)

emapa=hp.alm2map(galm,nside=64)          #kreacija mape iz ealm koeficientov

#emapa=emapa*10**-2                        #renormalizacija (še ne pravilna)

b=cm.seismic                                                         #      
b.set_under("w")                                                        #
hp.mollview(emapa, cmap=b, title='Planck Kappa mapa (ekvatorialni sistem)', cbar=True, xsize=1400)     #izris mape
ct.mollaxes()
hp.graticule(coord=('E'))                                               #
plt.show()

data=np.loadtxt('IceCube-59__Search_for_point_sources_using_muon_events.txt',skiprows=2)
dec,ra,time,log10E,angerr=data.T
print ("data len=",len(data))

theta=90-dec                #pretvorba deklinacije v sferično koordinato theta
theta*=(np.pi/180)
ra*=(np.pi/180)
n=len(data)

#Izris icecube podatkov na sferično karto v Mollweide projekciji

pixi=hp.ang2pix(Nside,theta,ra)            #določitev pikslov, katerim pripadajo koordinate
mapa=np.bincount(pixi,minlength=Npix)       #izdelava mape
print ("max,min,mean=",mapa.max(), mapa.min(),mapa.mean())
b=cm.BuPu                                                          #      
b.set_under("w")                                                        #
hp.mollview(mapa, cmap=b, title='Vsi dogodki', cbar=True, xsize=1400, unit='Gostota dogodkov [dogodkov/piksel]')     #izris mape
ct.mollaxes()
hp.graticule(coord=('E'))                                               #
plt.show()

delta=(mapa/np.mean(mapa))-1

alm=hp.map2alm(delta)

Cl=ct.crosscl(alm,galm,lmax=ic_lmax,ignorem0=True)

nbar=len(data)/(4*np.pi)
flat=[1/nbar]*(ic_lmax+1)

mapa1=ct.randmap(n,Nside,theta)
delta1=(mapa1/np.mean(mapa1))-1
alm1=hp.map2alm(delta1,lmax=lmax)
Cl1=ct.almtocl(alm1,lmax,ignorem0=True)

plt.figure()
plt.plot(Cl, label='Kros - spekter moči')
#plt.plot(flat)
#plt.title('Power spectrum of IceCube data')
#plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,np.max(l))
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell^m$')
plt.show()
