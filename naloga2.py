#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:52:08 2016

@author: mifridman
"""

from __future__ import print_function
import numpy as np
import healpy as hp
import matplotlib.cm as cm
import matplotlib.pyplot as plt


Nside=64                 #NSIDE
Npix=12*Nside**2

data=np.loadtxt('IceCube-59__Search_for_point_sources_using_muon_events.txt',skiprows=2)
dec,ra,time,log10E,angerr=data.T
print ("data len=",len(data))

theta=90-dec                #pretvorba deklinacije v sferično koordinato theta
theta*=(np.pi/180)
n=len(data)
phi=np.random.uniform(0,2*np.pi,n)

#icecube podatki

pixi=hp.ang2pix(Nside,theta,phi)            #določitev pikslov, katerim pripadajo koordinate
mapa=np.bincount(pixi,minlength=Npix)
print ("max,min,mean=",mapa.max(), mapa.min(),mapa.mean())
b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa, cmap=b, title='Vsi dogodki', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#histogram dogodkov po deklinaciji

bins = np.linspace(-90, 90, 50)
plt.hist(dec, bins, log=False)
plt.xlabel('dec')
plt.ylabel('Število dogodkov')
plt.xlim(-90,90)
plt.figure(figsize=(20,20))
plt.show()

# naključnih 10 map

sum_rand_mapa=np.zeros(len(mapa))

for i in range(60):
    ra=np.random.uniform(0,2*np.pi,n)
    u=np.random.uniform(0,1,n)
    
    theta1=np.arccos(2*u-1)
    
    pixi1=hp.ang2pix(Nside,theta1,ra)            
    sum_rand_mapa+=np.bincount(pixi1,minlength=Npix).T

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(sum_rand_mapa/60, cmap=b, title='Vsota naključnih map /60', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#####################

# najljučna mapa

ra=np.random.uniform(0,2*np.pi,n)
u=np.random.uniform(0,1,n)

theta1=np.arccos(2*u-1)

pixi1=hp.ang2pix(Nside,theta1,ra)
mapa1=np.bincount(pixi1,minlength=Npix)

#####################

delta=(mapa/(sum_rand_mapa/60))-1

#b=cm.Greys                                                           #      
#b.set_under("w")                                                        #
#hp.mollview(delta, cmap=b, title='Delta polje dogodkov', cbar=True, xsize=1400)     #izris mape
#hp.graticule(coord=('E'))                                               #
#plt.show()

alm=hp.map2alm(delta)

S=0
Cl=[]

for l in range(3*Nside):
    for m in range(1,l+1):      #m od 1 dalje, da zanemarimo al0
        #if m<193:
            #print(alm[m])
        w=(l+(m/2)*(383-m))//1
        S=S+(np.abs(alm[w]))**2
    Cl.append((1/(l+1))*S)
    S=0

Cl=np.array(Cl)
nbar=len(data)/(4*np.pi)
flat=[1/nbar]*192
#Cl=hp.anafast(delta)

Cl1=hp.alm2cl(alm)
    
plt.figure()
plt.plot(np.log10(Cl), label='Power spectrum (zanka)')
plt.plot(np.log10(flat), label='1/nbar (logaritem)')
#plt.plot(np.log10(Cl1), label='Power spectrum (alm2cl)')
plt.legend(loc=4)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$log_{10}C_\ell$')
plt.show()    
