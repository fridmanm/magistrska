#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:49:18 2016
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
theta*=(np.pi/180)     #v radiane
ra*=(np.pi/180)         #
pixi=hp.ang2pix(Nside,theta,ra)            #določitev pikslov, katerim pripadajo koordinate
mapa=np.bincount(pixi,minlength=Npix)
print ("max,min,mean=",mapa.max(), mapa.min(),mapa.mean())
b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa, cmap=b, title='Vsi dogodki', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()                    


print(min(log10E), max(log10E))
bins = np.linspace(2, 7, 50)
plt.hist(log10E, bins, log=True)
plt.xlabel('log10($E_\mu$)')
plt.ylabel('Število dogodkov')
plt.xlim(2,7)
plt.figure(figsize=(20,20))
plt.show()

dec1=[]
ra1=[]

for i in range(len(data)):
    if data[i][3]<=4.5:
        dec1.append(data[i][0])
        ra1.append(data[i][1])
dec1=np.array(dec1)
ra1=np.array(ra1)

theta1=90-dec1                #pretvorba deklinacije v sferično koordinato theta
theta1*=(np.pi/180)     #v radiane
#ra1*=(np.pi/180) 

pixi1=hp.ang2pix(Nside,theta1,ra1)            #določitev pikslov, katerim pripadajo koordinate
mapa1=np.bincount(pixi1,minlength=Npix)
print ("max,min,mean=",mapa1.max(), mapa1.min(),mapa1.mean())
b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa1, cmap=b, title='Dogodki z energijami $log_{10}E_\mu<4,5$', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()


dec2=[]
ra2=[]

for i in range(len(data)):
    if data[i][3]>4.5:
        dec2.append(data[i][0])
        ra2.append(data[i][1])
dec2=np.array(dec2)
ra2=np.array(ra2)

theta2=90-dec2                #pretvorba deklinacije v sferično koordinato theta
theta2*=(np.pi/180)     #v radiane
#ra1*=(np.pi/180) 

pixi2=hp.ang2pix(Nside,theta2,ra2)            #določitev pikslov, katerim pripadajo koordinate
mapa2=np.bincount(pixi2,minlength=Npix)
print ("max,min,mean=",mapa2.max(), mapa2.min(),mapa2.mean())
b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa2, cmap=b, title='Dogodki z energijami $log_{10}E_\mu>4,5$', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#analiza spektra moči

delta=((mapa.T)/np.mean(mapa))-1

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(delta, cmap=b, title='Delta polje dogodkov', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

alm=hp.map2alm(delta)

S=0
Cl=[]

for l in range(3*Nside):
    for m in range(1,l+1):      #m od 1 dalje, da zanemarimo al0
        #if m<193:
            #print(alm[m])
        w=l+(m/2)*(383-m)
        S=S+(np.abs(alm[w]))**2
    Cl.append((1/(l+1))*S)
    S=0

Cl=np.array(Cl)
nbar=len(data)/(4*np.pi)
fit=[1/nbar]*192
#Cl=hp.anafast(delta)
    
plt.figure()
plt.plot(np.log10(Cl), label='Power spectrum (zanka)')
plt.plot(np.log10(fit), label='1/nbar (logaritem)')
plt.legend(loc=4)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$log_{10}C_\ell$')
plt.show()