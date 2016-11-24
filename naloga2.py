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
ra*=(np.pi/180)
n=len(data)
#phi=np.random.uniform(0,2*np.pi,n)

#icecube podatki

pixi=hp.ang2pix(Nside,theta,ra)            #določitev pikslov, katerim pripadajo koordinate
mapa=np.bincount(pixi,minlength=Npix)
print ("max,min,mean=",mapa.max(), mapa.min(),mapa.mean())
b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa, cmap=b, title='Vsi dogodki', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#histogram dogodkov po deklinaciji

bins = np.linspace(-1, 1, 50)
plt.hist(np.sin(dec), bins, log=False)
plt.xlabel('$\sin{(dec)}$')
plt.ylabel('Število dogodkov')
plt.xlim(-1,1)
plt.figure(figsize=(20,20))
plt.show()

# naključnih 10 map

sum_rand_mapa=np.zeros(len(mapa))

for i in range(10):
    phi=np.random.uniform(0,2*np.pi,n)
    
    pixi1=hp.ang2pix(Nside,theta,phi)            
    sum_rand_mapa+=np.bincount(pixi1,minlength=Npix).T

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(sum_rand_mapa/10, cmap=b, title='Vsota naključnih map', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#####################

# najljučna mapa

phi=np.random.uniform(0,2*np.pi,n)
#u=np.random.uniform(0,1,n)

#theta1=np.arccos(2*u-1)

pixi1=hp.ang2pix(Nside,theta,phi)
mapa1=np.bincount(pixi1,minlength=Npix)

#####################

delta=(mapa/np.mean(sum_rand_mapa.T/10))-1
delta1=(mapa1/np.mean(sum_rand_mapa.T/10))-1

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(delta, cmap=b, title='Delta polje dogodkov', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

alm=hp.map2alm(delta)
alm1=hp.map2alm(delta1)

S=0
Cl=[]

for l in range(3*Nside):
    for m in range(1,l+1):      #m od 1 dalje, da zanemarimo al0
        w=(l+(m/2)*(383-m))//1
        S=S+(np.abs(alm[w]))**2
    if l==0:
        Cl.append(0)
    else:
        Cl.append((1/(l))*S)
    S=0

Cl=np.array(Cl)
nbar=len(data)/(4*np.pi)
flat=[1/nbar]*192
#Cl=hp.anafast(delta)

#Cl1=hp.alm2cl(alm)
S=0
Cl1=[]

for l in range(3*Nside):
    for m in range(1,l+1):      #m od 1 dalje, da zanemarimo al0
        w=(l+(m/2)*(383-m))//1
        S=S+(np.abs(alm1[w]))**2
    if l==0:
        Cl1.append(0)
    else:
        Cl1.append((1/(l))*S)
    S=0

Cl1=np.array(Cl1)
    
plt.figure()
#plt.plot(np.log10(Cl), label='Power spectrum icecube podatkov')
#plt.plot(np.log10(flat), label='$log_{10}(1/\overline{n})$ ')
#plt.plot(np.log10(Cl1), label='Power spectrum random mape po $\phi$')
plt.plot(np.log10(Cl)-np.log10(Cl1), label=' Power spectrum icecube\n podatkov brez shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$log_{10}C_\ell$')
plt.show()    

Clp=Cl-Cl1
C_l=np.zeros(12)
s=0

for i in range(12):
    for j in range(16):
        q=i*16+j
        s=s+Clp[q]
    C_l[i]=s/16
    s=0
    
plt.figure()
plt.plot((C_l), label=' Povprečen power spectrum\n icecube podatkov')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('16 $\ell$ na bin ')
plt.ylabel('$C_\ell$ bin')
plt.show()  

plt.figure()
plt.plot((Cl-Cl1), label=' Power spectrum icecube\n podatkov brez shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()
