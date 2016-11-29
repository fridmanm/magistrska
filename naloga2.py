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


Nside=64
Npix=12*Nside**2        #Število pikslov na sferi

#Branje podatkov iz icecube eksperimenta

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
b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa, cmap=b, title='Vsi dogodki', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#histogram dogodkov po deklinaciji

dec*=(np.pi/180)

bins = np.linspace(-1, 1, 50)
plt.hist(np.sin(dec), bins, log=False)
plt.xlabel('$\sin{(dec)}$')
plt.ylabel('Število dogodkov')
plt.xlim(-1,1)
plt.figure(figsize=(20,20))
plt.show()

# naključnih 10 map icecube podatkov po phi

sum_rand_mapa=np.zeros(len(mapa))

for i in range(10):
    phi=np.random.uniform(0,2*np.pi,n)
    
    pixi1=hp.ang2pix(Nside,theta,phi)            
    sum_rand_mapa+=np.bincount(pixi1,minlength=Npix).T

#Izris povprečne mape iz 10 map

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(sum_rand_mapa/10, cmap=b, title='Vsota naključnih map', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

# najljučna mapa icecube podatkov po phi

phi=np.random.uniform(0,2*np.pi,n)

pixi1=hp.ang2pix(Nside,theta,phi)
mapa1=np.bincount(pixi1,minlength=Npix)

#delta polji originalne mape icecube podatkov in naklučna mapa po phi

delta=(mapa/np.mean(sum_rand_mapa.T/10))-1
delta1=(mapa1/np.mean(sum_rand_mapa.T/10))-1

#izris delta polja icecube podatkov

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(delta, cmap=b, title='Delta polje dogodkov', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#določitev koeficientov razvoja po sfernih harmonikih za obe delta polji

alm=hp.map2alm(delta)
alm1=hp.map2alm(delta1)

#izračun spektra moči brez načinov m=0 za icecube podatke

S=0
Cl=np.zeros(3*Nside)            #map2alm določi 3*Nside l-jev v splošnem in tako dolg bo graf Cl

for l in range(3*Nside):        #l indeks od alm
    for m in range(1,l+1):      #m indeks od alm, m od 1 dalje, da zanemarimo al0
        i=int(l+(m/2)*(6*Nside-1-m))      #indeks v 1D tabeli alm, s strukturo alm=(a00,a01,a02,...,a11,a12,...a22,...). mora biti int, ker gre za indeks
        S=S+(np.abs(alm[i]))**2     #vsota kvadratov alm po m za nek l
    if l==0:
        Cl[0]=0                     #za l=0 je Cl=0, ker ni a00 člena
    else:
        Cl[l]=((1/(l))*S)           #izračun Cl=(1/l)*sum_m |alm|^2
    S=0

#konstantna funkcija 1/nbar
    
nbar=len(data)/(4*np.pi)
flat=[1/nbar]*192

#izračun spektra moči brez načinov m=0 za naključno mapo po phi

S=0
Cl1=np.zeros(3*Nside)

for l in range(3*Nside):        
    for m in range(1,l+1):      
        i=int(l+(m/2)*(6*Nside-1-m))
        S=S+(np.abs(alm1[i]))**2
    if l==0:
        Cl1[0]=0
    else:
        Cl1[l]=((1/(l))*S)
    S=0
    
#Izris spektra moči    
    
plt.figure()
#plt.plot(np.log10(Cl), label='Power spectrum icecube podatkov')
#plt.plot(np.log10(flat), label='$log_{10}(1/\overline{n})$ ')
#plt.plot(np.log10(Cl1), label='Power spectrum random mape po $\phi$')
plt.plot(np.log10(Cl)-np.log10(Cl1), label=' Power spectrum (log) icecube\n podatkov brez shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$log_{10}C_\ell$')
plt.show()    

#povprečenje po l

Clp=Cl-Cl1              #odštetje shot-noise
Cls_in_bin=16           #število l v enem binu
C_l=np.zeros(3*Nside/Cls_in_bin)    #dolžina nove tabele povprečenih podatkov
error=np.zeros(3*Nside/Cls_in_bin)      #tabela za napake
s=0
w=0

for i in range(3*Nside//Cls_in_bin):
    for j in range(Cls_in_bin):
        l=i*Cls_in_bin+j
        s=s+l*Clp[l]
        w=w+l
    C_l[i]=s/w
    error[i]=np.sqrt(2/nbar/w)          #izračun napak
    s=0
    w=0
    
#izris povprčenega in shot-noise free spektra moči
    
l=np.linspace(0,3*Nside/Cls_in_bin-1,num=3*Nside/Cls_in_bin)    #določitev osi zaradi izrisa napak

plt.figure()
plt.plot((C_l), label=' Povprečen power spectrum\n icecube podatkov')
plt.errorbar(l,C_l,yerr=error,fmt='o')
plt.legend(loc=0)
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
