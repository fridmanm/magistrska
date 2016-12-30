#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 13:52:12 2016

@author: mifridman
"""

from __future__ import print_function
import numpy as np
import healpy as hp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

Nside=128
Npix=12*Nside**2        #Število pikslov na sferi

#Branje podatkov iz icecube eksperimenta

data=np.loadtxt('IceCube-59__Search_for_point_sources_using_muon_events.txt',skiprows=2)
dec,ra,time,log10E,angerr=data.T
print ("data len=",len(data))

n=len(data)
           
rand_mapa=np.random.normal(0.,1.,Npix)

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(rand_mapa, cmap=b, title='Naključna porazdelitev', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

alm=hp.map2alm(rand_mapa)

def almtocl(alm):                   #funkcija za izračun cl iz alm
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
    return Cl
    
#spekter moči
    
Cl=almtocl(alm)
nbar=Npix/(4*np.pi)
flat=[1/nbar]*3*Nside

plt.figure()
plt.plot(Cl, label=' Power spectrum')
plt.plot(flat, label=' Shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()

Cls_in_bin=16          #število l v enem binu

def clavg(cl):                          #funkcija za povprečenje
    C_l=np.zeros(3*Nside//Cls_in_bin)    #dolžina nove tabele povprečenih podatkov
    error=np.zeros(3*Nside//Cls_in_bin)      #tabela za napake
    s=0
    w=0
    for i in range(3*Nside//Cls_in_bin):
        for j in range(Cls_in_bin):
            l=i*Cls_in_bin+j
            s=s+l*cl[l]
            w=w+l
        C_l[i]=s/w
        error[i]=np.sqrt(2/w)/nbar          #izračun napak
        s=0
        w=0
    return C_l, error
    
C_l,error=clavg(Cl-(1/nbar))

l=np.linspace(Cls_in_bin/2-1,3*Nside-Cls_in_bin/2-1,num=3*Nside/Cls_in_bin)    #določitev osi zaradi izrisa napak

plt.figure()
plt.plot(l,(C_l), label=' Povprečen power spectrum')
plt.errorbar(l,C_l,yerr=error,fmt='o')
plt.legend(loc=0)
plt.xlabel('$\ell$ ')
plt.ylabel('$\overline{C_\ell}$')
plt.show()  

sig=np.median(angerr)*np.pi/180
smoothed=hp.sphtfunc.smoothing(rand_mapa,sigma=sig)

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(smoothed, cmap=b, title='Naključna porazdelitev', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

alm1=hp.map2alm(smoothed)
Cl1=almtocl(alm1)

ell=np.linspace(0,3*Nside-1,3*Nside)

plt.figure()
plt.plot(ell,Cl1, label=' Smoothed power spectrum')
plt.plot(ell,Cl, label=' Power spectrum')
plt.plot(ell,Cl1/(np.exp(-(1)*(sig*ell)**2)), label=' Normiran smoothed power spectrum')
plt.plot(flat, label=' Shot noise')
plt.legend(loc=1)
plt.ylim([0,2*10**-4])
plt.xlim(0,300)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()

C_l1,error1=clavg(Cl1-(1/nbar))
C_l2,error2=clavg(Cl1/(np.exp(-(1)*(sig*ell)**2))-(1/nbar))
error2=error1/np.exp(-(1)*(sig*l)**2)

plt.figure(figsize=(7,5))
plt.plot(l,(C_l1), label=' Povprečen smoothed\n power spectrum')
plt.plot(l,(C_l), label=' Povprečen power spectrum')
plt.plot(l,(C_l2), label=' Povprečen normiran smoothed\n power spectrum')
plt.errorbar(l,C_l,yerr=error,fmt='o')
plt.errorbar(l,C_l1,yerr=error1,fmt='^')
plt.errorbar(l,C_l2,yerr=error2,fmt='s')
plt.legend(loc=2)
plt.ylim([-7*10**-5,7*10**-5])
plt.xlim(0,300)
plt.xlabel('$\ell$ ')
plt.ylabel('$\overline{C_\ell}$')
plt.show()  
