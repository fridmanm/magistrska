#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 22:52:15 2018

@author: mitja
"""

from __future__ import print_function, division
import numpy as np
import healpy as hp

def mollaxes():                 #axes labeling on the mollweide projection in equatorial coordinates
    hp.projtext(-168., 2., '12h', lonlat=True, coord='E')
    hp.projtext(-144., 2., '14h', lonlat=True, coord='E')
    hp.projtext(-114., 2., '16h', lonlat=True, coord='E')
    hp.projtext(-84., 2., '18h', lonlat=True, coord='E')
    hp.projtext(-54., 2., '20h', lonlat=True, coord='E')
    hp.projtext(-24., 2., '22h', lonlat=True, coord='E')
    hp.projtext(6., 2., '0h', lonlat=True, coord='E')
    hp.projtext(36., 2., '2h', lonlat=True, coord='E')
    hp.projtext(66., 2., '4h', lonlat=True, coord='E')
    hp.projtext(96., 2., '6h', lonlat=True, coord='E')
    hp.projtext(126., 2., '8h', lonlat=True, coord='E')
    hp.projtext(156., 2., '10h', lonlat=True, coord='E')
    hp.projtext(179., 2., '12h', lonlat=True, coord='E')
    hp.projtext(180., 28., '$30^o$', lonlat=True, coord='E')
    hp.projtext(180., 60., '$60^o$', lonlat=True, coord='E')
    #hp.projtext(23., 78., '$90^o$', lonlat=True, coord='E')
    hp.projtext(180., -32., '  $-30^o$', lonlat=True, coord='E')
    hp.projtext(180., -63., '     $-60^o$', lonlat=True, coord='E')
    #hp.projtext(57., -84., '$-90^o$', lonlat=True, coord='E')
    hp.projtext(180., 0., ' $0^o$', lonlat=True, coord='E')
    hp.projtext(14.,-6.,'RA',lonlat=True, coord='E')
    hp.projtext(-180.,0.,'       Dec',lonlat=True, coord='E')
    
#izdelava naključne mape s specifično strukturo deklinacije ali brez
    
def randmap(n,Nside,theta):     
    Npix=12*Nside**2
    phi=np.random.uniform(0,2*np.pi,n)
    if np.sum(theta)==0:
        u=np.random.uniform(0,1,n)
        theta=np.arccos(2*u-1)
    
    pixi1=hp.ang2pix(Nside,theta,phi)
    randmap=np.bincount(pixi1,minlength=Npix)
    return randmap

#izračun spektra moči z in brez načinov m=0

def almtocl(alm,lmax,ignorem0=False):                   #funkcija za izračun cl iz alm
    S=0
    Cl=np.zeros(lmax+1)            #map2alm določi 3*Nside l-jev v splošnem in tako dolg bo graf Cl
    if ignorem0:
        start=1
        sym=1
    else:
        start=0
        sym=2
    
    for l in range(lmax+1):        #l indeks od alm
        for m in range(start,l+1):      #m indeks od alm; m od 1 dalje, da zanemarimo al0
            i=int(l+(m/2)*(2*(lmax+1)-1-m))      #indeks v 1D tabeli alm, s strukturo alm=(a00,a10,a20,...,a11,a21,...a22,...). mora biti int, ker gre za indeks
            if m==0:
                S=S+(np.abs(alm[i]))**2     #vsota kvadratov alm po m za nek l
            else:
                S=S+sym*(np.abs(alm[i]))**2
        if l==0 and ignorem0:
            Cl[0]=0                    #za l=0 je Cl=0, ker ni a00 člena
        elif ignorem0:
            Cl[l]=((1/(l))*S)           #izračun Cl=(1/l)*sum_m |alm|^2
        else:
            Cl[l]=((1/(2*l+1))*S)
        S=0
    return Cl

#poškatlano povprečje

def clavg(cl,lmax,Cls_in_bin,nbar):                          #funkcija za povprečenje
    C_l=np.zeros((lmax+1)//Cls_in_bin)    #dolžina nove tabele povprečenih podatkov
    error=np.zeros((lmax+1)//Cls_in_bin)      #tabela za napake
    s=0
    w=0
    for i in range((lmax+1)//Cls_in_bin):    
        for j in range(Cls_in_bin):     #izračun povprečja škatle
            l=i*Cls_in_bin+j
            s=s+(l)*cl[l]
            w=w+(l)
        C_l[i]=s/w
        error[i]=np.sqrt(2/(w))/nbar          #izračun napak
        s=0
        w=0
    return C_l, error

#chi2 test

def chi2(num, err, hipoteza):                   #definicija funkcije chi square
    chi2=0
    for i in range(len(num)):                   #vsota kvadratov normalno porazdeljenih številk
        chi2+=(num[i]-hipoteza)**2/(err[i]**2)
    return chi2

def crosscl(alm,alm2,lmax,ignorem0=False):
    S=0
    Cl=np.zeros(lmax+1)            #map2alm določi 3*Nside l-jev v splošnem in tako dolg bo graf Cl
    if ignorem0:
        start=1
        sym=1
    else:
        start=0
        sym=2
    
    for l in range(lmax+1):        #l indeks od alm
        for m in range(start,l+1):      #m indeks od alm; m od 1 dalje, da zanemarimo al0
            i=int(l+(m/2)*(2*(lmax+1)-1-m))      #indeks v 1D tabeli alm, s strukturo alm=(a00,a10,a20,...,a11,a21,...a22,...). mora biti int, ker gre za indeks
            if m==0:
                S=S+(alm[i]*np.conjugate(alm2[i]))     #vsota produktov alm od različnih map po m za nek l
            else:
                S=S+sym*(alm[i]*np.conjugate(alm2[i]))
        if l==0 and ignorem0:
            Cl[0]=0                    #za l=0 je Cl=0, ker ni a00 člena
        elif ignorem0:
            Cl[l]=((1/(l))*S)           #izračun Cl=(1/l)*sum_m |alm|^2
        else:
            Cl[l]=((1/(2*l+1))*S)
        S=0
    return Cl

def rearange_alm(lmax,alm):     #to preoblikovanje velja samo za začetno obliko: a00, a10, a11, a20, a21, a22, a30, a31, a32, a33, ...
    almmatrix=np.zeros((lmax+1,lmax+1))+np.zeros((lmax+1,lmax+1))*1j

    for i in range(lmax+1):
        for j in range(lmax+1):
            if j<=i:
                almmatrix[i][j]=alm[(i*(i+1)//2)+j]
            else:
                almmatrix[i][j]=0+0*1j
                
    almmatrix=np.transpose(almmatrix)
    
    alm1=almmatrix[np.triu_indices(lmax+1)]
    
    return alm1                     #preoblikuje jo v obliko, kot je definirana v funkcijah almtocl in crosscl

def eq2gal(alpha,delta):
    alpha=alpha*(np.pi/180)
    delta=delta*(np.pi/180)
    
    alphaG=192.85948*(np.pi/180)
    deltaG=27.12825*(np.pi/180)
    lomega=32.93192*(np.pi/180)
    
    b=np.arcsin(np.sin(delta)*np.sin(deltaG)+np.cos(delta)*np.cos(deltaG)*np.cos(alpha-alphaG))
    l=np.arccos(np.cos(delta)*np.sin(alpha-alphaG)/np.cos(b))+lomega
    aomega=np.arcsin(np.cos(lomega))+alphaG
    
    return b, l, aomega