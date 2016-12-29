#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 12:44:02 2016

@author: mifridman
"""

from __future__ import print_function
import numpy as np
import healpy as hp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

Nside=64
Npix=12*Nside**2        #Število pikslov na sferi

#Branje podatkov iz icecube eksperimenta

data=np.loadtxt('IceCube-59__Search_for_point_sources_using_muon_events.txt',skiprows=2)
dec,ra,time,log10E,angerr=data.T
print ("data len=",len(data))

n=len(data)

#histogram dogodgov porazeljen po ločljivosti

print(min(angerr), max(angerr))
bins = np.linspace(0, 10, 50)
plt.hist(angerr, bins, log=True)
plt.xlabel('Napaka [deg]')
plt.ylabel('Število dogodkov')
plt.xlim(0,10)
plt.figure(figsize=(20,20))
plt.show()

print(np.median(angerr),np.mean(angerr))

#deljenje dogodkov na dve območji ločljivosti 

dec1=[]         #tuple array, ker ni znana dolžina tabel pri deljenju
ra1=[]
dec2=[]
ra2=[]

errlim=1.5

for i in range(len(data)):      #zanka za filtriranje podatkov
    if angerr[i]<=errlim:          #določitev meje na roko iz histograma
        dec1.append(dec[i])
        ra1.append(ra[i])
    else:
        dec2.append(dec[i])
        ra2.append(ra[i])
        
dec1=np.array(dec1)             #pretvorba iz tuple v numpy array
ra1=np.array(ra1)
dec2=np.array(dec2)             #pretvorba iz tuple v numpy array
ra2=np.array(ra2)


theta1=90-dec1                #pretvorba deklinacije v sferično koordinato theta
theta1*=(np.pi/180)         #v radiane
ra1*=(np.pi/180) 
theta2=90-dec2                
theta2*=(np.pi/180)

#Izris mape z dogodki pri boljši ločljivosti

pixi1=hp.ang2pix(Nside,theta1,ra1)            #določitev pikslov, katerim pripadajo koordinate
mapa1=np.bincount(pixi1,minlength=Npix)
print ("max,min,mean=",mapa1.max(), mapa1.min(),mapa1.mean())
b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa1, cmap=b, title='Dogodki z napakami $\sigma_{ang}\leq%d$' %errlim, cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#Izris mape z dogodki pri slabši ločljivosti

pixi2=hp.ang2pix(Nside,theta2,ra2)            #določitev pikslov, katerim pripadajo koordinate
mapa2=np.bincount(pixi2,minlength=Npix)
print ("max,min,mean=",mapa2.max(), mapa2.min(),mapa2.mean())
b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(mapa2, cmap=b, title='Dogodki z napakami $\sigma_{ang}>%f$' %errlim, cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

# naključnih 10 map icecube podatkov s boljšo ločljivostjo po phi

sum_rand_mapa=np.zeros(len(mapa1))

for i in range(10):
    phi=np.random.uniform(0,2*np.pi,len(theta1))
    
    pixi1=hp.ang2pix(Nside,theta1,phi)            
    sum_rand_mapa+=np.bincount(pixi1,minlength=Npix)/10
'''
s=pd.Series(sum_rand_mapa)
indeks_za_norm=s[s != 0].index[-1]      #izračun indeksa od katerega so vsi piksli v mapa1 enaki 0, za normalizacijo v delta mapi
print(indeks_za_norm)               #
'''

#Izris povprečne mape iz 10 map

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(sum_rand_mapa, cmap=b, title='Normirana vsota naključnih map za boljšo ločljivost', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

# najljučna mapa icecube podatkov po phi

def randmap():
    phi=np.random.uniform(0,2*np.pi,len(theta1))
    
    pixi1=hp.ang2pix(Nside,theta1,phi)
    randmap=np.bincount(pixi1,minlength=Npix)
    return randmap
    
mapa3=randmap()

#delta polji originalne mape icecube podatkov in naklučna mapa po phi

delta1=(mapa1/np.mean(sum_rand_mapa))-1
delta3=(mapa3/np.mean(sum_rand_mapa))-1

#izris delta polja icecube podatkov

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(delta1, cmap=b, title='Delta polje dogodkov', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#določitev koeficientov razvoja po sfernih harmonikih za obe delta polji

alm1=hp.map2alm(delta1)
alm3=hp.map2alm(delta3)

#spekter moči za icecube podatke
    
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

Cl1=almtocl(alm1)

#konstantna funkcija 1/nbar
    
nbar=len(theta1)/(4*np.pi)
flat=[1/nbar]*192

#speketr moči za naključno mapo po phi

Cl3=almtocl(alm3)

plt.figure()
plt.plot(Cl1, label=' Power spectrum icecube\n podatkov')
plt.plot(Cl3, label=' Power spectrum random\n podatkov')
plt.plot(flat, label=' Shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()

Clp=Cl1-Cl3              #odštetje shot-noise
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
    
C_l,error=clavg(Clp)
    
#izris povprčenega (s fitom) in shot-noise free spektra moči

fit=np.loadtxt('theory_fit.txt')
ell,fit_Cl=fit.T

plt.figure()
plt.plot(Clp, label=' Power spectrum icecube\n podatkov z boljšo ločljivostjo\n brez shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()

l=np.linspace(Cls_in_bin/2-1,3*Nside-Cls_in_bin/2-1,num=3*Nside/Cls_in_bin)    #določitev osi zaradi izrisa napak

plt.figure()
plt.plot(l,(C_l), label=' Povprečen power spectrum\n icecube podatkov (boljša ločljivost)')
plt.plot(ell,3*fit_Cl,label='Teoretična krivulja')
plt.errorbar(l,C_l,yerr=error,fmt='o')
plt.legend(loc=0)
plt.xlabel('$\ell$ ')
plt.ylabel('$\overline{C_\ell}$')
plt.show()

def chi2(num, err, hipoteza):                   #definicija funkcije chi square
    chi2=0
    for i in range(len(num)):                   #vsota kvadratov normalno porazdeljenih številk
        chi2+=(num[i]-hipoteza)**2/(err[i]**2)
    return chi2
    
print('chi2(high res) =',chi2(C_l,error,0))

# naključnih 10 map icecube podatkov s slabšo ločljivostjo po phi

sum_rand_mapa=np.zeros(len(mapa2))

for i in range(10):
    phi=np.random.uniform(0,2*np.pi,len(theta2))
    
    pixi2=hp.ang2pix(Nside,theta2,phi)            
    sum_rand_mapa+=np.bincount(pixi2,minlength=Npix)/10

s=pd.Series(sum_rand_mapa)
indeks_za_norm=s[s != 0].index[-1]      #izračun indeksa od katerega so vsi piksli v mapa1 enaki 0, za normalizacijo v delta mapi
print(indeks_za_norm)               #

#Izris povprečne mape iz 10 map

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(sum_rand_mapa, cmap=b, title='Normirana vsota naključnih map za slabšo ločljivost', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

# najljučna mapa icecube podatkov po phi

def randmap():
    phi=np.random.uniform(0,2*np.pi,len(theta2))
    
    pixi2=hp.ang2pix(Nside,theta2,phi)
    randmap=np.bincount(pixi2,minlength=Npix)
    return randmap
    
mapa3=randmap()

#delta polji originalne mape icecube podatkov in naklučna mapa po phi

delta2=(mapa2/np.mean(sum_rand_mapa))-1
delta3=(mapa3/np.mean(sum_rand_mapa))-1

#izris delta polja icecube podatkov

b=cm.Greys                                                           #      
b.set_under("w")                                                        #
hp.mollview(delta2, cmap=b, title='Delta polje dogodkov', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#določitev koeficientov razvoja po sfernih harmonikih za obe delta polji

alm2=hp.map2alm(delta2)
alm3=hp.map2alm(delta3)

#spekter moči za icecube podatke

Cl2=almtocl(alm2)

#konstantna funkcija 1/nbar
    
nbar=len(theta2)/(4*np.pi)
flat=[1/nbar]*192

#speketr moči za naključno mapo po phi

Cl3=almtocl(alm3)

plt.figure()
plt.plot(Cl2, label=' Power spectrum icecube\n podatkov')
plt.plot(Cl3, label=' Power spectrum random\n podatkov')
plt.plot(flat, label=' Shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()

Clp2=Cl2-Cl3              #odštetje shot-noise
    
C_l2,error2=clavg(Clp2)
    
#izris povprčenega (s fitom) in shot-noise free spektra moči

fit=np.loadtxt('theory_fit.txt')
ell,fit_Cl=fit.T

plt.figure()
plt.plot(Clp2, label=' Power spectrum icecube\n podatkov s slabšo ločljivostjo\n brez shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()

l=np.linspace(Cls_in_bin/2-1,3*Nside-Cls_in_bin/2-1,num=3*Nside/Cls_in_bin)    #določitev osi zaradi izrisa napak

plt.figure()
plt.plot(l,(C_l2), label=' Povprečen power spectrum\n icecube podatkov (slabša ločljivost)')
plt.plot(ell,30*fit_Cl,label='Teoretična krivulja')
plt.errorbar(l,C_l2,yerr=error2,fmt='o')
plt.legend(loc=0)
plt.xlabel('$\ell$ ')
plt.ylabel('$\overline{C_\ell}$')
plt.show()

def chi2(num, err, hipoteza):                   #definicija funkcije chi square
    chi2=0
    for i in range(len(num)):                   #vsota kvadratov normalno porazdeljenih številk
        chi2+=(num[i]-hipoteza)**2/(err[i]**2)
    return chi2
    
print('chi2(low res) =',chi2(C_l2,error2,0))

