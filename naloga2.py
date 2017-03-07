#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:52:08 2016

@author: mifridman
"""

from __future__ import print_function, division
import numpy as np
import healpy as hp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.stats as st
import claw
#from scipy.special import gamma

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
b=cm.BuPu                                                          #      
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
    sum_rand_mapa+=np.bincount(pixi1,minlength=Npix)/10

#Izris povprečne mape iz 10 map

b=cm.BuPu                                                           #      
b.set_under("w")                                                        #
hp.mollview(sum_rand_mapa, cmap=b, title='Normirana vsota naključnih map', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

# najljučna mapa icecube podatkov po phi
def randmap():
    phi=np.random.uniform(0,2*np.pi,n)
    
    pixi1=hp.ang2pix(Nside,theta,phi)
    randmap=np.bincount(pixi1,minlength=Npix)
    return randmap
    
mapa1=randmap()

#delta polji originalne mape icecube podatkov in naklučna mapa po phi

delta=(mapa/np.mean(sum_rand_mapa))-1
delta1=(mapa1/np.mean(sum_rand_mapa))-1

#izris delta polja icecube podatkov

b=cm.BuPu                                                           #      
b.set_under("w")                                                        #
hp.mollview(delta1, cmap=b, title='Delta polje dogodkov', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

#določitev koeficientov razvoja po sfernih harmonikih za obe delta polji

alm=hp.map2alm(delta)
alm1=hp.map2alm(delta1)

#izračun spektra moči brez načinov m=0

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
    
#spekter moči za icecube podatke
    
Cl=almtocl(alm)

#konstantna funkcija 1/nbar
    
nbar=len(data)/(4*np.pi)
flat=[1/nbar]*192

#speketr moči za naključno mapo po phi

Cl1=almtocl(alm1)
    
#Izris spektra moči (logaritemsko)
    
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
plt.plot(Clp, label=' Power spectrum icecube\n podatkov brez shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()

l=np.linspace(Cls_in_bin/2-1,3*Nside-Cls_in_bin/2-1,num=3*Nside/Cls_in_bin)    #določitev osi zaradi izrisa napak

plt.figure()
plt.plot(l,(C_l), label=' Povprečen power spectrum\n icecube podatkov')
plt.plot(ell,3.5*fit_Cl,label='Teoretična krivulja')
plt.errorbar(l,C_l,yerr=error,fmt='o')
plt.legend(loc=0)
plt.xlabel('$\ell$ ')
plt.ylabel('$\overline{C_\ell}$')
plt.show()  

# izračun chi square

def chi2(num, err, hipoteza):                   #definicija funkcije chi square
    chi2=0
    for i in range(len(num)):                   #vsota kvadratov normalno porazdeljenih številk
        chi2+=(num[i]-hipoteza)**2/(err[i]**2)
    return chi2
    
print('chi2(icecube_svoj) =',chi2(C_l,error,0))      #izračun chi square za ice cube podatke

#random povprečeni power spectrumi (za preverjanje porazdelitve chi square)
'''
st_randmap=1000

rand_stat=np.zeros(st_randmap)

for i in range(st_randmap):
    mapa2=randmap()
    delta2=(mapa2/np.mean(sum_rand_mapa))-1
    alm2=hp.map2alm(delta2)
    Cl2=almtocl(alm2)
    Clp1=Cl2-Cl1
    C_l1,error1=clavg(Clp1)
    rand_stat[i]=(chi2(C_l1,error1,0))
    #print(i)
    
#rand_stat=np.array(rand_stat)    

nula=[0]*192

plt.figure()
plt.plot(l,(C_l1), label=' Povprečen power spectrum\n random podatkov')
plt.plot(ell,nula)
plt.errorbar(l,C_l1,yerr=error1,fmt='o')
plt.legend(loc=0)
plt.xlabel('$\ell$ ')
plt.ylabel('$\overline{C_\ell}$')
plt.show()

#print('chi² =',chi2(C_l1,error1,0))

plt.figure()
plt.plot(Clp1, label=' Power spectrum random\n podatkov brez shot noise')
plt.legend(loc=1)
#plt.ylim([-6,-3])
#plt.xlim(0,100)
plt.xlabel('$\ell$')
plt.ylabel('$C_\ell$')
plt.show()

bins = np.linspace(20, 90, 50)
chiaxis=np.linspace(0,90,120)

plt.figure()
plt.hist(rand_stat, bins, log=False, label='Naključne mape')
plt.plot(chiaxis,17*64*st.chi2.pdf(chiaxis,3*Nside/Cls_in_bin),label='Porazdelitev $\chi^2$ za dof=%d' % int(3*Nside/Cls_in_bin))
#plt.plot(ell,(17*25*(1/2**(3*Nside/Cls_in_bin/2))*(1/gamma(3*Nside/Cls_in_bin/2))*(ell**(3*Nside/Cls_in_bin/2-1))*(np.exp(-(ell/2)))))
plt.legend()
plt.xlabel('$\chi^2$')
plt.ylabel('Število naključnih map')
#plt.title('Povprečeno z 8 $C_\ell$-ji $\rightarrow$ dof=8')
#plt.xlim(0,40)
#plt.ylim(0,55)
plt.show()
'''

# Primerjava chi2 testa s claw chi 2 testom

covmat=np.dot(np.diag(error),np.diag(error))     #kovariančna matrika za podatke iz prejšnje analize

c=claw.Cl(lmaxdl=(3*Nside,Cls_in_bin),vals=C_l,cov=covmat)  #icecube Cl
p=claw.Cl(lmaxdl=(3*Nside,Cls_in_bin))      #teoretični Cl (v tem primeru enak 0 za vse ell)

print('chi2(icecube_claw)= ',c.chi2(p))

# Izračun nbar po "vsi phi na vsako deklinacijo" metodi

theta1,phi1=hp.pix2ang(Nside,np.arange(Npix))
print(len(np.unique(theta1)),len(theta1), len(np.unique(phi1)))


thetaun=np.unique(theta1)

thetaall=np.zeros(len(thetaun)*len(data))
phiall=np.zeros(len(thetaun)*len(data))


'''
for i in range(len(thetaun)):      #daljša metoda
    for j in range(len(data)):
        thetaall[i*len(data)+j]=thetaun[i]
        phiall[i*len(data)+j]=ra[j]
'''


for i in range(len(thetaun)):      #krajša metoda
    thetaall[i*len(data):(i+1)*len(data)]=thetaun[i]
    phiall[i*len(data):(i+1)*len(data)]=ra

pixi1=hp.ang2pix(Nside,thetaall,phiall)            #določitev pikslov, katerim pripadajo koordinate
nrmapa=np.bincount(pixi1,minlength=Npix)/len(thetaun)       #izdelava mape
print ("max,min,mean=",nrmapa.max(), nrmapa.min(),nrmapa.mean())
b=cm.BuPu                                                          #      
b.set_under("w")                                                        #
hp.mollview(nrmapa, cmap=b, title='New random', cbar=True, xsize=1400)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()

nbar1=np.mean(nrmapa) #=2.18849690755 (2182817.6044)

print("nbar(na piksel), nbar(na steradian-nenormiran)= ",nbar1,",",len(data)*len(thetaun)/(4*np.pi))

# Claw analiza

M=claw.MeasureCl(c,nbar1,np.sqrt(1/nbar1),m0neg=True)

#mapa=np.float64(mapa)

#M.getNoiseBias()
M.getCovMat(c)
#M.getCouplingMat()

ClM=M.getEstimate(mapa,m0neg=True)
'''
Tukaj vrne error in ne vem zakaj:
    
File "/home/mifridman/anaconda3/lib/python3.5/site-packages/claw/MeasureCl.py", line 53, in _getIM
    return self.bnorm*np.bincount(self.binlist,weights=almsq)

ValueError: operands could not be broadcast together with shapes (13,) (12,) 
'''
plt.figure()
plt.plot(ClM.vals)
plt.show()
