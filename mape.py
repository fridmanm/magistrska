#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:49:18 2016

@author: mifridman
"""

import numpy as np
import healpy as hp
import matplotlib.cm as cm
import matplotlib.pyplot as plt

data=np.loadtxt('IceCube-59__Search_for_point_sources_using_muon_events.txt',skiprows=2)

n=107569                #število vseh dogodkov
N=512                   #NSIDE
ra=[]
dec=[]

for i in range(n):                  #pisanje rektascenzije in deklinacije v ločena seznama
    for j in range(2):              #
        if j==0:                    #
            dec.append(data[i][j])  #
        elif j==1:                  #
            ra.append(data[i][j])   #

dect=90-np.array(dec)               #pretvorba deklinacije v sferično koordinato theta
dect=np.array(dect)*(np.pi/180)     #v radiane
ra=np.array(ra)*(np.pi/180)         #

#print(dect)

m1=hp.ang2pix(N,dect,ra)            #določitev pikslov, katerim pripadajo koordinate



m=[0 for i in range(12*N**2)]       #pisanje v tabelo: 0 na pikslu, kjer ni izvora in 1 na pikslu, kjer je izvor
for j in range(len(m1)):            #
    t=m1[j] 	                   #
    m[t]=1                          #
m=np.array(m)                       #
#print(len(m))

b=cm.plasma                                                             #      
b.set_under("w")                                                        #
hp.mollview(m, cmap=b, title='Vsi dogodki', cbar=False, xsize=1000)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()                                                              #

#E=[]

#for i in range(n):
#    E.append(data[i][3])
#E=np.array(E)

#E=10**np.array(E)
#E1=np.around(E, decimals=0)
#E1=E1.astype(int)
#print(min(E), max(E))

#A=np.bincount(E1)


#H=np.histogram(A, bins=10)
#plt.hist(H)