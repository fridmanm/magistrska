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


Nside=64                   #NSIDE
Npix=12*Nside**2

data=np.loadtxt('IceCube-59-_Search_for_point_sources_using_muon_events.txt',skiprows=2)
dec,ra,time,log10E,angerr=data.T
print ("data len=",len(data))

theta=90-dec                #pretvorba deklinacije v sferično koordinato theta
theta*=(np.pi/180)     #v radiane
ra*=(np.pi/180)         #
pixi=hp.ang2pix(Nside,theta,ra)            #določitev pikslov, katerim pripadajo koordinate
mapa=np.bincount(pixi,minlength=Npix)
print ("max,min,mean=",mapa.max(), mapa.min(),mapa.mean())
b=cm.plasma                                                             #      
b.set_under("w")                                                        #
hp.mollview(mapa, cmap=b, title='Vsi dogodki', cbar=False, xsize=1000)     #izris mape
hp.graticule(coord=('E'))                                               #
plt.show()                                                              #

