#
# Object that actually measures Cls
#

from __future__ import print_function, division

import numpy as np
import healpy as hp
import scipy.linalg as la
import copy
class MeasureCl:
    def __init__(self, Cl, Weight, Noise,Ng=1000, m0neg=False):
        """
          When initializing, set it up with Cl object for binning
          and cov matrix prediction.
          Weight is the weight map. It can be arbirary, but Weight=1/Noise**2 is optimal.
          Ng is the number of random samples it takes to get coupling matrix. It needs to be
          surprisingly large.
        """
          
        self.Cl=copy.deepcopy(Cl)
        self.m0neg=m0neg
        self.Weight=Weight
        self.Noise=Noise
        self.Nside=Cl.Nside
        self.lmax=self.Nside*3
        self.Npix=12*self.Nside**2
        self.nbins=self.Cl.nbins
        self.mbins=self.nbins+1 ## mbins includes the edge bin that we are throwing away
        self._getbinlist()
        self.bnorm=1./np.bincount(self.binlist)
        self.Ng=Ng

    def _getbinlist(self):
        lar=[]
        for m in range(0,self.lmax):
            for l in range (m,self.lmax):
                lar.append(l)
        ## now get mapping from lmax to bins
        self.binlist=self.Cl.ndx[lar]
        self.initbinlist=self.binlist

    def _getIM(self,mp, addN=False, m0neg=False):
        if addN:
            mp+=np.random.normal(0,self.Noise)
        mp*=self.Weight
        almsq=abs(hp.map2alm(mp)**2)
        self.binlist=self.initbinlist
        if m0neg:
            index=np.arange(self.lmax-1)    #first 3*Nside elements corespond to m=0 modes
            almsq=np.delete(almsq,index)
            self.binlist=np.delete(self.binlist,index)
        return self.bnorm*np.bincount(self.binlist,weights=almsq)

    def getNoiseBias(self):
        nv=[]
        for cc in range(self.Ng):
            nv.append(self._getIM(np.zeros(self.Npix),addN=True,m0neg=self.m0neg))
        nv=np.array(nv)
        self.nbias=nv.mean(axis=0)
        self.ncov=np.cov(nv,rowvar=False)

    def setIdentityCoupling(self):
        self.coupmat=np.identity(self.mbins)
        self.icoupmat=np.identity(self.mbins)
        
    def getCouplingMat(self):
        cmat=np.zeros((self.mbins,self.mbins))
        for i in range(self.mbins):
            for cc in range(self.Ng):
                clx=np.zeros(self.lmax)
                if (i==self.nbins):
                    clx[self.Cl.lmax[-1]:]=1.0
                else:
                    clx[self.Cl.lmin[i]:self.Cl.lmax[i]]=1.0
                m=hp.synfast(clx,self.Nside,verbose=False)
                cmat[i,:]+=self._getIM(m,m0neg=self.m0neg)
        self.coupmat=cmat/self.Ng
        self.icoupmat=la.inv(self.coupmat)

    def getCovMat(self,thCl=None):
        if not hasattr(self,"coupmat"):
            self.getCouplingMat()
        if (hasattr(self,"nbias") and (thCl is None)):
            tcov=self.ncov
        else:
            clx=thCl.Cl
            nv=[]
            for cc in range(self.Ng):
                m=hp.synfast(clx,self.Nside,verbose=False)
                r=self._getIM(m,addN=True,m0neg=self.m0neg)
                nv.append(r)

            nv=np.array(nv)
            tcov=np.cov(nv,rowvar=False)
        cov=np.dot(self.icoupmat,np.dot(tcov,self.icoupmat.T))
        self.Cl.setCov(cov[:self.nbins, :self.nbins])

    def getEstimate(self,mp,m0neg):
        if not hasattr(self,"coupmat"):
            self.getCouplingMat()
        if not hasattr(self,"nbias"):
            self.getNoiseBias()
        v=self._getIM(mp,m0neg=m0neg)
        self.Cl.setVals(np.dot(self.icoupmat,v-self.nbias)[:-1])
        return self.Cl

