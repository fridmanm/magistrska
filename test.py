#!/usr/bin/env python
from __future__ import print_function, division
import claw
import numpy as np, healpy as hp,matplotlib.pyplot as plt, pickle as cp
from os.path import isfile
import scipy.stats as st


flatWeight=True
zeroPower=False
Ng=1000
Ngtest=10000
ignorem0=True
pname="./MeasureCl_%i_%i_%i.pickle"%(flatWeight,zeroPower,Ng)


def main():
    ClT=makeTestCl()
    weight,noise=makeTestWindowNoise(ClT)
    M=getMeasureCl(ClT,weight,noise)
    test(M,ClT)
    
def makeTestCl():
    if zeroPower:
        return claw.Cl(lmaxdl=(60,10))
    else:
        return claw.Cl(lmaxdl=(60,10),vals=[3.,2.,5.,10.,3.,1.])

def makeTestWindowNoise(ClT):
    Nside=ClT.Nside
    Npix=12*Nside**2
    theta,phi=hp.pix2ang(Nside,np.arange(Npix))
    noise=10.*(1.+theta/np.pi)
    if flatWeight:
        weight=1
    else:
        weight=1./noise**2
    return weight, noise

def getMeasureCl(ClT,weight,noise):
    if (isfile(pname)):
        print("Loading pickled version... (delete ",pname,"if changing params.")
        M=cp.load(open(pname,'rb'))
    else:
        M=claw.MeasureCl(ClT,weight,noise,Ng=Ng,ignorem0=ignorem0)
        print ("Getting noise bias...")
        M.getNoiseBias()
        if flatWeight:
            M.setIdentityCoupling()
        else:
            print("Getting coupling matrix...")
            M.getCouplingMat()
        print("Getting covariance matrix...")
        M.getCovMat(ClT)
        cp.dump(M,open(pname,'wb'),-1)

    return M

def test(M,ClT):    
    print("Testing Algorithm")
    res=[]
    chi2=[]
    chi2d=[]
    for cc in range(Ngtest):
        #generate test problem
        cls=ClT.Cl
        mp=hp.synfast(cls,ClT.Nside,verbose=False)
        mp+=np.random.normal(0.,M.Noise)
        ClM=M.getEstimate(mp)
        res.append(ClM.vals)
        chi2.append(ClM.chi2(ClT))
        chi2d.append(ClM.chi2diag(ClT))
        print("Go #",cc,"\r",end="")
    print ("\n")
    chi2=np.array(chi2)
    chi2d=np.array(chi2d)
    print(chi2.mean(), chi2.var())
    print(chi2d.mean(), chi2d.var())
    #plt.hist(chi2,bins=50)
    chiaxis=np.linspace(0,90,120)
    histoy,histox,_=plt.hist(chi2, bins=50)
    chiaxy=st.chi2.pdf(chiaxis,3*ClM.Nside/ClM.dl)
    
    plt.figure()
    plt.hist(chi2, bins=50, label='Naključne mape')
    plt.plot(chiaxis,(histoy.max()/chiaxy.max())*st.chi2.pdf(chiaxis,(3*ClM.Nside/ClM.dl)),label='Porazdelitev $\chi^2$ za dof=%d' % int(3*ClM.Nside/ClM.dl))
    #plt.plot(ell,(17*25*(1/2**(3*Nside/Cls_in_bin/2))*(1/gamma(3*Nside/Cls_in_bin/2))*(ell**(3*Nside/Cls_in_bin/2-1))*(np.exp(-(ell/2)))))
    plt.legend()
    plt.xlabel('$\chi^2$')
    plt.ylabel('Število naključnih map')
    #plt.title('Povprečeno z 8 $C_\ell$-ji $\rightarrow$ dof=8')
    plt.xlim(0,40)
    #plt.ylim(0,55)
    plt.show()

    res=np.array(res)
    print ("Truth:",ClT.vals)
    print ("Mean:",res.mean(axis=0))
    print ("Err:",np.sqrt(np.cov(res,rowvar=False).diagonal()/Ngtest))
    print ("Oerr:",np.sqrt(ClM.cov.diagonal()/Ngtest))


if __name__=="__main__":
    main()
