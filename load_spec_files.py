import numpy as np
import pyfits as py
import pandas as pd
import os

def checkfiles(specfiles):
    worked=np.zeros(len(specfiles),dtype='bool')
    for ispec,specfile in zip(np.arange(len(specfiles)),specfiles):
        worked[ispec]=os.path.isfile(specfile)
    return worked

def make_spec_names(plates,mjds,fibers,prefix=''):
    return np.array(['{}spec-{:04d}-{:05d}-{:04d}.fits'.format(prefix,x,y,z) for x,y,z in zip(plates,mjds,fibers)],dtype='|S{}'.format(25+len(prefix)))

def load_spec_files(redshifts,plates=None,mjds=None,fibers=None,spec_dir='./spec',smooth_wid=10,wavstep=None,wavmin=0,wavmax=10000,savefile=None,ids=None):
    inpcheck=np.sum(np.array([plates==None,mjds==None,fibers==None]))
    if inpcheck>0:
        if inpcheck<3:
            print 'plates, mjds, and fibers must ALL be provided'
        try:
            specfiles=os.listdir(spec_dir)
        except OSError:
            print "ls to the directory '{}' failed".format(spec_dir)
            return
        plates,mjds,fibers=np.zeros(len(specfiles),dtype='<i4'),np.zeros(len(specfiles),dtype='<i4'),np.zeros(len(specfiles),dtype='<i4')
    else:
        if ((len(plates)!=len(mjds))|(len(plates)!=len(fibers))):
            print 'plates, fibers, and mjds must all have same length'
            return
        specfiles=make_spec_names(plates,mjds,fibers)
    if wavstep==None: wavstep=smooth_wid*2
    wavelengths=np.arange(wavmin,wavmax,wavstep)
    fluxes=np.zeros((len(specfiles),len(wavelengths)))
    for i in range(0,len(specfiles)):
        spechdu=py.open('{}/{}'.format(spec_dir,specfiles[i]))
        specdata=spechdu[1].data
        tmp_flux,tmp_wav=specdata['flux'],10**(specdata['loglam'])
        #Transform wavelength to rest-frame using redshifts
        tmp_wav/=(1.+redshifts[i])
        for wavcentmp,iw in zip(wavelengths,np.arange(len(wavelengths))):
            fluxes[i][iw]=np.sum(tmp_flux[np.abs(tmp_wav-wavcentmp)<=smooth_wid])
    fluxdf=pd.DataFrame(fluxes,columns=wavelengths)
    if ids!=None:
        fluxdf['ID']=ids
        fluxdf.set_index('ID',inplace=True)
    if savefile!=None:
        if ids!=None:
            fluxdf.to_csv(savefile)
        else:
            fluxdf.to_csv(savefile,index=False)
    return fluxdf
