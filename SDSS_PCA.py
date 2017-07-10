import numpy as np
import pandas as pd
from load_spec_files import load_spec_files as LSF
from load_spec_files import checkfiles,make_spec_names
from sklearn import decomposition,neighbors
from sklearn.neighbors import KDTree,BallTree
from plotting import *

class SDSS_PCA:
    def __init__(self,masterfile='./random_SDSS_specs.csv',fluxdf=None,inputfile=None):
        #Read in master file with info on each object
        self.master=pd.read_csv(masterfile)
        if fluxdf!=None:
            self.fluxdf=fluxdf
        elif inputfile!=None:
            self.fluxdf=pd.read_csv(inputfile)

    def cut_master(self,imax):
        #Cut master down to size
        if imax>len(self.master): print 'imax greater than length of master'
        self.master=self.master[:imax]
        
    def cut_on_z(self,zmin=0,zmax=8):
        #Make a cut in redshift space
        self.master=self.master[(self.master.z.values>=zmin)&(self.master.z.values<=zmax)]

    def set_flux(self,infile,indexkey='ID',na_values=None):
        #Load in flux array from a csv file, rather than
        #loading in spectra fits files and applying smoothing
        self.fluxdf=pd.read_csv(infile,na_values=na_values)
        try:
            self.fluxdf.set_index(indexkey,inplace=True)
        except KeyError:
            '{} not found in {}'.format(indexkey,infile)

    def prune_master(self,spec_dir='./spec_dir/'):
        #Deletes rows without corresponding spectrum file
        plates,mjds,fibers,ids=self.master.plate.values,self.master.mjd.values,self.master.fiberid.values,self.master.specobjid.values
        specfiles=make_spec_names(plates,mjds,fibers,prefix=spec_dir)
        files_exist=checkfiles(specfiles)
        self.master=self.master[files_exist]
    

    def set_wavelengths(self,wavmin=0,wavmax=10000):
        self.wavelengths=np.linspace(wavmin,wavmax,np.shape(self.fluxdf)[1])

    def load_spec_files(self,spec_dir='./spec_dir',smooth_wid=10,wavstep=None,wavmin=0,wavmax=10000,savefile=None):
        #Load in files for each spectra. Corrects all spectra to rest-frame
        #and smooths down to a grid defined by wavstep, wavmin, and wavmax
        #Set savefile to write output to a file so it can be loaded later
        redshifts,plates,mjds,fibers,ids=self.master.z.values,self.master.plate.values,self.master.mjd.values,self.master.fiberid.values,self.master.specobjid.values
        self.fluxdf=LSF(redshifts,plates,mjds,fibers,spec_dir=spec_dir,smooth_wid=smooth_wid,wavstep=wavstep,wavmin=wavmin,wavmax=wavmax,savefile=savefile,ids=ids)
        self.wavelengths=np.linspace(wavmin,wavmax,np.shape(self.fluxdf)[1])

    def DoPCA(self,n_components='mle'):
        #Perform PCA decomposition of flux array, then transform
        #flux array into the space of PCA components
        try:
            self.fluxdf
        except AttributeError:
            print 'Must set fluxdf'
            return
        X = self.fluxdf.values
        if n_components=='mle':
            svd_solver == 'full'
        else:
            svd_solver='auto'
        self.pca = decomposition.PCA(n_components=n_components,svd_solver=svd_solver)
        self.flux_pca=self.pca.fit_transform(X)

    def check_target(self,target):
        #Set target to master.class values if unset
        if np.shape(target)==():
            if target==None:
                try:
                    return self.master['class'].values
                except:
                    print 'master not set up correctly: no class column'
                    return
        else:
            return target
    
    def NNClassify(self,target=None,train_perc=0.9,n_neighbors=5, algorithm='kd_tree',weights='distance'):
        #Perform nearest neighbor classification in PCA component space
        #Uses a fraction of the flux array as the training set, defined
        #by train_perc. The rest is used as the test set
        try:
            self.flux_pca,self.master
        except AttributeError:
            print 'Must set flux_pca and master'
            return
        target=self.check_target(target)
        train_df=self.flux_pca[:int(train_perc*np.shape(self.flux_pca)[0])]
        train_X,train_y=train_df,target[:int(train_perc*len(self.flux_pca))]
        test_X,test_y=self.flux_pca[int(train_perc*np.shape(self.flux_pca)[0]):],target[int(train_perc*np.shape(self.flux_pca)[0]):]
        self.clf=neighbors.KNeighborsClassifier(n_neighbors,weights=weights,algorithm=algorithm)
        self.clf.fit(train_X,train_y)
        self.predicted_y=self.clf.predict(test_X)
        
    def ComparePredictions(self,target=None,check_values=None,imax=None,verbose=False,returnperc=False):
        #Compares predictions to true values. Set verbose to True to
        #print out every comparison. Otherwise, just prints out the
        #percentage predicted correctly
        try:
            self.predicted_y,self.master
        except AttributeError:
            print 'Must set predicted_y and master'
            return
        target=self.check_target(target)
        if check_values==None:check_values=target[-len(self.predicted_y):]
        if not(verbose):
            perc_correct=np.sum(check_values==self.predicted_y)*1./len(self.predicted_y)*100
            print 'Predicted Correctly: {:.2f}'.format(perc_correct)
            if returnperc: return perc_correct
        else:
            masterlen=len(target)
            if imax==None: imax=len(self.predicted_y)
            if imax>masterlen:imax=masterlen
            for i in range(0,imax):
                print check_values[i],self.predicted_y[i]
            
    def PlotPCAComponent(self,component,doShow=False,clear=True,fignum=None):
        #Plots the specificed PCA component
        try:
            self.pca.components_
        except AttributeError:
            print 'Must run pca first'
            return
        try:
            plot_spectrum(self.wavelengths,self.pca.components_[component],doShow=doShow,clear=clear,fignum=fignum)
        except:
            plot_spectrum(self.wavelengths,component,doShow=doShow,clear=clear,fignum=fignum)

    def PlotPCADecomp(self,lightcurve,max_components=None,savefile=None,colors=['cyan','blue','magenta','red','pink','orange','yellow','green','gray','brown','purple','silver'],fignum=None):
        #Plots the PCA decomposition of one lightcurve
        try:
            self.pca.components_,self.wavelengths
        except AttributeError:
            print 'Must run pca first;set wavelengths'
            return
        if fignum==None: fignum=1
        plot_spectrum(self.wavelengths,self.fluxdf.iloc[lightcurve],color='k',lw=2,fignum=fignum,clear=False)
        if max_components==None: max_components=np.shape(self.pca.components_)[0]
        if max_components>np.shape(self.pca.components_)[0]:max_components=np.shape(self.pca.components_)[0]
        for icomp in range(0,max_components):
            plot_spectrum(self.wavelengths,self.pca.components_[icomp]*self.flux_pca[lightcurve][icomp],color=colors[icomp%len(colors)],lw=1,fignum=fignum,clear=False)
        if savefile!=None: plt.savefig(savefile)
        
    def PlotPCAModel(self,lightcurve,max_components=None,savefile=None,colors=['cyan','blue','magenta','red','pink','orange','yellow','green','gray','brown','purple','silver'],fignum=None,plottype=True,ignore_components=[]):
        #Plots the combined model for a lightcurve from all
        #PCA components and compares to data
        try:
            self.pca.components_,self.wavelengths
        except AttributeError:
            print 'Must run pca first;set wavelengths'
            return
        if fignum==None: fignum=1
        plot_spectrum(self.wavelengths,self.fluxdf.iloc[lightcurve],color='k',lw=2,fignum=fignum,clear=True)
        if max_components==None: max_components=np.shape(self.pca.components_)[0]
        if max_components>np.shape(self.pca.components_)[0]:max_components=np.shape(self.pca.components_)[0]
        model=np.zeros(len(self.wavelengths))
        for icomp in np.delete(np.arange(0,max_components),ignore_components):
            model+=self.pca.components_[icomp]*self.flux_pca[lightcurve][icomp]
        plot_spectrum(self.wavelengths,model,color='red',lw=1,fignum=fignum,clear=False)
        if plottype:
            ax=plt.gca()
            plt.text(0.05,0.95,'Pred: {}\nTrue: {}'.format(self.predicted_y[lightcurve],self.master['class'].values[lightcurve-len(self.predicted_y)]),transform=ax.transAxes,horizontalalignment='left',verticalalignment='top')
        if savefile!=None: plt.savefig(savefile)

    def Plot2DComp(self,comp1,comp2,target=None,train_perc=0.9,colors=['pink','green','cyan'],alpha=0.25,savefile=None):
        #Plots the training and test set in a 2D space defined
        #by two of the PCA components.
        try:
            self.predicted_y,self.master
        except AttributeError:
            print 'Must set predicted_y and master'
            return
        plt.clf()
        target=self.check_target(target)
        train_df=self.flux_pca[:int(train_perc*np.shape(self.flux_pca)[0])]
        train_X,train_y=train_df,target[:int(train_perc*len(self.flux_pca))]
        test_X,test_y=self.flux_pca[int(train_perc*np.shape(self.flux_pca)[0]):],target[int(train_perc*np.shape(self.flux_pca)[0]):]
        classfs=np.unique(np.append(train_y,test_y))
        c_dict={classfs[x]: colors[x] for x in np.arange(len(classfs))}

        for classf in classfs:
            plt.scatter(train_X[train_y==classf][:,comp1],train_X[train_y==classf][:,comp2],color=c_dict[classf],s=6,alpha=alpha)
        for classf in classfs:
            plt.scatter(test_X[self.predicted_y==classf][:,comp1],test_X[self.predicted_y==classf][:,comp2],s=26,marker='d',color=c_dict[classf],label=classf)
        for classf in classfs:
            plt.scatter(test_X[(test_y==classf)&(self.predicted_y!=classf)][:,comp1],test_X[(test_y==classf)&(self.predicted_y!=classf)][:,comp2],s=26,marker='x',color=c_dict[classf],edgecolor='None',lw=2)
        plt.legend(frameon=False)
        if savefile!=None: plt.savefig(savefile)

    def find_wrong(self):
        bool_wrong=self.predicted_y!=self.master['class'][-len(self.predicted_y):]
        self.wrong=np.arange(len(self.bool_wrong))[bool_wrong.values]

