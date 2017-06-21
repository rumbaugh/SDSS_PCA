import numpy as np
import pandas as pd
from load_spec_files import load_spec_files as LSF
from sklearn import decomposition,neighbors
from sklearn.neighbors import KDTree,BallTree

class SDSS_PCA:
    def __init__(self,masterfile='./random_SDSS_specs.csv',fluxdf=None,inputfile=None):
        self.master=pd.read_csv(masterfile)
        if fluxdf!=None:
            self.fluxdf=fluxdf
        elif inputfile!=None:
            self.fluxdf=pd.read_csv(inputfile)

    def cut_master(self,imax):
        if imax>len(self.master): print 'imax greater than length of master'
        self.master=self.master[:imax]

    def set_flux(self,infile):
        self.fluxdf=pd.read_csv(infile)

    def load_spec_files(self,spec_dir='./spec_dir',smooth_wid=10,wavstep=None,wavmin=0,wavmax=10000,savefile=None):
        redshifts,plates,mjds,fibers,ids=self.master.z.values,self.master.plate.values,self.master.mjd.values,self.master.fiberid.values,self.master.specobjid.values
        self.fluxdf=LSF(redshifts,plates,mjds,fibers,spec_dir=spec_dir,smooth_wid=smooth_wid,wavstep=wavstep,wavmin=wavmin,wavmax=wavmax,savefile=savefile,ids=ids)

    def DoPCA(self,n_components='mle'):
        try:
            self.fluxdf
        except NameError:
            print 'Must set fluxdf'
            return
        X = self.fluxdf.values
        if n_components=='mle':
            svd_solver == 'full'
        else:
            svd_solver='auto'
        self.pca = decomposition.PCA(n_components=n_components,svd_solver=svd_solver)
        self.pca.fit(X)
        self.flux_pca=self.pca.transform(X)
    
    def KDTreeClassify(self,train_perc=0.9,n_neighbors=5, leaf_size=30, metric='euclidean',weights='distance'):
        try:
            self.flux_pca,self.master
        except NameError:
            print 'Must set flux_pca and master'
            return
        #try:
        #    self.sourcetype
        train_df=self.flux_pca[:int(train_perc*np.shape(self.flux_pca)[0])]
        train_X,train_y=train_df,self.master['class'].values[:int(train_perc*len(self.flux_pca))]
        test_X,test_y=self.flux_pca[int(train_perc*np.shape(self.flux_pca)[0]):],self.master['class'].values[int(train_perc*np.shape(self.flux_pca)[0]):]
        self.clf=neighbors.KNeighborsClassifier(n_neighbors,weights=weights,algorithm='kd_tree')
        self.clf.fit(train_X,train_y)
        self.predicted_y=self.clf.predict(test_X)
        
    def ComparePredictions(self,check_values=None,imax=None,verbose=False):
        try:
            self.predicted_y,self.master
        except NameError:
            print 'Must set predicted_y and master'
            return
        if check_values==None:check_values=self.master['class'].values[-len(self.predicted_y):]
        if not(verbose):
            perc_correct=np.sum(check_values==self.predicted_y)*1./len(self.predicted_y)*100
            print 'Predicted Correctly: {:.2f}'.format(perc_correct)
        else:
            masterlen=len(self.master['class'].values)
            if imax==None: imax=len(self.predicted_y)
            if imax>masterlen:imax=masterlen
            for i in range(0,imax):
                print check_values[i],self.predicted_y[i]
            
