import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(wav,flux,xmin=None,xmax=None,ymin=None,ymax=None,labels=None,plotfile=None,doShow=False,fignum=None,clear=True,color='k',lw=1):
    fluxshared=False
    if np.shape(wav)!=np.shape(flux):
        if (len(np.shape(wav))==1):
            if len(wav)!=np.shape(flux)[1]:
                print 'wav and flux must have same primary dimensions'
                return
            fluxshared=True
        else:
            print 'wav and flux must have same dimensions'
            return
    if plotfile!=None:
        import matplotlib.backends.backend_pdf as bpdf
        pdf=bpdf.PdfPages(plotfile)
    if fignum==None:
        fig=plt.figure()
    else:
        fig=plt.figure(fignum)
    if clear:plt.clf()
    plt.rc('axes',linewidth=2)
    plt.fontsize = 14
    plt.tick_params(which='major',length=8,width=2,labelsize=14)
    plt.tick_params(which='minor',length=4,width=1.5,labelsize=14)
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    try:
        wavlen=np.shape(wav)[1]
        numspec=np.shape(wav)[0]
        if labels==None:
            labels=np.full(numspec,None)
        else:
            if len(labels)!=numspec:
                print "labels length doesn't match"
                labels=np.full(numspec,None)
        for i in range(0,numspec):
            if plotfile!=None:plt.clf()
            if fluxshared:
                plt.plot(wav[i],flux,label=labels[i],color=color,lw=lw)
            else:
                plt.plot(wav[i],flux[i],label=labels[i],color=color,lw=lw)
            if plotfile!=None:fig.savefig(pdf,format='pdf')
        if (plotfile==None) & (np.count_nonzero(labels)>0): plt.legend(frameon=False)
    except:
        plt.plot(wav,flux,color=color,lw=lw)
        if plotfile!=None:fig.savefig(pdf,format='pdf')
    if plotfile!=None: pdf.close()
    if doShow:plt.show(block=False)
    return
