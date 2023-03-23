
##
## By Elim Thompson (11/27/2020)
##
## This is an AI class that simulates a triage AI device given an empirical
## ROC curve via a CSV file. The data points in CSV file are used to re-
## build the diseased and non-diseased histograms assuming a bi-normal
## distribution. In this assumption, non-diseased histogram is centered
## at 0 with a standard deviation of 1. The mu and sigma of the diseased
## histogram are determined based on the fit. All fits are performed when
## the AI object is first initiated. During simulation, the AI instance
## will determine the AI-call of a given patient based on its disease
## truth status.
## 
##
## 10/28/2022
## ----------
## * Add in a new class method to accept a single operating threshold
###########################################################################

################################
## Import packages
################################ 
import numpy, pandas, scipy, matplotlib, os
import statsmodels.api as sm

matplotlib.use ('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore")

import patient

############################################
## Functions related to normal distribution
############################################
norm_cdf  = lambda x, mu, sigma: scipy.stats.norm.cdf (x, mu, sigma)
norm_pdf  = lambda x, mu, sigma: scipy.stats.norm.pdf (x, mu, sigma)
norm_sf   = lambda x, mu, sigma: scipy.stats.norm.sf (x, mu, sigma)
norm_roc  = lambda x, a, b: scipy.stats.norm.cdf (a + b*scipy.stats.norm.ppf (x))
norm_iroc = lambda y, a, b: scipy.stats.norm.cdf (scipy.stats.norm.ppf (y)/b - a/b)

################################
## Define AI class
################################ 
class AI (object):

    ''' A class to simulate AI's diagnostic performance. It can be
        initialized via a CSV file with raw data points along an ROC
        or a single operating Se, Sp threshold. If using raw data
        points, a bi-normal distribution is assumed to fit the ROC
        curve. During simulation, the key function is is_positive()
        which generates AI call based on the disease (truth) status
        of the simulated patient.    
    '''

    def __init__ (self):
        
        ''' Initialize an AI object. Will be built from either
            empirical ROC or a single operation threshold.
        '''
        
        ## AI name
        self._AIname = None
        ## SeThresh and SpThresh are the operating threshold
        self._SeThresh, self._SpThresh = None, None

        ## Parameters based on class method
        #  1. If reading data points of an empirical ROC curve 
        self._from_empiricalROC = False
        self._rocFile = None
        self._rocFPFs, self._rocTPFs = None, None
        self._r2, self._norm_a, self._norm_b = None, None, None
        #  2. If given an operating point
        self._from_opThresh = False

    @classmethod
    def build_from_empiricalROC (cls, AIname, rocFile, SeThresh):

        ''' Initialize the AI object with an input CSV file with data points
            along an empirical ROC curve. To obtain the underlying diseased
            and non-diseased distributions, a bi-normal distribution is
            assumed. Non-diseased distribution is a normal distribution
            centered at 0 with a width of 1, and the mu and sigma of the
            diseased distribution are fitted results. During simulation, the
            AI call of a patient is randomly assigned from the corresponding
            distribution.  
            
            inputs
            ------
            AIname (str): a unique name for this AI device
            rocFile (str): path to the csv file with the empirical ROC with
                           two columns and no header. First column must be
                           true-positive rate (Se), and second column is
                           false-positive rate (1-Sp).
            SeThresh (float): sensitivity threshold. SpThresh will be obtained
                              from the fitted ROC curve.
                              
            output
            ------
            anAI (AI): an AI instance
        '''
        
        ## Initialize an instance
        anInstance = cls ()
        ## Set parameters
        anInstance._AIname = AIname
        anInstance.rocFile = rocFile
        anInstance.SeThresh = SeThresh
        anInstance._from_empiricalROC = True
        
        return anInstance

    @classmethod
    def build_from_opThresh (cls, AIname, SeThresh, SpThresh):
        
        ''' Initialize the AI object with an operating point of Se and Sp
            
            inputs
            ------
            AIname (str): a unique name for this AI device
            SeThresh (float): sensitivity threshold
            SpThresh (float): specificity threshold

            output
            ------
            anAI (AI): an AI instance
        '''        
        
        ## Initialize an instance
        anInstance = cls ()
        ## Set parameters
        anInstance._AIname = AIname
        anInstance.SeThresh = SeThresh
        anInstance.SpThresh = SpThresh
        anInstance._from_opThresh = True
        
        return anInstance

    @property
    def doPlot (self): return self._doPlot
    @doPlot.setter
    def doPlot (self, doPlot):
        if not isinstance (doPlot, bool):
            raise IOError ('Input doPlot is not a boolean (either True or False)')
        self._doPlot = doPlot

    @property
    def rocFile (self): return self._rocFile
    @rocFile.setter
    def rocFile (self, rocFile):
        try:
            if not os.path.exists (rocFile):
                raise IOError ('Input rocFile does not exist: {0}'.format (rocFile))
        except:
            raise IOError ('Input rocFile is not a file path: {0}'.format (rocFile))
        
        self._rocFile = rocFile

    @property
    def AIname (self): return self._AIname

    @property
    def norm_mu (self): return self._norm_a / self._norm_b

    @property
    def norm_sigma (self): return 1 / self._norm_b

    @property
    def SeThresh (self): 
        return self._SeThresh
    @SeThresh.setter
    def SeThresh (self, SeThresh):
        self._SeThresh = SeThresh

    @property
    def SpThresh (self):
        if self._from_empiricalROC:
            return 1-norm_iroc (self.SeThresh, self._norm_a, self._norm_b)
        else:
            return self._SpThresh
    @SpThresh.setter
    def SpThresh (self, SpThresh):
        self._SpThresh = SpThresh

    @property
    def ratingThreshold (self):
        thresh = scipy.stats.norm.ppf (self.SpThresh, 0, 1)   
        return numpy.nan_to_num (thresh)

    def _read_AI_dataset (self):

        ''' Read the CSV file with ROC data points. The file is expected to
            have two columns; first column is the false-positive rate, and
            the second column is the true-positive rate.
        '''

        try:
            df = pandas.read_csv (self.rocFile, names=['FPF', 'TPF'])
            sorted_index = numpy.argsort (df['FPF'])
            return df['FPF'][sorted_index], df['TPF'][sorted_index]
        except:
            raise IOError ('Failed to read rocFile. Make sure the file has two columns (TPR, FPR) without no header.')
            
    def _format_ROC_subplot (self, axis, xlim=[0, 1], ylim=[0.5, 1], xticks=None, yticks=None,
                             xticklabels=None, yticklabels=None, ylabel=None, xlabel=None,
                             title=None, do_legend=False, legend_loc=4, legend_ncol=1):
        
        ''' Format axes in each subplot in the PDF file when requested. 
        '''
        
        axis.set_xlim (xlim)
        axis.set_ylim (ylim)
        
        xticks = numpy.linspace (xlim[0], xlim[-1], 5) if xticks is None else xticks 
        yticks = numpy.linspace (ylim[0], ylim[-1], 5) if yticks is None else yticks
        axis.set_yticks (yticks)
        axis.set_xticks (xticks)
    
        for ytick in axis.get_yticks():
            axis.axhline (y=ytick, color='gray', alpha=0.2, linestyle=':', linewidth=0.2)
        for xtick in axis.get_xticks():
            axis.axvline (x=xtick, color='gray', alpha=0.2, linestyle=':', linewidth=0.2)

        if xticklabels is not None: axis.set_xticklabels (xticklabels, fontsize=10)
        if yticklabels is not None: axis.set_yticklabels (yticklabels, fontsize=10)
        
        if do_legend: axis.legend (loc=legend_loc, ncol=legend_ncol, prop={'size':12})
        if title is not None: axis.set_title (title, fontsize=15)
        if xlabel is not None: axis.set_xlabel (xlabel, fontsize=12)
        if ylabel is not None: axis.set_ylabel (ylabel, fontsize=12)        

    def _plot_ROC (self, outPath):
        
        ''' Generate a PDF file with the empirical data points and the fitted results.
            Left  : diseased and non-diseased distribution from fitted results 
            Middle: raw data points and fitted curve in fitted space
            Right : raw data points and fitted curve in ROC space
            
            input
            -----
            outPath (str): Output path where the PDF file is stored.
        ''' 
        
        h  = plt.figure (figsize=(22.5, 6))
        gs = gridspec.GridSpec (1, 3, wspace=0.3, hspace=0.3)
        gs.update (bottom=0.1)
       
        # +-----------------------------------------------------
        # | Left: diseased and non-diseased PDFs
        # +-----------------------------------------------------
        axis = h.add_subplot (gs[0])
        # Plot PDFs using fitted parameter
        xs = numpy.linspace (-15, 15, 1000)
        for gtype in ['diseased', 'non-diseased']:
            ys = norm_pdf (xs, 0, 1) if gtype == 'non-diseased' else \
                 norm_pdf (xs, self.norm_mu, self.norm_sigma)
            fit_label = r"non-disease ($\mu$ = 0; $\sigma$ = 1)" if gtype == 'non-diseased' else \
                        r"disease ($\mu$ = {0:.2f}; $\sigma$ = {1:.2f})".format (self.norm_mu, self.norm_sigma)
            linestyle = '-' if gtype == 'diseased' else '--'
            color = '#d95f02' if gtype == 'diseased' else '#1b9e77'
            axis.plot (xs, ys, color=color, linestyle=linestyle, linewidth=2.0, label=fit_label)
        # Plot rating threshold
        axis.axvline (self.ratingThreshold, color='darkgray', linestyle=':', linewidth=2.0)
        # Format this plot
        self._format_ROC_subplot (axis, xlim=[min (xs), max(xs)], ylim=[0, 0.8], 
                                  ylabel='PDF', xlabel='relative rating',
                                  title='PDFs using fitted params',
                                  do_legend=True, legend_loc=1, legend_ncol=1)

        # +-----------------------------------------------------
        # | Middle: ROC in fitting scale
        # +-----------------------------------------------------
        axis = h.add_subplot (gs[1])
        # Plot raw data points  
        axis.scatter (scipy.stats.norm.ppf  (self._rocFPFs), scipy.stats.norm.ppf  (self._rocTPFs),
                      marker='o', s=20, color='#1f78b4', label='Normal; {0}'.format (self.AIname))
        # Plot fitted curves
        xs = numpy.linspace (0.000001, 1, 1000)
        ys = norm_roc (xs, self._norm_a, self._norm_b)
        xvalues, yvalues = scipy.stats.norm.ppf(xs), scipy.stats.norm.ppf(ys)
        fit_label = r"fit ($\mu$ = {0:.2f}; $\sigma$ = {1:.2f})".format (self.norm_mu, self.norm_sigma)            
        axis.plot (xvalues, yvalues, color='darkgray', linestyle='-', linewidth=2.0, label=fit_label)
        # Text R2
        axis.text (0.05, 0.95, r'$R^2$ = {0:.2f}'.format (self._r2), horizontalalignment='left',
                   fontsize=12, verticalalignment='center', transform = axis.transAxes, color='black')
        # Format this plot
        is_valid = numpy.logical_and (numpy.isfinite (xvalues), numpy.isfinite (yvalues))
        self._format_ROC_subplot (axis, xlim=[min (xvalues[is_valid]), max(xvalues[is_valid])],
                                  ylim=[min (yvalues[is_valid]), max(yvalues[is_valid])], 
                                  ylabel='Inverse Gauss CDF ({0})'.format ('TPF'),
                                  xlabel='Inverse Gauss CDF ({0})'.format ('FPF'),
                                  title='ROC in fitting space',
                                  do_legend=True, legend_loc=4, legend_ncol=1)

        # +-----------------------------------------------------
        # | Last ROC in regular scale
        # +-----------------------------------------------------
        axis = h.add_subplot (gs[2])
        # Plot raw data points 
        axis.plot (self._rocFPFs, self._rocTPFs, color='#1f78b4', linestyle='-',
                    linewidth=2.0, label='Normal; {0}'.format (self.AIname))
        # Plot operating threshold
        axis.scatter (1-self.SpThresh, self.SeThresh, marker='x', s=50, color='#1f78b4',
                      label='thresh; {0}'.format (self.AIname))
        # Plot fitted curves
        ys = numpy.linspace (0.000001, 1, 1000)
        xs = norm_iroc (ys, self._norm_a, self._norm_b)        
        fit_label = r"Norm ($\mu$ = {0:.2f}; $\sigma$ = {1:.2f})".format (self.norm_mu, self.norm_sigma)
        axis.plot (xs, ys, color='#1b9e77', linestyle='--', linewidth=2.0, label=fit_label)
        # Plot thresh from fitted params
        y = self.SeThresh
        x = norm_iroc (self.SeThresh, self._norm_a, self._norm_b)
        axis.scatter (x, y, marker='x', s=50, color=color)
        # Format this plot
        self._format_ROC_subplot (axis, xlim=[0, 1], ylim=[0, 1], ylabel='TPF', xlabel='FPF',
                                  title='ROC in regular space', do_legend=True, legend_loc=4, legend_ncol=1)

        h.savefig (outPath + 'ROC_' + self._AIname + '.pdf')
        plt.close('all')        

    def fit_ROC (self, doPlots=False, outPath=None):

        ''' Fit a and b from the ROC assuming bi-normal distribution.
                a = |mu+ - mu-|/sigma+
                b = sigma-/sigma+
            where mu+ and sigma+ are the mean and standard deviation of the
            diseased histogram, whereas mu- and sigma- are the ones for the
            non-diseased histogram. Because we are interested in the separation
            between the two histograms (not the absolute locations of both
            histograms), mu- and sigma- are set at 0 and 1.
            
            If asked to do plot, call self._plot_ROC to plot the fit info.
            
            inputs
            ------
            doPlots (bool): set to True if want to get more information 
                            about the fit. Default False.
            outPath (str): path of the output plot. Required if doPlot is True.            
        '''

        ## Read in ROC raw data points
        self._rocFPFs, self._rocTPFs = self._read_AI_dataset()

        ## Clean up the data points before fitting 
        xvalues = scipy.stats.norm.ppf (self._rocFPFs)
        yvalues = scipy.stats.norm.ppf (self._rocTPFs)
        is_valid = numpy.logical_and (numpy.isfinite (xvalues), numpy.isfinite(yvalues)) 
        
        ## Call statsmodel package to perform the fit
        xvalues = sm.add_constant (xvalues) 
        model = sm.OLS (yvalues[is_valid], xvalues[is_valid])
        results = model.fit()
        
        ## Store the fitted results and r2 value
        self._r2 = results.rsquared
        self._norm_a, self._norm_b = results.params

        ## Generate plot if asked
        if not doPlots: return
        ## Check if outPath is valid
        try:
            if not os.path.exists (outPath):
                raise IOError ('Input outPath does not exist: {0}'.format (outPath))
        except:
            raise IOError ('Input outPath is not a path: {0}'.format (outPath))
        ## Create PDF plot
        self._plot_ROC(outPath)
        
    def is_positive (self, is_diseased=False):
        
        ''' Generate AI call of a patient based on its diseased (truth) status.
        
            input
            -----
            is_diseased (bool): truth status of the patient
            
            output
            ------
            is_positive (bool): AI-call of the patient
        '''
        
        if self._from_empiricalROC:
            ## If using empirical ROC data points, is_positive score
            ## is from normal distribution, whose mean and standard
            ## deviation depend on the diseased (truth) status
            mu = 0 if not is_diseased else self.norm_mu
            sigma = 1 if not is_diseased else self.norm_sigma
            score = numpy.random.normal (loc=mu, scale=sigma)
            return bool (int (score > self.ratingThreshold))
        
        elif self._from_opThresh:
            ## If using single operating (Se, Sp), score is generated
            ## using uniform distribution from 0 to 1. 
            score = numpy.random.uniform ()
            if is_diseased: 
                return bool (int (score <= self.SeThresh))
            return bool (int (score > self.SpThresh))
        
        ## If neither from ROC data points nor signle operating point,
        ## this AI instance is not fully initialized.        
        raise IOError ('AI is not initialized. Please provide either an empirical ROC via a CSV file or an operating Se, Sp point.')

    def run_pivotal (self, nPatients=5000, prevalence=0.1, fractionED=0.1, arrivalRate=0.1):

        ''' Run pivotal study to quickly check AI diagnostic performance from
            simulation
            
            inputs
            ------
            nPatients (int): number of simulated patients
            prevalence (float): number of diseased / total number of simulated non-emergent patients
            fractionED (float): number of emergent / total number of simulated patients
            arrivalRate (float): patient's overall arrival rate i.e. inverse of average interarrival times
 
            output
            ------
            diagnostic performance (dict): true-positive, true-negative, false-positive, false-negative rates  
        '''

        TP, TN, FP, FN = 0, 0, 0, 0
        timestamp = pandas.to_datetime ('2020-01-01 00:00')
    
        for i in range (nPatients):
            apatient = patient.patient (i, prevalence, fractionED, arrivalRate, timestamp)
            apatient.is_positive = self.is_positive (is_diseased=apatient.is_diseased)
            if apatient.is_diseased and apatient.is_positive: TP += 1
            if not apatient.is_diseased and not apatient.is_positive: TN += 1
            if not apatient.is_diseased and apatient.is_positive: FP += 1
            if apatient.is_diseased and not apatient.is_positive: FN += 1
        
        return {'TP':TP/(TP+FN), 'TN':TN/(TN+FP), 'FP':FP/(TN+FP), 'FN':FN/(TP+FN)}

