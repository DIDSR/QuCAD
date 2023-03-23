##
## By Elim Thompson (11/27/2020)
##
## This script includes functions that plot image workflow and various
## distributions such as number of patient iamges in queue and their wait
## time for different priority classes.
###########################################################################

################################
## Import packages
################################ 
import numpy, matplotlib

matplotlib.use ('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore")

import calculator

################################
## Define constants
################################
ymin = 1e-5
day_to_second = 60 * 60 * 24
hour_to_second = 60 * 60 
minute_to_second = 60  

queuetypes = ['fifo', 'preresume']
colors = {qtype: color for qtype, color in zip (queuetypes, plt.get_cmap ('Set2').colors)}
colors['diseased'] = '#1f78b4' 
colors['non-diseased'] = '#b2df8a'
colors['interrupting'] = '#b41f78' 
colors['positive'] = '#1f78b4' 
colors['negative'] = '#78b41f'
for gtype, color in zip (['TP', 'FN', 'TN', 'FP'], plt.get_cmap ('Set2').colors):
    colors[gtype] = color
colors['theory'] = 'magenta'
colors['simulation'] = 'darkgray'

################################
## Define lambdas
################################
convert_time = lambda time, time0: (time - time0).total_seconds() / hour_to_second

################################
## Define plotting functions
################################
## +------------------------------------------
## | For timing of cases
## +------------------------------------------
def plot_timing (outFile, records, time0, n=200, qtype='fifo'):
    
    ''' Function to plot image timestamp workflows for the first N cases.

        inputs
        ------
        outFile (str): filename of the output plot including path
        records (dict): key = 'fifo' or 'preresume' for without and with CADt scenarios
                        Each value is a list of patient instances.
        time0 (pandas Timestamp): simulation start time
        n (int): number of cases from the beginning of simulation to be included 
        qtype (str): either 'fifo' or 'preresume' to be plotted
    '''

    ## Extract the first N number of cases (default 200)
    records = sorted (records[qtype][:n])
    
    ## Set up canvas
    h  = plt.figure (figsize=(15, 15))
    gs = gridspec.GridSpec (1, 1)
    gs.update (bottom=0.1)
    axis = h.add_subplot (gs[0])

    ## Extract timestamps (patient arrival trigger time, radiologist open and class time)
    ## and the doctor that is reading the case
    xvalues, yvalues, trigger, close = [], [], [], []
    caseIDs = {'text':[], 'yvalue':[], 'xvalue':[]}
    doctors = {'text':[], 'yvalue':[], 'xvalue':[]} 
    for index, record in enumerate (records):
        pclass = 'E' if record.is_interrupting else \
                 'TP' if record.is_positive and record.is_diseased else \
                 'FP' if record.is_positive and not record.is_diseased else \
                 'FN' if not record.is_positive and record.is_diseased else \
                 'TN'
        caseIDs['text'].append ('#{0} ({1})'.format (record.caseID[-3:], pclass))
        caseIDs['yvalue'].append (len (records) - index)
        caseIDs['xvalue'].append (convert_time (record.trigger_time, time0))
        for n in range (len (record.open_times)):
            openTime, closeTime = record.open_times[n], record.close_times[n]
            xvalues.append (convert_time (openTime, time0))
            yvalues.append (index)
            begin = record.trigger_time if n==0 else record.close_times[n-1]
            trigger.append (convert_time (begin, time0))
            close.append (convert_time (closeTime, time0))
            doctors['text'].append (' {0}'.format (record.doctors[n].split('_')[-1]))
            doctors['yvalue'].append (len (records) - index)
            doctors['xvalue'].append (convert_time (closeTime, time0))            
    # Time bar from patient image arrival to radiologist start reading
    xlower = [x-t for t, x in zip (trigger, xvalues)]
    axis.errorbar (xvalues, len (records) - numpy.array (yvalues), marker=None, color='blue',
                   xerr=[xlower, numpy.zeros(len (xvalues))], label='waiting',
                   elinewidth=2, alpha=0.5, ecolor='blue', linestyle='')
    for x, y, t in zip (caseIDs['xvalue'], caseIDs['yvalue'], caseIDs['text']):
        axis.text (x, y, t, horizontalalignment='right', verticalalignment='center', color='black', fontsize=2)
    # Time bar from radiologist start reading to radiologist closing the case
    xupper = [c-x for c, x in zip (close, xvalues)]
    axis.errorbar (xvalues, len (records) - numpy.array (yvalues), marker=None, color='red',
                   xerr=[numpy.zeros(len (xvalues)), xupper], label='serving',
                   elinewidth=2, alpha=0.5, ecolor='red', linestyle='')
    # Print the doctor index (0, 1, 2, etc.). If only 1 radiologist, always "doctor 0".
    for x, y, t in zip (doctors['xvalue'], doctors['yvalue'], doctors['text']):
        axis.text (x, y, t, horizontalalignment='right', verticalalignment='center', color='black', fontsize=2)    

    ## Format plotting style
    #  x-axis
    axis.set_xlim (0, numpy.ceil (close[-1]))
    axis.set_xlabel ('Time lapse from start time [hr]', fontsize=15)
    axis.tick_params (axis='x', labelsize=12)
    for xtick in axis.get_xticks():
        axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    #  y-axis
    axis.set_ylim (0, len (records)+2)
    for ytick in axis.get_yticks():
        axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
    axis.get_yaxis().set_ticks([])
    #  legend and title
    axis.legend (loc=1, ncol=1, prop={'size':15})
    axis.set_title ('Timing of top {0} cases in {1}'.format (len (records), qtype), fontsize=15)

    h.savefig (outFile)
    plt.close('all')
    
## +------------------------------------------
## | For state prob distributions
## +------------------------------------------
def get_stats (values, weights=None):

    ''' Obtain statistics from a distribution of input values.
        Statistics include 95% C.I. and 1 sigma ranges, as
        well as median and mean.
        
        inputs
        ------
        values (array): values from which a distribution is built
        weights (array): weights of elements in `values`. Must
                            have same length as `values`.
                            
        output
        ------
        stats (dict): statistics of the distribution from values.
                        Keys: 'lower_95cl', 'lower_1sigma', 'median',
                            'mean', 'upper_1sigma', 'upper_95cl'
    '''

    ## If no valid values, return 0's
    sample = numpy.array(values)[numpy.isfinite(values)]
    if len (sample) == 0:
        return {'lower_95cl':0, 'lower_1sigma':0, 'median':0,
                'upper_1sigma':0, 'upper_95cl':0, 'mean':0}
    
    ## Massage input values and apply weights 
    indices = numpy.argsort (sample)
    sample = numpy.array ([sample[i] for i in indices])
    if weights is None: weights = numpy.ones (len (sample))
    w = numpy.array ([weights[i] for i in indices])
        
    ## Extract statistics from cumulative PDF
    stat = {}
    cumulative = numpy.cumsum(w)/numpy.sum(w)
    stat['lower_95cl']   = sample[cumulative>0.025][0] # lower part is at 50%-47.5% = 2.5%        
    stat['lower_1sigma'] = sample[cumulative>0.16][0]  # lower part is at 50%-34% = 16%
    stat['median']       = sample[cumulative>0.50][0]  # median
    stat['mean']         = numpy.average (sample, weights=w)
    stat['upper_1sigma'] = sample[cumulative>0.84][0]  # upper is 50%+34% = 84%
    stat['upper_95cl']   = sample[cumulative>0.975][0] # upper is 50%+47.5% = 97.5%
    
    return stat

def plot_n_patient_distributions (ext, sim_nPatients, stats, params, oneTrial=False):

    ''' Function to plot distributions of the observed number of patients in
        the system *right before a new patient arrives*.

        inputs
        ------
        ext (str): extension of output file name
        sim_nPatients (pandas DataFrame): observed number of patients in system for every patient
                                          typically `trialGen.n_patients_system_df`
        stats (dict): mean, lower/upper 1 sigma and 95% C.I. 
                      typically `trialGen.n_patients_system_stats`
        params (dict): settings for all simulations
        oneTrial (bool): If true, plot one random trial instance
    '''

    ## Show one random trial
    if oneTrial:
        ## Obtain a random trial ID
        allTrialIDs = sim_nPatients['non-interrupting']['trial_id']
        randonInteger = numpy.random.randint (0, len (allTrialIDs))
        trial_id = allTrialIDs[randonInteger]
        ## Extract the patients from that one ID
        sim_nPatients = {gtype: df[df.trial_id==trial_id] for gtype, df in sim_nPatients.items()}
    
    ## 1. Plot # patients in system without CADt when a new patient enters
    ##    Priority classes: interrupting (not in reading list) and interruted (in reading list)
    outFile = params['plotPath'] + 'n_patients_system_distribution_without_CADt' + ext
    plot_n_patient_distribution_total (outFile, sim_nPatients, stats, params, doLogY=True)

    ## 2. Plot # patients in system with CADt when a new patient enters
    ##    Priority classes: interrupting (not in reading list) and AI positive and AI negative in reading list
    outFile = params['plotPath'] + 'n_patients_system_distribution_with_CADt' + ext
    plot_n_patient_distribution_classes (outFile, sim_nPatients, stats, params, doLogY=True)
    
def plot_n_patient_distribution_total (outFile, nPatients, stats, params, doLogY=True):
    
    ''' Function to plot distributions of the observed number of patients in
        the system *right before a new patient arrives* without CADt.

        inputs
        ------
        outFile (str): output plot file name
        nPatients (pandas DataFrame): observed number of patients in system for every patient
                                      typically `trialGen.n_patients_system_df`
        stats (dict): mean, lower/upper 1 sigma and 95% C.I. 
                      typically `trialGen.n_patients_system_stats`
        params (dict): settings for all simulations
        doLogY (bool): If true, plot log y
    '''

    ## 2 plots: interrupting (outside of reading list) and interrupted (in reading list)
    h  = plt.figure (figsize=(16, 6))
    gs = gridspec.GridSpec (1, 2, wspace=0.4, hspace=0.4)
    
    for index, aclass in enumerate (['interrupting', 'non-interrupting']):

        ## Skip the plot for interrupting class if fractionED = 0 i.e. no interrupting patients
        if round (params['fractionED'],4) == 0 and aclass == 'interrupting': continue

        ## Set up subplots and xticks
        subgs = gs[index].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
        values = nPatients[aclass]['fifo']
        hist_bins = numpy.linspace (0, values.max(), int (values.max()+1))
        xticks = hist_bins[::2] if max (values) < 30 else \
                 hist_bins[::4] if max (values) < 70 else \
                 hist_bins[::8] if max (values) < 200 else \
                 hist_bins[::16] if max (values) < 300 else hist_bins[::32]
         
        ## Top plot: distributions
        axis = h.add_subplot (subgs[0])
        #  Simulation
        hist, edges = numpy.histogram (values, bins=hist_bins)
        hist_sum = numpy.sum (hist)
        hist = numpy.r_[hist[0], hist]
        yvalues = hist/hist_sum
        axis.plot (edges[:-1], yvalues[1:], label='sim', color=colors['simulation'], drawstyle='steps-mid',
                   linestyle='-', linewidth=2.0, alpha=0.6)
        yvalues = hist[1:]/hist_sum
        yerrors = numpy.sqrt (hist[1:])/hist_sum
        axis.errorbar (edges[:-1], yvalues, marker=None, color=colors['simulation'], yerr=yerrors,
                       elinewidth=1.5, alpha=0.5, ecolor=colors['simulation'], linestyle='')
        # Theory
        if aclass == 'interrupting':          
            pred = calculator.get_state_pdf_MMn (hist_bins, params['nRadiologists'],
                                                 params['lambdas']['interrupting'],
                                                 params['mus']['interrupting'])
            xvalues, yvalues = edges, pred
        else: 
            cal = calculator.def_cal_MMs (params, lowPriority='non-interrupting',
                                          doDynamic=params['nRadiologists']>3)
            pred = cal.solve_prob_distributions (len (hist_bins))
            ps = numpy.array ([numpy.sum (p) for p in pred])
            xvalues, yvalues = range (0, len (ps)), ps
        axis.plot (xvalues, yvalues, label='theory', color=colors['theory'], linestyle='--', linewidth=2.0, alpha=0.7)
        
        # Format x-axis
        axis.set_xlim (hist_bins[0], hist_bins[-1])
        axis.set_xticks (xticks)
        axis.set_xticklabels ([int (x) for x in xticks], fontsize=7)
        for xtick in axis.get_xticks():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
        axis.get_xaxis().set_ticks([])
        # Format y-axis
        if doLogY: axis.set_yscale("log")
        axis.set_ylim (min (hist/hist_sum), max (hist/hist_sum) + 0.05)        
        axis.set_ylabel ('Normalized counts', fontsize=9)
        for ytick in axis.get_yticks():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
        # Format others
        axis.legend (loc='best', ncol=1, prop={'size':9})
        axis.set_title ('Without CADt; {0}'.format (aclass), fontsize=9)

        ## Bottom plot: mean/95CI
        axis = h.add_subplot (subgs[1])
        #  Simulation
        xlower = [stats['fifo'][aclass]['mean'] - stats['fifo'][aclass]['lower_95cl']]
        xupper = [stats['fifo'][aclass]['upper_95cl'] - stats['fifo'][aclass]['mean']]
        axis.errorbar (stats['fifo'][aclass]['mean'], 1, marker="x", markersize=10, color=colors['simulation'],
                       xerr=[xlower, xupper], elinewidth=3, alpha=0.8, ecolor=colors['simulation'], linestyle='')
        #  Theory
        if aclass == 'interrupting':
            pred = calculator.get_state_pdf_MMn (numpy.linspace (0, 1000, 1001), params['nRadiologists'],
                                                 params['lambdas']['interrupting'],
                                                 params['mus']['interrupting'])
        else:
            cal = calculator.def_cal_MMs (params, lowPriority='non-interrupting',
                                          doDynamic=params['nRadiologists']>3)
            pred = cal.solve_prob_distributions (1001)
            pred = numpy.array ([numpy.sum (p) for p in pred])
        theoryStats = get_stats (numpy.linspace (0, 1000, 1001), pred)
        xlower = [theoryStats['mean'] - theoryStats['lower_95cl']]
        xupper = [theoryStats['upper_95cl'] - theoryStats['mean']]
        axis.errorbar (theoryStats['mean'], 2, marker="x", color=colors['theory'], markersize=10,
                       xerr=[xlower, xupper], elinewidth=3, alpha=0.8, ecolor=colors['theory'], linestyle='')
        
        # Format x-axis
        axis.set_xlim (hist_bins[0], hist_bins[-1])
        axis.set_xlabel ('number of patients ({0} class only)'.format (aclass), fontsize=9)
        axis.set_xticks (xticks)
        axis.set_xticklabels ([int (x) for x in xticks], fontsize=7)
        for xtick in axis.get_xticks():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.1)
        # Format y-axis
        axis.set_ylim ([0, 3])
        axis.set_yticks ([1, 2])
        axis.set_yticklabels ([r'sim (95%)', r'theory (95%)'], fontsize=7)
        for ytick in axis.get_yticks():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.1)

    plt.suptitle ('number of total patients in system per class', fontsize=10)
    h.savefig (outFile)
    plt.close('all')

def plot_n_patient_distribution_classes (outFile, nPatients, stats, params, doLogY=True):

    ''' Function to plot distributions of the observed number of patients in
        the system *right before a new patient arrives* with CADt.

        inputs
        ------
        outFile (str): output plot file name
        nPatients (pandas DataFrame): observed number of patients in system for every patient
                                      typically `trialGen.n_patients_system_df`
        stats (dict): mean, lower/upper 1 sigma and 95% C.I. 
                      typically `trialGen.n_patients_system_stats`
        params (dict): settings for all simulations
        doLogY (bool): If true, plot log y
    '''

    ## 3 plots (all pre-resume): interrupting, Positive, Negative
    h  = plt.figure (figsize=(25, 6))
    gs = gridspec.GridSpec (1, 3, wspace=0.4, hspace=0.4)

    for index, aclass in enumerate (['interrupting', 'positive', 'negative']):

        ## Skip the plot for interrupting class if fractionED = 0 i.e. no interrupting patients
        if round (params['fractionED'],4) == 0 and aclass == 'interrupting': continue
        ## Skip the plot for positive if Se is 0 i.e. all are classified as AI negative
        if params['TPFThresh'] == 0 and aclass == 'positive': continue
        ## Skip the plot for negative if Sp is 0 i.e. all are classified as AI positive
        if params['TPFThresh'] == 1 and aclass == 'negative': continue        

        ## Set up subplots and xticks
        subgs = gs[index].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
        values = nPatients[aclass]['preresume']
        hist_bins = numpy.linspace (0, values.max(), int (values.max()+1))
        xticks = hist_bins[::2] if max (values) < 30 else \
                 hist_bins[::4] if max (values) < 70 else \
                 hist_bins[::8] if max (values) < 200 else \
                 hist_bins[::16] if max (values) < 300 else hist_bins[::32]

        ## Top plot: distribution
        axis = h.add_subplot (subgs[0])
        #  Simulation
        hist, edges = numpy.histogram (values, bins=hist_bins)
        hist_sum = numpy.sum (hist)
        hist = numpy.r_[hist[0], hist]
        yvalues = hist/hist_sum
        axis.plot (edges[:-1], yvalues[1:], label='sim', color=colors['simulation'], drawstyle='steps-mid',
                   linestyle='-', linewidth=2.0, alpha=0.6)
        yvalues = hist[1:]/hist_sum
        yerrors = numpy.sqrt (hist[1:])/hist_sum
        axis.errorbar (edges[:-1], yvalues, marker=None, color=colors['simulation'], yerr=yerrors,
                       elinewidth=1.5, alpha=0.5, ecolor=colors['simulation'], linestyle='')
        #  Theory
        if aclass == 'interrupting':
            pred = calculator.get_state_pdf_MMn (numpy.linspace (0, 1000, 1001), params['nRadiologists'],
                                                 params['lambdas']['interrupting'],
                                                 params['mus']['interrupting'])
            xvalues, yvalues = numpy.linspace (0, 1000, 1001), pred
        else:
            if aclass == 'positive':
                cal = calculator.def_cal_MMs (params, lowPriority='positive',
                                              doDynamic=params['nRadiologists']>3)
                pred = cal.solve_prob_distributions (1001)
                ps = numpy.array ([numpy.sum (p).real for p in pred])
                xvalues, yvalues = range (0, len (ps)), ps
            else: 
                xvalues, yvalues = None, None
                if params['nRadiologists'] <= 2:
                    cal = calculator.get_cal_lowest (params)
                    pred = cal.solve_prob_distributions (1001)
                    ps = numpy.array ([numpy.sum (p).real for p in pred])
                    xvalues, yvalues = range (0, len (ps)), ps
                elif params['fractionED'] == 0.0:
                    cal = calculator.def_cal_MMs (params, lowPriority='negative',
                                                  highPriority='positive',
                                                  doDynamic=params['nRadiologists']>3)
                    pred = cal.solve_prob_distributions (1001)
                    ps = numpy.array ([numpy.sum (p).real for p in pred])
                    xvalues, yvalues = range (0, len (ps)), ps        
        if xvalues is not None:
            axis.plot (xvalues, yvalues, label='theory', color=colors['theory'], linestyle='--', linewidth=2.0, alpha=0.7)
                                
        # Format x-axis
        axis.set_xlim (hist_bins[0], hist_bins[-1])
        axis.set_xticks (xticks)
        axis.set_xticklabels ([int (x) for x in xticks], fontsize=7)
        for xtick in axis.get_xticks():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
        axis.get_xaxis().set_ticks([])
        # Format y-axis
        if doLogY: axis.set_yscale("log")
        axis.set_ylim (min (hist/hist_sum), max (hist/hist_sum) + 0.05)
        axis.set_ylabel ('Normalized counts', fontsize=9)
        for ytick in axis.get_yticks():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
        # Format others
        axis.legend (loc='best', ncol=1, prop={'size':9})
        axis.set_title ('With CADt; {0}'.format (aclass), fontsize=9)

        ## Bottom plot: mean/1sigma
        axis = h.add_subplot (subgs[1])
        #  Simulation
        xlower = [stats['preresume'][aclass]['mean'] - stats['preresume'][aclass]['lower_95cl']]
        xupper = [stats['preresume'][aclass]['upper_95cl'] - stats['preresume'][aclass]['mean']]
        axis.errorbar (stats['preresume'][aclass]['mean'], 1, marker="x", markersize=10, color=colors['simulation'],
                       xerr=[xlower, xupper], elinewidth=3, alpha=0.8, ecolor=colors['simulation'], linestyle='')
        #  Theory
        if aclass == 'interrupting':
            pred = calculator.get_state_pdf_MMn (numpy.linspace (0, 1000, 1001), params['nRadiologists'],
                                                 params['lambdas']['interrupting'],
                                                 params['mus']['interrupting'])
        else:
            if aclass == 'positive':
                cal = calculator.def_cal_MMs (params, lowPriority='positive',
                                              doDynamic=params['nRadiologists']>3)
                pred = cal.solve_prob_distributions (1001)
                pred = numpy.array ([numpy.sum (p).real for p in pred])
            else: 
                pred = None
                if params['nRadiologists'] <= 2:
                    cal = calculator.get_cal_lowest (params)
                    pred = cal.solve_prob_distributions (1001)
                    pred = numpy.array ([numpy.sum (p).real for p in pred])
                elif params['fractionED'] == 0.0:
                    cal = calculator.def_cal_MMs (params, lowPriority='negative',
                                                  highPriority='positive',
                                                  doDynamic=params['nRadiologists']>3)
                    pred = cal.solve_prob_distributions (1001)
                    pred = numpy.array ([numpy.sum (p).real for p in pred])                    
        if pred is not None:
            theoryStats = get_stats (numpy.linspace (0, 1000, 1001), pred)
            xlower = [theoryStats['mean'] - theoryStats['lower_95cl']]
            xupper = [theoryStats['upper_95cl'] - theoryStats['mean']]
            axis.errorbar (theoryStats['mean'], 2, marker="x", color=colors['theory'], markersize=10,
                        xerr=[xlower, xupper], elinewidth=3, alpha=0.8, ecolor=colors['theory'], linestyle='')

        # Format x-axis
        axis.set_xlim (hist_bins[0], hist_bins[-1])
        axis.set_xlabel ('number of patients ({0} class only)'.format (aclass), fontsize=9)
        axis.set_xticks (xticks)
        axis.set_xticklabels ([int (x) for x in xticks], fontsize=7)
        for xtick in axis.get_xticks():
            axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle=':', linewidth=0.1)
        # Format y-axis
        axis.set_ylim ([0, 3])
        axis.set_yticks ([1, 2])
        axis.set_yticklabels ([r'sim (95%)', r'theory (95%)'], fontsize=7)
        for ytick in axis.get_yticks():
            axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle=':', linewidth=0.1)

    plt.suptitle ('number of total patients in system per class', fontsize=10)
    h.savefig (outFile)
    plt.close('all')

## +------------------------------------------
## | For waiting time distributions
## +------------------------------------------
def plot_waiting_time_distributions (ext, waitTimesDF, stats, params):

    ''' Function to plot distributions of the waiting time and wait-time difference for
        diseased / non-diseased (radiologist diagnosis) and AI positive / negative
        subgroups in both with and without CADt scenarios.

        inputs
        ------
        ext (str): extension of output file name
        waitTimesDF (pandas DataFrame): waiting time for every patient in both with and without
                                        CADt scenario. Typically `trialGen.waiting_times_df`
        stats (dict): mean, lower/upper 1 sigma and 95% C.I. 
                      typically `trialGen.waiting_times_stats`
        params (dict): settings for all simulations
    '''

    ## Calculate time difference per patient between with and without CADt
    waitTimesDF['delta'] = waitTimesDF['preresume'] - waitTimesDF.fifo

    ## Plot waiting time grouped by AI call
    outFile = params['plotPath'] + 'waiting_time_distribution_AIcall' + ext
    plot_waiting_time_distribution (outFile, waitTimesDF, stats, params, byDiagnosis=False)

    ## Plot waiting time grouped by radiologist's diagnosis
    outFile = params['plotPath'] + 'waiting_time_distribution_radDiagnosis' + ext
    plot_waiting_time_distribution (outFile, waitTimesDF, stats, params, byDiagnosis=True)

def plot_waiting_time_distribution (outFile, waitTimesDF, stats, params, byDiagnosis=False):

    ''' Function to plot distributions of the waiting time and wait-time difference.

        inputs
        ------
        outFile (str): output plot file name
        waitTimesDF (pandas DataFrame): waiting time for every patient in both with and without
                                        CADt scenario. Typically `trialGen.waiting_times_df`
        stats (dict): mean, lower/upper 1 sigma and 95% C.I. 
                      typically `trialGen.waiting_times_stats`
        params (dict): settings for all simulations
        byDiagnosis (bool): If true, plot diseased and non-diseased (radiologist diagnosis)
                            subgroups. If False, plot AI positive and negative subgroups.
    '''

    #########################################################
    ## If by AI Call: 
    ## +-------------|----------------|---------------+
    ## | Int w/o AI  | non-int w/o AI |    (empty)    |
    ## +-------------|----------------|---------------+
    ## | Int w/ AI   |      AI +      |      AI -     |
    ## +-------------|----------------|---------------+
    ## | Int delta   |   AI + delta   |   AI - delta  |
    ## +-------------|----------------|---------------+
    ##
    ## If by radiologist diagnosis: 
    ## +-------------|----------------|--------------------+
    ## | Int w/o AI  | non-int w/o AI |       (empty)      |
    ## +-------------|----------------|--------------------+
    ## | Int w/ AI   |    Diseased    |     Non-diseased   |
    ## +-------------|----------------|--------------------+
    ## | Int delta   | Diseased delta | Non-diseased delta |
    ## +-------------|----------------|--------------------+   
    #########################################################

    h  = plt.figure (figsize=(20, 20))
    gs = gridspec.GridSpec (3, 3, wspace=0.2, hspace=0.2)
    gs.update (bottom=0.1)

    gindex = 0
    ## Looping through rows
    for qtype in ['fifo', 'preresume', 'delta']:

        gtypes = ['interrupting', 'non-interrupting', ''] if qtype == 'fifo' else \
                 ['interrupting', 'diseased', 'non-diseased'] if byDiagnosis else \
                 ['interrupting', 'positive', 'negative'] 
        xlabel = r'$\delta$ waiting time (With CADt - Without CADt) [min]' if gindex>5 else 'waiting time [min]'

        # For a row, loop through column
        for gtype in gtypes:

            # skip empty plot on top right
            ignore = params['fractionED'] == 0 and gtype == 'interrupting'
            if gtype == '' or ignore:
                gindex += 1
                continue

            xticks = [-250, -200, -150, -100, -50, 0, 50] if qtype=='delta' and gtype == 'positive' else \
                     [-200, -100, 0, 100, 200] if qtype=='delta' and gtype in ['diseased', 'non-diseased'] else \
                     [-50, 0, 50, 100, 150, 200, 250] if qtype=='delta' and gtype == 'negative' else \
                     [-50, -25, 0, 25, 50] if qtype=='delta' else \
                     [0, 10, 20, 30, 40, 50] if gtype == 'interrupting' else \
                     [0, 25, 50, 75, 100] if gtype == 'positive' else \
                     [0, 100, 200, 300, 400] ## non-interrupting or negative or diseased or non-diseased
            hist_bins = numpy.linspace (xticks[0], xticks[-1], 21)

            subgs = gs[gindex].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05) 

            # +----------------------------------------------
            # | Top: distributions
            # +----------------------------------------------
            axis = h.add_subplot (subgs[0])
            # Simulation
            if gtype == 'interrupting':
                values = waitTimesDF[qtype][waitTimesDF.is_interrupting]
            else:
                nonInterrupting = waitTimesDF[~waitTimesDF.is_interrupting]
                values = nonInterrupting[qtype]
                if gtype == 'positive': values = values[nonInterrupting.is_positive]
                if gtype == 'negative': values = values[~nonInterrupting.is_positive]
                if gtype == 'diseased': values = values[nonInterrupting.is_diseased]
                if gtype == 'non-diseased': values = values[nonInterrupting.is_diseased==False]

            hist, edges = numpy.histogram (values, bins=hist_bins)
            hist_sum = numpy.sum (hist)
            hist = numpy.r_[hist[0], hist]
            yvalues = hist/hist_sum
            axis.plot (edges[:-1], yvalues[1:], label='sim', color=colors['simulation'], drawstyle='steps-mid',
                       linestyle='-', linewidth=2.0, alpha=0.6)
            yvalues = hist[1:]/hist_sum
            yerrors = numpy.sqrt (hist[1:])/hist_sum
            axis.errorbar (edges[:-1], yvalues, marker=None, color=colors['simulation'], yerr=yerrors,
                        elinewidth=1.5, alpha=0.5, ecolor=colors['simulation'], linestyle='')

            # Theory
            # No theoretical predictions for the waiting time distributions. Only means are predicted.

            # If delta time, add a vertical line at x = 0
            if qtype=='delta': axis.axvline (x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)

            # Format x-axis
            axis.set_xlim (xticks[0], xticks[-1])
            axis.set_xticks (xticks)
            axis.set_xticklabels ([int (x) for x in xticks], fontsize=10)
            for xtick in axis.get_xticks():
                axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
            axis.get_xaxis().set_ticks([])
            # Format y-axis
            axis.set_yscale("log")
            axis.set_ylim (ymin, max (hist/hist_sum) + 10)    
            axis.set_ylabel ('Normalized counts', fontsize=12)
            for ytick in axis.get_yticks():
                axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
            # Format others
            axis.set_title ('{0} patients ({1})'.format (gtype, qtype), fontsize=12)

            # +----------------------------------------------
            # | Bottom: Mean / CI
            # +----------------------------------------------
            axis = h.add_subplot (subgs[1])

            #  Simulation
            thisStats = stats['fifo']['waitTime'][gtype] if qtype == 'fifo' else \
                        stats['preresume']['waitTime'][gtype] if qtype == 'preresume' else \
                        stats['preresume']['diff'][gtype]
            xlower = [max (0, thisStats['mean'] - thisStats['lower_95cl'])]
            xupper = [max (0, thisStats['upper_95cl'] - thisStats['mean'])]
            axis.errorbar (thisStats['mean'], 1, marker="x", markersize=10, color=colors['simulation'], label='sim',
                           xerr=[xlower, xupper], elinewidth=3, alpha=0.8, ecolor=colors['simulation'], linestyle='')

            # Plot theoretical mean
            theory = calculator.get_theory_waitTime (gtype, qtype, params)
            if theory is not None:
                axis.scatter (theory, 2, s=60, marker="+", color=colors['theory'], alpha=0.5, label='theory')

            # If delta time, add a vertical line at x = 0
            if qtype=='delta': axis.axvline (x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)

            # Format x-axis
            axis.set_xlim (xticks[0], xticks[-1])
            axis.set_xlabel (xlabel, fontsize=12)
            axis.set_xticks (xticks)
            axis.set_xticklabels ([int (x) for x in xticks], fontsize=10)
            for xtick in axis.get_xticks():
                axis.axvline (x=xtick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
            # Format y-axis
            axis.set_ylim ([0, 3])
            axis.set_yticks ([1, 2])
            axis.set_yticklabels ([r'sim (95%)', r'theory'], fontsize=7)
            for ytick in axis.get_yticks():
                axis.axhline (y=ytick, color='gray', alpha=0.3, linestyle='--', linewidth=0.3)
            # Format others
            axis.legend (loc=1, ncol=1, prop={'size':6})

            gindex += 1

    group = 'radiologist diagnosis' if byDiagnosis else 'AI call'
    plt.suptitle ('Distributions of waiting time and wait-time difference by {0}'.format (group), fontsize=15)
    h.savefig (outFile)
    plt.close('all')
