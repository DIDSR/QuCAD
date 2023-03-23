
##
## By Elim Thompson 12/15/2020
##
## This is the main python script to simulate radiology reading workflow at a specific clinical
## setting with a CADt diagnostic performance. This simulation software handles a simplified
## scenario with 1 AI that is trained to identify 1 disease condition from 1 modality and
## anatomy. Patients in the reading queue either have the disease condition or not. User can
## either use argument flags like below or use `../inputs/config.dat` to specify user input values.
##
## Method 1: via individual argument flags
##
## $ python run_sim.py --traffic 0.8 --TPFThresh 1.0 --prevalence 0.1
##                     --nRadiologists 1 --fractionED 0.0 
##                     --meanServiceTimeDiseasedMin 10
##                     --meanServiceTimeNonDiseasedMin 10
##                     --meanServiceTimeInterruptingMin 5
##                     --statsFile /path/to/outputs/stats/stats.p
##                     --nTrials 10 --nPatientsTarget 1000 
##                     (--FPFThresh 0.1) # if using a Se, Sp threshold point
##                     (--rocFile /path/to/inputs/exampleROC.dat) # if using a parameterized ROC curve
##                     (--runtimeFile /path/to/outputs/runTime.txt) # if print out runtime performance
##                     (--plotPath /path/to/outputs/plots/) # if generate plots
##                     (--verbose) # if print out progress 
##
## Method 2: via a configuration file
##
## $ python run_sim.py --configFile ../inputs/config.dat
#######################################################################################################

################################
## Import packages
################################ 
import cProfile, pstats, io, os, sys, time, pickle

sys.path.insert(0, os.getcwd()+'\\tools')
from tools import inputHandler, AI, simulator, trialGenerator, plotter

#import logging
#logging.basicConfig(level=logging.DEBUG)

################################
## Define lambdas
################################ 
get_n_positive_patients = lambda oneSim, qtype:len (oneSim.get_positive_records(qtype))
get_n_negative_patients = lambda oneSim, qtype:len (oneSim.get_negative_records(qtype))
get_n_interrupting_patients = lambda oneSim, qtype:len (oneSim.get_interrupting_records(qtype))

################################
## Define functions
################################ 
def create_AI (TPFThresh, FPFThresh=None, rocFile=None, doPlots=False, plotPath=None):

    ''' Function to create a CADt device either at a threshold or from 
        an input ROC file. If provided an ROC File, the file should have
        two columns. First is false positive fraction (FPF), and second
        is true positive fraction (TPF). Note that either FPFThresh or
        rocFile should be provided. If both are provided, use FPFThresh
        and ignore ROC file.

        inputs
        ------
        TPFThresh (float): CADt Se operating point to be used for simulation
        FPFThresh (float): CADt 1-Sp operating point to be used for simulation
        rocFile (str): File to ROC curve that will be parameterized
        doPlot (bool): If true, generate plots for ROC parameterization 
        plotPath (str): Path where plots generated will live

        output
        ------
        anAI (AI): CADt with a diagnostic performance from user input
    '''

    ## When FPF is provided, use a single operating point
    if FPFThresh is not None:
        return AI.AI.build_from_opThresh('anAI', TPFThresh, 1-FPFThresh)

    ## If FPF is not provided, but emperical ROC is provided, parameterize
    ## the ROC curve based on bi-normal distribution.
    anAI = AI.AI.build_from_empiricalROC ('anAI', rocFile, TPFThresh)
    anAI.fit_ROC (doPlots=doPlots, outPath=plotPath)
    ## Make sure the CADt operates at the user-input Se Threshold 
    anAI.SeThresh = TPFThresh
    return anAI

def print_AI_sim_performance (params, oneSim):

    ''' Function to print AI performance 
    '''

    # Check with prevalence
    p_sim = len (oneSim.get_diseased_records ('fifo')) / len (oneSim.get_noninterrupting_records ('fifo'))
    print ('+-----------------------------------------------')
    print ('| prevalence setting : {0:.3f}'.format (params['prevalence']))
    print ('| prevalence from sim: {0:.3f}'.format (p_sim))
    print ('|')

    # From simulation
    TP, FN = len (oneSim.get_TP_records ('preresume')), len (oneSim.get_FN_records ('preresume'))
    TN, FP = len (oneSim.get_TN_records ('preresume')), len (oneSim.get_FP_records ('preresume'))
    ppv_sim = 0 if TP + FP == 0 else TP / (TP + FP)
    npv_sim = 0 if TN + FN == 0 else TN / (TN + FN)
    print ('| PPV from theory: {0:.7f}'.format (params['ppv']))
    print ('| PPV from sim   : {0:.7f}'.format (ppv_sim))
    print ('| NPV from theory: {0:.7f}'.format (params['npv']))
    print ('| NPV from sim   : {0:.7f}'.format (npv_sim))
    print ('+-----------------------------------------------')

################################
## Script starts here!
################################ 
if __name__ == '__main__':

    ## Gather user-specified settings
    params = inputHandler.read_args()

    pr = None
    if params['doRunTime']:
        pr = cProfile.Profile()
        pr.enable()

    ## Create an AI object and update parameters
    anAI = create_AI (params['TPFThresh'], FPFThresh=params['FPFThresh'], doPlots=params['doPlots'],
                      rocFile=params['rocFile'], plotPath=params['plotPath'])
    params = inputHandler.add_params(anAI, params)

    if not params['doTrialOnly']:
        ## Check AI performance with one trial
        oneSim = simulator.simulator ()
        oneSim.set_params (params)
        oneSim.track_log = False ## Make sure it is False for optimal runtime
        oneSim.simulate_queue (anAI)
        print_AI_sim_performance (params, oneSim)
        params['n_patients_per_class'] = {qtype:{aclass:get_n_interrupting_patients (oneSim, qtype) if aclass=='interrupting' else \
                                                        get_n_positive_patients  (oneSim, qtype) if aclass=='positive' else \
                                                        get_n_negative_patients  (oneSim, qtype)
                                                 for aclass in ['interrupting', 'positive', 'negative']}
                                          for qtype in params['qtypes'][1:]}
        ## If do-plots, generate plots for one simulation
        if params['doPlots']:
            # Timing flow with first 200 cases for both with and withoutCADt
            for qtype in params['qtypes']:
                scenario = 'withCADt' if qtype == 'preresume' else 'withoutCADt'
                outFile = params['plotPath'] + 'patient_timings_' + scenario + '.pdf'
                plotter.plot_timing (outFile, oneSim.all_records, params['startTime'], n=200, qtype=qtype)

    ## Run trials
    t0 = time.time()
    trialGen = trialGenerator.trialGenerator ()
    trialGen.set_params (params)
    trialGen.simulate_trials (anAI)
    params['runTimeMin'] = (time.time() - t0)/60 # minutes
    print ('{0} trials took {1:.2f} minutes'.format (params['nTrials'], params['runTimeMin']))

    # Plot histograms with all patients from all trials
    if params['doPlots']:
        # Plot n patients histograms
        plotter.plot_n_patient_distributions ('.pdf', trialGen.n_patients_system_df, trialGen.n_patients_system_stats, params)
        # Plot waiting time histograms
        plotter.plot_waiting_time_distributions ('.pdf', trialGen.waiting_times_df, trialGen.waiting_times_stats, params)
        
    ## Gather data for dict
    data = {'params':params,
            'lpatients':trialGen.n_patients_system_df,
            'wpatients':trialGen.waiting_times_df,
            'lstats':trialGen.n_patients_system_stats,
            'wtstats':trialGen.waiting_times_stats}
    with open (params['statsFile'], 'wb') as f:
        pickle.dump (data, f)

    if params['doRunTime']:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open (params['runtimeFile'], 'w+') as f:
            f.write (s.getvalue())