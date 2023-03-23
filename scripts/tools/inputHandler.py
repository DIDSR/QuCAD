## 
## Elim Thompson (03/10/2023)
##
## This script contains functions that handle input user values, either as
## argument flags or as an input file. Based on that, an output params is
## returned for simulation to run on.
################################################################################

################################
## Import packages
################################
import numpy, pandas, os, argparse
from calculator import get_theory_waitTime

################################
## Define constants
################################
## +------------------------
## | Clinical / AI setting
## +------------------------
traffic = 0.8        # Hospital busyness
TPFThresh = 1.0      # AI operating TPF (i.e. Se); Sp is defined by the ROC curve
FPFThresh = 0.1      # None if use ROC curve
prevalence = 0.1     # Disease prevalence
nRadiologists = 1    # Number of radiologists on-site
fractionED = 0.0     # Fraction of interrupting patients to all patients
doPlots = False      # Flag to generate plots for this run
## Radiologist service process in minutes
##  * Diseased    : Average service time when the radiologist calls a case diseased
##  * Non-diseased: Average service time when the radiologist calls a case non-diseased
##  * Interrupting: Average service time for interruting patients
meanServiceTimes = {'diseased':10, 'non-diseased':10, 'interrupting':5}

## +------------------------
## | Path / File locations
## +------------------------
rocFile     = None
statsFile   = '../outputs/stats/stats.p'
plotPath    = '../outputs/plots/'
runtimeFile = '../outputs/stats/runTime.txt'

## +------------------------
## | Workflow setting
## +------------------------
verbose = False                # Print 
doTrialOnly = False            # Skip the oneSim
nTrials = 1                    # Number of trials
nPatientsTarget = 20000        # Rough number of patients per trial
qtypes = ['fifo', 'preresume'] # 'fifo' = without CADt scenario; 'preresume' = with CADt scenario
rhoThresh = 0.95               # Maximum allowed hospital busyness
nPatientsPads = [0, 1]         # Chop off the first and last 100 patients
startTime = pandas.to_datetime ('2020-01-01 00:00') # Simulation Start time 

################################
## Define lambdas
################################ 
get_ppv = lambda p, Se, Sp: p*Se / (p*Se + (1-p)*(1-Sp))
get_npv = lambda p, Se, Sp: 1 - p*(1-Se) / (p*(1-Se) + (1-p)*Sp)

get_n_positive_patients = lambda oneSim, qtype:len (oneSim.get_positive_records(qtype))
get_n_negative_patients = lambda oneSim, qtype:len (oneSim.get_negative_records(qtype))
get_n_interrupting_patients = lambda oneSim, qtype:len (oneSim.get_interrupting_records(qtype))

get_timeWindowDay = lambda arrivalRate, nPatientsTarget: int (numpy.ceil (nPatientsTarget / arrivalRate / (24*60)))

get_is_positive = lambda params: params['prevalence']*(1-params['fractionED'])*params['SeThresh'] + \
                                 (1-params['prevalence'])*(1-params['fractionED'])*(1-params['SpThresh'])
get_is_negative = lambda params: params['prevalence']*(1-params['fractionED'])*(1-params['SeThresh']) + \
                                 (1-params['prevalence'])*(1-params['fractionED'])*params['SpThresh']
get_mu_effective = lambda params: 1/(params['prob_isPositive'] / params['mus']['positive'] + \
                                     params['prob_isNegative'] / params['mus']['negative'] + \
                                     params['prob_isInterrupting'] / params['mus']['interrupting'])
get_mu_positive = lambda params:1/(params['ppv']/params['serviceRates']['diseased'] + \
                                   (1-params['ppv'])/params['serviceRates']['non-diseased'])
get_mu_negative = lambda params:1/(params['npv']/params['serviceRates']['non-diseased'] + \
                                   (1-params['npv'])/params['serviceRates']['diseased'])
get_mu_nonInterrupting = lambda params:1/(params['prevalence']/params['serviceRates']['diseased'] + \
                                       (1-params['prevalence'])/params['serviceRates']['non-diseased'])
get_lambda_effective = lambda params: params['traffic']*params['nRadiologists']*params['mu_effective']

get_service_rate = lambda service_time: numpy.nan_to_num (numpy.inf) if service_time==0 else 1/service_time

################################
## Define functions
################################ 
def check_user_inputs (params):

    ''' Function to check input traffic, FPFThresh, rocFile, and existence of
        input ROC file (rocFile) and output files / paths (plotPlot, statsFile,
        runtimeFile).

        input
        -----
        params (dict): dictionary capsulating all user inputs
    '''

    ## Checks on traffic
    if params['traffic'] > rhoThresh:
        print ('ERROR: Input traffic {0:.3f} is too high.'.format (params['traffic']))
        raise IOError ('Please limit traffic below {0:.3f}.'.format (rhoThresh))

    ## Checks on values that are between 0 and 1
    for key in ['traffic', 'TPFThresh', 'FPFThresh', 'prevalence', 'fractionED']:
        if key == 'FPFThresh' and params[key] is None: continue
        if params[key] > 1:
            print ('ERROR: Input {0} {1:.3f} is too high.'.format (key, params[key]))
            raise IOError ('Please provide a {0} between 0 and 1.'.format (key))
        if params[key] < 0:
            print ('ERROR: Input {0} {1:.3f} is too low.'.format (key, params[key]))
            raise IOError ('Please provide a {0} between 0 and 1.'.format (key))            

    ## Checks on number of radiologists
    if params['nRadiologists'] > 2 and params['fractionED'] > 0.0:
        print ('WARN: There are more than 2 radiologits with presence of interrupting images. Theoretical values for AI negative and diseased/non-diseased subgroups will not be available.')

    ## Checks on FPFThresh and rocFile
    #  1. Both are provided
    if params['FPFThresh'] is not None and params['rocFile'] is not None:
        print ('WARN: Received both FPFThresh and rocFile. Taking FPFThresh and ignoring rocFile.')
        params['rocFile'] = None
    #  2. Neither are provided
    if params['FPFThresh'] is None and params['rocFile'] is None:
        print ('ERROR: Neither FPFThresh nor rocFile is provided.')
        raise IOError ('Please provide either FPFThresh or rocFile.')

    ## Checks if file/folders exists
    if params['rocFile'] is not None:
        if not os.path.exists (params['rocFile']):
            print ('ERROR: Input rocFile does not exist.')
            raise IOError ('Please provide a valid rocFile with two columns (first is true-positive fraction, and second is false-positive fraction).')
    for location in ['statsFile', 'runtimeFile', 'plotPath']:
        if params[location] is not None:
            if not os.path.exists (os.path.dirname (params[location])):
                print ('ERROR: Path does not exist: {0}'.format (params[location]))
                try:
                    print ('ERROR: Trying to create the folder ...')
                    os.mkdir (os.path.dirname (params[location]))
                except:
                    raise IOError ('Cannot create the folder.\nPlease provide a valid {0} path.'.format (location))

def read_args (): 

    ''' Function to read user inputs. Adding doPlot and doRunTime based on whether
        user provides the paths.

        output
        ------
        params (dict): dictionary capsulating all user inputs
    '''

    parser = argparse.ArgumentParser(description='Read user input setting params.')

    parser.add_argument('--traffic', type=float, default=traffic, help='overall traffic intensity')
    parser.add_argument('--TPFThresh', type=float, default=TPFThresh, help='AI TPF (Se) threshold')
    parser.add_argument('--FPFThresh', type=float, default=FPFThresh, help='AI FPF (1-Sp) threshold')
    parser.add_argument('--prevalence', type=float, default=prevalence, help='Disease prevalence')
    parser.add_argument('--nRadiologists', type=int, default=nRadiologists, help='number of radiologists on-site')    
    parser.add_argument('--fractionED', type=float, default=fractionED, help='Fraction of interrupting images')
    parser.add_argument('--meanServiceTimeDiseasedMin', type=float, default=meanServiceTimes['diseased'],
                        help='Mean service time [min] a radiologist takes to complete a image that s/he calls diseased')
    parser.add_argument('--meanServiceTimeNonDiseasedMin', type=float, default=meanServiceTimes['non-diseased'],
                        help='Mean service time [min] a radiologist takes to complete a image that s/he calls non-dieased')
    parser.add_argument('--meanServiceTimeInterrutingMin', type=float, default=meanServiceTimes['interrupting'],
                        help='Mean service time [min] a radiologist takes to complete an interrupting image')

    parser.add_argument('--configFile', type=str, default=None, help='User input configuration data file with all parameters')
    parser.add_argument('--rocFile', type=str, default=None, help='Input ROC curve with TPR and FPR that will be parameterized')    
    parser.add_argument('--statsFile', type=str, default=statsFile, help='Path to output stats .p pickled file')
    parser.add_argument('--runtimeFile', type=str, default=runtimeFile, help='Output runtime performance .txt text file')    
    parser.add_argument('--plotPath', type=str, default=plotPath, help='Output folder to store all plots')

    parser.add_argument('--verbose', action='store_true', default=False, help='Print out simulation progress')
    parser.add_argument('--doTrialOnly', action='store_true', default=False, help='Skip one simulation for checking AI performance')
    parser.add_argument('--nTrials', type=int, default=nTrials, help='Number of trials to perform')
    parser.add_argument('--nPatientsTarget', type=int, default=nPatientsTarget, help='targeted # patients per trial')
    
    args = parser.parse_args()
    
    ## Put everything in a dictionary
    params = {'traffic':args.traffic, 'TPFThresh':args.TPFThresh, 'FPFThresh':args.FPFThresh,
              'prevalence':args.prevalence, 'nRadiologists':args.nRadiologists, 'fractionED':args.fractionED, 
              'meanServiceTimes':{'diseased':args.meanServiceTimeDiseasedMin,
                                  'non-diseased':args.meanServiceTimeNonDiseasedMin,
                                  'interrupting':args.meanServiceTimeInterrutingMin},
              'rocFile':args.rocFile, 'statsFile':args.statsFile, 'runtimeFile':args.runtimeFile,
              'plotPath':args.plotPath, 'verbose':args.verbose, 'doTrialOnly':args.doTrialOnly,
              'configFile':args.configFile, 'nTrials':args.nTrials, 'nPatientsTarget':args.nPatientsTarget,
              'qtypes':qtypes, 'nPatientsPads':nPatientsPads, 'startTime':startTime}

    if args.configFile is not None:
        params = read_configFile (args.configFile, params)

    if params['verbose']:
        print ('Reading user inputs:')
        print ('+------------------------------------------')
        ## Put user inputs into `params` 
        for key in params.keys():
            if key in ['qtypes', 'nPatientsPads', 'startTime']: continue
            if key == 'meanServiceTimes':
                for subgroup in params[key].keys():
                    print ('| {0} {1}: {2}'.format (key, subgroup, params[key][subgroup]))
                continue
            print ('| {0}: {1}'.format (key, params[key]))

    ## Add a few flags 
    params['doPlots'] = params['plotPath'] is not None
    if params['verbose']: print ('| doPlots: {0}'.format (params['doPlots']))
    params['doRunTime'] = params['runtimeFile'] is not None
    if params['verbose']: print ('| doRunTime: {0}'.format (params['doRunTime']))

    if params['verbose']:
        print ('+------------------------------------------')
        print ('')

    check_user_inputs (params)
    return params

def read_configFile (configFile, params):

    ''' Function with all user input values for user to feed to simulation software.

        input
        -----
        configFile (str): path and filename of the user input file
                          For an example, see ../../inputs/config.dat
        params (dict): dictionary capsulating all user inputs                          
        
        outputs
        -------
        params (dict): update dictionary capsulating all user inputs from config file
    '''


    with open (configFile, 'r') as f:
        config = f.readlines()
    f.close ()

    ## Extract the key, values as inputs
    inputs = [line.strip().split('#')[0].strip() for line in config
              if len (line.strip())>0 and line.strip()[0]!='#']
    inputs = {line.split()[0]:line.split()[1] for line in inputs}

    ## Change value types accordingly based on parameters name
    for key, value in inputs.items():
        # These keys should be float
        if key in ['traffic', 'fractionED', 'meanServiceTimeDiseasedMin', 'meanServiceTimeNonDiseasedMin',
                   'meanServiceTimeInterruptingMin', 'prevalence', 'TPFThresh']:
            inputs[key] = float (value)
        # These keys should be integers
        if key in ['nRadiologists', 'nTrials', 'nPatientsTarget']:
            inputs[key] = int (value)
        # These keys should be boolean
        if key in ['doTrialOnly', 'verbose']:
            if not inputs[key] in ['True', 'False']:
                raise IOError ('ERROR: {0} must be either True or False.'.format (key))
            inputs[key] = eval (value)
        # These keys may be None
        if key in ['FPFThresh', 'rocFile', 'runTimeFile', 'plotPath']:
            if value == 'None': inputs[key] = None

    ## Update params with user values
    for key, value in inputs.items ():
        ## If mean service time, put in sub-key
        if 'meanServiceTime' in key:
            if 'NonDiseased' in key:
                params['meanServiceTimes']['non-diseased'] = inputs[key]
            elif 'Diseased' in key:
                params['meanServiceTimes']['diseased'] = inputs[key]
            else: ## interrupting
                params['meanServiceTimes']['interrupting'] = inputs[key]
            continue
        params[key] = inputs[key]

    return params

def add_params (anAI, params):

    ''' Function to add additional parameters from user inputs. This includes
        probabilities that the next patient belongs to a certain priority class,
        as well as the arrival and service rates.

        inputs
        ------
        params (dict): dictionary with user inputs
        anAI (AI): CADt with a diagnostic performance from user input

        output
        ------
        params (dict): dictionary capsulating all simulation parameters
    '''

    ## Probabilities
    params['SeThresh'] = anAI.SeThresh
    params['SpThresh'] = anAI.SpThresh
    params['ppv'] = 0 if params['SeThresh']==0 else get_ppv (params['prevalence'], params['SeThresh'], params['SpThresh'])
    params['npv'] = 0 if params['SeThresh']==1 else get_npv (params['prevalence'], params['SeThresh'], params['SpThresh'])
    params['prob_isInterrupting'] = params['fractionED']
    params['prob_isDiseased'] = params['prevalence']*(1-params['fractionED'])
    params['prob_isNonDiseased'] = (1-params['prevalence'])*(1-params['fractionED'])
    params['prob_isPositive'] = get_is_positive (params)
    params['prob_isNegative'] = get_is_negative (params)
    params['prob_isTP'] = params['prevalence']*(1-params['fractionED'])*params['SeThresh']
    params['prob_isFP'] = (1-params['prevalence'])*(1-params['fractionED'])*(1-params['SpThresh'])
    params['prob_isFN'] = params['prevalence']*(1-params['fractionED'])*(1-params['SeThresh'])
    params['prob_isTN'] = (1-params['prevalence'])*(1-params['fractionED'])*params['SpThresh']

    ## Service rates
    #  Convert diseased/non-diseased service time from probabilities
    params['serviceRates'] = {key: get_service_rate (value) for key, value in params['meanServiceTimes'].items()}
    params['mus'] = params['serviceRates']
    params['mus']['positive'] = get_mu_positive (params)
    params['mus']['negative'] = get_mu_negative (params)
    params['mus']['non-interrupting'] = get_mu_nonInterrupting (params)
    params['mu_effective'] = get_mu_effective (params)
    
    ## Arrival rates
    params['lambda_effective'] = get_lambda_effective (params)
    params['meanArrivalTime'] = 1/params['lambda_effective']
    params['arrivalRates'] = {'interrupting':params['prob_isInterrupting']*params['lambda_effective'],
                              'non-interrupting':(1-params['prob_isInterrupting'])*params['lambda_effective'],
                              'diseased':params['prob_isDiseased']*params['lambda_effective'],
                              'non-diseased':params['prob_isNonDiseased']*params['lambda_effective'],
                              'positive':params['prob_isPositive']*params['lambda_effective'],
                              'negative':params['prob_isNegative']*params['lambda_effective'],
                              'diseased_positive'    :params['prob_isTP']*params['lambda_effective'],
                              'non-diseased_positive':params['prob_isFP']*params['lambda_effective'],
                              'diseased_negative'    :params['prob_isFN']*params['lambda_effective'],
                              'non-diseased_negative':params['prob_isTN']*params['lambda_effective']}
    
    ## Simulation times
    nPatientsPerTrial = params['nPatientsTarget'] + sum (params['nPatientsPads'])
    params['timeWindowDays'] = get_timeWindowDay (params['lambda_effective'], nPatientsPerTrial)
    params['endTime'] = params['startTime'] + pandas.offsets.Day (params['timeWindowDays'])
    
    ## Setting per priority class    
    params['lambdas'] = {'interrupting':params['arrivalRates']['interrupting'],
                         'non-interrupting':params['arrivalRates']['non-interrupting'],
                         'positive':params['arrivalRates']['positive'],
                         'negative':params['arrivalRates']['negative']}
    params['rhos'] = {key: params['lambdas'][key]/params['mus'][key]/params['nRadiologists']
                      for key in params['lambdas'].keys()}

    ## Get theoretical waiting time and delta time (i.e. wait-time-saving)
    params['theory'] = {}
    for aclass in ['interrupting', 'non-interrupting', 'diseased', 'non-diseased', 'positive', 'negative']:
        params['theory'][aclass] = {}
        for variable in ['fifo', 'preresume', 'delta']:
            key = 'waitTimeWithoutCADt' if variable == 'fifo' else \
                  'waitTimeWithCADt' if variable == 'preresume' else 'waitTimeSaving'
            params['theory'][aclass][key] = get_theory_waitTime (aclass, variable, params)    

    return params
