
##
## By Elim Thompson (12/15/2020)
##
## This script simulates one trial of reading flow. Incoming patients are
## placed in both with- and without-CADt scenarios. In both scenarios, a 
## preemptive resume scheduling is assumed. Each patient has a unique ID
## based on its order of arrival, and the ID is compared when action points
## are reached (e.g. when a radiologist reads a new case, when it closes a
## case, when it is interrupted by higher-priority case, and when a new
## patient enters the system, etc.). At each action time point, the queue
## is changed (e.g. adding new patient, putting interrupted patient back
## to the queue, removing a patient from the queue, etc.).
##
## This simulation software keeps track of all information from all patients
## in the queueing system, including the disease/interrupting status, the AI
## call, the reading duration (i.e. service time), the various timestamps
## (arrival, case open/closed, interrupted, etc.), the waiting time, etc.
## When a patient arrives, the number of patients in the system per priority
## class *right before the patient's arrival* are recorded to form the
## state probability distribution for this simulation trial. 
##
## Besides the simulations itself, this class also has functions for
## debugging purposes, including the option to print a log file of all
## patients and the functions to extract patients of specific subgroups.
##
## Note that emergency and interrupting are interchangeable in this script.
## Emergency patients do NOT refer to patients in emergency department.
## Instead, "emergency" refers to interrupting (emergent, urgent) cases
## that require immediate attention and interrupt the radiologist's
## reading the images in the reading queue. CADt devices are meant to
## only triage cases in the reading queue not interrupting cases.
###########################################################################

################################
## Import packages
################################ 
import numpy, pandas, queue, logging
from copy import deepcopy

import patient, radiologist

################################
## Define constants
################################
## Time conversion
day_to_second = 60 * 60 * 24
hour_to_second = 60 * 60 
minute_to_second = 60  

## Names of queue types for with and without CADt scenarios. Due to
## historical reasons, fifo means without-CADt scenario. In the past,
## without-CADt scenario did not include emergency subgroup. But now,
## with this highest-priority group, without-CADt scenario is a
## 2-priority-class system and is no longer fifo. However, this
## software still calls this without-CADt scenario "fifo". 
queuetypes = ['fifo', 'preresume'] 
## Names of priority classes in with CADt scenario. Positive
## and negative patients will be lumped into one priority
## class in the without-CADt scenario.
priorityClasses = ['interrupting', 'positive', 'negative']

## Simulation default parameters
##  1. number of days during simulation. Default: 1 month
timeWindowDays = 30
##  2. number of radiologists. Default: 1. Same number for both
##     with and without CADt scenario. For now, average reading
##     time per patient subgroup is the same for all radiologists
##     if multiple radiologists are simulated. 
nRadiologists = 1
##  2. number of patients to be padded before and after the
##     simulation periods. It is found to have no impacts
##     on the state probability distribution.
nPatientsPads = [100, 100]
##  3. simulation start timestamp. Actual time doesn't
##     really matter, but it is needed for the first
##     patient's arrival.
startTime = pandas.to_datetime ('2020-01-01 00:00')

################################
## Define lambdas
################################ 
extract_waitTime = lambda p: [p.wait_time_duration, p.is_interrupting, p.is_diseased, bool (p.is_positive)]
remove_nan = lambda array: array[numpy.isfinite (array)].astype (bool)

################################
## Define class
################################ 
class simulator (object):
    
    def __init__ (self):
        
        ''' A simulator doesn't need any parameters to be initialized. To update
            the parameters of a simulator instance, use set_params().
        '''
        
        ## Parameters related to the queues themselves. These parameters
        ## will not be updated when a user changes input parameters.
        ##  1. Radiologists for the two scenarios 
        self._fionas  = None # without CADt
        self._palmers = None # with CADt
        ##  2. Names of priority classes and queue types
        self._classes = priorityClasses
        self._qtypes = queuetypes 
        ##  3. Queue holder for both scenario
        self._aqueue = None
        ##  4. Patient holder for all patients 
        self._patient_records = None
        ##  5. Holder for "future" patients with arrival timestamps
        ##     after current action time point 
        self._future_patients = None
        ##  6. Holder for busy periods in without-CADt scenario with
        ##     only one radiologist. Note: this only agrees with
        ##     theory in without-CADt scenario with 1 radiologist.
        self._busy_periods_fifo = None

        ## Parameters related to debugging
        ##  1. Does user turn on logging?
        self._track_log = False ## Make sure it is False for optimal runtime
        ##  2. Holder for a long string with information of all patients
        self._log_string = ''
        ##  3. Set logger instance that works with other classes 
        self._logger = logging.getLogger ('simulator.simulator')  
        
        ## Parameters related to simulation results. These parameters will
        ## not be updated when a user changes input parameters.
        ##  1. Keep track of number of patients in the system per priority
        ##     class right before a new patient arrives
        self._n_patients = None
        ##  2. Keep track of individual patient instances
        self._subgroup_records = None
        ##  3. Patient waiting time and number of patients in two scenarios
        self._waiting_time_dataframe = None
        self._n_patient_total_dataframe = None
        self._n_patient_queue_dataframe = None

        ## Parameters that will be updated when a user changes input params.
        self._startTime = startTime                # simulation start timestamp
        self._nRadiologists = nRadiologists        # number of radiologists 
        self._timeWindowDays = timeWindowDays      # simulation duration in days
        self._nPatientsPadStart = nPatientsPads[0] # number of patients to be padded before counting results 
        self._nPatientsPadEnd = nPatientsPads[1]   # number of patients to be padded after counting results

        self._prevalence = None                    # disease prevalence within the non-emergent subgroup
        self._fractionED = None                    # fraction of emergent patient in all patients
        self._arrivalRate = None                   # overall patient arrival rate regardless of subgroups 
        self._serviceTimes = None                  # mean reading time by interrupting, diseased, and non-diseased
        
    ## +----------------------------------------
    ## | Class properties
    ## +----------------------------------------
    @property
    def qtypes (self): return self._qtypes
    @property
    def classes (self): return self._classes
    @property
    def all_records (self): return self._patient_records
    @property
    def log_string (self): return self._log_string
    @property
    def n_patients (self): return self._n_patients
    @property
    def n_patient_total_dataframe (self): return self._n_patient_total_dataframe
    @property
    def n_patient_queue_dataframe (self): return self._n_patient_queue_dataframe
    @property
    def waiting_time_dataframe (self): return self._waiting_time_dataframe
    @property
    def startTime (self): return self._startTime
    @startTime.setter
    def startTime (self, startTime):
        if isinstance (startTime, str):
            try:
                startTime = pandas.to_datetime (startTime)
            except:
                raise IOError ('Input startTime string is invalid.')
        if not isinstance (startTime, pandas._libs.tslibs.timestamps.Timestamp):
            raise IOError ('Input startTime must be a pandas Timestamp object.')
        self._startTime = startTime
    @property
    def endTime (self):
        if self._timeWindowDays >= 1:
            return self._startTime + pandas.offsets.Day (self._timeWindowDays)
        return self._startTime + pandas.offsets.Minute (int (self._timeWindowDays*24*60))
    @property
    def timeWindowDays (self): return self._timeWindowDays
    @timeWindowDays.setter
    def timeWindowDays (self, timeWindowDays):           
        self._timeWindowDays = timeWindowDays
    @property
    def nPatientPadsStart (self): return self._nPatientPadsStart
    @nPatientPadsStart.setter
    def nPatientPadsStart (self, nPatientPadsStart):
        if not isinstance (nPatientPadsStart, int):
            raise IOError ('Input nPatientPadsStart must be an integer.')         
        self._nPatientPadsStart = nPatientPadsStart
    @property
    def nPatientPadsEnd (self): return self._nPatientPadsEnd
    @nPatientPadsEnd.setter
    def nPatientPadsEnd (self, nPatientPadsEnd):
        if not isinstance (nPatientPadsEnd, int):
            raise IOError ('Input nPatientPadsEnd must be an integer.')            
        self._nPatientPadsEnd = nPatientPadsEnd
    @property
    def prevalence (self): return self._prevalence
    @prevalence.setter
    def prevalence (self, prevalence):
        if not isinstance (prevalence, float):
            raise IOError ('Input prevalence must be a float.')
        if not (prevalence >= 0.0 and prevalence <= 1.0):
            raise IOError ('Input prevalence must be between 0 and 1.')
        self._prevalence = prevalence
    @property
    def fractionED (self): return self._fractionED
    @fractionED.setter
    def fractionED (self, fractionED):
        if not isinstance (fractionED, float):
            raise IOError ('Input fractionED must be a float.')
        if not (fractionED >= 0.0 and fractionED <= 1.0):
            raise IOError ('Input fractionED must be between 0 and 1.')
        self._fractionED = fractionED
    @property
    def arrivalRate (self):
        return self._arrivalRate
    @arrivalRate.setter
    def arrivalRate (self, arrivalRate):
        if not isinstance (arrivalRate, float):
            raise IOError ('Input arrivalRate must be a float.')
        self._arrivalRate = arrivalRate
    @property
    def serviceTimes (self): return self._serviceTimes
    @serviceTimes.setter
    def serviceTimes (self, serviceTimes):
        # 1. It must be a dictionary
        if not isinstance (serviceTimes, dict):
            raise IOError ('Input serviceTimes must be a dictionary.')
        # 2. Three keys are expected: interrupting, diseased, and non-diseased
        for key in ['interrupting', 'diseased', 'non-diseased']:
            if not key in serviceTimes:
                raise IOError ('Input serviceTimes must include an "{0}" key.'.format (key))
        self._serviceTimes = serviceTimes
    @property
    def nRadiologists (self): return self._nRadiologists
    @nRadiologists.setter
    def nRadiologists (self, nRadiologists):
        if not isinstance (nRadiologists, int):
            raise IOError ('Input nRadiologists must be an integer.')        
        self._nRadiologists = nRadiologists
    @property
    def track_log (self): return self._track_log
    @track_log.setter
    def track_log (self, track_log):
        if not isinstance (track_log, bool):
            raise IOError ('Input track_log must be a boolean.')
        self._track_log = track_log
      
    ## +---------------------------------------------
    ## | Public functions to get simulation results
    ## +---------------------------------------------
    def get_all_records (self, qtype):
        
        ''' Public function to obtain all patient instances excluding
            padding patients from either with or without CADt scenarios.
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from all subgroups
        '''
        
        return self._subgroup_records[qtype]['all']

    def get_interrupting_records (self, qtype):

        ''' Public function to obtain only emergent patient instances
            excluding padding patients from either with or without
            CADt scenarios.
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from interrupting subgroups only
        '''        
        
        return self._subgroup_records[qtype]['interrupting']

    def get_noninterrupting_records (self, qtype):

        ''' Public function to obtain only non-emergent patient
            instances excluding padding patients from either with
            or without CADt scenarios.
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from non-interrupting subgroups only
        '''         
        
        return self._subgroup_records[qtype]['non-interrupting']

    def get_diseased_records (self, qtype):
        
        ''' Public function to obtain only diseased (signal presence)
            patient instances excluding padding patients from either
            with or without CADt scenarios. 
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from diseased subgroups only
        ''' 
        
        return self._subgroup_records[qtype]['diseased']

    def get_nondiseased_records (self, qtype):

        ''' Public function to obtain only non-diseased (signal
            absence) patient instances excluding padding patients
            from either with or without CADt scenarios. 
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from non-diseased subgroups only
        ''' 
        
        return self._subgroup_records[qtype]['non-diseased']

    def get_positive_records (self, qtype):

        ''' Public function to obtain only AI positive patient
            instances excluding padding patients from either
            with or without CADt scenarios. 
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from AI positive subgroups only
        '''         
        
        return self._subgroup_records[qtype]['positive']

    def get_negative_records (self, qtype):
        
        ''' Public function to obtain only AI negative patient
            instances excluding padding patients from either
            with or without CADt scenarios. 
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from AI negative subgroups only
        '''           
        
        return self._subgroup_records[qtype]['negative']

    def get_TP_records (self, qtype):
        
        ''' Public function to obtain only true-positive patient
            instances excluding padding patients from either
            with or without CADt scenarios. 
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from AI true-positive
                             subgroups only
        '''           
        
        return self._subgroup_records[qtype]['TP']

    def get_FN_records (self, qtype):

        ''' Public function to obtain only false-negative patient
            instances excluding padding patients from either
            with or without CADt scenarios. 
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from AI false-negative
                             subgroups only
        '''               
        
        return self._subgroup_records[qtype]['FN']
 
    def get_TN_records (self, qtype):

        ''' Public function to obtain only true-negative patient
            instances excluding padding patients from either
            with or without CADt scenarios. 
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from AI true-negative
                             subgroups only
        '''               
        
        return self._subgroup_records[qtype]['TN']
 
    def get_FP_records (self, qtype):
        
        ''' Public function to obtain only false-positive patient
            instances excluding padding patients from either
            with or without CADt scenarios. 
        
            input
            -----
            qtype (str): either 'fifo' for without-CADt scenario
                         or 'preresume' for with-CADt scenario
            
            output
            ------
            patients (list): patients from AI false-positive
                             subgroups only
        '''
        
        return self._subgroup_records[qtype]['FP']
   
    def get_n_patients_in_system (self, gtype):
        
        ''' Public function to obtain a summary dataframe of the
            number of patients of a specific subgroup in the system
            *right before* a new patient enters. This function is
            used to obtain the state probability distribution from
            simulation, which can be checked with the calculated
            distribution. Note that the numbers in these dataframes
            are expected to be small (a lot of 0s and 1s) because
            it is not the number of patients that have arrived.
            Rather, it is the number of patients (of a specific
            subgroup) in the system *when a new patient arrives*.
            The most important subgroups are 
                * for without-CADt: interrupting and non-interrupting
                * for with CADt: interrupting, positive, and negative
            Note that for positive and negative subgroups, only
            the counts from with-CADt scenario is available.
        
            input
            -----
            gtype (str): subgroup name: 'all', 'non-interrupting',
                         'interrupting', 'diseased', 'non-diseased',
                         'positive', 'negative'
            
            output
            ------
            dataframe (pandas.dataframe): number of subgroup
                                          patients in the system
                                          *right before* a new
                                          patient arrives.
        '''
        
        return self._n_patient_total_dataframe[gtype]

    def get_n_patients_in_queue (self, gtype):
        
        ''' Public function to obtain a summary dataframe of the
            number of patients of a specific subgroup in the queue
            *right before* a new patient enters. This function is
            no longer used since the RDR method has the state
            defined as the number of patients in the system.
        
            input
            -----
            gtype (str): subgroup name: 'all', 'non-interrupting',
                         'interrupting', 'diseased', 'non-diseased',
                         'positive', 'negative'
            
            output
            ------
            dataframe (pandas.dataframe): number of subgroup
                                          patients in the queue
                                          *right before* a new
                                          patient arrives.
        '''
        
        return self._n_patient_queue_dataframe[gtype]   

    ## +---------------------------------------------
    ## | Private functions to simulate queues
    ## +---------------------------------------------
    def _count_nPatients_in_queue (self, original, noCADt=False):

        ''' Count the number of patients currently in the queue. This
            function is meant to be called right before a new patient
            arrives. Note that in the queue module, once an item is
            "get" from the queue, the item is automatically removed
            from the queue. Since we need to count the number of
            patients with different priority classes, patients in the
            queue need to be pulled to get the priority information. 
            Therefore, a new queue is created such that the patient
            is copied to the new queue after it is pulled from the
            old queue.
            
            This function was written when without-CADt is a simple
            FIFO. Now that both with and withou CADt scenarios have
            more than one priority classes, this function could be
            simplied.
            
            inputs
            ------
            original (queue.PriorityQueue): current queue from which
                                            patients are read
            noCADt (bool): True if counting the queue in the without
                           CADt scenario 
            
            outputs
            -------
            copied (queue.PriorityQueue): copied queue with the same
                                          patients in the input queue
            nPatients (dict): number of patients in different subgroups,
                              including 'all', 'interrupting', 'diseased',
                              'non-interrupting', and 'non-diseased'. For
                              with-CADt scenario, 'positive', 'negative',
                              'TP', 'FP', 'TN', 'FN' are also included.
        '''

        ## Set counters
        nClass1, nClass2, nAll = 0, 0, 0
        nDiseased, nNonDiseased = 0, 0
        nTP, nTN, nFP, nFN = 0, 0, 0, 0
        nEmergency, nNonEmergency = 0, 0
        ## Separate variable to keep track of the patient class
        pclass = None

        # Holder for the copied queue
        copied = queue.PriorityQueue()

        ## For without CADt scenario, only two priority classes:
        ## interrupting and non-interrupting. Within the non-interrupting
        ## class, count the numebr of diseased vs non-diseased
        ## patients, both of which have the same priority classes.
        if noCADt:
            while not original.empty():
                nAll += 1
                # Pull out the patient instance
                p = original.get()[2]
                # Emergency class has the highest priority i.e. 1 
                if p.is_interrupting:
                    nEmergency += 1
                    pclass = 1
                # Diseased patient is non-interrupting class i.e. 2
                elif p.is_diseased:
                    nNonEmergency += 1
                    nDiseased += 1
                    pclass = 2
                # Non-diseased patient is non-interrupting class i.e. 2                    
                else:
                    nNonEmergency += 1
                    nNonDiseased += 1
                    pclass = 2
                # Copy this patient to the new queue. Prioritized based
                # on patient's priority class, and then by its arrival
                # time. The last entry is the patient instance.
                copied.put ((pclass, p.trigger_time, p))
            
            return copied, {'all':nAll, 'non-interrupting':nNonEmergency, 'interrupting':nEmergency,
                            'diseased':nDiseased, 'non-diseased':nNonDiseased}
    
        ## For with CADt scenario, there are three priority classes:
        ## interrupting and positive and negative. Within the positive
        ## and negative classes, also count the number of diseased
        ## (i.e. TP or FN) and non-diseased (i.e. FP or TN) patients,
        ## all of which have the same priority classes.
        while not original.empty():
            nAll += 1
            # Pull out the patient instance
            p = original.get()[2]
            # Emergency class has the highest priority i.e. 1 
            if p.is_interrupting:
                nEmergency += 1
                pclass = 1
            # Positive class has a middle priority i.e. 2
            elif p.priority_class == 2:
                nNonEmergency += 1
                nClass1 += 1
                pclass = 2
                if p.is_diseased:
                    nDiseased += 1
                    nTP += 1
                else:
                    nNonDiseased += 1
                    nFP += 1
            # Negative class has the lowest priority i.e. 3
            elif p.priority_class == 3:
                nNonEmergency += 1
                nClass2 += 1
                pclass = 3
                if p.is_diseased:
                    nDiseased += 1
                    nFN += 1
                else:
                    nNonDiseased += 1
                    nTN += 1
            # Copy this patient to the new queue. Prioritized based
            # on patient's priority class, and then by its arrival
            # time. The last entry is the patient instance.                    
            copied.put ((pclass, p.trigger_time, p))
    
        return copied, {'all':nAll, 'non-interrupting':nNonEmergency, 'interrupting':nEmergency,
                        'positive':nClass1, 'negative':nClass2,
                        'diseased':nDiseased, 'non-diseased':nNonDiseased,
                        'TP':nTP, 'TN':nTN, 'FP':nFP, 'FN':nFN}

    def _count_nPatients (self, qtype, newArrivalTime):
        
        ''' Count the number of patients currently in the system. This
            function is meant to be called right before a new patient
            arrives. The number of patients in the system is the sum
            of number of patients in the queue and number of patients
            currently read be the radiologists. The counts will be
            stored in self._n_patients.
                        
            inputs
            ------
            qtype (str): either 'fifo' meaning without-CADt scenario
                         or 'preresume' meaning with-CADt scenario
            newArrivalTime (pandas.datetime): the time to check if a
                                              radiologist is busy
        '''
        
        ## Update logger message
        self._logger.debug ('+---------------------------------------------------------------')
        self._logger.debug ('| _count_nPatiets {0}: '.format (qtype))        
        
        ## Get the booleans whether each doctor is treating diseased,
        ## non-diseased, positive, negative, or interrupting patients 
        #   1. Get the doctors for this scenario (either with or without CADt)
        doctors = self._fionas if qtype=='fifo' else self._palmers
        #   2. Holders for booleans
        doctor_is_busy, doctor_treating_interrupting = [], []
        doctor_treating_diseased, doctor_treating_nondiseased = [], [] 
        doctor_treating_positive, doctor_treating_negative = [], []
        #   3. Loop through each doctor (in case of multiple radiologists)
        #      and get the subgroup of its current patient 
        for doctor in doctors:
            #  Is this doctor currently busy? If not, a value of None is
            #  appended to all holders instead of True or False.
            is_busy = doctor.is_busy (newArrivalTime)
            doctor_is_busy.append (is_busy)
            #  Check whether the current patient is interrupting, diseased,
            #  non-diseased, positive, or negative.
            interrupting   = numpy.nan if not is_busy else doctor.current_patient.is_interrupting
            diseased    = numpy.nan if not is_busy else False if doctor.current_patient.is_interrupting else \
                          doctor.current_patient.is_diseased
            nondiseased = numpy.nan if not is_busy else False if doctor.current_patient.is_interrupting else \
                          not doctor.current_patient.is_diseased                       
            positive    = numpy.nan if not is_busy else False if doctor.current_patient.is_interrupting else \
                          doctor.current_patient.is_positive
            negative    = numpy.nan if not is_busy else False if doctor.current_patient.is_interrupting else \
                          not doctor.current_patient.is_positive
            #  Append to boolean holders
            doctor_treating_interrupting.append (interrupting)
            doctor_treating_diseased.append (diseased)
            doctor_treating_nondiseased.append (nondiseased)
            doctor_treating_positive.append (positive)
            doctor_treating_negative.append (negative)
        #  4. Drop any None values in the boolean arrays 
        doctor_is_busy = numpy.array (doctor_is_busy)
        doctor_treating_interrupting   = remove_nan (numpy.array (doctor_treating_interrupting))
        doctor_treating_positive    = remove_nan (numpy.array (doctor_treating_positive))
        doctor_treating_negative    = remove_nan (numpy.array (doctor_treating_negative))
        doctor_treating_diseased    = remove_nan (numpy.array (doctor_treating_diseased))
        doctor_treating_nondiseased = remove_nan (numpy.array (doctor_treating_nondiseased))
        
        ## Count the number of patients currently in the queue
        self._aqueue[qtype], nPatients = self._count_nPatients_in_queue (self._aqueue[qtype], noCADt=qtype=='fifo')
        
        ## Count the number of patients currently in the system
        for group, nqueue in nPatients.items():
            
            # # patients in queue only
            self._n_patients[qtype][group]['queue'].append (nqueue)
            
            # total # patients in system
            #  If all doctors are in idle (i.e. not busy), # total is the same as # queue i.e. 0
            if  numpy.all (doctor_is_busy==False):
                # Update self._n_patients
                self._n_patients[qtype][group]['total'].append (nqueue)
                self._logger.debug ('|     * None of the doctors is busy ({0}). {1} nqueue: {2}'.format (newArrivalTime, group, nqueue))
                continue

            #  Otherwise, # total is # queue + # busy radiologists
            #  is_group is the boolean of whether the doctors are treating patients of this group.            
            if group == 'all':
                is_group = doctor_is_busy
            elif qtype == 'fifo':
                is_group = doctor_treating_interrupting   if group=='interrupting' else \
                           ~doctor_treating_interrupting  if group=='non-interrupting' else \
                           doctor_treating_diseased    if group=='diseased' else \
                           doctor_treating_nondiseased if group=='non-diseased' else \
                           numpy.zeros (len (doctor_treating_diseased)).astype (bool)
            else:
                is_group = doctor_treating_interrupting   if group=='interrupting'     else \
                           ~doctor_treating_interrupting  if group=='non-interrupting' else \
                           doctor_treating_diseased    if group=='diseased'      else \
                           doctor_treating_nondiseased if group=='non-diseased'  else \
                           doctor_treating_positive    if group=='positive'      else \
                           doctor_treating_negative    if group=='negative'      else \
                           numpy.logical_and (~doctor_treating_interrupting, numpy.logical_and (doctor_treating_diseased   , doctor_treating_positive)) if group=='TP' else \
                           numpy.logical_and (~doctor_treating_interrupting, numpy.logical_and (doctor_treating_nondiseased, doctor_treating_positive)) if group=='FP' else \
                           numpy.logical_and (~doctor_treating_interrupting, numpy.logical_and (doctor_treating_nondiseased, doctor_treating_negative)) if group=='TN' else \
                           numpy.logical_and (~doctor_treating_interrupting, numpy.logical_and (doctor_treating_diseased   , doctor_treating_negative)) ## FN
            #  Count the number of True in is_group i.e. number of patients in this subgroup
            #  currently being served by the radiologists
            is_group = is_group.astype (bool)
            nserving = len (is_group[is_group])
            #  Update self._n_patients
            self._n_patients[qtype][group]['total'].append (nqueue + nserving)
            self._logger.debug ('|     * {0} nservings, nqueue: {1}, {2}'.format (group, nserving, nqueue))

    def _get_next_patient (self, qtype, doctor_name):

        ''' This function returns the next patient to be seen. For
            previously interrupted patient, it was programmed to 
            find previous doctor and wait until it is free such
            that the same patient would not be read by different
            radiologists. However, this is different from the
            preemptive resume assumption where no work is re-done.
            Therefore, in this function, only the next patient in
            the queue is returned. But in the debugging, a message
            will be shown informing us whether this patient is
            seen by different doctor than before.
            
            inputs
            ------
            qtype (str): either 'fifo' meaning without-CADt scenario
                         or 'preresume' meaning with-CADt scenario
            doctor_name (str): the current doctor that is in idle
                               and ready to read the new case
                               
            output
            ------
            p (patient): the next patient to be read 
        '''

        ## Get the next patient in the queue
        p = self._aqueue[qtype].get()[2]
        
        ## If this patient is previously interrupted, show in logger
        ## whether it is now read by a different doctor.
        while p.doctors is not None:
            self._logger.debug ('| | Dealing with previously interrupted patient {0}'.format (p.caseID))
            self._logger.debug ('| | p.doctors: {0}'.format (p.doctors))
            if p.doctors[0] == doctor_name:
                self._logger.debug ('| | This doctor {0} is the same as before :)'.format (doctor_name))
            else:
                # The previously interrupted patient was not seen by this doctor.
                self._logger.debug ('| | This doctor, {0}, is different than before {1}'.format (doctor_name, p.doctors[0]))
            return p

        self._logger.debug ('| | Doctor {0} is seeing patient {1}'.format (doctor_name, p.caseID))
        return p

    def _append_log_string_radiologist (self, qtype, aTime, original):
 
        ''' If asked to track log, form a text string to be appended
            in the log string. In doing so, we need to extract data
            from the patients in the queue. To avoid the issue in 
            getting but not removing entries in the queue module,
            the original queue is copied to a new queue, which is
            then returned as an output. 
            
            inputs
            ------
            qtype (str): either 'fifo' meaning without-CADt scenario
                         or 'preresume' meaning with-CADt scenario
            aTime (pandas.datetime): the time to be logged
            original (queue.PriorityQueue): current queue from which
                                            patients are read
            
            output
            ------
            copied (queue.PriorityQueue): copied queue with the same
                                          patients in the input queue            
        '''
 
        ## If not asked to log, don't need to do anything.
        if not self._track_log: return original

        ## Print out the current timestamp in seconds from start time
        aTimeSec = (aTime - self._startTime).total_seconds() 
        summary = 'At {0:.1f}\n'.format (aTimeSec)
        
        ## Status of each doctor: which patient (with its priority
        ## class) and the closing time in second from start time
        doctors = self._fionas  if qtype=='fifo' else self._palmers
        for doctor in doctors:
            name = doctor.radiologist_name
            has_patient = doctor.current_patient is not None
            closeTimeSec, caseID, priority_class = 0, '', ''
            if has_patient:
                closeTimeSec = (doctor.current_patient.latest_close_time - self._startTime).total_seconds()
                ## Only show the patient id if doctor is currently being served 
                if closeTimeSec > aTimeSec:
                    caseID = doctor.current_patient.caseID
                    priority_class = doctor.current_patient.priority_class
            summary += '* {0:8} ({1:10.1f}): {2:7}-{3}\n'.format (name, closeTimeSec, caseID, priority_class)

        ## Status of each patient in the queue: patient name and its
        ## priroity class
        summary += '{0:6} queue: '.format (name[:-2])
        copied = queue.PriorityQueue()
        while not original.empty():
            p = original.get()[2]
            if p.is_interrupting:
                pclass = 1
            elif p.is_positive:
                pclass = 2  
            else:
                pclass = 2 if qtype == 'fifo' else 3
            copied.put ((pclass, p.trigger_time, p))
            ## Only show in list if this patient actually arrives the system by report time
            if p.trigger_time <= aTime:
                summary += '{0:7}-{1}, '.format (p.caseID, p.priority_class)
        summary += '\n'
        
        ## Update the log string
        if self._track_log: self._log_string += summary
        return copied
            
    def _append_log_string_patient (self, aPatient):

        ''' If asked to track log, form a text string to be appended
            in the log string about the input patient instance.
            
            input
            -----
            aPatient (patient): log the information related to the
                                input patient
        '''

        ## If not asked to log, don't need to do anything.
        if not self._track_log: return
        ## Form debug string log text
        summary  = '+----------------------------------------------\n'
        summary += '| Patient ({0:7})\n'.format (aPatient.caseID)
        summary += '|  * Priority class? {0}\n'.format (aPatient.priority_class)
        triggerSec =  (aPatient.trigger_time - self._startTime).total_seconds() 
        summary += '|  * Arrived at {0:.3f} seconds \n'.format (triggerSec)
        summary += '|  * Served for {0:.3f} seconds \n'.format (aPatient.service_duration*60)
        summary += '|  Emergency, Non-Emergency: {0}, {1}\n'.format (self._n_patients['fifo']['interrupting']['total'][-1],
                                                                     self._n_patients['fifo']['non-interrupting']['total'][-1])
        if len (self._n_patients['preresume']['interrupting']['total']) > 0:
            summary += '|  Emergency, Positive, Negative: {0}, {1}, {2}\n'.format (self._n_patients['preresume']['interrupting']['total'][-1],
                                                                                   self._n_patients['preresume']['positive']['total'][-1],
                                                                                   self._n_patients['preresume']['negative']['total'][-1])
        summary += '+----------------------------------------------\n'
        ## Update log string
        if self._track_log: self._log_string += summary

    def _radiologist_do_work (self, qtype, newArrivalTime):
    
        ''' An important function during simulation to mimic radiologist
            steps when a new case arrives. Because this is an important
            function, logger is constantly updated for debugging purposes.
            
            1. Radiologist needs to read everything before the new patient 
            2. Radiologist may resume reading of previously interrupted case
            
            Note that the special part here is related to the busy period.
            In simulation, the busy period can be collected from any input
            clinical setting. However, theoretically, I can only calculate
            the busy period in without-CADt scenario with only one
            radiologist.
            
            inputs
            ------
            qtype (str): either 'fifo' meaning without-CADt scenario
                         or 'preresume' meaning with-CADt scenario
            newArrivalTime (pandas.datetime): the arrival time of the
                                              new patient
                                              
            output
            ------
            drTime (float): the latest closing time among the current
                            patients that are being served by doctors
        '''
    
        ## Get the doctors for this queue type 
        doctors = self._fionas if qtype=='fifo' else self._palmers 
        
        ## Update logger
        self._logger.debug ('+====================================================')
        self._logger.debug ('| In _radiologist_do_work() for {0} ...'.format (qtype))
        self._logger.debug ('| ')
        self._logger.debug ('| {0}'.format ([doctor.radiologist_name for doctor in doctors]))
        self._logger.debug ('| New arrival time: {0}'.format (newArrivalTime))
        
        ## Get doctor closing time i.e. for each doctor, its current patient's closing time
        ## Note that, Between current and new patient, doctor may complete reading 0 or many patients
        drTimes = [doctor.current_patient.latest_close_time if doctor.current_patient is not None else \
                   self._startTime for doctor in doctors]
        drTime = min (drTimes) 
        self._logger.debug ('| All doctor Times before new arrival: {0}'.format (drTimes))
        
        ## Between the earliest closing time and the arrival time of the new patient,
        ## radiologist needs to keep reading patients that came before the new patient.
        while drTime < newArrivalTime:
            self._logger.debug ('| +-------------------------------------')
            self._logger.debug ('| | New Arrival Time: {0}'.format (newArrivalTime))
            self._logger.debug ('| | Current dr time: {0}'.format (drTime))
            # No more patients in line. The new patient will enter an empty system.
            if self._aqueue[qtype].qsize() == 0:
                self._logger.debug ('| | No patient in line.')
                break
            # Which doctor that finishes its last patient first
            # i.e. the one with the smallest closing time of the last patient
            self._logger.debug ('| | Which doctor? the {0}-th  i.e. {1}'.format (numpy.argmin (drTimes),
                                                                                 doctors[numpy.argmin (drTimes)].radiologist_name))
            doctor = doctors[numpy.argmin (drTimes)] 
            # Handle next patients
            p = self._get_next_patient (qtype, doctor.radiologist_name)
            ############################################################################################################
            # Special part starts: busy period for without CADt scenario
            #                      this is to be checked with the theoretical busy period
            #                      distribution with only one radiologist.
            # if qtype == 'fifo':
            #     # Check if the next patient is the first low-class patient
            #     # 1. Doctor is seeing higher class patient 
            #     old_is_high = None if doctor.current_patient is None else doctor.current_patient.priority_class == 1
            #     # 2. Next patient is lower class patient 
            #     new_is_low  = None if p is None else p.priority_class == 2
            #     # Busy period starts at patient arrival time when
            #     if p is not None and p.priority_class == 1:
            #         starttime = (p.trigger_time - self._startTime).total_seconds()
            #         if doctor.current_patient is None:  
            #             self._busy_periods_fifo['start'].append (starttime)
            #             self._log_string += '*** BUSY PERIOD STARTS AT {0}.\n'.format (starttime)
            #         elif doctor.current_patient.priority_class == 2:
            #             self._busy_periods_fifo['start'].append (starttime)
            #             self._log_string += '*** BUSY PERIOD STARTS AT {0}.\n'.format (starttime)
            #     ## Busy period ends at doctor end time when
            #     #   1. Old patients high, No patients now
            #     #   2. Old patients high, new is low
            #     if doctor.current_patient is not None and doctor.current_patient.priority_class == 1:
            #         endtime = (doctor.current_patient.latest_close_time - self._startTime).total_seconds()
            #         if p is None:   
            #             self._busy_periods_fifo['end'].append (endtime)
            #             self._log_string += '*** BUSY PERIOD ENDS AT {0}.\n'.format (endtime)
            #         elif p.priority_class == 2:   
            #             self._busy_periods_fifo['end'].append (endtime)
            #             self._log_string += '*** BUSY PERIOD ENDS AT {0}.\n'.format (endtime)
            # # Special part ends
            ############################################################################################################
            # If the next patient is not available, it means there is no more patient.
            if p is None: break
            # Otherwise, read this patient
            doctor.read_new_patient (p)
            self._logger.debug ('| | This doctor new end time: {0}'.format (doctor.current_patient.latest_close_time))
            # Append this patient instance for record keeping
            self._patient_records[qtype].append (doctor.current_patient)
            # Update logger for debugging
            report_time = max (doctor.current_patient.open_times[-1], p.trigger_time) 
            if self._track_log: self._log_string += 'from _radiologist_do_work while loop: \n'
            self._aqueue[qtype] = self._append_log_string_radiologist (qtype, report_time, self._aqueue[qtype])
            message = '| | {0} closed case, {1}, at {2} after waiting for {3:.4f} minutes'
            self._logger.debug (message.format (doctor.radiologist_name, doctor.current_patient.caseID,
                                                doctor.current_patient.latest_close_time,
                                                doctor.current_patient.wait_time_duration))
            # Get the updated earliest doctor closing time
            drTimes = [doctor.current_patient.latest_close_time if doctor.current_patient is not None else \
                       self._startTime for doctor in doctors]
            drTime = min (drTimes) ## Default min (drTimes)
            self._logger.debug ('| | New dr times: {0}'.format (drTimes))
            self._logger.debug ('| | New min dr time: {0}'.format (drTime))
        self._logger.debug ('| +-------------------------------------')

        ## Right before reading the newest patient, a doctor may be now available for previously
        ## interrupted patient. 
        if self._aqueue[qtype].qsize() > 0:
            ## Update logger
            self._logger.debug ('| | Special check for pre-resume ...')
            self._logger.debug ('| +-------------------------------------')
            self._logger.debug ('| | New Arrival Time: {0}'.format (newArrivalTime))
            # For each doctor, check if it is 
            for doctor in doctors:
                # If no more patient in the queue, nothing needs to be done
                if self._aqueue[qtype].qsize() == 0: break
                # If the doctor is busy with a higher priority patient, it is not available
                if doctor.current_patient is None: continue
                drTime = doctor.current_patient.latest_close_time
                if not drTime < newArrivalTime: continue
                self._logger.debug ('| | Doctor {0} has a closing time < new arrival time'.format (doctor.radiologist_name))
                self._logger.debug ('| | Current dr time: {0}'.format (drTime))
                   
                # Handle next patients
                p = self._get_next_patient (qtype, doctor.radiologist_name)
                ############################################################################################################
                # Special part starts: busy period for without CADt scenario
                #                      this is to be checked with the theoretical busy period
                #                      distribution with only one radiologist.
                # if qtype == 'fifo':
                #     # Check if the next patient is the first low-class patient
                #     # 1. Doctor is seeing higher class patient 
                #     old_is_high = None if doctor.current_patient is None else doctor.current_patient.priority_class == 1
                #     # 2. Next patient is lower class patient 
                #     new_is_low  = None if p is None else p.priority_class == 2
                #     # Busy period starts at patient arrival time when
                #     #busy_period_starts1 = doctor.current_patient is None and p.priority_class == 1 # No patients now, new is high
                #     #busy_period_starts2 = doctor.current_patient.priority_class == 2 and p.priority_class == 1 # Old patients low, new is high 
                #     if p is not None and p.priority_class == 1:
                #         starttime = (p.trigger_time - self._startTime).total_seconds()
                #         if doctor.current_patient is None:  
                #             self._busy_periods_fifo['start'].append (starttime)
                #             self._log_string += '*** BUSY PERIOD STARTS AT {0}.\n'.format (starttime)
                #         elif doctor.current_patient.priority_class == 2:
                #             self._busy_periods_fifo['start'].append (starttime)
                #             self._log_string += '*** BUSY PERIOD STARTS AT {0}.\n'.format (starttime)
                #     ## Busy period ends at doctor end time when
                #     #   1. Old patients high, No patients now
                #     #   2. Old patients high, new is low
                #     if doctor.current_patient is not None and doctor.current_patient.priority_class == 1:
                #         endtime = (doctor.current_patient.latest_close_time - self._startTime).total_seconds()
                #         if p is None:   
                #             self._busy_periods_fifo['end'].append (endtime)
                #             self._log_string += '*** BUSY PERIOD ENDS AT {0}.\n'.format (endtime)
                #         elif p.priority_class == 2:   
                #             self._busy_periods_fifo['end'].append (endtime)
                #             self._log_string += '*** BUSY PERIOD ENDS AT {0}.\n'.format (endtime)
                # # Special part ends
                ############################################################################################################
                # If the next patient is not available, it means there is no more patient.
                if p is None: continue
                # Read this patient
                doctor.read_new_patient (p)
                self._logger.debug ('| | This doctor new end time: {0}'.format (doctor.current_patient.latest_close_time))
                # Append this patient instance for record keeping
                self._patient_records[qtype].append (doctor.current_patient)
                # Update logging for debugging
                if self._track_log: self._log_string += 'from _radiologist_do_work interruption: \n'
                self._aqueue[qtype] = self._append_log_string_radiologist (qtype, doctor.current_patient.trigger_time, self._aqueue[qtype])
                message = '| | {0} closed case, {1}, at {2} after waiting for {3:.4f} minutes'
                self._logger.debug (message.format (doctor.radiologist_name, doctor.current_patient.caseID,
                                                    doctor.current_patient.latest_close_time,
                                                    doctor.current_patient.wait_time_duration))
            self._logger.debug ('| +-------------------------------------')


        ## Print to log file if more than one patient in queue
        if self._aqueue[qtype].qsize() > 0:
            if self._track_log: self._log_string += 'from _radiologist_do_work end: \n'
            self._aqueue[qtype] = self._append_log_string_radiologist (qtype, newArrivalTime, self._aqueue[qtype])
            
        ## Return the closing time of the *last* patient served
        drTimes = [doctor.current_patient.latest_close_time if doctor.current_patient is not None else \
                   self._startTime for doctor in doctors]
        drTime = max (drTimes) 
        self._logger.debug ('| ')
        self._logger.debug ('| New dr times: {0}'.format (drTimes))
        self._logger.debug ('| New max dr time: {0}'.format (drTime))
        self._logger.debug ('+====================================================')
        self._logger.debug ('| ')        
        return drTime

    def _put_future_patients_in_queue (self, qtype, drTime, future_patient):
    
        ''' Future patients here refers to patients who arrives after the
            certain action time points (e.g. when radiologist closes a
            case, or when radiologist is interrupted, etc.). We have
            future patients because arrival time is randomly generated
            and sometimes before these patients come, radiologists can
            finish reading multiple cases in the queue before its arrival.
            Therefore, instead of putting this future patient in the queue,
            it is stored separately until certain time point is reached.
            This function is called when this time point is reached. So,
            now the future patients will be added to the normal queue.
            
            inputs
            ------
            qtype (str): either 'fifo' meaning without-CADt scenario
                         or 'preresume' meaning with-CADt scenario
            drTime (float): the latest closing time among the current
                            patients that are being served by doctors
            future_patient (list): list of the patient instances that
                                   arrive too far in the future
                                      
            output
            ------
            future_patient (list): updated list of patient instances
                                   that arrive too far in the future            
        '''
    
        ## If the current list of future patients is empty, no need
        ## to do anything.
        if len (future_patient) == 0: return future_patient
    
        ## For each future patient in the list, compare the arrival
        ## time with the latest closing time among the current patients
        ## being served by the doctors. If it is before the latest
        ## closing time, this future patient should be added to the queue. 
        for p in future_patient[:]:
            if self._aqueue[qtype].qsize()==0 or p.trigger_time < drTime:
                # Count # patients in queue and in system *right before* arrival
                self._count_nPatients (qtype, p.trigger_time)
                # Put this patient in the queue
                pclass = 1 if p.is_interrupting else 2 if qtype=='fifo' else p.priority_class
                self._aqueue[qtype].put((pclass, p.trigger_time, p))
                # Remove this patient from the list of future patient
                future_patient.remove (p)
                # Update logging
                if self._track_log:
                    self._log_string += 'from put_future_patients_in_queue for {0}: \n'.format (qtype)
                    self._append_log_string_patient (p)
        
        return future_patient

    def _read_newest_urgent_patient (self, qtype, apatient):

        ''' Called to see if the input patient needs to be read immediately.
            If so, update doctor's (lower-priority) current patient
            
            inputs
            ------
            qtype (str): either 'fifo' meaning without-CADt scenario
                         or 'preresume' meaning with-CADt scenario
            apatient (patient): a patient instance that may interrupt
                                radiologist's workflow
        '''

        ## If no input patient, no need to do anything.
        if apatient is None: return 
        ## Also no interruption if the new patient is of lower classes
        if qtype=='fifo' and not apatient.priority_class == 1: return 
        if not qtype=='fifo' and apatient.priority_class == 3: return 

        ## This patient interrupts radiologists. Update logger
        self._logger.debug ('+====================================================')
        self._logger.debug ('| In _read_newest_urgent_patient() for {0} ...'.format (qtype))
        self._logger.debug ('| ')
        ## Get the doctors for this queue type
        doctors = self._fionas  if qtype=='fifo' else self._palmers 
        ## Check the current patient in each doctor
        for doctor in doctors:
            if doctor.current_patient is not None:
                # Nothing special to be done if one of the doctors is already treating this
                # priority patient. Within the same priority class, the current patient has
                # a higher priority (based on arrival time). 
                if apatient == doctor.current_patient:
                    self._logger.debug ('| Doctor {0} is already reading this new patient.'.format (doctor.radiologist_name))
                    self._logger.debug ('+====================================================')
                    self._logger.debug ('| ')
                    return
                # No interruption if this priority patient arrives when one of the doctors is in idle.
                # This patient would be picked by this doctor without interruption (via regular procedure).
                if apatient.trigger_time > doctor.current_patient.latest_close_time:
                    self._logger.debug ('| Doctor {0} is in idle when this new patient arrives.'.format (doctor.radiologist_name))
                    self._logger.debug ('| This patient will be picked up in regular procedure without interruption.')
                    self._logger.debug ('+====================================================')
                    self._logger.debug ('| ')
                    return
            
        ## Sort doctors such that the one with the lowest priority comes first
        sort_indices = numpy.argsort ([2 if qtype == 'fifo' and d.current_patient.priority_class==3 else 
                                       d.current_patient.priority_class for d in doctors])[::-1]
        doctors = numpy.array (doctors)[sort_indices]
        
        ## For each doctor, see if 
        for doctor in doctors:
            # Update logger for this doctor
            self._logger.debug ('| +-------------------------------------')
            self._logger.debug ('| | {0}'.format (doctor.radiologist_name))
            # Next doctor if this doctor has no patient i.e. no need to interrupt.
            if doctor.current_patient is None:
                self._logger.debug ('| | This doctor has no patient. No interruption')
                continue
            
            # Next doctor if this doctor is treating the same (high) priority class.
            pclass = 1 if apatient.is_interrupting else 2 if qtype=='fifo' else apatient.priority_class
            doc_pclass = 1 if doctor.current_patient.is_interrupting else 2 if qtype=='fifo' else doctor.current_patient.priority_class
            if doc_pclass <= pclass:
                self._logger.debug ('| | This doctor is reading a higher-priority patient. (next doctor)')
                continue
            
            # Now, interruption happens.
            self._logger.debug ('| | This doctor is interrupted by the new arrival.')
            ################################################################################################
            # Special code for busy period starts: Only works in without CADt scenario with one radiologist
            #old_is_low = doc_pclass == 2
            # Special code for busy period ends
            ################################################################################################
            
            # Current patient is back in the queue.
            p = doctor.stop_reading (apatient.trigger_time)
            pclass = 1 if p.is_interrupting else 2 if qtype=='fifo' else p.priority_class
            self._aqueue[qtype].put ((pclass, p.trigger_time, p))
            # Remove this patients from the list of patient records
            self._patient_records[qtype].remove (p)
            self._logger.debug ('| | Case {0} is back in queue.'.format (p.caseID))
            # Now, read the highest priority patient.
            doctor.read_new_patient (self._aqueue[qtype].get()[2])
            
            ################################################################################################
            # Special code for busy period starts:  
            #if qtype == 'fifo':
            #    if old_is_low and doctor.current_patient.priority_class == 1:
            #        starttime = (doctor.current_patient.trigger_time - self._startTime).total_seconds()
            #        self._busy_periods_fifo['start'].append (starttime)
            #        if self._track_log: self._log_string += '*** BUSY PERIOD STARTS (interrupted) AT {0}.\n'.format (starttime)
            # Special code for busy period ends
            ################################################################################################
                        
            # Append current patient instance for record keeping 
            self._patient_records[qtype].append (doctor.current_patient)
            # Print to log file 
            message = '| | {0} closes case, {1}, at {2} after waiting for {3:.4f} minutes'
            if self._track_log: self._log_string += 'from _read_newest_urgent_patient: \n'
            self._aqueue[qtype] = self._append_log_string_radiologist (qtype, doctor.current_patient.trigger_time, self._aqueue[qtype])            
            self._logger.debug (message.format (doctor.radiologist_name, doctor.current_patient.caseID,
                                                doctor.current_patient.latest_close_time,
                                                doctor.current_patient.wait_time_duration))
            
            # The new interrupting patient is handled. No need to check other doctors.
            break
        
        self._logger.debug ('+====================================================')
        self._logger.debug ('| ')

    def _handle_next_patient (self, apatient):

        ''' When a new patient arrives, it is first put in the list of
            future patient. If conditions are met, this patient is then
            moved to the queue. Before putting it in the queue, we need
            to count the number of patients *right before* the new
            patient arrives.
            
            input
            -----
            apatient (patient): new patient instance
        '''

        ## No action if input patient is None.
        if apatient is None: return
        ## Update logging
        self._logger.debug ('+====================================================')
        self._logger.debug ('| In _handle_next_patient() ...')
        self._logger.debug ('| ')

        ## This very same patient is handled in two scenarios
        for qtype in self._qtypes:
            
            self._logger.debug ('| In {0},'.format (qtype))
            # Append this patient to future patients
            self._future_patients[qtype].append (deepcopy (apatient))
            # Get the doctors for this queue type
            doctors = self._fionas if qtype=='fifo' else self._palmers 
                     
            ## Put new patients in queue - ONLY those that is before last-leaving radiologist's current
            #  close time. This is to avoid scenario where future high-class patients are treated before
            #  actually entering the system. The only exception is when there is down time
            #  i.e. fast-forward to the timestamp when the patient comes in.
            # Between current and next patient, each doctor may complete 0, 1, 2, etc patients
            drTimes = [doctor.current_patient.latest_close_time if doctor.current_patient is not None else \
                       self._startTime for doctor in doctors]
            drTime = min (drTimes) 
            
            ## For each future patient, decide if it needs to be put in the queue 
            for p in self._future_patients[qtype][:]:
                if self._aqueue[qtype].qsize()==0 or p.trigger_time < drTime:
                    ## Count # patients in queue and in system *right before* arrival
                    self._count_nPatients (qtype, p.trigger_time)
                    # Put this patient in the queue
                    self._logger.debug ('|   - putting patient {0} in queue.'.format (p.caseID))
                    pclass = 1 if p.is_interrupting else 2 if qtype=='fifo' else p.priority_class
                    self._aqueue[qtype].put((pclass, p.trigger_time, p))
                    # Remove this patient from the list of future patient
                    self._future_patients[qtype].remove (p)
                    # Update logging
                    if self._track_log: 
                        self._log_string += 'from handle_next_patient for {0}: \n'.format (qtype)
                        self._append_log_string_patient (p)

            self._logger.debug ('| * {0} patients currently in queue.'.format (self._aqueue[qtype].qsize()))
            self._logger.debug ('| * {0} future patients remain: {1}'.format (len (self._future_patients[qtype]),
                                                                              ', '.join ([p.caseID for p in self._future_patients[qtype]])))        


        self._logger.debug ('+====================================================')
        self._logger.debug ('| ')

    def _divide_subgroups (self):
        
        ''' Divide patient records into subgroup. This function is meant to
            be called at the end after simulation is completed. If padding
            was requested, patient records are trimmed at the beginning 
            and at the end.
            
            output
            ------
            subgroup_records (dict): patient records by qtype: 'fifo' and
                                     'preresume'. For each qtype, records
                                     are grouped by subgroup: 'all',
                                     'interrupting', 'non-interrupting',
                                     'diseased', 'non-diseased', 'positive',
                                     'negative', 'TP', 'FP', 'TN', 'FN'
        '''
        
        ## Define holder for patient records
        subgroup_records = {}
        
        ## Handle each queue type correspondingly
        for qtype, records in self._patient_records.items():
            # Define holders
            interrupting, noninterrupting = [], []
            diseased, nondiseased = [], []
            positive, negative = [], []
            TP, FP, TN, FN = [], [], [], []            
            # Trim patient records as requested
            trimmed_records = records[self.nPatientPadsStart:-self.nPatientPadsEnd]
            # Populate patient records individually to the corresponding subgroups
            for p in trimmed_records:
                if p.is_interrupting:
                    interrupting.append (p)
                elif p.is_diseased:
                    noninterrupting.append (p)
                    diseased.append (p)
                    if p.is_positive:
                        positive.append (p)
                        TP.append (p)
                    else:
                        negative.append (p)
                        FN.append (p)
                else:
                    noninterrupting.append (p)
                    nondiseased.append (p)
                    if p.is_positive:
                        positive.append (p)
                        FP.append (p)
                    else:
                        negative.append (p)
                        TN.append (p)
            # Save to subgroup_records
            subgroup_records[qtype] = {'all':trimmed_records,
                                       'interrupting':interrupting, 'non-interrupting':noninterrupting,
                                       'diseased':diseased, 'non-diseased':nondiseased,
                                       'positive':positive, 'negative':negative,
                                       'TP':TP, 'FN':FN, 'TN':TN, 'FP':FP} 
    
        return subgroup_records

    def _collect_waiting_times (self):
        
        ''' Put all patient waiting times into a dataframe. This function
            is meant to be called at the end after simulation is completed.
            Each row in the dataframe is a patient, who may have different
            waiting time in with and without CADt scenarios.
        '''
    
        ## Basis is without-CADt scenario. Extract the waiting time per
        ## patient in the without-CADt scenario. Put them into dataframe.
        ## Besides the waiting time, the interrupting status, truth status,
        ## and AI call are also extracted per patient.
        adict = {r.caseID:extract_waitTime(r) for r in self._subgroup_records['fifo']['all']}
        df = pandas.DataFrame.from_dict (adict, orient='index',
                                         columns=['fifo', 'is_interrupting', 'is_diseased', 'is_positive'])
        
        ## For other queue types, extract the same information and merge
        ## the columns by patient ID.
        for qtype in self._qtypes:
            if qtype == 'fifo': continue
            adict = {r.caseID:extract_waitTime(r) for r in self._subgroup_records[qtype]['all']}
            adf = pandas.DataFrame.from_dict (adict, orient='index',
                                              columns=[qtype, 'is_interrupting', 'is_diseased', 'is_positive'])
            df = pandas.merge (df, adf.drop (axis=1, columns=['is_interrupting', 'is_diseased', 'is_positive']),
                               right_index=True, left_index=True, how='inner')
    
        ## Store the data frame as a private variable
        self._waiting_time_dataframe = df

    def _collect_n_patients (self):
    
        ''' Put the number of patients per subgroup right before a new patient
            arrives into two dataframes: one counts the number of patients in
            the queue, the other counts that in the system. This function is
            meant to be called at the end after simulation is completed.
            Each row in the dataframe is a patient, who may have different
            number of patients per subgroup in with and without CADt scenarios.
        '''
    
        ## For # patients in system
        dataframes = {}
        for subgroup in self._n_patients['preresume'].keys():
            adict = {}
            for qtype in queuetypes:
                if qtype == 'fifo' and not subgroup in ['all', 'non-interrupting', 'interrupting', 'diseased', 'non-diseased']: continue
                adict[qtype] = self._n_patients[qtype][subgroup]['total']
            dataframes[subgroup] = pandas.DataFrame.from_dict (adict, orient='index').transpose()
        self._n_patient_total_dataframe = dataframes

        ## For # patients in queue
        dataframes = {}
        for subgroup in self._n_patients['preresume'].keys():
            adict = {}
            for qtype in queuetypes:
                if qtype == 'fifo' and not subgroup in ['all', 'non-interrupting', 'interrupting', 'diseased', 'non-diseased']: continue
                adict[qtype] = self._n_patients[qtype][subgroup]['queue']
            dataframes[subgroup] = pandas.DataFrame.from_dict (adict, orient='index').transpose()
        self._n_patient_queue_dataframe = dataframes

    ## +--------------------------------------------------------------
    ## | Public functions to set parameters and start simulation
    ## +--------------------------------------------------------------
    def reset (self):

        ''' Define empty holders. Called right before simulation. 
        '''

        ## Parameters related to logging
        self._logger.debug ('Resetting simulation')
        self._log_string = ''

        ## Define new radiologists
        self._fionas  = [radiologist.radiologist ('Fiona_'+str(i) , self._serviceTimes)
                         for i in range (self._nRadiologists)]
        self._palmers = [radiologist.radiologist ('Palmer_'+str(i), self._serviceTimes)
                         for i in range (self._nRadiologists)] 

        ## Set holders for all patient records and queues
        self._patient_records = {qtype:[] for qtype in self._qtypes}
        self._aqueue = {qtype:queue.PriorityQueue() for qtype in self._qtypes}
        self._future_patients = {qtype:[] for qtype in self._qtypes}
        self._busy_periods_fifo = {'start':[], 'end':[]}
        
        ## Set holders for simulation results
        groups = ['all', 'non-interrupting', 'diseased', 'non-diseased', 'TP', 'TN', 'FP', 'FN'] + list (self._classes)
        self._n_patients = {qtype:{group:{'total':[], 'queue':[]} for group in groups} for qtype in self._qtypes}
        self._waiting_time_dataframe = None  
        self._n_patient_total_dataframe = None
        self._n_patient_queue_dataframe = None    
    
    def set_params (self, params):
        
        ''' Public function to be called to reset simulation parameters.
            
            input
            -----
            params (dict): simulation parameters. It must have the following keys:
                            startTime, timeWindowDays, prevalence, fractionED,
                            meanArrivalTime, meanServiceTimes, nRadiologist, and
                            nPatientsPads
        '''
        
        self.startTime = params['startTime']
        self.timeWindowDays = params['timeWindowDays']
        self.prevalence = params['prevalence']
        self.fractionED = params['fractionED']
        self.arrivalRate = 1/params['meanArrivalTime']
        self.serviceTimes = params['meanServiceTimes']
        self.nRadiologists = params['nRadiologists']
        self.nPatientPadsStart = params['nPatientsPads'][0]
        self.nPatientPadsEnd = params['nPatientsPads'][1]

    def simulate_queue (self, anAI):
        
        ''' Public function to perform a single trial of simulation.
        
            input 
            -----
            anAI (AI): the CADt device used in the with-CADt scenario
        '''

        ## Reset parameters        
        self.reset()
        
        ## Simulation starts! Each timestamp is when a new patient arrives.
        apatient = None
        patient_counter = 0
        newArrivalTime = self._startTime
        
        ## Keep creating new patients before the simulation end time
        while newArrivalTime < self.endTime:
            # Update logging
            self._logger.debug ('+------------------------------------------------------')
            self._logger.debug ('| Current newArrivalTime: {0}'.format (newArrivalTime))
            
            # +-------------------------------
            # | Radiologists do their work
            # +-------------------------------
            for qtype in self._qtypes:
                # Read images between current and right before the patient actually arrives
                drTime = self._radiologist_do_work (qtype, newArrivalTime)
                # Update future patients - this is when the new arrival(s) actally arrives
                self._future_patients[qtype] = self._put_future_patients_in_queue (qtype, drTime,
                                                                                   self._future_patients[qtype])            
                # Check if any interruption due to the new arrival.
                # If a doctor is interrupted, read the latest (high priority) patient
                self._read_newest_urgent_patient (qtype, apatient)
        
            # +-------------------------------
            # | New customer *will* arrive
            # +-------------------------------
            apatient = patient.patient (patient_counter, self._prevalence, self._fractionED,
                                        self.arrivalRate, newArrivalTime)
            #  Decide the AI call and service duration by the first Fiona
            #  Positive remains None if interrupting - doesn't go through AI
            if not apatient.is_interrupting:
                apatient.is_positive = anAI.is_positive (is_diseased=apatient.is_diseased)
            apatient.service_duration = self._fionas[0].determine_service_duration(is_diseased=apatient.is_diseased,
                                                                                   is_interrupting=apatient.is_interrupting)
            apatient.assigned_service_time = deepcopy (apatient.service_duration)
            #  Update logging
            message = '| Case {0} (class {1}) arrives at {2} with duration of {3} min'
            self._logger.debug(message.format (apatient.caseID, apatient.priority_class,
                                               apatient.trigger_time, apatient.service_duration))
            #  This patient may be put in queue later or now
            self._handle_next_patient (apatient)

            # +-------------------------------
            # | Update for next patient
            # +-------------------------------
            newArrivalTime = apatient.trigger_time
            patient_counter += 1
        
        ## Sort records
        self._patient_records = {qtype: sorted (r) for qtype, r in self._patient_records.items()}
        #  Divide each queue's records into subgroup 
        self._subgroup_records = self._divide_subgroups()
        #  Collect waiting times & n_patients
        self._collect_waiting_times()
        self._collect_n_patients()
