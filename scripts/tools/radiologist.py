
##
## By Elim Thompson (11/27/2020)
##
## This script encapsulates a server (radiologist) class. Each radiologist
## has a name and the current patient (an instance that contains all the
## information about the patient case in this radiologist's hand). The
## "current patient" instance will be updated as the radiologist gets
## interrupted or reads a new case. 
###########################################################################

################################
## Import packages
################################ 
import numpy, pandas
from copy import deepcopy

################################
## Define reader class
################################ 
minutes_to_microsec = lambda minute: round (minute * 60 * 10**3 * 10**3)

class radiologist (object):
    
    def __init__ (self, name, serviceTimes):
        
        ''' A radiologist is initialized with a name and the average reading
            (service) times for diseased, non-diseased and interrupting subgroups.
            
            inputs
            ------
            name (str): a unique name for this radiologist
            serviceTimes (dict): {subgroup: average reading time} where the
                                  subgroup keys are diseased, non-diseased,
                                  and interrupting. The values (mean reading
                                  times) are in minutes.
        '''
        
        self._radiologist_name = name
        self._serviceTimes = serviceTimes
        
        ## Keep a record of the patient that is currently read by this radiologist 
        self._current_patient = None
        
    @property
    def _get_serviceRate (self):
        return self._serviceTimes
    
    @property
    def radiologist_name (self):
        return self._radiologist_name

    @property
    def current_patient (self):
        return self._current_patient
    
    def is_busy (self, thisTime):
        
        ''' Is this radiologist busy at the input time? Yes if the current
            patient has a closing time after the input time.
            
            input
            -----
            thisTime (pandas.datetime): the time to check if this rad is busy
            
            output
            ------
            is_busy (bool): True if this radiologist is busy at input time
                            False if this radiologist is in idle at that time
        '''
        
        # If no patient, s/he is not busy
        if self._current_patient is None: return False
        
        # If the current patient's closing time is after the input time,
        # s/he is busy. Otherwise, s/he is in idle.
        return self._current_patient.latest_close_time > thisTime
    
    def determine_service_duration (self, is_diseased=False, is_interrupting=False):
        
        ''' Randomly generate a service reading time for a patient with input
            interrupting status and disease (truth) status.
            
            inputs
            ------
            is_diseased (bool): True if disease status is diseased (signal presence)
                                False if disease status is non-diseased
            is_interrupting (bool): True if interrupting status is interrupting
            
            output
            ------
            reading time (float): random reading time in minutes from an
                                  exponential distribution
        '''
        
        ## average reading time depends on the interrupting and disease status 
        averageReadTime = self._serviceTimes['interrupting'] if is_interrupting else \
                          self._serviceTimes['diseased']  if is_diseased  else \
                          self._serviceTimes['non-diseased']
        ## randomly generate a number from the average reading time
        return numpy.random.exponential (averageReadTime)
        
    def read_new_patient (self, apatient):
        
        ''' Called when reading a brand new patient or when resuming a previously
            interrupted patient. This function updates the patient information,
            including this radiologist name as well as the open and close times
            by this radiologist. The updated patient instance is kept as in this
            radiologist instance for later time comparison. 
            
            input
            -----
            apatient (patient): a patient instance to be read by this radiologist
        '''
        
        # Determine case open time of the patient. 
        #  * If radiologist doesn't have any current patient, openTime is the new
        #    patient's trigger time.
        #  * If the new patient's trigger time is after the current patient's close
        #    time, it is also the new patient's trigger time.
        #  * If the new patient's trigger time is before the current patient's close
        #    time, this new patient has to wait until the current case is closed.
        openTime = apatient.trigger_time if self.current_patient is None else \
                   apatient.trigger_time if self.current_patient.latest_close_time <= apatient.trigger_time else \
                   self.current_patient.latest_close_time
        
        # Update the timing information of this new patient
        #  1. the open time
        apatient.open_times.append (openTime)
        #  2. the close time
        readTimeDuration = minutes_to_microsec (apatient.service_duration)
        apatient.close_times.append (openTime + pandas.offsets.Micro (readTimeDuration))
        #  3. this radiologist's name
        apatient.add_doctor (self._radiologist_name)
        
        # Update current patient to the new patient
        self._current_patient = apatient

    def stop_reading (self, stop_reading_time):

        ''' Called when the reading of current patient is interrupted (by another
            patient of higher priority).
            
            input 
            -----
            stop_reading_time (pandas.datetime): time when the current patient is
                                                 interrupted, likely the arrival
                                                 time of a higher-priority patient
                                                 
            output
            ------
            apatient (patient): the current (lower-priority) patient with updated
                                remaining service duration and close time
        '''

        # Make a copy of the current patient to avoid changing data
        apatient = deepcopy (self._current_patient)
        
        # For preemptive resume, we need to check the remaining service time
        # i.e. service duration of this patient - (diff between open times of
        #      next and this patient)
        new_service_duration = apatient.service_duration - \
                               (stop_reading_time - apatient.latest_open_time).total_seconds()/60
        # For debugging purposes, need to take a look if the new duraction is negative
        if new_service_duration < 0:
            print ('New service time for Case {0} is negative?'.format (apatient.caseID))
            
        # Update the timing information of the lower-priority patient
        #  1. remaining service time 
        apatient.service_duration = new_service_duration
        #  2. case close time
        apatient.add_close_time (stop_reading_time)
        
        # Re-set this radiologist current patient for the interrupted patient
        self._current_patient = None
        
        # Return the lower-priority patient instance
        return apatient