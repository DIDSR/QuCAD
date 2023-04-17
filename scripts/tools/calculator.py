
##
## By Elim Thompson (11/27/2020)
##
## This script contains all theoretical calculations to predict the state
## probability of the number of patients in system (observed by every
## in-coming patient). For more information, please visit
## 
## 1. Osogami et al (2005) Closed form solutions for mapping general
##    distributions to quasi-minimal PH distributions 
##    https://www.cs.cmu.edu/~harchol/Papers/quasi-minimal-PH.pdf
## 2. Harchol-Balter et al Multi-server queueing systems with multiple
##    priority classes
##    https://www.cs.cmu.edu/~harchol/Papers/questa.pdf
## 3. Osogami's thesis (2005) Analysis of Multi-server Systems via
##    Dimensionality Reduction of Markov Chains
##    http://reports-archive.adm.cs.cmu.edu/anon/2005/CMU-CS-05-136.pdf
###########################################################################

################################
## Import packages
################################ 
import numpy, scipy, math
from numpy.linalg import matrix_power
from scipy.linalg import lu, eig, inv

#######################################################
## Common calculator to be used to get mean wait-time
#######################################################
def get_theory_waitTime (aclass, variable, params):

    ''' Function to obtain theoretical waiting time and wait-time difference.
        If input number of radiologist is greater than 2, average waiting time
        for AI negative subgroup cannot be calculated (hence, waiting time
        and wait-time difference for radiologist diagnosis diseased and
        non-diseased subgroups). The theoretical values will not be shown,
        but the simulation results will be available.

        inputs
        ------
        aclass (str): patient subgroup. Either interrupting, non-interrupting,
                      diseased, non-diseased, positive, or negative.
        variable (str): what is being outputted? Either "fifo" for patient
                        waiting time without CADt scenario, "preresume" for
                        patient waiting time with CADt scenario, or "delta"
                        for wait-time difference between the two scenarios.
        params (dict): dictionary capsulating all simulation parameters
    '''

    hist_bins = numpy.linspace (0, 1000, 1001)

    if aclass == 'interrupting' and variable == 'delta':
        return 0
    if aclass == 'interrupting':
        interrupting_predL = get_state_pdf_MMn (hist_bins, params['nRadiologists'],
                                                params['lambdas']['interrupting'],
                                                params['mus']['interrupting'])
        interrupting_predL = numpy.sum ([i*p for i, p in enumerate (interrupting_predL)]).real
        interrupting_predW = interrupting_predL / params['lambdas']['interrupting']
        interrupting_predW = interrupting_predW - 1/params['mus']['interrupting']
        return interrupting_predW

    ## For without-CADt, lower class = non-interrupting
    nonInterrupting_cal = def_cal_MMs (params, lowPriority='non-interrupting',
                                       doDynamic=params['nRadiologists']>3)
    nonInterrupting_predL = nonInterrupting_cal.solve_prob_distributions (len (hist_bins))
    nonInterrupting_predL = numpy.array ([numpy.sum (p) for p in nonInterrupting_predL])
    nonInterrupting_predL = numpy.sum ([i*p for i, p in enumerate (nonInterrupting_predL)]).real
    nonInterrupting_predW = nonInterrupting_predL / params['lambdas']['non-interrupting']
    nonInterrupting_predW = nonInterrupting_predW - 1/params['mus']['non-interrupting']
    if aclass == 'non-interrupting': return nonInterrupting_predW
    if aclass in ['diseased', 'non-diseased'] and variable == 'fifo': return nonInterrupting_predW
    if aclass in ['positive', 'negative'] and variable == 'fifo': return nonInterrupting_predW

    ## For with-CADt, non-interrupting class becomes positive and negative
    positive_cal = def_cal_MMs (params, lowPriority='positive',
                                doDynamic=params['nRadiologists']>3)
    positive_predL = positive_cal.solve_prob_distributions (len (hist_bins))
    positive_predL = numpy.array ([numpy.sum (p) for p in positive_predL])
    positive_predL = numpy.sum ([i*p for i, p in enumerate (positive_predL)]).real
    positive_predW = positive_predL / params['lambdas']['positive']
    positive_predW = positive_predW - 1/params['mus']['positive']
    if aclass == 'positive' and variable == 'preresume': return positive_predW
    if aclass == 'positive' and variable == 'delta': return positive_predW - nonInterrupting_predW

    ## If input number of radiologists is more than 2 *and* fractionED is non-zero, cannot
    ## calculate theoretical values for negative subgroup, hence, diseased and non-diseased subgroups.
    if params['nRadiologists'] > 2 and params['fractionED'] >  0:
        print ('WARN: Cannot calculate theoretical values for AI negative subgroup when more than 2 radiologists with presence of interrupting patient cases.')
        return None

    ## Negative
    negative_cal = get_cal_lowest (params) if params['fractionED'] > 0.0 else \
                   def_cal_MMs (params, lowPriority='negative', highPriority='positive',
                                doDynamic=params['nRadiologists']>3)
    negative_predL = negative_cal.solve_prob_distributions (len (hist_bins))
    negative_predL = numpy.array ([numpy.sum (p).real for p in negative_predL])
    negative_predL = numpy.sum ([i*p for i, p in enumerate (negative_predL)]).real
    negative_predW = negative_predL / params['lambdas']['negative']
    negative_predW = negative_predW - 1/params['mus']['negative']
    if aclass == 'negative' and variable == 'preresume': return negative_predW
    if aclass == 'negative' and variable == 'delta': return negative_predW - nonInterrupting_predW

    ## Diagnosis diseased
    diseased_predW = positive_predW*params['SeThresh'] + negative_predW*(1-params['SeThresh'])
    if aclass == 'diseased' and variable == 'preresume': return diseased_predW
    if aclass == 'diseased' and variable == 'delta': return diseased_predW - nonInterrupting_predW    

    ## Diagnosis non-diseased
    nondiseased_predW = positive_predW*(1-params['SpThresh']) + negative_predW*params['SpThresh']
    if aclass == 'non-diseased' and variable == 'preresume': return nondiseased_predW
    if aclass == 'non-diseased' and variable == 'delta': return nondiseased_predW - nonInterrupting_predW   

    print ('Should not land here!')

########################################
## Classic queueing state probabilities
########################################
def get_state_pdf_MMn (ns, s, aLambda, aMu):
    
    ''' Function to obtain a state probability for a M/M/n system.

        inputs
        ------
        ns (array): number of customers in system
        s (int): number of servers
        aLambda (float): customer arrival rate
        aMu (float): server reading rate

        output
        ------
        pred (array): state probability
    '''

    rho = aLambda / s / aMu
    
    p0_first = numpy.sum ([(s*rho)**(i)/numpy.math.factorial (i) for i in numpy.linspace (0, s-1, s)])
    p0_second = (s*rho)**s / numpy.math.factorial  (s) / (1-rho)
    p0 = 1/(p0_first + p0_second)
    
    pred = []
    for n in ns:
        p = (s*rho)**n * p0 / numpy.math.factorial  (n) if n <= s else \
            s**s*rho**n * p0 / numpy.math.factorial  (s)
        pred.append (p)
    
    return numpy.array (pred)

########################################
## RDR method
########################################
def DR_2_phase_coxian (muG1, muG2, muG3):

    ''' Closed form solutions for mapping general distributions to
        quasi-minimal PH distributions with only 2 Coxian phases.

        See https://www.cs.cmu.edu/~harchol/Papers/quasi-minimal-PH.pdf
        for more details.

        inputs
        ------
        muG1 (float): first moment of busy period distribution
        muG2 (float): second moment of busy period distribution
        muG3 (float): third moment of busy period distribution
    '''

    mG2 = muG2 / muG1**2
    mG3 = muG3 / muG1 / muG2
    denom = 3*mG2 - 2*mG3
    if denom == 0: denom += 0.001
    u = (6-2*mG3) / denom
    v = (12-6*mG2) / (mG2*denom)
    if round (u**2 - 4*v, 10) == 0: 
        lambdaX1 = (u + numpy.sqrt (0)) / (2*muG1)
        lambdaX2 = (u - numpy.sqrt (0)) / (2*muG1)
    else:
        lambdaX1 = (u + numpy.sqrt (u**2 - 4*v)) / (2*muG1)
        lambdaX2 = (u - numpy.sqrt (u**2 - 4*v)) / (2*muG1)
    pX = lambdaX2 * (lambdaX1 * muG1 - 1) / lambdaX1
    if lambdaX1 == 0: pX = 1
    # n = 2; p = 1; lambdaY = 0
    return [2, 1, 0, lambdaX1, lambdaX2, pX]

def DR_simple_solution (muG1, muG2, muG3):

    ''' Closed form solutions for mapping general distributions to
        quasi-minimal PH distributions (simple solutions).

        See https://www.cs.cmu.edu/~harchol/Papers/quasi-minimal-PH.pdf
        for more details.

        inputs
        ------
        muG1 (float): first moment of busy period distribution
        muG2 (float): second moment of busy period distribution
        muG3 (float): third moment of busy period distribution
    '''

    mG2 = muG2 / muG1**2
    mG3 = muG3 / muG1 / muG2
    ## In U0 and M0 (shaded area in thesis Fig 2.12)
    ## 2-phase coxian is sufficient
    if mG2 > 2 and mG3 > 2*mG2 - 1:
        return DR_2_phase_coxian (muG1, muG2, muG3)
    # left side (thesis Fig. 2.12): p = 1
    p = 1 if not mG3 < 2*mG2 - 1 else 1/(2*mG2 - mG3)
    muW1 = muG1 / p
    mW2 = p*mG2
    mW3 = p*mG3
    n = numpy.floor (mW2/(mW2 - 1) + 1)
    
    mX2 = ((n-3)*mW2 - (n-2)) / \
          ((n-2)*mW2 - (n-1))
    muX1 = muW1 / \
           ((n-2)*mX2 - (n-3))
    alpha = (n-2)*(mX2-1)*(n*(n-1)*mX2**2 - n*(2*n-5)*mX2 + (n-1)*(n-3))
    beta = ((n-1)*mX2-(n-2))*((n-2)*mX2-(n-3))**2
    mX3 = (beta*mW3 - alpha) / mX2
    
    u = (6-2*mX3) / (3*mX2 - 2*mX3)
    v = (12-6*mX2) / (mX2*(3*mX2 - 2*mX3))
    lambdaX1 = (u + numpy.sqrt (u**2 - 4*v)) / (2*muX1)
    lambdaX2 = (u - numpy.sqrt (u**2 - 4*v)) / (2*muX1)
    pX = lambdaX2 * (lambdaX1 * muX1 - 1) / lambdaX1
    lambdaY = 1/(muX1*(mX2-1))

    return [n, p, lambdaY, lambdaX1, lambdaX2, pX]

def def_cal_MMs (p, highPriority='interrupting', lowPriority='non-interrupting', doDynamic=False):
    
    ''' Obtain a calculator for the lower priority class with only
        two priority classes. Note that this function also applies
        to the second high priority class when more than two priority
        classes in a preemptive-resume setting because the presence
        or absence of lower priority customers do not affect the
        second high priority class.

        inputs
        ------
        params (dict): settings for all simulations
        highPriority (str): name of the higher priority class above the second high class
        lowPriority (str): name of the lower priority class of interest
        doDynamic (bool): use the dynamic matrix formation (esp used when more than
                          3 radiologists).
    '''

    cal = traditional_calculator()
    rhoH = p['rhos'][highPriority]
    bB = p['mus'][highPriority]*p['nRadiologists']
    muG1 = 1 / bB / (1-rhoH)
    muG2 = 2 / bB**2 / (1-rhoH)**3
    muG3 = 6 * (1+rhoH) / bB**3 / (1-rhoH)**5
    DR = DR_2_phase_coxian (muG1, muG2, muG3)

    if doDynamic:
        cal.form_DR_M_M_s_matrix(p['nRadiologists'],
                     p['lambdas'][lowPriority],
                     p['lambdas'][highPriority],
                     p['mus'][lowPriority],
                     p['mus'][highPriority],
                     (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
    elif p['nRadiologists'] == 1:
        cal.form_DR_M_M_1_matrix(p['lambdas'][lowPriority],
                     p['lambdas'][highPriority],
                     p['mus'][lowPriority],
                     p['mus'][highPriority],
                     (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
    elif p['nRadiologists'] == 2:
        cal.form_DR_M_M_2_matrix(p['lambdas'][lowPriority],
                     p['lambdas'][highPriority],
                     p['mus'][lowPriority],
                     p['mus'][highPriority],
                     (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
    elif p['nRadiologists'] == 3:
        cal.form_DR_M_M_3_matrix(p['lambdas'][lowPriority],
                     p['lambdas'][highPriority],
                     p['mus'][lowPriority],
                     p['mus'][highPriority],
                     (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])

    return cal

def get_cal_lowest (params):

    ''' Obtain a calculator for the loweset priority, AI-negative class in the
        with CADt scenario. Note that there is valid theoretical predictions
        when number of radiologists is 1 or 2. None will be returned if
        nRadiologists > 3.

        inputs
        ------
        params (dict): settings for all simulations
    '''

    if params['nRadiologists'] > 3: return None

    ## Get busy periods and cond. probabilities from middle class
    MidCal = MG1_calculator()
    rhoH = params['rhos']['interrupting']
    bB = params['mus']['interrupting']*params['nRadiologists']
    muG1 = 1 / bB / (1-rhoH)
    muG2 = 2 / bB**2 / (1-rhoH)**3
    muG3 = 6 * (1+rhoH) / bB**3 / (1-rhoH)**5
    DR = DR_2_phase_coxian (muG1, muG2, muG3)
    if params['nRadiologists'] == 1:
        MidCal.form_DR_M_M_1_matrix(params['lambdas']['positive'],
                                    params['lambdas']['interrupting'],
                                    params['mus']['positive'],
                                    params['mus']['interrupting'],
                                    (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
    elif params['nRadiologists'] == 2:
        MidCal.form_DR_M_M_2_matrix(params['lambdas']['positive'],
                                    params['lambdas']['interrupting'],
                                    params['mus']['positive'],
                                    params['mus']['interrupting'],
                                    (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
        t12_M = DR[5]*DR[3]

    MidCal.get_Zs()
    ## These are non-repeating
    Gs = MidCal._G_nonrep
    Zs = MidCal.Zs
    
    ## For 1 radiologist: 2 busy periods
    if params['nRadiologists'] == 1:
        ## Use [0,0] and [1,0] elements in Zs / Gs for 1 rad
        ## For [0,0]: (0,1,l) -> (0,0,l)
        DR = DR_2_phase_coxian (Zs['Z1'][0][0], Zs['Z2'][0][0], Zs['Z3'][0][0])
        t1_M, t12_M, t2_M = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_M = Gs[0][0]
        
        ## For [1,0]: (1,0,l) -> (0,0,l)
        DR = DR_2_phase_coxian (Zs['Z1'][1][0], Zs['Z2'][1][0], Zs['Z3'][1][0])
        t1_H, t12_H, t2_H = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_H = Gs[1][0]
    
        ## Get calculator for lower class
        cal = traditional_calculator()
        cal.form_DR_M_M_1_3classes_matrix(params['lambdas']['negative'],
                                          params['lambdas']['positive'],
                                          params['lambdas']['interrupting'],
                                          params['mus']['negative'],
                                          p_M, t1_M, t12_M, t2_M,
                                          p_H, t1_H, t12_H, t2_H)
        return cal
    
    ## For 2 radiologists; 6 busy periods
    if params['nRadiologists'] == 2:
        ## 1. [0,0]: (0,2,l) -> (0,1,l)
        # 8.215, 0.512, 1.05 ##
        DR = DR_simple_solution (Zs['Z1'][0][0], Zs['Z2'][0][0], Zs['Z3'][0][0])
        t1_2M_1M, t12_2M_1M, t2_2M_1M = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_2M_1M = Gs[0][0]
        
        ## 2. [0,1]: (0,2,l) -> (1,0,l) !!!
        DR = DR_simple_solution (Zs['Z1'][0][1], Zs['Z2'][0][1], Zs['Z3'][0][1])
        t0_2M_1H, t01_2M_1H = (1-DR[1])*DR[2], DR[1]*DR[2]
        t1_2M_1H, t12_2M_1H, t2_2M_1H = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_2M_1H = Gs[0][1]
        
        ## 3. [1,0]: (1,1,l) -> (0,1,l)
        DR = DR_simple_solution (Zs['Z1'][1][0], Zs['Z2'][1][0], Zs['Z3'][1][0])
        t1_1M1H_1M, t12_1M1H_1M, t2_1M1H_1M = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_1M1H_1M = Gs[1][0]
    
        ## 4. [1,1]: (1,1,l) -> (1,0,l)
        DR = DR_simple_solution (Zs['Z1'][1][1], Zs['Z2'][1][1], Zs['Z3'][1][1])
        t1_1M1H_1H, t12_1M1H_1H, t2_1M1H_1H = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_1M1H_1H = Gs[1][1]
    
        ## 5. [2,0]: (2+,0,l) -> (0,1,l) !!!
        DR = DR_simple_solution (Zs['Z1'][2][0], Zs['Z2'][2][0], Zs['Z3'][2][0])
        t0_2H_1M, t01_2H_1M = (1-DR[1])*DR[2], DR[1]*DR[2]
        t1_2H_1M, t12_2H_1M, t2_2H_1M = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_2H_1M = Gs[2][0]
        
        ## 6. [2,1]: (2+,0,l) -> (1,0,l)
        DR = DR_simple_solution (Zs['Z1'][2][1], Zs['Z2'][2][1], Zs['Z3'][2][1])
        t1_2H_1H, t12_2H_1H, t2_2H_1H = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_2H_1H = Gs[2][1]

        ## Get calculator for lower class
        cal = traditional_calculator()
        cal.form_DR_M_M_2_3classes_matrix(params['lambdas']['negative'],
                                          params['lambdas']['positive'],
                                          params['lambdas']['interrupting'],
                                          params['mus']['negative'],
                                          params['mus']['positive'],
                                          params['mus']['interrupting'],
                                          p_2M_1M  , t1_2M_1M  , t12_2M_1M  , t2_2M_1M  ,
                                          p_2M_1H  , t1_2M_1H  , t12_2M_1H  , t2_2M_1H  , t0_2M_1H, t01_2M_1H, 
                                          p_1M1H_1M, t1_1M1H_1M, t12_1M1H_1M, t2_1M1H_1M,
                                          p_1M1H_1H, t1_1M1H_1H, t12_1M1H_1H, t2_1M1H_1H,
                                          p_2H_1M  , t1_2H_1M  , t12_2H_1M  , t2_2H_1M  , t0_2H_1M, t01_2H_1M,
                                          p_2H_1H  , t1_2H_1H  , t12_2H_1H  , t2_2H_1H  )
        
        return cal

############################################
## Calculators 
############################################
class stateProb_calculator (object):

    def __init__ (self):
        
        self._A0, self._A1, self._A2 = None, None, None
        self._B00, self._B01, self._B10 = None, None, None

        self._pis = None

    @property
    def pis (self): return self._pis

    @property
    def A0 (self): return self._A0
    @A0.setter
    def A0 (self, A0):
        if not self._matrix_is_square (A0):
            print ('Please provide a square matrix for A0.')
            return
        self._A0 = A0 
        
    @property
    def A1 (self): return self._A1
    @A1.setter
    def A1 (self, A1):
        if not self._matrix_is_square (A1):
            print ('Please provide a square matrix for A1.')
            return
        self._A1 = A1 
        
    @property
    def A2 (self): return self._A2
    @A2.setter
    def A2 (self, A2):
        if not self._matrix_is_square (A2):
            print ('Please provide a square matrix for A2.')
            return
        self._A2 = A2 
        
    @property
    def B00 (self): return self._B00
    @B00.setter
    def B00 (self, B00):
        good_B00 = self._matrix_is_square (B00) or isinstance (B00, float) \
                   or isinstance (B00, int)
        if not good_B00:
            print ('Please provide a value or square matrix for B00.')
            return
        self._B00 = B00 

    @property
    def B10 (self): return self._B10
    @B10.setter
    def B10 (self, B10): self._B10 = B10 

    @property
    def B01 (self): return self._B01
    @B01.setter
    def B01 (self, B01): self._B01 = B01
    
    #######################################################
    ## Check system
    #######################################################
    def _matrix_is_square (self, matrix):
        
        shape = matrix.shape
        if not len (shape) == 2: return False
        return shape[0] == shape[1]
        
    def _have_matching_shapes (self):
        
        shape = self._A1.shape
        # A0 and A2 must have the same shape
        if not self._A0.shape == shape:
            print ('A0 and A1 have different shape.')
            return False
        if not self._A2.shape == shape:
            print ('A2 and A1 have different shape.')
            return False
        
        # B01 must have the same # columns as A1
        if not self._B01.shape[1] == shape[1]:
            print ('B01 and A1 have different # columns.')
            return False
        
        # B10 must have the same # rows as A1
        if not self._B10.shape[0] == shape[0]:
            print ('B10 and A1 have different # rows.')
            return False

        # B00 must have the same # rows as B01
        # and same # columns as B10
        if not self._B00.shape[0] == self._B01.shape[0]:
            print ('B00 and B01 have different # rows.')
            return False
        if not self._B00.shape[1] == self._B10.shape[1]:
            print ('B00 and B10 have different # columns.')
            return False

        return True

    def _is_valid_system (self):

        for element in ['A0', 'A1', 'A2', 'B10', 'B01', 'B00']:
            matrix = eval ('self._' + element)
            if matrix is None:
                print ('Please provide a valid {0} matrix.'.format (element))
                return False

        # Make sure matrix shapes are matched
        if not self._have_matching_shapes (): return False
        
        return True

    def _negated_sum (self, matrix):
        
        for i, row in enumerate (matrix):
            matrix[i][i] = -numpy.sum (numpy.append (row[:i], row[i+1:]))
 
        return matrix

    def _combine_matrix (self, a00, a01, a10, a11):
        return numpy.vstack([numpy.hstack([a00, a01]), numpy.hstack([a10, a11])])
    
    def _solve_LH_vector (self, p):
        # Find left-handed eigenvalues / vectors
        vals, vecs = eig (p, left=True, right=False)
        # Get the value closest to 1 - should be the largest
        closest_to_1_idx = vals.argsort()[::-1][0]
        val = numpy.round (vals[closest_to_1_idx], 4)
        if val < 0 or val > 1:
            print ('Largest eigenvalue, {0}, found is invalid.'.format (val))
            return None
        # Get the corresponding eigenvector
        vec = vecs[:,closest_to_1_idx]
        # Normalize it
        return vec / numpy.sum (vec)
    
class traditional_calculator (stateProb_calculator):

    def __init__ (self):    
        super(stateProb_calculator, self).__init__ ()
        self._R = None
        
        self._is_MH2C2 = False
        self._is_MC2 = False
        self._nRad = 1

    @property
    def R (self): return self._R

    #######################################################
    ## Form matrix
    #######################################################
    def form_DR_M_M_1_3classes_matrix (self, lambdaL, lambdaM, lambdaH, muL,
                                       p_M, t1_M, t12_M, t2_M,
                                       p_H, t1_H, t12_H, t2_H):
        
        self._B00 = numpy.array ([[-lambdaH*p_H-lambdaM*p_M-lambdaL,         lambdaM*p_M,             0,         lambdaH*p_H,             0],
                                  [                            t1_M, -lambdaL-t1_M-t12_M,         t12_M,                   0,             0],
                                  [                            t2_M,                   0, -lambdaL-t2_M,                   0,             0],
                                  [                            t1_H,                   0,             0, -lambdaL-t1_H-t12_H,         t12_H],
                                  [                            t2_H,                   0,             0,                   0, -lambdaL-t2_H]])
        self._B01 = numpy.array ([[lambdaL,       0,       0,       0,       0],
                                  [      0, lambdaL,       0,       0,       0],
                                  [      0,       0, lambdaL,       0,       0],
                                  [      0,       0,       0, lambdaL,       0],
                                  [      0,       0,       0,       0, lambdaL]])
        self._B10 = numpy.array ([[muL, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0]])
        
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = self._B10
        self._A1 = numpy.array ([[-lambdaH*p_H-lambdaM*p_M-lambdaL-muL,         lambdaM*p_M,             0,         lambdaH*p_H,             0],
                                 [                                t1_M, -lambdaL-t1_M-t12_M,         t12_M,                   0,             0],
                                 [                                t2_M,                   0, -lambdaL-t2_M,                   0,             0],
                                 [                                t1_H,                   0,             0, -lambdaL-t1_H-t12_H,         t12_H],
                                 [                                t2_H,                   0,             0,                   0, -lambdaL-t2_H]])
        self._A2 = self._B01

    def form_DR_M_M_2_3classes_matrix (self, lambdaL, lambdaM, lambdaH, muL, muM, muH,
                                       p_2M_1M  , t1_2M_1M  , t12_2M_1M  , t2_2M_1M  ,
                                       p_2M_1H  , t1_2M_1H  , t12_2M_1H  , t2_2M_1H  , t0_2M_1H, t01_2M_1H, 
                                       p_1M1H_1M, t1_1M1H_1M, t12_1M1H_1M, t2_1M1H_1M,
                                       p_1M1H_1H, t1_1M1H_1H, t12_1M1H_1H, t2_1M1H_1H,
                                       p_2H_1M  , t1_2H_1M  , t12_2H_1M  , t2_2H_1M  , t0_2H_1M, t01_2H_1M,
                                       p_2H_1H  , t1_2H_1H  , t12_2H_1H  , t2_2H_1H ):
        
        self._B00 = numpy.array ([[-lambdaH-lambdaM-lambdaL,                                                                         lambdaM,                                                                         lambdaH,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                     muM,-lambdaM*p_2M_1M-lambdaM*p_2M_1H-lambdaH*p_1M1H_1M-lambdaH*p_1M1H_1H-lambdaL-muM,                                                                               0,            lambdaM*p_2M_1M,                0,            lambdaM*p_2M_1H,                          0,                0,              lambdaH*p_1M1H_1M,                  0,              lambdaH*p_1M1H_1H,                  0,                          0,                          0,                0,                          0,                0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                     muH,                                                                               0,-lambdaM*p_1M1H_1M-lambdaM*p_1M1H_1H-lambdaH*p_2H_1M-lambdaH*p_2H_1H-lambdaL-muH,                          0,                0,                          0,                          0,                0,              lambdaM*p_1M1H_1M,                  0,              lambdaM*p_1M1H_1H,                  0,            lambdaH*p_2H_1M,                          0,                0,            lambdaH*p_2H_1H,                0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                        t1_2M_1M,                                                                               0,-lambdaL-t1_2M_1M-t12_2M_1M,        t12_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                        t2_2M_1M,                                                                               0,                          0,-lambdaL-t2_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                               0,                                                                        t0_2M_1H,                          0,                0,-lambdaL-t0_2M_1H-t01_2M_1H,                  t01_2M_1H,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],                                  
                                  [                       0,                                                                               0,                                                                        t1_2M_1H,                          0,                0,                          0,-lambdaL-t1_2M_1H-t12_2M_1H,        t12_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                               0,                                                                        t2_2M_1H,                          0,                0,                          0,                          0,-lambdaL-t2_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                      t1_1M1H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,-lambdaL-t1_1M1H_1M-t12_1M1H_1M,        t12_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                      t2_1M1H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,                              0,-lambdaL-t2_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                               0,                                                                      t1_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,-lambdaL-t1_1M1H_1H-t12_1M1H_1H,        t12_1M1H_1H,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                               0,                                                                      t2_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,-lambdaL-t2_1M1H_1H,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0],
                                  [                       0,                                                                        t0_2H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,-lambdaL-t0_2H_1M-t01_2H_1M,                  t01_2H_1M,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0],                                  
                                  [                       0,                                                                        t1_2H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,-lambdaL-t1_2H_1M-t12_2H_1M,        t12_2H_1M,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0],
                                  [                       0,                                                                        t2_2H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,-lambdaL-t2_2H_1M,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0],
                                  [                       0,                                                                               0,                                                                        t1_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,-lambdaL-t1_2H_1H-t12_2H_1H,        t12_2H_1H,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0],
                                  [                       0,                                                                               0,                                                                        t2_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,-lambdaL-t2_2H_1H,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL],
                                  [muL,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-lambdaH-lambdaM-lambdaL-muL,                                                                             lambdaM,                                                                             lambdaH,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0, muL,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                         muM,-lambdaM*p_2M_1M-lambdaM*p_2M_1H-lambdaH*p_1M1H_1M-lambdaH*p_1M1H_1H-lambdaL-muL-muM,                                                                                   0,            lambdaM*p_2M_1M,                0,            lambdaM*p_2M_1H,                          0,                0,              lambdaH*p_1M1H_1M,                  0,              lambdaH*p_1M1H_1H,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0, muL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                         muH,                                                                                   0,-lambdaM*p_1M1H_1M-lambdaM*p_1M1H_1H-lambdaH*p_2H_1M-lambdaH*p_2H_1H-lambdaL-muL-muH,                          0,                0,                          0,                          0,                0,              lambdaM*p_1M1H_1M,                  0,              lambdaM*p_1M1H_1H,                  0,            lambdaH*p_2H_1M,                          0,                0,            lambdaH*p_2H_1H,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t1_2M_1M,                                                                                   0,-lambdaL-t1_2M_1M-t12_2M_1M,        t12_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t2_2M_1M,                                                                                   0,                          0,-lambdaL-t2_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t0_2M_1H,                          0,                0,-lambdaL-t0_2M_1H-t01_2M_1H,                  t01_2M_1H,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],                                  
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t1_2M_1H,                          0,                0,                          0,-lambdaL-t1_2M_1H-t12_2M_1H,        t12_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t2_2M_1H,                          0,                0,                          0,                          0,-lambdaL-t2_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                          t1_1M1H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,-lambdaL-t1_1M1H_1M-t12_1M1H_1M,        t12_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                          t2_1M1H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,-lambdaL-t2_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                          t1_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,-lambdaL-t1_1M1H_1H-t12_1M1H_1H,        t12_1M1H_1H,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                          t2_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,-lambdaL-t2_1M1H_1H,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t0_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,-lambdaL-t0_2H_1M-t01_2H_1M,                  t01_2H_1M,                0,                          0,                0],                                  
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t1_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,-lambdaL-t1_2H_1M-t12_2H_1M,        t12_2H_1M,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t2_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,-lambdaL-t2_2H_1M,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t1_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,-lambdaL-t1_2H_1H-t12_2H_1H,        t12_2H_1H],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t2_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,-lambdaL-t2_2H_1H]])
        self._B01 = numpy.array ([[      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL]])
        self._B10 = numpy.array ([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*muL,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, muL,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0, muL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])        
        
        
        ## Repeating starts at level 2
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = numpy.array ([[2*muL,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0, muL,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0, muL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self._A1 = numpy.array ([[-lambdaH-lambdaM-lambdaL-2*muL,                                                                             lambdaM,                                                                             lambdaH,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                           muM,-lambdaM*p_2M_1M-lambdaM*p_2M_1H-lambdaH*p_1M1H_1M-lambdaH*p_1M1H_1H-lambdaL-muL-muM,                                                                                   0,            lambdaM*p_2M_1M,                0,            lambdaM*p_2M_1H,                          0,                0,              lambdaH*p_1M1H_1M,                  0,              lambdaH*p_1M1H_1H,                  0,                          0,                          0,                0,                          0,                0],
                                 [                           muH,                                                                                   0,-lambdaM*p_1M1H_1M-lambdaM*p_1M1H_1H-lambdaH*p_2H_1M-lambdaH*p_2H_1H-lambdaL-muL-muH,                          0,                0,                          0,                          0,                0,              lambdaM*p_1M1H_1M,                  0,              lambdaM*p_1M1H_1H,                  0,            lambdaH*p_2H_1M,                          0,                0,            lambdaH*p_2H_1H,                0],
                                 [                             0,                                                                            t1_2M_1M,                                                                                   0,-lambdaL-t1_2M_1M-t12_2M_1M,        t12_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                            t2_2M_1M,                                                                                   0,                          0,-lambdaL-t2_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                            t0_2M_1H,                          0,                0,-lambdaL-t0_2M_1H-t01_2M_1H,                  t01_2M_1H,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                            t1_2M_1H,                          0,                0,                          0,-lambdaL-t1_2M_1H-t12_2M_1H,        t12_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                            t2_2M_1H,                          0,                0,                          0,                          0,-lambdaL-t2_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                          t1_1M1H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,-lambdaL-t1_1M1H_1M-t12_1M1H_1M,        t12_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                          t2_1M1H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,-lambdaL-t2_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                          t1_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,-lambdaL-t1_1M1H_1H-t12_1M1H_1H,        t12_1M1H_1H,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                          t2_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,-lambdaL-t2_1M1H_1H,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                            t0_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,-lambdaL-t0_2H_1M-t01_2H_1M,                  t01_2H_1M,                0,                          0,                0],                                 
                                 [                             0,                                                                            t1_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,-lambdaL-t1_2H_1M-t12_2H_1M,        t12_2H_1M,                          0,                0],
                                 [                             0,                                                                            t2_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,-lambdaL-t2_2H_1M,                          0,                0],
                                 [                             0,                                                                                   0,                                                                            t1_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,-lambdaL-t1_2H_1H-t12_2H_1H,        t12_2H_1H],
                                 [                             0,                                                                                   0,                                                                            t2_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,-lambdaL-t2_2H_1H]])
        self._A2 = numpy.array ([[lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL]])

    def form_DR_M_M_1_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        
        self._B00 = numpy.array ([[-lambdaH-lambdaL,          lambdaH,          0],
                                  [              t1, -(lambdaL+t1+t12),        t12],
                                  [              t2,                 0, -t2-lambdaL]])
        self._B01 = numpy.array ([[lambdaL,       0,       0],
                                  [      0, lambdaL,       0],
                                  [      0,       0, lambdaL]])
        self._B10 = numpy.array ([[muL, 0, 0],
                                  [  0, 0, 0],
                                  [  0, 0, 0]])
        
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = self._B10
        self._A1 = numpy.array ([[-lambdaH-lambdaL-muL,          lambdaH,          0],
                                 [                  t1, -(lambdaL+t1+t12),        t12],
                                 [                  t2,                 0, -t2-lambdaL]])
        self._A2 = self._B01

    def form_DR_M_M_2_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        
        self._B00 = numpy.array ([[    -lambdaH-lambdaL,              lambdaH,               0,           0,              lambdaL,                        0,               0,               0],
                                  [                 muH, -lambdaH-lambdaL-muH,         lambdaH,           0,                    0,                  lambdaL,               0,               0],
                                  [                   0,                   t1, -lambdaL-t1-t12,         t12,                    0,                        0,         lambdaL,               0],
                                  [                   0,                   t2,               0, -t2-lambdaL,                    0,                        0,               0,         lambdaL],
                                  
                                  [                 muL,                    0,               0,           0, -muL-lambdaH-lambdaL,                  lambdaH,               0,               0],
                                  [                   0,                  muL,               0,           0,                  muH, -muL-lambdaH-muH-lambdaL,         lambdaH,               0],
                                  [                   0,                    0,               0,           0,                    0,                       t1, -t1-t12-lambdaL,             t12],
                                  [                   0,                    0,               0,           0,                    0,                       t2,               0,     -t2-lambdaL]])
        self._B01 = numpy.array ([[      0,       0,       0,       0],
                                  [      0,       0,       0,       0],
                                  [      0,       0,       0,       0],
                                  [      0,       0,       0,       0],
                                  [lambdaL,       0,       0,       0],
                                  [      0, lambdaL,       0,       0],
                                  [      0,       0, lambdaL,       0],
                                  [      0,       0,       0, lambdaL]])
        self._B10 = numpy.array ([[0, 0, 0, 0, 2*muL,   0, 0, 0],
                                  [0, 0, 0, 0,     0, muL, 0, 0],
                                  [0, 0, 0, 0,     0,   0, 0, 0],
                                  [0, 0, 0, 0,     0,   0, 0, 0]])
        
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = numpy.array ([[2*muL,   0, 0, 0],
                                 [    0, muL, 0, 0],
                                 [    0,   0, 0, 0],
                                 [    0,   0, 0, 0]])
        self._A1 = numpy.array ([[-lambdaH-lambdaL-2*muL,                  lambdaH,               0,           0],
                                 [                   muH, -lambdaL-muL-muH-lambdaH,         lambdaH,           0],
                                 [                     0,                       t1, -lambdaL-t1-t12,         t12],
                                 [                     0,                       t2,               0, -t2-lambdaL]])
        self._A2 = numpy.array ([[lambdaL,       0,       0,       0],
                                 [      0, lambdaL,       0,       0],
                                 [      0,       0, lambdaL,       0],
                                 [      0,       0,       0, lambdaL]])

    def form_DR_M_M_3_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        
        self._B00 = numpy.array ([[-lambdaH-lambdaL,             lambdaH,                     0,              0,          0,             lambdaL,                       0,                         0,              0,          0,                     0,                         0,                         0,              0,          0],
                                  [             muH,-lambdaH-lambdaL-muH,               lambdaH,              0,          0,                   0,                 lambdaL,                         0,              0,          0,                     0,                         0,                         0,              0,          0],
                                  [               0,               2*muH,-lambdaH-lambdaL-2*muH,        lambdaH,          0,                   0,                       0,                   lambdaL,              0,          0,                     0,                         0,                         0,              0,          0],
                                  [               0,                   0,                    t1,-lambdaL-t1-t12,        t12,                   0,                       0,                         0,        lambdaL,          0,                     0,                         0,                         0,              0,          0],
                                  [               0,                   0,                    t2,              0,-t2-lambdaL,                   0,                       0,                         0,              0,    lambdaL,                     0,                         0,                         0,              0,          0],
                                  
                                  [             muL,                   0,                     0,              0,          0,-muL-lambdaH-lambdaL,                 lambdaH,                         0,              0,          0,               lambdaL,                         0,                        0,              0,          0],
                                  [               0,                 muL,                     0,              0,          0,                 muH,-muL-lambdaH-muH-lambdaL,                   lambdaH,              0,          0,                     0,                   lambdaL,                         0,              0,          0],
                                  [               0,                   0,                   muL,              0,          0,                   0,                   2*muH,-muL-lambdaH-2*muH-lambdaL,        lambdaH,          0,                     0,                         0,                   lambdaL,              0,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                        t1,-t1-t12-lambdaL,        t12,                     0,                         0,                         0,        lambdaL,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                        t2,              0,-t2-lambdaL,                     0,                         0,                         0,              0,    lambdaL],
                                  
                                  
                                  [               0,                   0,                     0,              0,          0,               2*muL,                       0,                         0,              0,          0,-2*muL-lambdaH-lambdaL,                   lambdaH,                         0,              0,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                   2*muL,                         0,              0,          0,                   muH,-2*muL-lambdaH-muH-lambdaL,                   lambdaH,              0,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                       muL,              0,          0,                     0,                     2*muH,-muL-lambdaH-2*muH-lambdaL,        lambdaH,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                         0,              0,          0,                     0,                         0,                        t1,-t1-t12-lambdaL,        t12],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                         0,              0,          0,                     0,                         0,                        t2,              0,-t2-lambdaL]])
        self._B01 = numpy.array ([[      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],                                  
                                  [lambdaL,       0,       0,       0,       0],
                                  [      0, lambdaL,       0,       0,       0],
                                  [      0,       0, lambdaL,       0,       0],
                                  [      0,       0,       0, lambdaL,       0],
                                  [      0,       0,       0,       0, lambdaL]])
        self._B10 = numpy.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*muL,     0,   0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 2*muL,   0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,     0, muL, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,     0,   0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,     0,   0, 0, 0]])
        
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = numpy.array ([[3*muL,     0,   0, 0, 0],
                                 [    0, 2*muL,   0, 0, 0],
                                 [    0,     0, muL, 0, 0],
                                 [    0,     0,   0, 0, 0],
                                 [    0,     0,   0, 0, 0]])
        self._A1 = numpy.array ([[-lambdaH-lambdaL-3*muL,                    lambdaH,                         0,               0,           0],
                                 [                   muH, -lambdaL-2*muL-muH-lambdaH,                   lambdaH,               0,           0],
                                 [                     0,                      2*muH,-lambdaL-muL-2*muH-lambdaH,         lambdaH,           0],
                                 [                     0,                          0,                        t1, -lambdaL-t1-t12,         t12],
                                 [                     0,                          0,                        t2,               0, -t2-lambdaL]])
        self._A2 = numpy.array ([[lambdaL,       0,       0,       0,       0],
                                 [      0, lambdaL,       0,       0,       0],
                                 [      0,       0, lambdaL,       0,       0],
                                 [      0,       0,       0, lambdaL,       0],
                                 [      0,       0,       0,       0, lambdaL]])
        
    def form_DR_M_M_s_matrix (self, s, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        
        ## In paper, A0 = B; A1 = L; A2 = F
        shape = (s+2, s+2)
        
        #  A0 i.e. B 
        A0 = numpy.identity (shape[0])
        for i in range (shape[0]):
            A0[i][i] = 0 if i in [shape[0]-1, shape[0]-2] else (s-i)*muL
        self._A0 = A0
        #  A2 i.e. F
        self._A2 = numpy.identity (shape[0])*lambdaL
        #  A1 i.e. L
        T = numpy.array ([[-t1-t12, t12],
                          [      0, -t2]])
        t = -numpy.sum (T, axis=1)
        QB = []
        for i in range (s):
            array = numpy.zeros (s+2)
            array[i+1] = lambdaH
            array[i] = -lambdaH
            if i > 0:
                array[i-1] = i*muH
                array[i] -= i*muH
            QB.append (array)
        last2rows = numpy.hstack ([numpy.zeros ((2, shape[0]-3)),
                                   numpy.vstack ([t, T.T]).T])
        QB = numpy.vstack ([QB, last2rows])
        self._A1 = QB - self._A0 - self._A2
        
        ## At Boundary
        #  B01
        self._B01 = numpy.vstack ([numpy.zeros (shape) for i in range (s-1)] + [self._A2])
        #  B10
        self._B10 = numpy.hstack ([numpy.zeros (shape) for i in range (s-1)] + [self._A0])
        #  B00
        B00 = []
        for i in range (s):
            L = QB - self._A2 if i == 0 else self._A1
            # if s = 1, L0 is B00. No need to stack matrices
            if s==1:
                B00 = L
                break 
            # First row block: L, F, 0, 0, 0, ...
            if i == 0:
                rowblock = numpy.hstack ([L, self._A2] + [numpy.zeros (shape) for i in range (s-2)])
                B00.append (rowblock)
                continue
            # Last row block: 0, 0, 0, ..., B, L
            if i == s-1:
                B = numpy.identity (shape[0]) 
                for r, row in enumerate (B):
                    B[r][r] = 0 if r in [shape[0]-1, shape[0]-2] else min (s-r, i)*muL
                rowblock = numpy.hstack ([numpy.zeros (shape) for i in range (s-2)] + [B, L])
                B00.append (rowblock)
                continue
            
            # Anything in between first and last: 0, 0, ..., 0, B, L, F, 0, ..., 0, 0
            B = numpy.identity (shape[0]) 
            for r, row in enumerate (B):
                B[r][r] = 0 if r in [shape[0]-1, shape[0]-2] else min (s-r, i)*muL
            rowblock = numpy.hstack ([numpy.zeros (shape) for i in range (i-1)] + [B, L, self._A2] +
                                     [numpy.zeros (shape) for i in range (s-3-(i-1))])
            B00.append (rowblock)
        # negated sum
        B00 =  numpy.vstack (B00)
        firstblock = self._negated_sum(numpy.hstack ([B00, self._B01]))
        for r, row in enumerate (B00):
            B00[r][r] = firstblock[r][r]
        self._B00= B00
 
    def _get_R (self):
        #  R is found iteratively
        #     R_(k+1) = -V - R^2_(k) W
        #  where V = A2 inv(A1)
        #        W = A0 inv(A1)
        #  and initial R_0 = 0
        # ** R is non-decreasing i.e. its values will only 
        #    grow until they converge.
        V = numpy.dot (self._A2, inv(self._A1))
        W = numpy.dot (self._A0, inv(self._A1))
        R = numpy.zeros_like (V)
        for niter in range (1000):
            R = -V - numpy.dot (matrix_power (R, 2), W)
        
        return R

    def _solve_boundary_solutions (self, R):
        bound = self._combine_matrix (self._B00, self._B01, self._B10,
                                      self._A1 + numpy.dot (R, self._A0))
        pi_bound = self._solve_LH_vector (bound)
        return pi_bound[:self._B01.shape[0]], pi_bound[self._B01.shape[0]:]

    def _find_norm (self, pi0, pi1, R):
        alpha = sum (pi0.T)
        try:
            alpha += numpy.dot (pi1, sum (inv (numpy.identity (R.shape[0])-R).T))
        except numpy.linalg.LinAlgError:
            for i in range (R.shape[0]):
                if R[i][i] == 1:  R[i][i] = 0.99999
            alpha += numpy.dot (pi1, sum (inv (numpy.identity (R.shape[0])-R).T))
        return alpha
    
    def solve_prob_distributions (self, n=1000):
    
        self._is_valid_system()

        R = self._get_R()
        pi0, pi1 = self._solve_boundary_solutions (R)

        # Normalize boundary states
        alpha = self._find_norm (pi0, pi1, R)
        pi0 /= alpha
        pi1 /= alpha

        pis = [pi0, pi1]
        ## If boundary condition has more states than repeating matrix
        if len (pi0) > self._A1.shape[0]:
            nstats = len (pi0)/self._A1.shape[0]
            pis = numpy.split (pi0, nstats) + [pi1]
            
        ## If M/H2C2/s, only 1 (0) for pi0. 
        if self._is_MH2C2:
            indices = []
            nstats = self._nRad
            for i in range (nstats):
                nCustomer = i
                nsubstats = numpy.math.factorial  (4+nCustomer-1) / (numpy.math.factorial  (nCustomer)*numpy.math.factorial  (4-1))
                index = int (nsubstats)
                if i > 0: index += 1
                indices.append (index)
            pis = numpy.split (pi0, indices[:-1]) + [pi1]

        ## If M/C2/s, only 1 (0) for pi0. 
        if self._is_MC2:
            indices = []
            nstats = self._nRad
            for i in range (nstats):
                nCustomer = i
                nsubstats = numpy.math.factorial  (2+nCustomer-1) / (numpy.math.factorial  (nCustomer)*numpy.math.factorial  (2-1))
                index = int (nsubstats)
                if i > 0: index += 1
                indices.append (index)
            pis = numpy.split (pi0, indices[:-1]) + [pi1]

        for i in range (n-len (pis)):
            pii = numpy.dot (pis[-1], R)
            pis.append (pii)
        
        self._R = R
        self._pis = pis
        
        return pis

class MG1_calculator (stateProb_calculator):

    def __init__ (self):    
        super(stateProb_calculator, self).__init__ ()
        
        self._nRad = None
        self._l_hat = None
        self._gamma = None
        self._Zs = None
        
        self._A3, self._B02 = None, None
        
        self._G = None
        self._G_nonrep = None
        self._G_nonrep2 = None
        self._Astar1, self._Astar2, self._Astar3 = None, None, None
        self._Bstar01, self._Bstar02 = None, None

    @property
    def A3 (self): return self._A3
    @A3.setter
    def A3 (self, A3):
        if not self._matrix_is_square (A3):
            print ('Please provide a square matrix for A3.')
            return
        self._A3 = A3 

    @property
    def B02 (self): return self._B02
    @B02.setter
    def B02 (self, B02): self._B02 = B02

    @property
    def G (self): return self._G

    @property
    def Zs (self): return self._Zs

    @property
    def Astar1 (self): return self._Astar1

    @property
    def Astar2 (self): return self._Astar2

    @property
    def Astar3 (self): return self._Astar3    

    @property
    def Bstar01 (self): return self._Bstar01

    @property
    def Bstar02 (self): return self._Bstar02    

    def form_DR_M_M_1_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        ## This function is different from the same function in traditional_calculator.
        ## This MG1_calculator is used when having m priority classes and/or non-
        ## exponential service process in priority classes.
        
        ## In this case, we keep track of level l-1
        ##  l - 1 = #_middle + Min (#_server, #_high)
        ## The input transition rates are used to formulate the transition probability,
        ## which are used to calculate G. The transition rates are also used to get
        ## the first 3 moments of the forward, local, and backward matrices in order
        ## to obtain the moments of G. 
        
        self._nRad = 1
        self._l_hat = 2
        
        B00 = numpy.array ([[-lambdaH-lambdaL]])
        self._B00 = numpy.array ([[0]])
        
        B01 = numpy.array ([[lambdaL, lambdaH, 0]])
        gamma = -B00[0][0]
        self._B01 = B01 / gamma
        
        B10 = numpy.array ([[muL],
                            [ t1],
                            [ t2]])
        gamma = numpy.array ([[muL+lambdaH+lambdaL], [t1+t12+lambdaL], [t2+lambdaL]])
        self._B10 = B10 / gamma
        
        ## In paper, A0 = B; A1 = L; A2 = F
        A0 = numpy.array ([[muL, 0, 0],
                           [ t1, 0, 0],
                           [ t2, 0, 0]])
        A1 = numpy.array ([[-lambdaH-lambdaL-muL,               0,            0],
                           [                   0, -lambdaL-t1-t12,          t12],
                           [                   0,                0, -t2-lambdaL]])
        A2 = numpy.array ([[lambdaL, lambdaH,       0],
                           [      0, lambdaL,       0],
                           [      0,       0, lambdaL]])
        gamma = numpy.array ([[muL+lambdaH+lambdaL], [t1+t12+lambdaL], [t2+lambdaL]])
        self._gamma = gamma
        self._A0 = A0 / gamma
        self._A1 = A1 / gamma
        self._A2 = A2 / gamma
        
        self._A1[0][0] += 1
        self._A1[1][1] += 1
        self._A1[2][2] += 1

    def form_DR_M_M_2_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        ## This function is different from the same function in traditional_calculator.
        ## This MG1_calculator is used when having m priority classes and/or non-
        ## exponential service process in priority classes.
        
        ## In this case, we keep track of level l-1
        ##  l - 1 = #_middle + Min (#_server, #_high)
        ## The input transition rates are used to formulate the transition probability,
        ## which are used to calculate G. The transition rates are also used to get
        ## the first 3 moments of the forward, local, and backward matrices in order
        ## to obtain the moments of G. 
        
        self._nRad = 2
        self._l_hat = 4
        
        ## L = 1: (0, 0)
        B00 = numpy.array ([[-lambdaH-lambdaL]])
        self._B00 = numpy.array ([[0]])
        
        B01 = numpy.array ([[lambdaL, lambdaH]])
        gamma = -B00[0][0]
        self._B01 = B01 / gamma
        
        self._gamma_1 = gamma

        ## L = 2: (0, 1), (1, 0)
        B10 = numpy.array ([[muL], [muH]])
        gamma = numpy.array ([[muL+lambdaH+lambdaL], [muH+lambdaH+lambdaL]])
        self._B10 = B10 / gamma
        
        B11 = numpy.array ([[-gamma[0][0],            0],
                            [           0, -gamma[1][0]]])
        self._B11 = B11 / gamma + numpy.identity (B11.shape[0])
        
        B12 = numpy.array ([[lambdaL, lambdaH,       0, 0],
                            [      0, lambdaL, lambdaH, 0]])
        self._B12 = B12 / gamma
        
        self._gamma_2 = gamma
        
        ## L = 3: (0, 2), (1, 1), (2+, 0), (x, 0)
        B20 = numpy.array ([[2*muL,   0],
                            [  muH, muL],
                            [    0,  t1],
                            [    0,  t2]])
        gamma = numpy.array ([[2*muL+lambdaH+lambdaL], [muH+muL+lambdaH+lambdaL], [t1+t12+lambdaL], [t2+lambdaL]])
        self._B20 = B20 / gamma
        
        B21 = numpy.array ([[-gamma[0][0],            0,            0,            0],
                            [           0, -gamma[1][0],            0,            0],
                            [           0,            0, -gamma[2][0],          t12],
                            [           0,            0,            0, -gamma[3][0]]])
        self._B21 = B21 / gamma + numpy.identity (B21.shape[0])
        
        B22 = numpy.array ([[lambdaL, lambdaH,       0,       0],
                            [      0, lambdaL, lambdaH,       0],
                            [      0,       0, lambdaL,       0],
                            [      0,       0,       0, lambdaL]])
        self._B22 = B22 / gamma
        
        self._gamma_3 = gamma
        
        ## L = 4 - start repeating: (0, 3), (1, 2), (2+, 1), (x, 1)
        ## In paper, A0 = B; A1 = L; A2 = F
        self._gamma = gamma
        
        A0 = numpy.array ([[2*muL,   0, 0, 0],
                           [  muH, muL, 0, 0],
                           [    0,  t1, 0, 0],
                           [    0,  t2, 0, 0]])
        A1 = numpy.array ([[-lambdaH-lambdaL-2*muL,                        0,               0,           0],
                           [                     0, -lambdaH-lambdaL-muH-muL,               0,           0],
                           [                     0,                        0, -lambdaL-t1-t12,         t12],
                           [                     0,                        0,               0, -t2-lambdaL]])
        A2 = B22
        
        self._A0 = A0 / gamma
        self._A1 = A1 / gamma + numpy.identity (A1.shape[0])
        self._A2 = A2 / gamma
    
    def _get_G (self):
        #  G is found iteratively
        #     G_(k+1) = sum_i=0^inf Ai G^i_k
        #  and initial G_0 = 0
        # ** G is non-decreasing i.e. its values will only 
        #    grow until they converge.
        G = numpy.zeros_like (self._A0)
        for niter in range (1000):
            G = self._A0 + numpy.dot (self._A1, G) + \
                numpy.dot (self._A2, matrix_power (G, 2))
        return G

    def _get_A_moments (self, r):
        
        P = numpy.identity (self._A0.shape[0])
        for nrow in range (self._A0.shape[0]):
            P[nrow][nrow] = numpy.math.factorial (r)/numpy.power (self._gamma[nrow][0], r)
        A0 = numpy.dot (P, self._A0)
        A1 = numpy.dot (P, self._A1)
        A2 = numpy.dot (P, self._A2)
        return A0, A1, A2

    def _get_G_1 (self, G, A0_1, A1_1, A2_1):
        
        G1 = numpy.zeros_like (A0_1)
        for niter in range (1000):
            G1 = A0_1 + numpy.dot (A1_1, G) + numpy.dot (self._A1, G1) + \
                 numpy.dot (A2_1, matrix_power (G, 2)) + \
                 numpy.dot (self._A2, numpy.dot (G1, G)) + \
                 numpy.dot (self._A2, numpy.dot (G, G1))        
        return G1
    
    def _get_G_2 (self, G, G1, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2):
        
        G2 = numpy.zeros_like (A0_2)
        for niter in range (1000):
            G2 = A0_2 + numpy.dot (A1_2, G) + 2*numpy.dot (A1_1, G1) + \
                 numpy.dot (self._A1, G2) + numpy.dot (A2_2, matrix_power (G, 2)) + \
                 2*numpy.dot (A2_1, numpy.dot (G1, G) + numpy.dot (G, G1)) + \
                 numpy.dot (self._A2, numpy.dot (G2, G) + 2*numpy.dot (G1, G1) + numpy.dot (G, G2))      
        return G2

    def _get_G_3 (self, G, G1, G2, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, A0_3, A1_3, A2_3):
        
        G3 = numpy.zeros_like (A0_3)
        for niter in range (1000):
            G3 = A0_3 + numpy.dot (A1_3, G) + 3*numpy.dot (A1_2, G1) + 3*numpy.dot (A1_1, G2) + \
                 numpy.dot (self._A1, G3) + numpy.dot (A2_3, matrix_power (G, 2)) + \
                 3*numpy.dot (A2_2, numpy.dot (G1, G) + numpy.dot (G, G1)) + \
                 3*numpy.dot (A2_1, numpy.dot (G2, G) + 2*numpy.dot (G1, G1) + numpy.dot (G, G2)) + \
                 numpy.dot (self._A2, numpy.dot (G3, G) + 3*numpy.dot (G2, G1) + 3*numpy.dot (G1, G2) + numpy.dot (G, G3))      
        return G3

    def _get_G_moments (self, G):

        A0_1, A1_1, A2_1 = self._get_A_moments (1)
        A0_2, A1_2, A2_2 = self._get_A_moments (2)
        A0_3, A1_3, A2_3 = self._get_A_moments (3)

        G1 = self._get_G_1 (G, A0_1, A1_1, A2_1)
        G2 = self._get_G_2 (G, G1, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2)
        G3 = self._get_G_3 (G, G1, G2, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, A0_3, A1_3, A2_3)

        return G1, G2, G3

    def _get_G_nonrepeating (self, G_rep, repeat=False):
        #  G is found iteratively
        #     G_(k+1) = sum_i=0^inf Ai G^i_k
        #  and initial G_0 = 0
        # ** G is non-decreasing i.e. its values will only 
        #    grow until they converge.
        
        ## B10 for 1 radiologist only
        ## l = 2

        B0 = self._B20 if self._nRad == 2 or (self._l_hat == 3 and not repeat) else \
             self._B10 #if self._nRad == 1
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2
        
        G = numpy.zeros_like (B0)
        for niter in range (1000):
            G = B0 + numpy.dot (A1, G) + \
                numpy.dot (A2, numpy.dot (G_rep, G))
        #for niter in range (1000):
        #    G = B0 + numpy.dot (self._A1, G) + \
        #        numpy.dot (self._A2, numpy.dot (G_rep, G))
        
        return G

    def _get_A_moments_nonrepeating (self, r, repeat=False):
        
        ## At least for exponential, the first non-repeating (one below repeating level)
        ## has the same L (i.e. A1) and F (i.e. A2). Only the B (i.e. A1) is different.
        ## Gamma values are also the same.
        
        B0 = self._B20 if self._nRad == 2 or (self._l_hat == 3 and not repeat) else \
             self._B10 #if self._nRad == 1
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2
        gamma = self._gamma_2 if repeat else self._gamma              

        P = numpy.identity (A1.shape[0])
        for nrow in range (A1.shape[0]):
            P[nrow][nrow] = numpy.math.factorial (r)/numpy.power (gamma[nrow][0], r)
        A0 = numpy.dot (P, B0)
        A1 = numpy.dot (P, A1)
        A2 = numpy.dot (P, A2)
        
        #P = numpy.identity (self._A1.shape[0])
        #for nrow in range (self._A1.shape[0]):
        #    P[nrow][nrow] = numpy.math.factorial (r)/numpy.power (self._gamma[nrow][0], r)
        #A0 = numpy.dot (P, B0)
        #A1 = numpy.dot (P, self._A1)
        #A2 = numpy.dot (P, self._A2)
        return A0, A1, A2

    def _get_G_1_nonrepeating (self, G_nonrep, G, G1, A0_1, A1_1, A2_1, repeat=False):
        
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2        
        
        G1_nonrep = numpy.zeros_like (A0_1)
        for niter in range (1000):
            G1_nonrep = A0_1 + numpy.dot (A1_1, G_nonrep) + numpy.dot (A1, G1_nonrep) + \
                        numpy.dot (A2_1, numpy.dot (G, G_nonrep)) + \
                        numpy.dot (A2, numpy.dot (G1, G_nonrep)) + \
                        numpy.dot (A2, numpy.dot (G, G1_nonrep))
        return G1_nonrep
    
    def _get_G_2_nonrepeating (self, G_nonrep, G1_nonrep, G, G1, G2, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, repeat=False):
        
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2          
        
        G2_nonrep = numpy.zeros_like (A0_2)
        for niter in range (1000):
            G2_nonrep = A0_2 + numpy.dot (A1_2, G_nonrep) + 2*numpy.dot (A1_1, G1_nonrep) + \
                        numpy.dot (A1, G2_nonrep) + numpy.dot (A2_2, numpy.dot (G, G_nonrep)) + \
                        2*numpy.dot (A2_1, numpy.dot (G1, G_nonrep) + numpy.dot (G, G1_nonrep)) + \
                        numpy.dot (A2, numpy.dot (G2, G_nonrep) + 2*numpy.dot (G1, G1_nonrep) + numpy.dot (G, G2_nonrep))      
        return G2_nonrep

    def _get_G_3_nonrepeating (self, G_nonrep, G1_nonrep, G2_nonrep, G, G1, G2, G3, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, A0_3, A1_3, A2_3, repeat=False):
        
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2          
        
        G3_nonrep = numpy.zeros_like (A0_3)
        for niter in range (1000):
            G3_nonrep = A0_3 + numpy.dot (A1_3, G_nonrep) + 3*numpy.dot (A1_2, G1_nonrep) + 3*numpy.dot (A1_1, G2_nonrep) + \
                        numpy.dot (A1, G3_nonrep) + numpy.dot (A2_3, numpy.dot (G, G_nonrep)) + \
                        3*numpy.dot (A2_2, numpy.dot (G1, G_nonrep) + numpy.dot (G, G1_nonrep)) + \
                        3*numpy.dot (A2_1, numpy.dot (G2, G_nonrep) + 2*numpy.dot (G1, G1_nonrep) + numpy.dot (G, G2_nonrep)) + \
                        numpy.dot (A2, numpy.dot (G3, G_nonrep) + 3*numpy.dot (G2, G1_nonrep) + 3*numpy.dot (G1, G2_nonrep) + numpy.dot (G, G3_nonrep))      
        return G3_nonrep

    def _get_G_moments_nonrepeating (self, G_nonrep, G, G1, G2, G3, repeat=False):

        ## l = 2 (repeating starts at l = 3)
        A0_1, A1_1, A2_1 = self._get_A_moments_nonrepeating (1, repeat=repeat)
        A0_2, A1_2, A2_2 = self._get_A_moments_nonrepeating (2, repeat=repeat)
        A0_3, A1_3, A2_3 = self._get_A_moments_nonrepeating (3, repeat=repeat)

        G1_nonrep = self._get_G_1_nonrepeating (G_nonrep, G, G1, A0_1, A1_1, A2_1, repeat=repeat)
        G2_nonrep = self._get_G_2_nonrepeating (G_nonrep, G1_nonrep, G, G1, G2, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, repeat=repeat)
        G3_nonrep = self._get_G_3_nonrepeating (G_nonrep, G1_nonrep, G2_nonrep, G, G1, G2, G3, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, A0_3, A1_3, A2_3, repeat=repeat)

        return G1_nonrep, G2_nonrep, G3_nonrep

    def get_Zs (self):
        
        ## Repeating parts
        G = self._get_G()
        self._G = G
        G1, G2, G3 = self._get_G_moments (G)
        
        ## Non-repeating parts - Only the first state below the repeating level
        ## e.g. if 1 radiologist, repeating level starts at 2. So, non-repeating parts return the Zs at 1
        ##      if 2 radiologist, repeating level starts at 4. So, non-repeating parts return the Zs at 3
        G_nonrep = self._get_G_nonrepeating (G)
        self._G_nonrep = G_nonrep
        G1_nonrep, G2_nonrep, G3_nonrep = self._get_G_moments_nonrepeating (G_nonrep, G, G1, G2, G3)
        Z1_nonrep = numpy.nan_to_num (G1_nonrep/G_nonrep)
        Z2_nonrep = numpy.nan_to_num (G2_nonrep/G_nonrep)
        Z3_nonrep = numpy.nan_to_num (G3_nonrep/G_nonrep)
        
        if self._l_hat == 3:
            G_nonrep2 = self._get_G_nonrepeating (G_nonrep, repeat=True)
            self._G_nonrep2 = G_nonrep2
            G1_nonrep, G2_nonrep, G3_nonrep = self._get_G_moments_nonrepeating (G_nonrep2, G_nonrep, G1_nonrep, G2_nonrep, G3_nonrep, repeat=True)
            Z1_nonrep = numpy.nan_to_num (G1_nonrep/G_nonrep2)
            Z2_nonrep = numpy.nan_to_num (G2_nonrep/G_nonrep2)
            Z3_nonrep = numpy.nan_to_num (G3_nonrep/G_nonrep2)        
        
        self._Zs = {'Z1':Z1_nonrep, 'Z2':Z2_nonrep, 'Z3':Z3_nonrep}

    def _get_stars (self, G):

        ## 1. A stars
        Astar1 = numpy.identity (G.shape[0]) - (self._A1 + numpy.dot (self._A2, G))
        Astar2 = - (self._A2)
        Astar3 = None
        if self._A3 is not None:
            Astar1 += - numpy.dot (self._A3, matrix_power (G, 2))
            Astar2 += - numpy.dot (self._A3, matrix_power (G, 1))
            Astar3 = - (self._A3)
        
        Bstar01 = self._B01
        Bstar02 = None
        if self._B02 is not None:
            Bstar01 += numpy.dot (self._B02, G)
            Bstar02 = self._B02

        return Astar1, Astar2, Astar3, Bstar01, Bstar02

    def _solve_boundary_solutions (self, Astar1, Bstar01):
        bound = numpy.identity (self._B00.shape[0]) - self._B00 - \
                numpy.dot (Bstar01, numpy.dot (inv (Astar1), self._B10))
        vecs = eig (bound, left=True, right=False)[1]
        for idx in range (bound.shape[0]):
            pi_bound = numpy.round (vecs[:, idx], 6)
            if not (pi_bound<0).any(): break
        return pi_bound ## Only pi0

    def _find_norm (self, pi0, Astar1, Astar2, Astar3, Bstar01, Bstar02):
        
        AstarSum = sum ([0 if a is None else a for a in [Astar1, Astar2, Astar3]])
        BstarSum = sum ([0 if b is None else b for b in [Bstar01, Bstar02]])
        
        alpha = numpy.sum (pi0) + \
                numpy.dot (pi0, numpy.dot (BstarSum, numpy.sum (inv (AstarSum), axis=1)))
        
        return alpha

    def solve_prob_distributions (self, n=100):
    
        self._is_valid_system()
        #if not self._check_ergodic():
        #    print ('This MC is not ergodic. Cannot solve for its pis.')
        #    return None   

        ## Repeating parts
        G = self._get_G()
        G1, G2, G3 = self._get_G_moments (G)
        Z1 = numpy.nan_to_num (G1/G)
        Z2 = numpy.nan_to_num (G2/G)
        Z3 = numpy.nan_to_num (G3/G)
        
        ## Non-repeating parts - For 1 radiologist only
        G_nonrep = self._get_G_nonrepeating (G)
        G1_nonrep, G2_nonrep, G3_nonrep = self._get_G_moments_nonrepeating (G_nonrep, G, G1, G2, G3)
        Z1_nonrep = numpy.nan_to_num (G1_nonrep/G_nonrep)
        Z2_nonrep = numpy.nan_to_num (G2_nonrep/G_nonrep)
        Z3_nonrep = numpy.nan_to_num (G3_nonrep/G_nonrep)
        
        self._Zs = {'Z1':Z1_nonrep, 'Z2':Z2_nonrep, 'Z3':Z3_nonrep}
        
        Astar1, Astar2, Astar3, Bstar01, Bstar02 = self._get_stars(G)
        pi0 = self._solve_boundary_solutions (Astar1, Bstar01)
        
        # Normalize boundary states
        alpha = self._find_norm (pi0, Astar1, Astar2, Astar3, Bstar01, Bstar02)
        pi0 /= alpha
    
        pis = [pi0]
        for i in range (n-1):
            if i == 0: # state = 1
                pii = numpy.dot (pis[-1], numpy.dot (Bstar01, inv (Astar1)))
            elif i == 1: # state = 2
                pii = -numpy.dot (pis[-1], numpy.dot (Astar2, inv (Astar1)))
                if not self._B02 is None:
                    pii += numpy.dot (pis[0], numpy.dot (Bstar02, inv (Astar1)))
            else:
                pii = -numpy.dot (pis[-1], numpy.dot (Astar2, inv (Astar1)))
                if not self._A3 is None:
                    pii += -numpy.dot (pis[-2], numpy.dot (Astar3, inv (Astar1)))
            pis.append (pii)
        
        self._G = G
        self._Astar1  = Astar1
        self._Astar2  = Astar2
        self._Astar3  = Astar3
        self._Bstar01 = Bstar01
        self._Bstar02 = Bstar02
        self._pis = pis
        
        return pis