import numpy as np
import scipy.special as math
import sys

##########################################################
# Parameter sets for the calcium and synaptic weight dynamics
def choseParameterSet(plasticityCase):
        #
        global tauCa, Cpre, Cpost, thetaP, thetaD, gammaP, gammaD, sigma, tau, rhoStar, D, beta, b
        if plasticityCase == 'DP':
                print('DP')
                tauCa = 0.02 # in sec
                Cpre = 1.
                Cpost = 2.
                thetaD  = 1.
                thetaP  = 1.3
                gammaD  = 200.
                gammaP  = 321.808
                sigma   = 2.8284
                tau     = 150.
                rhoStar = 0.5
                D       = 0.0137 # in sec
                beta    = 0.5
                b       = 5.
        elif plasticityCase == 'DPD':
                print('DPD')
                tauCa = 0.02 # in sec
                Cpre = 0.9
                Cpost = 0.9
                thetaD  = 1.
                thetaP  = 1.3
                gammaD  = 250.
                gammaP  = 550.
                sigma   = 2.8284
                tau     = 150.
                rhoStar = 0.5
                D       = 0.0046 # in sec
                beta    = 0.5
                b       = 5.
        elif plasticityCase == 'DPDprime':
                print('DPDprime')
                tauCa = 0.02 # in sec
                Cpre = 1.
                Cpost = 2.
                thetaD  = 1
                thetaP  = 2.5
                gammaD  = 50.
                gammaP  = 600.
                sigma   = 2.8284
                tau     = 150.
                rhoStar = 0.5
                D       = 0.0022 # in sec
                beta    = 0.5
                b       = 5.
        elif plasticityCase == 'P':
                print('P')
                tauCa = 0.02 # in sec
                Cpre = 2.
                Cpost = 2.
                thetaD  = 1.
                thetaP  = 1.3
                gammaD  = 160.
                gammaP  = 257.447
                sigma   = 2.8284
                tau     = 150.
                rhoStar = 0.5
                D       = 0.0 # in sec
                beta    = 0.5
                b       = 5.
        elif plasticityCase == 'D':
                print('D')
                tauCa = 0.02 # in sec
                Cpre = 0.6
                Cpost = 0.6
                thetaD  = 1.
                thetaP  = 1.3
                gammaD  = 500.
                gammaP  = 550.
                sigma   = 5.6568
                tau     = 150.
                rhoStar = 0.5
                D       = 0. # in sec
                beta    = 0.5
                b       = 5.
        elif plasticityCase == 'Dprime':
                print('Dprime')
                tauCa = 0.02 # in sec
                Cpre = 1.
                Cpost = 2.
                thetaD  = 1.
                thetaP  = 3.5
                gammaD  = 60.
                gammaP  = 600.
                sigma   = 2.8284
                tau     = 150.
                rhoStar = 0.5
                D       = 0. # in sec
                beta    = 0.5
                b       = 5.
        elif plasticityCase == 'hippocampal slices':
                print('hippocampal slices')
                tauCa = 0.0488373 # in sec
                Cpre = 1.
                Cpost = 0.275865
                thetaD  = 1.
                thetaP  = 1.3
                gammaD  = 313.0965
                gammaP  = 1645.59
                sigma   = 9.1844
                tau     = 688.355
                rhoStar = 0.5
                D       = 0.0188008 # in sec
                beta    = 0.7
                b       = 5.28145
        elif plasticityCase == 'hippocampal cultures':
                print('hippocampal cultures')
                tauCa = 0.0119536 # in sec
                Cpre = 0.58156
                Cpost = 1.76444
                thetaD  = 1.
                thetaP  = 1.3
                gammaD  = 61.141
                gammaP  = 113.6545
                sigma   = 2.5654
                tau     = 33.7596
                rhoStar = 0.5
                D       = 0.01 # in sec
                beta    = 0.5
                b       = 36.0263
        elif plasticityCase == 'cortical slices':
                print('cortical slices')
                tauCa = 0.0226936 # in sec
                Cpre = 0.5617539
                Cpost = 1.23964
                thetaD  = 1.
                thetaP  = 1.3
                gammaD  = 331.909
                gammaP  = 725.085
                sigma   = 3.3501
                tau     = 346.3615
                rhoStar = 0.5
                D       = 0.0046098 # in sec
                beta    = 0.5
                b       = 5.40988
        else:
                print('Choose from one of the available parameter sets!')
                sys.exit(1)



##########################################################
# calculate UP and DOWN transition probabilities
def transitionProbability(N,interval,rhoStar,rho0,rhoBar,sigmaRhoSquared,tauEff):
        # argument for the Error Function
        x1 =  -(rhoStar - rhoBar + (rhoBar - rho0)*np.exp(-N*interval/tauEff))/(np.sqrt(sigmaRhoSquared*(1.-np.exp(-2.*N*interval/tauEff))))
        # transition probability
        if rho0 == 0.:
                return (1. + math.erf(x1))/2.
        else:
                return (1. - math.erf(x1))/2.


##########################################################
# calculate the change in synaptic strength
def changeSynapticStrength(beta,UP,DOWN,b):
        return ((beta*(1.-UP) + (1.-beta)*DOWN) + (beta*UP+ (1.-beta)*(1.-DOWN))*b)/(beta + (1.-beta)*b)

