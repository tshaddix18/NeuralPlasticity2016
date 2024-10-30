"""
astro_models.py
*** requires Brian 2.x simulator

Modules containing models and methods to build neuron-glia synaptic interactions used in

De Pitta and Brunel, "Modulation of synaptic plasticity by glutamatergic gliotransmission: A model study. Neural Plasticity, 2016: 7607924

v2.0 Maurizio De Pitta, Basque Center for Mathematics, 2020
Translated and Debugged to run in Python 3.x

v1.0 Maurizio De Pitta, The University of Chicago, 2015
"""


from brian2 import *
from brian2.units.allunits import mole, umole, mmole
import os
# Define some global lambdas
peak_normalize = lambda peak, taur, taud : peak*(1./taud - 1./taur)/Hz/( (taur/taud)**(taud/(taud-taur))-(taur/taud)**(taur/(taud-taur)) )

def get_parameters(oscillations='AM', synapse='depressing', stdp=None):

    # Parameters from Tables 1-3, using Table 2 if the values differ from Table 1
    # Common parameters
    parameters = {
        # ----Input
        'f_in': 1.*Hz,              # Input frequency (synapse)
        'f_c' : 1.*Hz,              # Input frequency (gliotransmission)
        # 't_on' : 0*second,         # Start of synaptic stimulation (used in STDP)
        't_off' : Inf*second,      # End of astrocyte stimulation (used in standalone gliotransmission)
        # --- IP_3R kinectics
        'd_1': 0.13*umole,         # IP_3 binding affinity
        'O_2': 0.2/umole/second,   # Inactivating Ca^2+ binding rate
        'd_2': 1.05*umole,         # Inactivating Ca^2+ binding affinity
        'd_3': 0.9434*umole,       # IP_3 binding affinity (with Ca^2+ inactivation)
        'd_5': 0.08*umole,         # Activating Ca^2+ binding affinity
        # ---  Calcium fluxes
        'C_osc': 0.2*umole,        # Estimated Threshold for Ca^2+ oscillations
        'C_T': 2*umole,            # Total ER Ca^2+ content
        'rho_A': 0.18,             # ER-to-cytoplasm volume ratio
        'Omega_C': 6/second,       # Maximal Ca^2+ release rate by IP_3Rs
        'Omega_L': 0.1/second,     # Maximal Ca^2+ leak rate,
        'O_P': 0.9*umole/second,   # Maximal Ca^2+ uptake rate
        # K_P (see below)          # Ca^2+ affinity of SERCA pumps
        # --- IP_3 production
        # Omega_delta (see below)  # Maximal rate of IP_3 production by PLCdelta
        'K_delta': 0.5*umole,      # Ca^2+ affinity of PLCdelta
        'kappa_delta': 1.*umole,    # Inhibiting IP_3 affinity of PLCdelta
        # --- IP_3 degradation
        # Omega_5P (see below)     # Maximal rate of IP_3 degradation by IP-5P
        'O_3K': 4.5*umole/second,  # Maximal rate of IP_3 degradation by IP_3-3K
        'K_D': 0.5*umole,          # Ca^2+ affinity of IP3-3K
        'K_3K': 1.*umole,           # IP_3 affinity of IP_3-3K
        # --- IP_3 diffusion
        'F': 2.*umole/second,       # GJC IP_3 permeability (nonlinear)
        'I_Theta': 0.3*umole,      # Threshold IP_3 gradient for diffusion
        'omega_I': 0.05*umole,     # Scaling factor of diffusion
        # I_bias (see below)       # IP_3 bias
        # --- Agonist-dependent IP_3 production
        'O_beta': 1.*umole/second,  # Maximal rate of IP_3 production by PLCbeta
        'O_N': 0.3/umole/second,   # Agonist binding rate
        'Omega_N': 1.8/second,     # Inactivation rate of GPCR signalling
        'K_KC': 0.5*umole,         # Ca^2+ affinity of PKC
        'zeta': 2.,                # Maximal reduction of receptor affinity by PKC
        'n': 1.,                   # Cooperativity of agonist binding reaction
        # --- Gliotransmitter release and time course        
        'C_Theta': 0.5*umole,      # Ca^2+ threshold for exocytosis
        'Omega_A': 0.6/second,     # Gliotransmitter recycling rate
        'U_A': 0.6,                # Gliotransmitter release probability
        'G_T': 200.*mmole,         # Total vesicular gliotransmitter
        'rho_e': 6.5e-4,           # Ratio of astrocytic vesicle volume/ESS volume
        'Omega_e': 5./second,      # Gliotransmitter clearance rate (think about distributed release)
        # --- Synaptic dynamics
        # Omega_d (see below)      # Depression rate
        # Omega_f (see below)      # Facilitation rate,
        # U_0__star (see below)    # Basal synaptic release probability
        'Omega_c': 40./second,     # Neurotransmitter clearance rate
        'rho_c': 0.005,            # synaptic vesicle-to-extracellular space volume ratio
        'Y_T': 500.*mmole,         # Total neurotransmitter synaptic resource (in terms of vesicular concentration)
        # --- Presynaptic receptors
        'O_G': 1.5/umole/second,   # Agonist binding rate (activating)
        'Omega_G': 0.5/(60*second),# Agonist release rate (inactivating)
        # alpha (see below)        # Gliotransmitter effect on synaptic release
        # --- SIC/SOC
        'G_sic'     : 4.5*mV,      # Max SIC/SOC depolarization
        'tau_sic_r' : 30.*ms,      # SIC/SOC rise time constant
        'tau_sic' : 600.*ms,       # SIC/SOC decay time constant
    }

    if oscillations == 'AM':
        parameters.update({
            'K_P': 0.1*umole,
            'O_delta': 0.01*umole/second,
            'Omega_5P': 0.1/second,
            'I_bias': 0.8*umole
        })
    elif oscillations == 'FM':
        parameters.update({
            'K_P': 0.05*umole,
            'O_delta': 0.05*umole/second,
            'Omega_5P': 0.1/second,
            'I_bias': 1.*umole
        })
    else:
        raise ValueError('oscillations argument has to be "AM" or "FM"')

    if synapse == 'depressing':
        parameters.update({
            'Omega_d': 2./second,
            'Omega_f': 3.33/second,
            'U_0__star': 0.6,
            'alpha': 0.,
        })
    elif synapse == 'facilitating':
        parameters.update({
            'Omega_d': 2./second,
            'Omega_f': 2./second,
            'U_0__star': 0.15,
            'alpha': 1.,
        })
    elif synapse == 'neutral':
        parameters.update({
            'Omega_d': 3./second,
            'Omega_f': 3./second,
            'U_0__star': 0.5,
            'alpha': 1.,
        })
    else:
        raise ValueError('synapse argument has to be "depressing", "facilitating" or "neutral"')

    # Post-synaptic neuron parameters
    parameters.update({
        'G_e' : 2*mV,   # Max synaptic depolarization
        'tau_m': 20*ms,  # Membrane time constant
        'tau_r': 5*ms,   # Refractory time
        'tau_e_r': 0.5*ms, # excitatory conductance rise time
        'tau_e': 5*ms,   # excitatory conductance time constant
        'tau_i': 10*ms,  # inhibitory conductance time constant
        'E_L': -60*mV,   # reversal potential
        'V_th': -55*mV,   # Threshold
        'V_r': -57*mV,   # Reset
    })

    # parameters.update({
    # 'G_norm'     : normalize(1.0,parameters['tau_e_r'],parameters['tau_e']),
    # 'G_sic_norm' : normalize(1.0,parameters['tau_sic_r'],parameters['tau_sic_d'])
    # })
    # STDP parameters
    # Graupner and Brunel (PNAS 2012) / DP curve
    parameters.update({
        'tau_ca': 20.0*ms, # Intrasynaptic Ca2+ decay constant
        'Cpre'  : 1.0,     # Presynaptic Ca2+ increase per spk
        'Cpost' : 2.0,     # Postynaptic Ca2+ increase per spk
        'Theta_d' : 1.0,   # LTD threshold
        'Theta_p': 1.3,    # LTP threshold
        'gamma_d': 200.0,  # LTD learning rate
        'gamma_p': 321.808,# LTP learning rate
        'W_0'    : 0.5,    # LTP/LTD boundary
        'tau_w'  : 346.3615*second, # Time decay of synaptic weights
        'D'      : 13.7*ms,# Synaptic delay
        'sigma'  : 2.8284, # variance in the diffusion approx,
        'beta'   : 0.5,
        'b'      : 5.
    })
    if stdp=='linear':
        pass
    elif stdp=='nonlinear':
        parameters.update({
            'Cpre'  : 1.0,         # Presynaptic Ca2+ increase per spk
            'tau_pre_r' : 10*ms,   # Presynaptic Ca2+ rise time
            'tau_pre_d' : 30*ms,   # Presynaptic Ca2+ decay time
            'Cpost' : 2.5,         # Postynaptic Ca2+ increase per spk
            'tau_post_r': 2*ms,    # Postsynaptic Ca2+ rise time
            'tau_post_d': 12*ms,   # Postsynaptic Ca2+ decay time
            'eta'       : 4.0,     # Pre amplification factor
            'etap'    : 1.0,       # Post amplification factor
            'Theta_d' : 1.0,       # LTD threshold
            'Theta_p': 2.2,        # LTP threshold
            'gamma_d': 0.57,       # LTD learning rate
            'gamma_p': 1.32,       # LTP learning rate
            'W_0'    : 0.5,        # LTP/LTD boundary
            'tau_w'  : 1.5*second, # Time decay of synaptic weights
            # Update values
            'sigma'  : 0.1,
            'beta'   : 0.5,
            'b'      : 4
        })
        # # Compute normalization factors
        # parameters.update({
        #     'Cpre_norm'  : normalize(parameters['Cpre'], parameters['tau_pre_r'], parameters['tau_pre_d']),
        #     'Cpost_norm' : normalize(parameters['Cpost'], parameters['tau_post_r'], parameters['tau_post_d'])
        # })

    return parameters

def astrocyte_group(N, params, dt=1*msecond, ics=None):
    eqs = '''
    # Fraction of activated astrocyte receptors:
    dGamma_A/dt = O_N * (Y_bias+Y_extra)**n * (1 - Gamma_A) -
                  Omega_N*(1 + zeta * C/(C + K_KC)) * Gamma_A : 1

    # IP_3 dynamics:
    dI/dt = O_beta * Gamma_A + O_delta/(1 + I/K_delta) * C**2/(C**2 + K_delta**2) -
            O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) - Omega_5P*I +
            I_coupling + I_exogenous : mole

    # Exogenous stimulation applied to the cell
    delta_I_bias = I - I_bias: mole
    I_exogenous = -F/2*(1 + tanh((abs(delta_I_bias) - I_Theta)/omega_I))*sign(delta_I_bias) : mole/second
    # I_exogenous : mole/second
    # diffusion between astrocytes:
    I_coupling : mole/second

    # Ca^2+-induced Ca^2+ release:
    dC/dt = (Omega_C * m_inf**3 * h**3 + Omega_L) * (C_T - (1 + rho_A)*C) -
            O_P * C**2/(C**2 + K_P**2) : mole
    dh/dt = (h_inf - h)/tau_h : 1  # IP3R de-inactivation probability
    m_inf = I/(I + d_1) * C/(C + d_5) : 1
    h_inf = Q_2/(Q_2 + C) : 1
    tau_h = 1/(O_2 * (Q_2 + C)) : second
    Q_2 = d_2 * (I + d_1)/(I + d_3) : mole

    # External neurotransmitter stimulation
    Y_bias : mole
    # Neurotransmitter concentration in the extracellular space
    Y_extra : mole

    # Additional (optional) coordinates (for spatial network implementation)
    x : meter
    y : meter
    '''
    
    # The definition of a threshold and reset mechanism in the astrocyte group
    # allows to use SpikeMonitors to estimate the frequency of oscillations
    group = NeuronGroup(N, eqs,
                        threshold='C>C_osc',
                        refractory='C>C_osc',
                        method='rk4',
                        dt=dt,
                        namespace=params,
                        name='astrocyte*')

    # Random initialization of initial conditions
    if ics=='rand':
        group.Gamma_A = 'rand()'
        group.I = '3*rand()*umole'
        group.C = '1.5*rand()*umole'
        group.h = 'rand()'

    return group

def synapse(source, target, params, connect='i==j', ics=None,
            dt=None,
            linear=False, name='synapses*',
            stimulus=None,
            stdp=None, postc=None, sic=None,
            delay=None):
    # Synapses modelled as in Tsodyks (2005) with basal release probability
    # modulated by presynaptic receptors as in De Pitta' et al., PLoS Comput. Biol. (2011)
    #
    # IMPORTANT: 'postc' argument stands for 'post' in other methods, but because 'post' is a protected keyword in Synapse
    # it cannot be used in this module and 'postc' is used instead.

    eqs = Equations('''
        # Fraction of activated presynaptic receptors
        dGamma_S/dt = O_G * G_A * (1 - Gamma_S) - Omega_G * Gamma_S : 1
        # Usage of releasable neurotransmitter per single action potential:
        du_S/dt = -Omega_f * u_S : 1 (event-driven)
        # Fraction of synaptic neurotransmitter resources available for release:
        dx_S/dt = Omega_d *(1 - x_S) : 1 (event-driven)
        dY_S/dt = -Omega_c * Y_S : mole
        G_A : mole  # gliotransmitter concentration in the extracellular space
        U_0 : 1
        r_S : 1     # Because r_S is the product of u_S and x_S that are event-driven, it is itself event-driven too
        # Astrocyte ID for connection
        astro_index : integer
        # Per-synapse gliotransmitter-effect parameter
        alpha  : 1
        ''')

    # Forcing of parameters
    if stdp=='linear':
        if postc!=None: postc=='single-exp'
    elif stdp=='nonlinear':
        if postc!=None: postc=='double-exp'
    if postc!=None and sic!=None : sic=postc

    # Add case-specific code
    if postc=='exp' or postc==True:
        eqs += Equations('''
                         dge/dt = -ge/tau_e + Y_S*we/tau_e : 1
                         ''')
    elif postc=='double-exp':
        params['nge'] = peak_normalize(1.,params['tau_e_r'],params['tau_e'])
        eqs += Equations('''
                       dge/dt = -ge/tau_e_r + nge*B_syn : 1
                       dB_syn/dt = -B_syn/tau_e + Y_S*we : Hz  # Technically you need the concentration in the cleft, but you can use Y_S and rescale w/in 'we'
                       ''')

    if sic=='exp' or sic==True:
        eqs += Equations('''
                       dgsic/dt = -gsic/tau_sic + G_A_sic*wa/tau_sic : 1
                       G_A_sic : mole
                       ''')
    elif sic=='double-exp':
        params['nsic'] = peak_normalize(1.,params['tau_sic_r'],params['tau_sic'])
        eqs += Equations('''
                       dgsic/dt = -gsic/tau_sic_r + nsic*B_sic : 1
                       dB_sic/dt = -B_sic/tau_sic + G_A_sic*wa : Hz  # Technically you need the concentration in the cleft, but you can use Y_S and recale w/in 'we'
                       G_A_sic : mole
                       ''')
    # Add postsynaptic side voltage (when synaptic stimulation is not 'test')
    # if (not stdp) and (stimulus!='test'): # old version
    # if (not stdp) and (stimulus!='test') and (postc!=None):
    if (not stdp) and (postc != None):
        eqs += Equations('''
                         # Correspondent neural variable
                         g_e_post = ge*G_e : volt (summed)
                         ''')
        if (sic!=None): # THE EQUATION HERE WAS ORIGINALLY INSIDE
            eqs += Equations('''
                         # Correspondent neural variable
                         g_sic_post = gsic*G_sic : volt (summed)
                             ''')
    if stdp=='linear':
        if not sic:
            eqs += Equations('''
                             # Postsynaptic calcium
                             dC_syn/dt = -C_syn/tau_ca : 1
                             # Synaptic weight
                             #dW/dt = -W* (1-W) * (W_0 - W) / tau_w + gamma_p * (1-W) * (1+sign(C_syn-Theta_p))/tau_w/2.0 - gamma_d * W * (1+sign(C_syn-Theta_d))/tau_w/2.0 : 1
                             dW/dt = -W* (1-W) * (W_0 - W) / tau_w + gamma_p * (1-W) * int(C_syn>=Theta_p)/tau_w - gamma_d * W * int(C_syn>=Theta_d)/tau_w : 1
                             ''')
        else:
            eqs += Equations('''
                             # Postsynaptic calcium
                             dC_syn/dt = -C_syn/tau_ca + pknorm(Csic, tau_ca, tau_sic)*C_sic : 1
                             # SIC calcium
                             dC_sic/dt = -C_sic/tau_sic
                             # SIC Contribution to postsynaptic calcium
                             Csic : 1
                             # Synaptic weight
                             # dW/dt = -W* (1-W) * (W_0 - W) / tau_w + gamma_p * (1-W) * (1+sign(C_syn-Theta_p))/tau_w/2.0 - gamma_d * W * (1+sign(C_syn-Theta_d))/tau_w/2.0 : 1
                             dW/dt = -W* (1-W) * (W_0 - W) / tau_w + gamma_p * (1-W) * int(C_syn>=Theta_p)/tau_w - gamma_d * W * int(C_syn>=Theta_d)/tau_w : 1
                             ''')
    elif stdp=='nonlinear':
        params['npre'] = peak_normalize(params['Cpre'],params['tau_pre_r'],params['tau_pre_d'])
        params['npost'] = peak_normalize(params['Cpost'],params['tau_post_r'],params['tau_post_d'])
        if not sic:
            eqs += Equations('''
                             # Postsynaptic calcium (presynaptic contribution)
                             dA/dt = -A/tau_pre_r + npre*B : 1
                             dB/dt = -B/tau_pre_d + Y_S*we : Hz
                             # Postsynaptic calcium (postsynaptic contribution)
                             dP/dt = -P/tau_post_r + npost*F : 1
                             dF/dt = -F/tau_post_d : Hz
                             # Total postsynaptic calcium
                             C_syn = A + P : 1
                             # Synaptic weight
                             # dW/dt = -W* (1-W) * (W_0 - W) / tau_w + gamma_p * (1-W) * (1+sign(C_syn-Theta_p))/tau_w/2.0 - gamma_d * W * (1+sign(C_syn-Theta_d))/tau_w/2.0 : 1
                             dW/dt = -W* (1-W) * (W_0 - W) / tau_w + gamma_p * (1-W) * int(C_syn>Theta_p)/tau_w - gamma_d * W * int(C_syn>Theta_d)/tau_w : 1
                             ''')
        else:
            eqs += Equations('''
                             # Postsynaptic calcium (presynaptic contribution)
                             dA/dt = -A/tau_pre_r + npre*B : 1
                             dB/dt = -B/tau_pre_d + Y_S*we : Hz
                             # Postsynaptic calcium (postsynaptic contribution)
                             dP/dt = -P/tau_post_r + npost*F : 1
                             dF/dt = -F/tau_post_d : Hz
                             # Total postsynaptic calcium
                             C_syn = A + P + gsic*Csic : 1
                             # SIC Contribution to postsynaptic calcium
                             Csic : 1
                             # Synaptic weight
                             # dW/dt = -W* (1-W) * (W_0 - W) / tau_w + gamma_p * (1-W) * (1+sign(C_syn-Theta_p))/tau_w/2.0 - gamma_d * W * (1+sign(C_syn-Theta_d))/tau_w/2.0 : 1
                             dW/dt = -W* (1-W) * (W_0 - W) / tau_w + gamma_p * (1-W) * int(C_syn>Theta_p)/tau_w - gamma_d * W * int(C_syn>Theta_d)/tau_w : 1
                             ''')

    # Case-specific code
    if linear:
        # Instantaneous synapses (no plasticity / no regulation by gliotransmitter)
        # (Used as a pulsed stimulus in the simulations)
        if not stdp:
            pre  = '''
            Y_S += rho_c * Y_T
            '''
            post = None
        elif stdp=='linear':
            pre  = '''
            C_syn += Cpre
            Y_S += rho_c * Y_T
            '''
            post = '''
            C_syn += Cpost
            '''
            delay = 'D'
        elif stdp=='nonlinear':
            pre  = '''
            Y_S += rho_c * Y_T
            '''
            post = '''
            F += (etap + eta*A)/second
            '''
    else:
        if not stdp:
            pre = '''
            U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
            u_S += U_0 * (1 - u_S)
            r_S = u_S * x_S # released synaptic neurotransmitter resources
            x_S -= r_S
            Y_S += rho_c * Y_T * r_S
            '''
            post = None
        elif stdp=='linear':
            pre = '''
            U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
            u_S += U_0 * (1 - u_S)
            r_S = u_S * x_S # released synaptic neurotransmitter resources
            x_S -= r_S
            C_syn += r_S * Cpre
            Y_S += rho_c * Y_T * r_S
            '''
            post = '''
            C_syn += Cpost
            '''
            delay = 'D'
        elif stdp=='nonlinear':
            pre = '''
            U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
            u_S += U_0 * (1 - u_S)
            r_S = u_S * x_S # released synaptic neurotransmitter resources
            x_S -= r_S
            Y_S += rho_c * Y_T * r_S
            '''
            post = '''
            F += (etap + eta*A)/second
            '''

    synapses = Synapses(source, target, eqs,
                        on_pre=pre,
                        on_post=post,
                        namespace=params,
                        name=name,
                        dt=dt)
    synapses.connect(connect)
    # Standard initalization
    synapses.x_S = 1.0

    # Case-specific initialization
    if stdp=='linear':
        synapses.C_syn = 0.0
        synapses.W = 'W_0'
        synapses.delay = delay
    elif stdp=='nonlinear':
        synapses.A = 0.0
        synapses.B = 0.0
        synapses.P = 0.0
        synapses.F = 0.0
        synapses.W = 'W_0'

    # Random initialization of initial conditions
    if ics=='rand':
        synapses.u_S = 'rand()'
        synapses.x_S = 'rand()'
        Y_T = params['Y_T']
        synapses.Y_S = '1.2 * rho_c * Y_T * rand()'

    return synapses

def gliotransmitter_release(astro, params, ics=None, standalone=False, N_astro=1, stimulus=None):
    # Add gliotransmitter release from the astrocyte "astro".

    if not standalone:
        eqs = '''
        # Gliotransmitter
        C : mole (linked)
        dx_A/dt = Omega_A * (1 - x_A) : 1  # Fraction of gliotransmitter resources available for release
        dG_A/dt = -Omega_e*G_A : mole  # gliotransmitter concentration in the extracellular space
        '''
        gliot_release = '''
        G_A += rho_e * G_T * U_A * x_A
        x_A -= U_A *  x_A
        '''
        threshold = 'C>C_Theta'
        refractory = 'C>C_Theta'
        N = len(astro)
    else:
        N = N_astro
        if stimulus!='poisson':
            eqs = '''
            # Gliotransmitter
            dv_A/dt = f_c*int(t <= t_off) : 1
            f_c : Hz
            dx_A/dt = Omega_A * (1 - x_A) : 1  # Fraction of gliotransmitter resources available for release
            dG_A/dt = -Omega_e*G_A : mole  # gliotransmitter concentration in the extracellular space
            '''
            gliot_release = '''
            G_A += rho_e * G_T * U_A * x_A
            x_A -= U_A *  x_A
            v_A = 0
            '''
            threshold = 'v_A>1'
            refractory = False
        else:
            eqs = '''
            # Gliotransmitter
            f_c : Hz
            dx_A/dt = Omega_A * (1 - x_A) : 1  # Fraction of gliotransmitter resources available for release
            dG_A/dt = -Omega_e*G_A : mole  # gliotransmitter concentration in the extracellular space
            '''
            gliot_release = '''
            G_A += rho_e * G_T * U_A * x_A
            x_A -= U_A *  x_A
            '''
            threshold = 'rand()<f_c*dt'
            refractory = False

    # Gliotransmitter release is modelled in the same fashion regardless of the standlone mode

    group = NeuronGroup(N, eqs,
                        # The following formulation makes sure that a "spike" is
                        # only triggered at the first threshold crossing
                        threshold=threshold,
                        refractory=refractory,
                        # The gliotransmitter release happens when the threshold
                        # is crossed, in Brian terms it can therefore be
                        # considered a "reset"
                        reset=gliot_release,
                        method='rk4',
                        name='gliot_release*',
                        namespace=params)

    # Assign initial conditions
    group.x_A = 1
    group.G_A = 0.0*mole
    if not standalone : group.C = linked_var(astro, 'C')

    # Random initialization of initial conditions
    if ics=='rand':
        synapses.x_A = 'rand()'
        synapses.G_A = '1.2 * rho_e * G_T * rand()'

    return group

def extracellular_syn_to_astro(synapse, astro, params, connect='astro_index_pre == j'):
    # Coupling from synapses to astrocytes
    # The default connection is all synapses to the same astrocyte
    extra = Synapses(synapse,astro,
                     model='''
                     # neurotransmitter concentration in the extracellular space
                     Y_extra_post = Y_S_pre : mole (summed)
                     ''',
                     namespace=params,
                     name="ecs_syn_to_astro*")
    extra.connect(connect)

    return extra

def extracellular_astro_to_syn(astro, synapse, params, connect=True, sic=None):
    # Coupling from astrocytes to synapses (Currently does not allow to have both pre- and post- synaptic modulation)
    # The default connection is the same astrocyte to all synapses impinging on it
    if not sic :
        extra = Synapses(astro,synapse,
                         model='''
                         # gliotransmitter concentration in the extracellular space
                         G_A_post = G_A_pre : mole (summed)
                         ''',
                         # One astrocyte per synapse
                         # connect=connect,
                         namespace=params,
                         name="ecs_astro_to_syn*"
                         )
    else :
        extra = Synapses(astro,synapse,
                         model='''
                         # gliotransmitter concentration in the extracellular space
                         G_A_sic_post = G_A_pre : mole (summed)
                         ''',
                         # One astrocyte per synapse
                         #connect=connect, # no longer supported
                         namespace=params,
                         name="ecs_astro_to_syn*"
                         )

    extra.connect(connect)

    # To allow custom connections the astro_index in the synapse module has been introduced
    return extra

def synapse_model(params, N_syn=1, connect='i==j', name='synapses*',
                  dt=1*msecond,
                  ics=None, linear=False,
                  post=None, sic=None, stdp=None,
                  stimulus=None, spikes=(0)*second, spikes_post=(0)*second, stim_opts={'f_in' : [0.0], 't_step': 0.0}):
    '''
    Tsodyks-Markram stand alone synapse (repeated by N_syn)

    params  : Dictionary
     Model parameters
    N_syn   : Integer
     Number of synapses
    connect : Condition String
     Connections between pre-post
    stimulus : {None} | String
     If None produce periodic firing.
     If 'test' create synapses stimulated by the same train of spikes in 'spike' | in the 'stdp' mode allows to specify
     per and post spike trains by input arguments: spikes and spikes_post respectively.
     If 'poisson' make source neurons Poissonian.
    ics     : String
     If 'rand' uses random initial conditions for synaptic variables
    linear  : Boolean

    return:
    source_group : Input NeuronGroup
    target_group : Output NeuronGroup
    synapses     : Synapse Group
    '''
    if not stdp:
        # No STDP included
        # Input and output neurons
        if stimulus=='test':
            indices = arange(N_syn).repeat(len(spikes))
            spikes  = spikes[None,:].repeat(N_syn,axis=0).flatten() # Repeat the spike train as many as N_syn
        elif stimulus=='poisson':
            source_eqs = '''
                         f_in : Hz
                         '''
            thr = 'rand()<f_in*dt'
        elif stimulus=='steps':
            # require specification of stim_opts dict
            # Function to generate string for f_in
            f_step = lambda f_in, t_step : '+'.join([str(dnu)+'*Hz *0.5*(1+sign(t-'+str(i*t_step)+'*second))' for i,dnu in enumerate(append([f_in[0]],diff(f_in)))])
            source_eqs = 'f_in = '+f_step(stim_opts['f_in'], stim_opts['t_step'])+' : Hz'
            thr = 'rand()<f_in*dt'
        else:
            source_eqs = '''
                         dv/dt = f_in : 1
                         f_in : Hz
                         '''
            thr = 'v>=1'

        # Presynaptic group
        if stimulus=='test':
            source_group = SpikeGeneratorGroup(N_syn,
                                               indices=indices,
                                               times=spikes,
                                               dt=dt)
        else:
            source_group = NeuronGroup(N_syn,
                                       model=source_eqs,
                                       threshold=thr,
                                       reset='v=0',
                                       namespace=params,
                                       dt=dt)
        # Postsynaptic group
        if not post :
            eqs_target = '''
            dv/dt = (E_L - v)/tau_m : volt
            '''
        else :
            if not sic :
                eqs_target = '''
                dv/dt = (g_e+(E_L - v))/tau_m : volt (unless refractory)
                g_e : volt
                '''
            else:
                eqs_target = '''
                dv/dt = (g_e + g_sic+(E_L - v))/tau_m : volt (unless refractory)
                g_e   : volt
                g_sic : volt
                '''

        target_group = NeuronGroup(N_syn,eqs_target,
                                   threshold='v>V_th',
                                   reset='v=V_r',
                                   refractory='tau_r',
                                   namespace=params,
                                   dt=dt)
        target_group.v = 'E_L'

    else:
        if stimulus=='test':
            # in the STDP mode, 'test' allows to specify timing of pre and post spikes (useful for ad-hoc simulations)
            # Pre
            indices_pre = arange(N_syn).repeat(len(spikes))
            spikes_pre  = spikes[None,:].repeat(N_syn,axis=0).flatten() # Repeat the spike train as many as N_syn
            source_group = SpikeGeneratorGroup(N_syn,
                                               indices=indices_pre,
                                               times=spikes_pre,
                                               name='neu_pre*',
                                               dt=dt)
            # Post
            indices_post = arange(N_syn).repeat(len(spikes_post))
            spikes_post  = spikes_post[None,:].repeat(N_syn,axis=0).flatten() # Repeat the spike train as many as N_syn
            target_group = SpikeGeneratorGroup(N_syn,
                                               indices=indices_post,
                                               times=spikes_post,
                                               name='neu_post*',
                                               dt=dt)
        else:
            eqs = '''
            dv/dt = f_in : 1 (unless refractory)
            f_in : Hz
            '''
            thr = 'v>=1'
            reset = 'v=0'
            source_group = NeuronGroup(N_syn,
                                       model=eqs,
                                       threshold=thr,
                                       reset=reset,
                                       name='neu_pre*',
                                       refractory='tau_r',
                                       namespace=params,
                                       dt=dt)
            target_group = NeuronGroup(N_syn,
                                       model=eqs,
                                       threshold=thr,
                                       reset=reset,
                                       name='neu_post*',
                                       refractory='tau_r',
                                       namespace=params,
                                       dt=dt)

    # Create synaptic connections
    synapses = synapse(source_group, target_group, params,
                       dt=dt,
                       linear=linear, connect=connect, ics=ics, name=name,
                       stimulus=stimulus,
                       postc=post, sic=sic, stdp=stdp)

    return source_group,target_group,synapses

def astro_with_syn_model(params, N_syn=1, N_astro=1, linear=False, stimulus=None, ics=None):
    # N_astro astrocytes, each stimulated by N_cell synapses

    # Input and output neurons and synapses
    source_group,target_group,synapses = synapse_model(params, N_syn*N_astro, linear=linear, stimulus=stimulus, ics=ics)

    # Add astrocyte
    astro = astrocyte_group(N_astro, params, ics=ics)

    # Connect the synapse to the astrocyte: make all synapses as inputs to the astrocyte
    synapses.astro_index = 'i%N_astro'
    es_syn2astro = extracellular_syn_to_astro(synapses, astro, params, connect='i % '+repr(N_astro)+' == j')

    return source_group,target_group,synapses,astro,es_syn2astro

def astrosyn_model(params, N_syn=1, N_astro=1, linear=False, stimulus=None, ics=None):
    # Build a simple astrocyte-modulated synapse

    # Input and output neurons and synapses
    source_group,target_group,synapses = synapse_model(params, N_syn*N_astro, linear=linear, stimulus=stimulus, ics=ics)

    # Add astrocyte
    astro = astrocyte_group(N_astro, params, ics=ics)

    # Connect the astrocyte with the synapses: this is done in 3 steps
    # 1: Make N_syn synapses as inputs to each astrocyte
    synapses.astro_index = 'i%N_astro'
    es_syn2astro = extracellular_syn_to_astro(synapses, astro, params, connect='i%'+repr(N_astro)+'==j')
    # 2: Generate gliotransmitter release
    gliot = gliotransmitter_release(astro, params, ics=ics)
    # 3: Make gliotransmission from an astrocyte modulate the N_syn synapses that impinge on it
    es_astro2syn = extracellular_astro_to_syn(gliot, synapses, params, connect='i==j%'+repr(N_astro))

    return source_group,target_group,synapses,astro,es_syn2astro,gliot,es_astro2syn

def openloop_model(params, N_pts=1, N_syn=1, N_astro=1,
                   dt=1*msecond,
                   linear=False,
                   connect='default',
                   stimulus_syn=None, stimulus_glt=None,
                   spikes_pre=(0)*second, spikes_post=(0)*second,
                   stim_opts={'f_in' : [0.0], 't_step': 0.0},
                   ics=None,
                   post=None, sic=None,
                   stdp=None,
                   stdp_protocol='pairs', Dt_min=-160*ms, Dt=320*ms):
    # Build a simple astrocyte-modulated synapse considering only an open loop configuration and using frequency of
    # gliotransmitter release as a third port of the system

    # Default value
    if connect=='default' : connect = 'i==j%'+repr(N_astro)
    if stdp_protocol=='test' : stimulus_syn=stdp_protocol

    # Input and output neurons and synapses
    source_group,target_group,synapses = synapse_model(params, N_pts*N_syn*N_astro,
                                                       dt=dt,
                                                       linear=linear, ics=ics,
                                                       stimulus=stimulus_syn,
                                                       stim_opts=stim_opts,
                                                       spikes=spikes_pre, spikes_post=spikes_post,
                                                       post=post, sic=sic, stdp=stdp)

    if stdp:
        if stdp_protocol=='pairs':
            # Initialize firing (standard )
            source_group.f_in = params['f_in']
            target_group.f_in = params['f_in']
            source_group.v = 'int( (Dt_min+ i/N_astro/N_syn *Dt/(N_pts-1))>0*second ) + (1 + f_in*( Dt_min+ i/N_astro/N_syn *Dt/(N_pts-1)) )*int( (Dt_min+ i/N_astro/N_syn*Dt/(N_pts-1))<=0*second )'
            target_group.v = 'int( (Dt_min+ i/N_astro/N_syn *Dt/(N_pts-1))<=0*second ) + (1 - f_in*( Dt_min+ i/N_astro/N_syn *Dt/(N_pts-1)) )*int( (Dt_min+ i/N_astro/N_syn*Dt/(N_pts-1))>0*second )'

    # Add Gliotransmitter release (currently assumed only periodic)
    gliot = gliotransmitter_release(None, params, ics=ics, N_astro=N_astro, standalone=True, stimulus=stimulus_glt)

    # Make gliotransmission from an astrocyte modulate the N_syn synapses
    es_astro2syn = extracellular_astro_to_syn(gliot, synapses, params, connect=connect, sic=sic)

    return source_group,target_group,synapses,gliot,es_astro2syn