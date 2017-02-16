"""
astrocyte_simulations.py

Simulations and methods to build figures of the book chapter <ADD REF>
"""

# TODO: Add reference to book chapter in the headlines of all the files
# TODO: Add delays in the astro-only network (currently not implemented in BRIAN)
# TODO: (optional) Create stimulus as a pulsed wave in the simulation of astrocyte networks

from brian2 import *
from brian2.units.allunits import mole, umole, mmole

# Optional settings for faster compilations and execution by Brian
prefs.codegen.cpp.extra_compile_args = ['-Ofast', '-march=native']

# User-defined modules
import astrocyte_models as asmod
import analysis
import figures_template as figtem

# Temporary modules (uncomment)
import os, sys
# sys.path.append(os.path.join(os.path.expanduser('~'),'Dropbox/Ongoing.Projects/pycustommodules'))
# import save_utils as svu

def simulate(sim_id):
    # Basic interface to select different simulations (does not allow to modify simulations)
    return {'test'            : astrosyn_test,
            'gchi_astrocyte'  : gchi_astrocyte,
            'bifdiag'         : astro_bifdiag,
            'bifdiag_syn'     : astro_with_syn_bifdiag,
            'io_synapse'      : io_synapse,
            'COBA_with_astro' : COBA_with_astro,
            'COBA_with_astro_alternative' : COBA_with_astro_alternative, # Temporary
            'ring'            : astrocyte_ring,
            '2d-network'      : astrocyte_network_2d,
            'stdp'            : tripartite_stdp
            }.get(sim_id)()

def astrosyn_test():
    """
    Simulate a simple astrocyte-modulated synapse
    """

    # # This should give a great performance boost but does currently not work
    # # with synapses targeting synapses (which we need for our extra-cellular
    # # space)
    # set_device('cpp_standalone_simple')

    # params = asmod.get_parameters('AM', 'facilitating')
    params = asmod.get_parameters('FM', 'facilitating')
    source_group,target_group,synapses,astro,es_syn2astro,gliot,es_astro2syn = asmod.astrosyn_model(params)

    # Set up monitors
    mon = StateMonitor(astro, variables=['C'], record=0)
    syn_mon = StateMonitor(synapses, variables=['Y_S'], record=0)
    spikes = SpikeMonitor(source_group)

    # Run the simulation
    duration = 60*second
    run(duration, namespace={}, report='text')

    # Plots
    fig, ax = plt.subplots(2,1)
    ax[0].plot(mon.t/second, mon[0].C/umole, lw=2)
    ax[0].plot(mon.t/second, ones(len(mon.t))*params['C_Theta']/umole, 'r--', lw=2)
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('$C$ (${0}$)'.format(sympy.latex(umole)))

    # TODO: this is not shown correctly
    ax[1].plot(syn_mon.t/second, syn_mon[0].Y_S/umole, lw=2)
    ax[1].plot(spikes.t/second, np.zeros(len(spikes.t)), 'rx', clip_on=False)
    ax[1].set_xlim([0, syn_mon.t[-1]/second])
    ax[1].set_xlabel('Time (${0}$)'.format(sympy.latex(second)))
    ax[1].set_ylabel('$Y_S$ (${0}$)'.format(sympy.latex(umole)))

    plt.show()

def gchi_astrocyte():
    """
    Simulate a simple GChI astrocyte
    """
    # TODO: might be run faster if dt is increased -- but require dt as input argument in modules w/in analysis...

    # Generate two astrocytes w/ different Ca2+ oscillations parameter sets
    # params = asmod.get_parameters('AM')
    # params['Y_T'] /= 2.0
    # in_am,out_am,syn_am,astro_am,ecs_am = asmod.astro_with_syn_model(params, linear=True)
    in_am,out_am,syn_am,astro_am,ecs_am = asmod.astro_with_syn_model(asmod.get_parameters('AM'), linear=True)
    in_fm,out_fm,syn_fm,astro_fm,ecs_fm = asmod.astro_with_syn_model(asmod.get_parameters('FM'), linear=True)

    # Define input
    f_in = 0.1*Hz
    in_am.f_in = f_in
    in_fm.f_in = f_in

    # Set up monitors
    mon_am = StateMonitor(astro_am, variables=['Gamma_A','C','h','I'], record=True)
    mon_fm = StateMonitor(astro_fm, variables=['Gamma_A','C','h','I'], record=True)

    # Run the simulation
    duration = 120*second
    run(duration, namespace={}, report='text')

    # Plots
    # AM
    ax = analysis.plot_astro_signals(mon_am, ls='-', color='g')
    # Superimpose FM
    analysis.plot_astro_signals(mon_fm, ax=ax, ls='-')


    plt.show()

def astro_bifdiag():
    # Plot bifurcation diagrams for the G-ChI model.
    # Bifurcation diagrams can be thought as I/O characteristics of an astrocyte
    # as a function of an input parameter, e.g. the extracellular neurotransmitter
    # concentration Y_bias

    # Number of different values for the bifurcation parameter
    N_astro = 100
    N_trials = 10

    #-------------------------------------------------------------------------------------------------------------------
    # AM parameter set
    #-------------------------------------------------------------------------------------------------------------------
    params = asmod.get_parameters('AM')
    params['F'] = 0.0*umole/second
    params['O_N'] = 0.1/umole/second
    params['O_beta'] = 1.5*umole/second
    # Alternative parameter set
    # params['O_3K'] = 4.0*umole/second

    # Create astrocyte group
    astro = asmod.astrocyte_group(N_astro*N_trials, params, 10*msecond)

    # Run continuation problem
    analysis.build_bifdiag(astro,N_trials,params,
                           astro,'Y_bias',[0.1,20.0],umole,
                           1000*second,100*second,5,True)

    #-------------------------------------------------------------------------------------------------------------------
    # FM parameter set
    #-------------------------------------------------------------------------------------------------------------------
    params = asmod.get_parameters('FM')
    params['F'] = 0.0*umole/second
    params['O_N'] = 0.1/umole/second
    params['O_beta'] = 1.5*umole/second
    # Alternative parameter set
    # params['O_3K'] = 4.5*umole/second

    # Create astrocyte group
    astro = asmod.astrocyte_group(N_astro*N_trials, params, 10*msecond)

    # Run continuation problem
    ## (BUG?) Interesting: if I change the max of the range to >1000 umole it gives me a RuntimeWarning and it does not
    ## run simulation correctly, not considering points >1000 umole (have not checked it why in details...)
    analysis.build_bifdiag(astro,N_trials,params,
                           astro,'Y_bias',[0.1,1000],umole,
                           1000*second,100*second,5,True)

def astro_with_syn_bifdiag():
    # Plot bifurcation diagrams for the G-ChI model coupled with N_syn Tsodyks-Markarm synapses.
    # Compare the bifurcation diagrams for Ca2+ in the astrocyte as a function of the synaptic input spike rate both
    # w/out and w/ gliotransmission.

    # TODO: WARNING: Could not run simulations with higher number of synapses (needs lot of memory not available on my lap)

    # Model size
    N_syn = 10
    N_astro = 50
    N_trials = 10

    # Continuation parameters
    duration = 300*second
    transient = 50*second

    params = asmod.get_parameters('FM','depressing')
    # params = asmod.get_parameters('FM','facilitating')
    # Rescale synaptically released neurotransmitter so as the sum of all contributions to the astrocyte if ~O(Y_T)
    params['Y_T'] /= N_syn*params['U_0__star']
    params['F'] = 0.0*umole/second
    params['O_N'] = 0.1/umole/second
    params['O_beta'] = 1.5*umole/second

    # Build model of N_syn on one astrocyte (w/out gliotransmission)
    neuron_in,_,_,astro,_ = asmod.astro_with_syn_model(params,
                                                       N_syn=N_syn,
                                                       N_astro=N_astro*N_trials,
                                                       ics='rand')
    # Run bifurcation analysis
    analysis.build_bifdiag(astro,N_trials,params,
                           neuron_in,'f_in',[0.1,100],Hz,
                           duration,transient,5,True)

    # Build model of N_syn on one astrocyte (w/ gliotransmission)
    neuron_in,_,_,astro,_,_,es_astro2syn = asmod.astrosyn_model(params,
                                                                N_syn=N_syn,
                                                                N_astro=N_astro*N_trials,
                                                                ics='rand')
    # Run bifurcation analysis
    analysis.build_bifdiag(astro,N_trials,params,
                           neuron_in,'f_in',[0.1,100],Hz,
                           duration,transient,5,True)

def astrocyte_ring():
    '''
    Simulate a ring of N astrocytes
    '''

#    params = get_parameters('AM', 'facilitating')
    params = asmod.get_parameters('FM', 'facilitating')
    # Will make synapses essentially istantaneous

    # Create a synaptic connection from an input neuron to an output one
    source_group = NeuronGroup(1, 'dv/dt = f_in : 1', threshold='v>=1',
                               reset='v=0', namespace=params)
    target_group = NeuronGroup(1, 'dv/dt = (E_L - v)/tau_m : volt',
                               threshold='v>V_th',
                               reset='v=V_r', namespace=params)
    # No short-term plasticity is taken into account
    synapses = asmod.synapse(source_group, target_group, params, linear=True)
    synapses.connect(True)
    synapses.x_S = 1

    # Create astrocyte network
    N_cells = 10
    astro = asmod.astrocyte_group(N_cells, params, dt=1*msecond)

    # Specify connections
    astro_conn = asmod.astro_astro_network(astro, params)
    asmod.spatial_connections(astro,astro_conn,
                              connect='ring',
                              radius=120*umeter,
                              speed_propagation=50*umeter/second)

    # By default the synapse will impinge only on astrocyte '0'
    extra = asmod.extracellular_syn_to_astro(synapses, astro, params)

    # Set up monitors
    mon = StateMonitor(astro, variables=['C', 'I'], record=True, dt=0.1*second)

    # Run the simulation
    duration = 60*second
    run(duration, namespace={}, report='text')

    # Save data
    # TODO: final version should use native Brian methods and then reload data to build figures separately
    # svu.savedata([mon.t_[:],mon.C_[:],mon.I_[:],astro.x_[:],astro.y_[:],astro_conn.i[:],astro_conn.j[:]],'tmp_ring.pkl')

    # Plot connections
    analysis.plot_connections(model_group=astro,connections=astro_conn)

    # Plot dynamics (C)
    analysis.plot_traces(mon.t_[:],mon.C_[:],'C')

    # Plot dynamics (I)
    analysis.plot_traces(mon.t_[:],mon.I_[:],'I')

    # Build final figure
    # TODO: proper time sampling and optimal sizing for best visualization will be finalized at submission time
    analysis.plot_pseudo_df_f(mon.t_[:],mon.C_[:],
                              astro.x_[:],astro.y_[:],
                              astro_conn.i[:],astro_conn.j[:],
                              time_samples=arange(0, duration/second + 1, 15),
                              img_size=800*umeter)

    plt.show()

def astrocyte_network_2d():
    '''
    Simulate a 2D network of nr*nc astrocytes
    '''

#    params = get_parameters('AM', 'facilitating')
    params = asmod.get_parameters('FM', 'facilitating')

    # The stimulus need to be changed by a pulsed wave
    # Create a synaptic connection from an input neuron to an output one
    source_group = NeuronGroup(1, 'dv/dt = f_in : 1', threshold='v>=1',
                               reset='v=0', namespace=params)
    target_group = NeuronGroup(1, 'dv/dt = (E_L - v)/tau_m : volt',
                               threshold='v>V_th',
                               reset='v=V_r', namespace=params)
    # No short-term plasticity is taken into account
    synapses = asmod.synapse(source_group, target_group, params, linear=True)
    synapses.connect(True)
    synapses.x_S = 1

    # Create astrocyte network
    N_rows = 10
    N_cols = 10
    astro = asmod.astrocyte_group(N_rows*N_cols, params, dt=1*msecond)

    # Specify connections
    astro_conn = asmod.astro_astro_network(astro, params)
    asmod.spatial_connections(astro,astro_conn,connect='lattice_stoch',
                              grid_dist=60*umeter, distance=50*umeter,
                              rows=N_rows, cols=N_cols)

    # By default the synapse will impinge only on astrocyte '0'
    extra = asmod.extracellular_syn_to_astro(synapses, astro, params)

    # Set up monitors
    mon = StateMonitor(astro, variables=['C', 'I'], record=True,dt=0.1*second)

    # Run the simulation
    duration = 60*second
    run(duration, namespace={}, report='text')

    # Save data
    # TODO: final version should use native Brian methods and then reload data to build figures separately
    # svu.savedata([mon.t_[:],mon.C_[:],mon.I_[:],astro.x_[:],astro.y_[:],astro_conn.i[:],astro_conn.j[:]],'tmp_lattice.pkl')

    # Plot connections
    analysis.plot_connections(model_group=astro,connections=astro_conn)

    # Build final figure
    # TODO: proper time sampling and optimal sizing for best visualization will be finalized at submission time
    analysis.plot_pseudo_df_f(mon.t_[:],mon.C_[:],
                              astro.x_[:],astro.y_[:],
                              astro_conn.i[:],astro_conn.j[:],
                              time_samples=arange(0, duration/second + 1, 15),
                              img_size=650*umeter)

    plt.show()

def tripartite_syn(syn_type='depressing'):
    '''
    Show I/O characteristics in terms of released neurotransmitter resources of a synapse both w/out and w/ gliotransmission
    '''

    if syn_type=='depressing':
        params = asmod.get_parameters('FM',synapse='depressing')
        # params['xi'] = 0
    else:
        params = asmod.get_parameters('FM',synapse='facilitating')
        # params['xi'] = 1

    #-------------------------------------------------------------------------------------------------------------------
    # Run simulation at fixed input for synapse w/ and w/out the astrocyte
    #-------------------------------------------------------------------------------------------------------------------
    # We want to compare synaptic transmission w/out vs. w/ gliotransmission (open-loop configuration). For this reason
    # we consider two testing synapses (test_syn) whose only one receives gliotransmitter from a sorrounding astrocyte.
    # The astrocyte is instead stimulated independently. Just to create an event of neurotransmitter release.
    # There are possible different solutions to implement this system. The most compact one is to create an astrocyte w/
    # N_syn stimulating synapses. And the on the side create two independent synapses stimulated by a testing stimulus,
    # whose only one is affected by gliotransmission.

    # # Synapse w/out gliotransmission
    duration = 17.5*second
    # duration = 0.1*second
    f_in = 5*Hz

    # Testing synapses
    offset = 10*msecond
    transient = 16.5*second
    spikes = concatenate((arange(0.0,200,50.0),arange(300.0,390,10.0),(450.0,)))*msecond
    spikes += transient + offset
    # Even if test_out will not be used, to properly create the model, it must be explicitly assigned and made available
    # w/in the local scope
    test_in, test_out, syn_test = asmod.synapse_model(params, N_syn=2, connect='i==j',
                                                      stimulus='test',
                                                      spikes = spikes,
                                                      name='testing_synapse*')

    # Stimulated astrocyte
    neuron_stim, neuron_out, syn_stim, astro, ecs_syn = asmod.astro_with_syn_model(params, N_syn=1, N_astro=1, ics=None)
    neuron_stim.f_in = f_in

    # Gliotransmission
    gliot = asmod.gliotransmitter_release(astro, params)

    # Connect the astrocyte to one of the testing synapses (synapse 0)
    ecs_astro = asmod.extracellular_astro_to_syn(gliot, syn_test, params, connect='i==j')

    # Create monitors
    mon = StateMonitor(syn_test, variables=['u_S','x_S','r_S','Y_S'], record=True)

    run(duration,namespace={},report='text')
    # svu.savedata([mon.t_[:],mon.u_S_[:],mon.x_S_[:],mon.r_S_[:],mon.Y_S_[:]],'tmp_iosyn.pkl')

    # Retrieve figure template
    fig, ax = figtem.generate_figure('5x1_4S1L',figsize=(6.5,7.5),left=0.12,bottom=0.1)
    tlims = array([16.0,17.5]) # xlim

    # Plot spikes
    analysis.plot_release(mon.t_, mon[0].r_S_, var='spk',  ax=ax[0], redraw=True)
    ax[0].set_xlim(tlims)

    # Add u_S
    analysis.plot_release(mon.t_, mon[1].u_S_, var='u', tau=1.0/params['Omega_f'], ax=ax[1], color='r')
    analysis.plot_release(mon.t_, mon[0].u_S_, var='u', tau=1.0/params['Omega_f'], ax=ax[1], color='k', redraw=True, spine_position=5)
    ax[1].set_xlim(tlims)
    ax[1].set_xlabel('')

    # Add x_S
    analysis.plot_release(mon.t_, mon[1].x_S_, var='x', tau=1.0/params['Omega_d'], ax=ax[2], color='r')
    analysis.plot_release(mon.t_, mon[0].x_S_, var='x', tau=1.0/params['Omega_d'], ax=ax[2], color='k', redraw=True, spine_position=5)
    ax[2].set_xlim(tlims)
    ax[2].set_xlabel('')

    # Add Y_S
    analysis.plot_release(mon.t_-tlims[0], mon[1].Y_S / mmole, var='y', var_units=mmole, ax=ax[3], color='r')
    analysis.plot_release(mon.t_-tlims[0], mon[0].Y_S / mmole, var='y', var_units=mmole, ax=ax[3], color='k', redraw=True, spine_position=5)
    ax[3].set_xlim(tlims-tlims[0])
    ax[3].set_xticks(arange(0.0,1.6,0.5))
    ax[3].set_ylim([0.0,2.5])
    ax[3].set_yticks(arange(0.0,2.5,1.0))

    # # # TODO: remove (Reload data and test plots from them)
    # # t,u_S,x_S,r_S,Y_S = svu.loaddata('tmp_iosyn.pkl')
    # #
    # # # Retrieve figure template
    # # fig, ax = figtem.generate_figure('5x1_4S1L',figsize=(6.5,7.5),left=0.12,bottom=0.1)
    # # tlims = array([16.0,17.5]) # xlim
    # #
    # # # Plot spikes
    # # # analysis.plot_release(t, r_S[0], var='spk',  ax=ax[0], redraw=True)
    # # # ax[0].set_xlim(tlims)
    # #
    # # # Add u_S
    # # # analysis.plot_release(t, u_S[1], var='u', tau=1.0/params['Omega_f'], ax=ax[1], color='r')
    # # # analysis.plot_release(t, u_S[0], var='u', tau=1.0/params['Omega_f'], ax=ax[1], color='k', redraw=True, spine_position=5)
    # # # ax[1].set_xlim(tlims)
    # # # ax[1].set_xlabel('')
    # #
    # # # Add x_S
    # # # analysis.plot_release(t, x_S[1], var='x', tau=1.0/params['Omega_d'], ax=ax[2], color='r')
    # # # analysis.plot_release(t, x_S[0], var='x', tau=1.0/params['Omega_d'], ax=ax[2], color='k', redraw=True, spine_position=5)
    # # # ax[2].set_xlim(tlims)
    # # # ax[2].set_xlabel('')
    # #
    # # # Add Y_S
    # # # analysis.plot_release(t-tlims[0], Y_S[1], var='y', ax=ax[3], color='r')
    # # # analysis.plot_release(t-tlims[0], Y_S[0], var='y', ax=ax[3], color='k', redraw=True, spine_position=5)
    # # # ax[3].set_xlim(tlims-tlims[0])
    # # # # ax.set_ylim([0.0,1.0])
    # # # # ax.set_yticks([0.0,0.5,1.0])

    #-------------------------------------------------------------------------------------------------------------------
    # Mean-field r_S
    #-------------------------------------------------------------------------------------------------------------------
    # TODO: cannot understand why the errorbars are so small, when they used to be quite large...
    # Model size
    N_points = 20
    N_trials = 10

    # Simulation parameters
    duration = 60*second

    # NOTE: rr_mean invokes astrocyte_models: perhaps it would be better to build them outside rr_mean and then pass the
    # relevant modules (neuron_in and synapse) to it avoiding importing of astrocyte_models?
    # A faster implementation could be by running both continuations in parallel but requires more memory.

    # w/out gliotransmission
    analysis.rr_mean(params, N_points, N_trials, freq_range=[0.1,100], gliot=False,
                     duration=duration, color='k', fmt = 'o', ax=ax[4])

    # w/ gliotransmission (and superimposing on previous case)
    analysis.rr_mean(params, N_points, N_trials, freq_range=[0.1,100], gliot=True,
                     duration=duration, color='r', fmt = 'o', ax=ax[4])

    # Show plot
    plt.show()

def io_synapse():
    # Simulate and create figures for the tripartite synapse
    tripartite_syn('depressing')
    tripartite_syn('facilitating')

def COBA_with_astro():
    '''
    Simulate a COBA network in three cases:
    1. with ideal (instantaneous) synapses
    2. with Tsodyks-Markram synapses
    3. Together with an astrocyte network and gliotransmission

    The effort is to recode the COBA network by using as much as possible the modules that we have defined in
    astrocyte_models.py
    '''

    # TODO: To be replaced by "alternative" if this latter will be made to work

    # Network size
    N_n = 4000  # Total number of neurons in the network
    N_e = 3200  # Number of excitatory neurons
    N_a = 3600  # Number of astrocytes in the network

    # Retrieve model parameters
    params = asmod.get_parameters('FM', 'facilitating')
    params['E_l'] = -49*mV
    params['V_t'] = -50*mV
    params['V_r'] = -60*mV
    # We add we and wi to the namespace so that they can used in the pre string
    params['we'] = (60*0.27/10)*mV / (params['rho_c']*params['Y_T']) # excitatory synaptic weight (voltage)
    params['wi'] = (-20*4.5/10)*mV / (params['rho_c']*params['Y_T']) # inhibitory synaptic weight

    # Create neurons for three networks
    N0 = asmod.COBA_neuron_model(params, N_neu=N_n)
    N1 = asmod.COBA_neuron_model(params, N_neu=N_n)
    N2 = asmod.COBA_neuron_model(params, N_neu=N_n)

    #-------------------------------------------------------------------------------------------------------------------
    # Create COBA network w/ ideal (linear) synapses
    #-------------------------------------------------------------------------------------------------------------------
    # Create synapses
    S0_e = asmod.synapse(N0, N0, params, connect=False, name='exc_ideal', linear=True)
    S0_i = asmod.synapse(N0, N0, params, connect=False, name='inh_ideal', linear=True)

    # Extend the pre-code of the synapse to act on the post-synaptic ge/gi
    S0_e.pre.code += 'ge_post += Y_S*we'
    S0_i.pre.code += 'gi_post += Y_S*wi'

    # Create connections
    S0_e.connect('i<'+repr(N_e), p=0.02)
    S0_i.connect('i>='+repr(N_e), p=0.02)

    #-------------------------------------------------------------------------------------------------------------------
    # Create COBA network w/ TM synapses
    #-------------------------------------------------------------------------------------------------------------------
    # Create synapses
    S1_e = asmod.synapse(N1, N1, params, connect=False, name='exc_stp', linear=False)
    S1_i = asmod.synapse(N1, N1, params, connect=False, name='inh_stp', linear=True)

    # Extend the pre-code of the synapse to act on the post-synaptic ge/gi
    S1_e.pre.code += 'ge_post += Y_S*we'
    S1_i.pre.code += 'gi_post += Y_S*wi'

    # Create connections
    S1_e.connect(S0_e.i, post=S0_e.j)
    S1_i.connect(S0_i.i, post=S0_i.j)

    #-------------------------------------------------------------------------------------------------------------------
    # Create COBA network w/ TM synapses and gliotransmission
    #-------------------------------------------------------------------------------------------------------------------
    # Create synapses
    S2_e = asmod.synapse(N2, N2, params, connect=False, name='exc_gliot', linear=False)
    S2_i = asmod.synapse(N2, N2, params, connect=False, name='inh_gliot', linear=True)

    # Extend the pre-code of the synapse to act on the post-synaptic ge/gi
    S2_e.pre.code += 'ge_post += Y_S*we'
    S2_i.pre.code += 'gi_post += Y_S*wi'

    # Create connections
    S2_e.connect(S0_e.i, post=S0_e.j)
    S2_i.connect(S0_i.i, post=S0_i.j)
    S2_e.astro_index = 'i%N_a'

    # Create astrocyte network
    A = asmod.astrocyte_group(N_a, params, dt=0.5*msecond, ics=True)
    A.C = 'C_Theta + rand()*umole'    # Make sure that initially, all astrocytes "fire", i.e. release gliotransmitter

    # Specify connections
    N_rows = int(sqrt(N_a))
    N_cols = N_a/N_rows
    C = asmod.astro_astro_network(A, params)
    asmod.spatial_connections(A, C, connect='lattice_stoch',
                              grid_dist=60*umeter, distance=50*umeter, rows=N_rows, cols=N_cols)

    # Add gliotransmission
    G = asmod.gliotransmitter_release(A, params, ics=None)

    # Connect excitatory synapses to astrocytes
    ES = asmod.extracellular_syn_to_astro(S2_e, A, params, connect='astro_index_pre == j')

    # Connect gliotransmission to excitatory synapses in a homosynaptic fashion
    EA = asmod.extracellular_astro_to_syn(G, S2_e, params, connect='i==astro_index_post')

    #-------------------------------------------------------------------------------------------------------------------
    # Generate monitors
    #-------------------------------------------------------------------------------------------------------------------
    mon_s0 = SpikeMonitor(N0)
    mon_s1 = SpikeMonitor(N1)
    mon_s2 = SpikeMonitor(N2)

    # Run simulation (all 3 networks in parallel)
    run(1 * second)

    # Plot results
    analysis.plot_raster(mon_s0.t/ms, mon_s0.i, N_e=N_e)
    analysis.plot_raster(mon_s1.t/ms, mon_s1.i, N_e=N_e)
    analysis.plot_raster(mon_s2.t/ms, mon_s2.i, N_e=N_e)

    # Show plot
    plt.show()

def COBA_with_astro_alternative():
    '''
    Simulate a COBA network in three cases:
    1. with ideal (instantaneous) synapses
    2. with Tsodyks-Markram synapses
    3. Together with an astrocyte network and gliotransmission

    The effort is to recode the COBA network by using as much as possible the modules that we have defined in
    astrocyte_models.py
    '''

    # Retrieve model parameters
    params = asmod.get_parameters('FM', 'facilitating')
    # Network size
    params['N_n'] = N_n = 4000  # Total number of neurons in the network
    params['N_e'] = N_e = 3200  # Number of excitatory neurons
    params['N_a'] = N_a = 3600  # Number of astrocytes in the network
    params['E_l'] = -49*mV
    params['V_t'] = -50*mV
    params['V_r'] = -60*mV
    # We add we and wi to the namespace so that they can used in the pre string
    params['we'] = (60*0.27/10)*mV / (params['rho_c']*params['Y_T']) # excitatory synaptic weight (voltage)
    params['wi'] = (-20*4.5/10)*mV / (params['rho_c']*params['Y_T']) # inhibitory synaptic weight

    # Create three network objects
    N0,S0_e,S0_i = asmod.COBA_network(params, N_neu=N_n, linear=True)
    N1,S1_e,S1_i = asmod.COBA_network(params, N_neu=N_n, linear=False)
    N2,S2_e,S2_i = asmod.COBA_network(params, N_neu=N_n, linear=False)

    #-------------------------------------------------------------------------------------------------------------------
    # Define COBA network w/ ideal (linear) synapses
    #-------------------------------------------------------------------------------------------------------------------
    # Create connections
    S0_e.connect('i<N_e', p=0.02)
    S0_i.connect('i>=N_e', p=0.02)
    #-------------------------------------------------------------------------------------------------------------------
    # Define COBA network w/ TM synapses
    #-------------------------------------------------------------------------------------------------------------------
    # Create connections
    S1_e.connect(S0_e.i, post=S0_e.j)
    S1_i.connect(S0_i.i, post=S0_i.j)

    #-------------------------------------------------------------------------------------------------------------------
    # Define COBA network w/ TM synapses and gliotransmission
    #-------------------------------------------------------------------------------------------------------------------
    # Create connections
    S2_e.connect(S0_e.i, post=S0_e.j)
    S2_i.connect(S0_i.i, post=S0_i.j)
    S2_e.astro_index = 'i%N_a'

    # Create astrocyte network
    A = asmod.astrocyte_group(N_a, params, dt=0.5*msecond, ics=True)
    A.C = 'C_Theta + rand()*umole'    # Make sure that initially, all astrocytes "fire", i.e. release gliotransmitter

    # Specify connections
    N_rows = int(sqrt(N_a))
    N_cols = N_a/N_rows
    C = asmod.astro_astro_network(A, params)
    asmod.spatial_connections(A, C, connect='lattice_stoch',
                              grid_dist=60*umeter, distance=50*umeter, rows=N_rows, cols=N_cols)

    # Add gliotransmission
    G = asmod.gliotransmitter_release(A, params, ics=None)

    # Connect excitatory synapses to astrocytes
    ES = asmod.extracellular_syn_to_astro(S2_e, A, params, connect='astro_index_pre == j')

    # Connect gliotransmission to excitatory synapses in a homosynaptic fashion
    EA = asmod.extracellular_astro_to_syn(G, S2_e, params, connect='i==astro_index_post')

    #-------------------------------------------------------------------------------------------------------------------
    # Generate monitors
    #-------------------------------------------------------------------------------------------------------------------
    mon_s0 = SpikeMonitor(N0)
    mon_s1 = SpikeMonitor(N1)
    mon_s2 = SpikeMonitor(N2)

    # Run simulation
    run(1 * second, report='text', namespace={})

    # Plot results
    analysis.plot_raster(mon_s0.t/ms, mon_s0.i, N_e=N_e)
    analysis.plot_raster(mon_s1.t/ms, mon_s1.i, N_e=N_e)
    analysis.plot_raster(mon_s2.t/ms, mon_s2.i, N_e=N_e)

    # Show plot
    plt.show()

if __name__ == '__main__':
    # simulate('test')
    # simulate('gchi_astrocyte')
    # simulate('bifdiag')
    # simulate('bifdiag_syn')
    # simulate('ring')
    # simulate('2d-network')
    simulate('io_synapse')
    # simulate('COBA_with_astro')
    # simulate('COBA_with_astro_alternative')
    # simulate('stdp')
    # print profiling_summary()

