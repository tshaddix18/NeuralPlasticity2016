"""
analysis.py

Methods for analysis of results and building simulation-specific Figures.
- Plotting of astrocyte signal
- Bifurcation diagrams
- Plotting of released resources and synaptic I/O characteristics
- Pseudo-fluorescence Figures for Ca2+ signal propagation in astrocyte networks

v2.0 Maurizio De Pitta, Basque Center for Mathematics, 2020
Translated and Debugged to run in Python 3.x

v1.0 Maurizio De Pitta, The University of Chicago, 2015
"""

from brian2 import *
from brian2.units.allunits import mole, umole, mmole

import sympy
from scipy.signal import convolve2d

# User-defined modules
import astrocyte_models as asmod
import figures_template as figtem

import matplotlib.patches as ptc

#-----------------------------------------------------------------------------------------------------------------------
# Astrocyte plotting
#-----------------------------------------------------------------------------------------------------------------------
def plot_astro_signals(monitor, time_units=second, Gamma_units=1, C_units=umole, I_units=umole, h_units=1, cell=0,
                       ax=None, color='k', ls='-', lw=2, spine_position=5):
    """
    Plot GChI astrocyte traces.

    monitor  :  Object
     StateMonitor w/ Gamma_A, C, h, I variables
    ax       : Handle
     4x1 Axis handle
    color    :  String or Array
     Color of traces
    ls       :  String
     Line style
    lw       :  Float
     Line width
    spine_position : Integer
     Number of points for spines location
    """

    # Ad-hoc formatting of ylabel
    setyl = lambda ax, var_name, var_units : ax.set_ylabel('${0}$ (${1}$)'.format(var_name, sympy.latex(var_units))) if type(var_units)!=type(1) else ax.set_ylabel('${0}$'.format(var_name))

    # Generate figure if not given
    if not ax : _, ax = figtem.generate_figure('4x1',left=0.15)

    # Plot Gamma_A
    ax[0].plot(monitor.t/time_units, monitor[cell].Gamma_A/Gamma_units, color=color, ls=ls, lw=lw)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_ylim([0.0,1.02])
    ax[0].set_yticks([0.0,0.5,1.0])
    setyl(ax[0],'\Gamma_{A}',Gamma_units)

    # Plot I
    ax[1].plot(monitor.t/time_units, monitor[cell].I/I_units, color=color, ls=ls, lw=lw)
    figtem.adjust_spines(ax[1], ['left'], position=spine_position)
    ax[1].set_ylim([0.0,2.5])
    ax[1].set_yticks(arange(0.0,3.1,0.7))
    setyl(ax[1],'I',I_units)

    # Plot C
    ax[2].plot(monitor.t/time_units, monitor[cell].C/C_units, color=color, ls=ls, lw=lw)
    figtem.adjust_spines(ax[2], ['left'], position=spine_position)
    ax[2].set_ylim([0.0,1.2])
    ax[2].set_yticks(arange(0.0,1.6,0.5))
    setyl(ax[2],'C',C_units)

    # Plot h
    ax[3].plot(monitor.t/time_units, monitor[cell].h/h_units, color=color, ls=ls, lw=lw)
    figtem.adjust_spines(ax[3], ['left','bottom'], position=spine_position)
    ax[3].set_xlabel('Time (${0}$)'.format(sympy.latex(time_units)))
    ax[3].set_ylim([0.0,1.0])
    ax[3].set_yticks([0.0,0.5,1.02])
    setyl(ax[3],'h',h_units)

    return ax

#-----------------------------------------------------------------------------------------------------------------------
# Bifurcation analysis
#-----------------------------------------------------------------------------------------------------------------------
# def build_bifdiag(astro_group,N_trials,params,bifpar_name,bifpar_range,bifpar_units,
#                   duration,transient,osc_threshold,log_scale=False):
def build_bifdiag(astro_group,N_trials,params,
                  bifpar_group,bifpar_name,bifpar_range,bifpar_units,
                  duration,transient,osc_threshold,log_scale=False):
    '''
    astro_group   : Group
      Astrocyte_group of N components
    N_trials      : Integer
      Number of computations for the same value of the bifurcation parameter starting from random initial conditions
    params        : Dictionary
      Parameters of the astro_group
    bifpar_group  : Group
      The group where the bifurcation parameter explicitly appears in the equations as an attribute
    bifpar_name   : Str
      String of bifurcation parameter name
    bifpar_range  : Quantity
      1D 2-element array for the range of variation of the bifurcation parameter
    bifpar_units  : Units
      The units of the bifurcation parameter
    duration      : Quantity
      Time duration (with units) of the simulation for continuation (must be sufficiently large)
    transient     : Quantity
      Time transient (with units) to drop from simulation (it must be transient < duration)
    osc_threshold : Integer
      Minimum number of Ca2+ peaks to assume oscillations
    log_scale     : Bool
      If True, logarithmic spacing in defining the values of the bifurcation parameter is used.
    '''

    # A simple function to make bifpar_name plottable with correct subscripts when needed
    lbl = lambda name : name if name.find("_")==-1 else name[:name.find("_")+1]+'{'+name[name.find("_")+1:]+'}'

    # # Bifurcation is evaluated at as many points as the size of model_group
    N_points = len(astro_group)/N_trials # The number of points where to compute the bifurcation coincides w/ N_astro
    N_syn = len(bifpar_group)/len(astro_group) # If bifpar_group==astro_group then N_syn==1

    # Set bifurcation parameter with the option of logarithmic scaling
    if log_scale:
        bifpar_range = log10(bifpar_range)
        # Step of variation for the bifurcation parameter
        bifpar_step = 0.1*diff(bifpar_range)[0]/N_points
        # Compute value of bifpar for each object in the astro_group
        bifpar_vals = array([10**(bifpar_range[0]+ bifpar_step + diff(bifpar_range)[0]*double(i%N_points)/N_points) for i in range(N_points)])
    else:
        # Step of variation for the bifurcation parameter
        bifpar_step = 0.1*diff(bifpar_range)[0]/N_points
        # Compute value of bifpar for each object in the astro_group
        bifpar_vals = array([bifpar_range[0]+bifpar_step+diff(bifpar_range)[0]*double(i%N_points)/N_points for i in range(N_points)])
    # Set bifpar values
    bifpar_vals = tile(tile(bifpar_vals,(1,N_trials)),(1,N_syn))[0]
    setattr(bifpar_group, bifpar_name, bifpar_vals*bifpar_units)

    # Run preliminary simulation to get initial conditions possibly close to stable states or oscillations
    # The preliminary run is taken 1/3 of the full duration
    print("Computing Initial Conditions")
    run(duration/3, namespace={}, report='text')

    # Forward continuation
    if log_scale:
        setattr(bifpar_group, bifpar_name, 10**(log10(bifpar_vals)+bifpar_step)*bifpar_units)
    else:
        setattr(bifpar_group, bifpar_name, (bifpar_vals+bifpar_step)*bifpar_units)
    print("Running continuation")
    # Will drop the first 'transient' seconds
    run(transient, namespace={}, report='text')

    # # Setup monitors (after transient)
    mon = StateMonitor(astro_group, variables=['Gamma_A', 'I', 'C', 'h'], record=True, dt=0.1*second)
    osc = SpikeMonitor(astro_group)
    # Run for the last time
    run(duration-transient, namespace={}, report='text')

    # Plot results for forward continuation
    plt.subplot(1, 2, 1)
    plot_bifurcation_diagram(mon.t, mon.C, 'C', umole, 0.01*umole,
                             getattr(bifpar_group,bifpar_name), lbl(bifpar_name), bifpar_units,
                             transient=0.0*second,log_scale=log_scale)
    plt.subplot(1, 2, 2)
    plot_oscillation_frequency(getattr(bifpar_group,bifpar_name), lbl(bifpar_name), bifpar_units,
                               osc.count, osc_threshold, duration=duration, log_scale=log_scale)

    # Plot settings
    plt.tight_layout()
    plt.show()

def plot_bifurcation_diagram(t, var, var_name, var_unit, threshold,
                             bifpar, bifpar_name, bifpar_unit,
                             transient=10.0*second,
                             color='k',log_scale=False):
    '''
    Plots a bifurcation diagram.

    Parameters
    ----------
    t           : `Quantity`
        The recording times for `var`
    var         : `Quantity`
        A 2-dimensional array of the recorded values (time as second dimension)
    var_name    : str
        The name of the variable `var` (used for the axis label).
    var_unit    : `Unit`
        The unit that should be used for plotting `var`.
    threshold   : `Quantity`
        The minimum difference between the minimum and maximum of `var` that
        to classify it as oscillating. Has to have the same units as `var`.
    bifpar      : `Quantity`
        The values of the bifurcation parameter.
    bifpar_name : str
        The name of the bifurcation parameter (used for the axis label)
    bifpar_unit : `Unit`
        The unit that should be used for plotting `bifpar`
    transient   : `Quantity`
        The time until which values of `var` should be ignored
    '''

    # Considers only solutions for t>transient
    data = var[:, t > transient]
    data_max = data.max(axis=1)
    data_min = data.min(axis=1)
    # For stable fixed points the min and max for t-->inf are the same
    # Conversely, if oscillations are detected, then min!=max
    oscillating = (data_max - data_min) > threshold
    # Plot bifurcation diagram
    # Plot stable fixed points
    plt.plot(bifpar[~oscillating]/bifpar_unit, data_max[~oscillating]/var_unit,linestyle='None',marker='x',color=color, lw=2.0)
    # Plot min-max envelope of oscillations
    plt.plot(bifpar[oscillating]/bifpar_unit, data_max[oscillating]/var_unit,linestyle='None',marker='o',color=color, lw=2.0)
    plt.plot(bifpar[oscillating]/bifpar_unit, data_min[oscillating]/var_unit,linestyle='None',marker='o',color=color, lw=2.0)
    plt.xlabel('${0}$ (${1}$)'.format(bifpar_name, sympy.latex(bifpar_unit)))
    plt.ylabel('${0}$ (${1}$)'.format(var_name, sympy.latex(var_unit)))
    if log_scale:
        plt.xscale('log')
        # TBD Add decimal axis labelling

    # Return logical indexes for bifpar_values where oscillations are detected
    # return oscillating

def plot_oscillation_frequency(bifpar, bifpar_name, bifpar_unit,
                               spk_count, spk_threshold, duration, log_scale):
    '''
    Plot the oscillation frequency over a bifurcation parameter.

    Parameters
    ----------
    bifpar      : `Quantity`
        The values of the bifurcation parameter.
    bifpar_name : str
        The name of the bifurcation parameter (used for the axis label)
    bifpar_unit : `Unit`
        The unit that should be used for plotting `bifpar`
    transient   : `Quantity`
        The time until which values of `var` should be ignored
    spk_count   : `ndarray`
        The total number of spikes per neuron/astrocyte
    duration    : `Quantity`
        The time during which `spk_count` was obtained.
    '''
    # Plots frequency of Ca^2+ oscillations
    idx_osc = spk_count > spk_threshold
    freq = spk_count[idx_osc] / duration

    plt.plot(bifpar[idx_osc]/bifpar_unit, freq/Hz, 'ko', lw=2.0)
    plt.xlabel('${0}$ (${1}$)'.format(bifpar_name, sympy.latex(bifpar_unit)))
    plt.ylabel('$f$ (Hz)')
    if log_scale:
        plt.xscale('log')
        # TBD Add decimal axis labelling

#-----------------------------------------------------------------------------------------------------------------------
# Synaptic/Gliotransmitter release
#-----------------------------------------------------------------------------------------------------------------------
def build_uxsyn(t, ux_solution, tau, var='x'):

    # Retrieve parameters w/out units
    tau /= second

    # Functions to build the time solution
    u_sol = lambda u_n,dt : u_n*exp(-dt/tau)
    x_sol = lambda x_n,dt : x_n*exp(-dt/tau)+(1-exp(-dt/tau))

    ux, ispk, nval = unique(ux_solution, return_index=True, return_counts=True, )
    nval = nval[argsort(ispk)]
    ispk = ispk[argsort(ispk)]
    tspk = []
    for i in range(len(ispk)) : tspk.extend(repeat(ispk[i],nval[i]))
    dt = t-t[tspk]

    if var=='x':
        # return x_sol(ux,dt)
        return x_sol(ux_solution,dt)
    else:
        # return u_sol(ux,dt)
        return u_sol(ux_solution,dt)

def plot_release(t, syn_signal, var='r', time_units=second, var_units=1, tau=1*second,
                 ax=None, color='k', linestyle='-', linewidth=1.5, lfs=16, afs=14,
                 redraw=False, spine_position=0):
    '''
    Plot of synaptic variables.

    t          : Quantity
     time (w/ units)
    syn_signal : Quantity
     Monitored synaptic variable (w/ units)
    var        : String
     String of synaptic variable name
    var_units  : Unit
     Units of synaptic variable
    ax         : Object
     Axis object
    color      : String or Array
     Color of 'x', 'u', 'Y' traces
    linestyle  : String
     Line style of 'x', 'u', 'Y' traces
    linewidth  : Quantity
     Line width in pts
    lfs        : Quantity
     Label font size
    afs        : Quantity
     Axis font size
    redraw     : Boolean
     If True use 'spine' to draw axis (for publication)
    spine_position : Integer
     Specify the number of points for position of spines
    '''

    if not ax:
        # Create a new figure in case no axis object is provided with two panels: spikes (top) and var (bottom)
        fig, ax = plt.subplots(2,1)
        plot_release(t,syn_signal,var='spk',ax=ax[0])
        ax[0].set_xticklabels([])
        plot_release(t,syn_signal,var=var,ax=ax[1])
        return

    if var=='spk':
        # "Presynaptic" spikes (a quick way to retrieve them w/out the need of a SpikeMonitor) (works fine if syn_signal = r_S)
        _, ispk = unique(syn_signal, return_index=True)
        if _[0]==0 : ispk = ispk[1:] # drop the first element because it is not a real spike (no release)
        l, m, b = ax.stem(t[ispk], ones(size(ispk)), linefmt='k', markerfmt='', basefmt='')
        plt.setp(m, 'linewidth', linewidth)
        plt.setp(l, 'linestyle', 'none')
        plt.setp(b, 'linestyle', 'none')
        ax.yaxis.set_visible(False)
        if redraw : figtem.adjust_spines(ax, [])
    if var=='r':
        # Released neurotransmitter resources
        r, ispk = unique(syn_signal, return_index=True)
        l, m, b = ax.stem(t[ispk], r, linefmt=color, markerfmt='', basefmt='')
        plt.setp(m, 'linewidth', linewidth)
        plt.setp(l, 'linestyle', 'none')
        plt.setp(b, 'linestyle', 'none')
        if redraw : figtem.adjust_spines(ax, ['left'], position=spine_position)
        ax.set_xlabel('Time (${0}$)'.format(sympy.latex(time_units)))
        ax.set_ylim([0.0,1.0])
        ax.set_yticks([0.0,0.5,1.0])
        ax.set_ylabel('$r_S$')
    if var=='x':
        # Available neurotransmitter for release
        x = build_uxsyn(t, syn_signal, tau, 'x')
        ax.plot(t, x, ls=linestyle, lw=linewidth, color=color)
        if redraw : figtem.adjust_spines(ax, ['left'], position=spine_position)
        ax.set_xlabel('Time (${0}$)'.format(sympy.latex(time_units)))
        ax.set_ylim([0.0,1.02])
        ax.set_yticks([0.0,0.5,1.0])
        ax.set_ylabel('$x_S$')
    if var=='u':
        # Intrasynaptic calcium
        u = build_uxsyn(t, syn_signal, tau, 'u')
        ax.plot(t, u, ls=linestyle, lw=linewidth, color=color)
        if redraw : figtem.adjust_spines(ax, ['left'], position=spine_position)
        ax.set_xlabel('Time (${0}$)'.format(sympy.latex(time_units)))
        ax.set_ylim([0.0,1.02])
        ax.set_yticks([0.0,0.5,1.0])
        ax.set_ylabel('$u_S$')
    if var=='y':
        # Released neurotransmitter
        ax.plot(t, syn_signal, ls=linestyle, lw=linewidth, color=color)
        if redraw : figtem.adjust_spines(ax, ['left','bottom'], position=spine_position)
        ax.set_xlabel('Time (${0}$)'.format(sympy.latex(time_units)), fontsize=lfs)
        ax.set_ylabel('$Y_S$ (${0}$)'.format(sympy.latex(var_units)), fontsize=lfs)


def rr_mean(params, N_points=1, N_trials=1, freq_range=[0.1,100], gliot=False,
            duration=60*second, color='k', fmt = 'o', ax=None, plot_results=True, ics='2*C_Theta'):
    '''
    Compute and plot mean-field released-resources of a synapse as a function of the input rate (a.k.a. the frequency
    response of the synapse).

    params     : Dictionary
     Model parameters
    N_points   : Integer
     Number of frequency points where to compute the average released resources
    N_trials   : Integer
     Number of trials over which to average
    freq_range : Quantity
     2-element array for the frequency range of exploration [fmin,fmax] (w/out units)
    gliot      : Boolean
     If True include gliotransmission
    duration   : Quantity
     Duration (w/ units) of each trial
    color      : String or Quantity
     Marker Color
    fmt        : String
     Marker type
    ax         : Object
     Axis object to use for plotting
    plot_reuslts     : Boolean
     Show results of simulation (only frequency response)
    '''

    # Functions to compute mean and std of released resources
    r_vals = lambda r : [mean(unique(x[x>0])) for x in r]                # mean unique r_S per trial
    r_mean = lambda rv, npts : [mean(rv[i::npts]) for i in range(npts)] # mean r_S per freq value
    r_std  = lambda rv, npts : [std(rv[i::npts]) for i in range(npts)]  # std on r_S per freq value

    # if sim:
    # Build synaptic model (w/out gliotransmission)
    if not gliot:
        # No gliotransmission
        neuron_in, neuron_out,synapses = asmod.synapse_model(params, N_syn=N_points*N_trials, stimulus='poisson')
    else:
        # w/ gliotransmission
        neuron_in, neuron_out, synapses, astro, es_syn2astro, gliot, es_astro2syn = asmod.astrosyn_model(params, N_syn=1, N_astro=N_points*N_trials, stimulus='poisson')
        synapses.alpha = params['alpha']
        astro.C = ics         # This make sure that even at small frequencies, there's an effect on synaptic release
                              # w/out this term and starting from "C=0, you may actually show the "bistability case!!

    # Set frequency values
    freq_range = log10(freq_range)
    # Step of variation for the bifurcation parameter
    freq_step = 0.1*diff(freq_range)[0]/N_points
    # Compute value of bifpar for each object in the astro_group
    freq_vals = array([10**(freq_range[0]+ freq_step + diff(freq_range)[0]*double(i%N_points)/N_points) for i in range(N_points)])
    freq_vals = tile(freq_vals,(1,N_trials))[0]
    setattr(neuron_in,'f_in',freq_vals*Hz)

    # Setup synaptic monitor
    mon = StateMonitor(synapses, variables=['U_0','r_S'], record=True, dt=2*ms)

    # Run simulation
    run(duration, namespace={}, report='text')

    R = {}
    R['U_0'] = r_vals(mon.U_0_[:])
    R['r_S'] = r_vals(mon.r_S_[:])

    if plot_results:
        # Retrieve mean unique values per trials
        # rv = r_vals(mon.r_S_[:])
        # rv = r_vals(mon['r_S'][:])

        # Effective plotting
        if not ax: fig, ax = plt.subplots(1,1)
        ax.errorbar(unique(freq_vals),r_mean(rv,N_points),yerr=r_std(rv,N_points),fmt=fmt,color=color)
        ax.set_xscale('log')
        ax.set_xlabel('$f_{0}$ (${1}$)'.format('{in}',sympy.latex(Hz)))
        ax.set_ylim([0.0,1.0])
        ax.set_yticks(arange(0.0,1.1,0.5))
        ax.set_ylabel(r'$\langle r_S\rangle$')

        return R, unique(freq_vals), ax
    else:
        return R, unique(freq_vals)

def synaptic_filter(params, N_points=1, N_trials=1, freq_range=[0.1,100], gliot=False,
                    duration=60*second, color='k', fmt = 'o', ax=None, sim=False, filename='tmp.pkl'):
    '''
    Compute frequency response (filter) of the synapse

    params     : Dictionary
     Model parameters
    N_points   : Integer
     Number of frequency points where to compute the average released resources
    N_trials   : Integer
     Number of trials over which to average
    freq_range : Quantity
     2-element array for the frequency range of exploration [fmin,fmax] (w/out units)
    gliot      : Boolean
     If True include gliotransmission
    duration   : Quantity
     Duration (w/ units) of each trial
    color      : String or Quantity
     Marker Color
    fmt        : String
     Marker type
    ax         : Object
     Axis object to use for plotting
    sim        : Boolean
     If True run simulation to compute all data points. If False, load from file filename and plot
    filename   : string
     String with file name to save/load data to/from. It must include also the extension '.pkl'
    '''

    # Functions to compute mean and std of released resources
    r_vals = lambda r : [mean(unique(x[x>0])) for x in r]                # mean unique r_S per trial
    r_mean = lambda rv, npts : [mean(rv[i::npts]) for i in range(npts)] # mean r_S per freq value
    r_std  = lambda rv, npts : [std(rv[i::npts]) for i in range(npts)]  # std on r_S per freq value

    if sim:
        # Build synaptic model (w/out gliotransmission)
        if not gliot:
            # No gliotransmission
            neuron_in, neuron_out,synapses = asmod.synapse_model(params, N_syn=N_points*N_trials, stimulus='poisson')
        else:
            # w/ gliotransmission
            neuron_in, neuron_out, synapses, astro, es_syn2astro, gliot, es_astro2syn = asmod.astrosyn_model(params, N_syn=N_points*N_trials, N_astro=1, stimulus='poisson')

        # Set frequency values
        freq_range = log10(freq_range)
        # Step of variation for the bifurcation parameter
        freq_step = 0.1*diff(freq_range)[0]/N_points
        # Compute value of bifpar for each object in the astro_group
        freq_vals = array([10**(freq_range[0]+ freq_step + diff(freq_range)[0]*double(i%N_points)/N_points) for i in range(N_points)])
        freq_vals = tile(freq_vals,(1,N_trials))[0]
        setattr(neuron_in,'f_in',freq_vals*Hz)

        # Setup synaptic monitor
        mon = StateMonitor(synapses, variables=['r_S'], record=True)

        # Run simulation
        run(duration, namespace={}, report='text')

        return mon

#-----------------------------------------------------------------------------------------------------------------------
# Postsynaptic firing
#-----------------------------------------------------------------------------------------------------------------------
def freq_out(params, sim=True, gliot=False, sync='double-exp', duration=30*second,
             N_points=1, N_trials=1, freq_range=[0.1,100], freq_rel=0*Hz,
             show=False, color='k', fmt = 'o', ax=None):
    '''
    Computes I/O of a neuron in terms of f_out vs. f_in or f_out vs. f_c (gliot=True). Produce results only when sim=True

    params     : Dictionary
     Model parameters
    N_points   : Integer
     Number of frequency points where to compute the average released resources
    N_trials   : Integer
     Number of trials over which to average
    freq_range : Quantity
     2-element array for the frequency range of exploration [fmin,fmax] (w/out units)
    gliot      : Boolean
     If True include gliotransmission
    duration   : Quantity
     Duration (w/ units) of each trial
    color      : String or Quantity
     Marker Color
    fmt        : String
     Marker type
    ax         : Object
     Axis object to use for plotting
    '''

    # Functions to compute mean and std of released resources
    v_mean = lambda v, npts : [mean(v[i::npts]) for i in range(npts)] # mean r_S per freq value
    v_std  = lambda v, npts : [std(v[i::npts]) for i in range(npts)]  # std on r_S per freq value


    if sim :
        if not gliot:
            # Provide f_out vs. f_in @ f_c = const
            neuron_in,neuron_out,synapses,gliotr,es_astro2syn = asmod.openloop_model(params, N_syn=N_points*N_trials, N_astro=1, ics=None,
                                                                                    linear=True, stimulus_syn='poisson', stimulus_glt=None,
                                                                                    post=sync, sic=sync, stdp=False)

            gliotr.f_c = freq_rel
            neuron_out.v = params['E_L']
        else:
            # Provide f_out vs. f_c @ f_in = const
            neuron_in,neuron_out,synapses,gliotr,es_astro2syn = asmod.openloop_model(params, N_syn=1, N_astro=N_points*N_trials, ics=None,
                                                                                    linear=True, stimulus_syn=None, stimulus_glt='poisson',
                                                                                    post=sync, sic=sync, stdp=False)
            neuron_in.f_in = freq_rel
            neuron_out.v = params['E_L']

        # Set frequency values
        freq_range = log10(freq_range)
        # Step of variation for the bifurcation parameter
        freq_step = 0.1*diff(freq_range)[0]/N_points
        # Compute value of bifpar for each object in the gliotransmission group
        freq_vals = array([10**(freq_range[0]+ freq_step + diff(freq_range)[0]*double(i%N_points)/N_points) for i in range(N_points)])
        freq_vals = tile(freq_vals,(1,N_trials))[0]
        if not gliot :
            setattr(neuron_in,'f_in',freq_vals*Hz)
        else :
            setattr(gliotr,'f_c',freq_vals*Hz)

        # Setup synaptic monitor
        mon = SpikeMonitor(neuron_out, record=True)

        # Run simulation
        run(duration, namespace={}, report='text')

        # Retrieve mean unique values per trials
        f_out = mon.count[:] / duration

        # Build output data
        data = vstack((unique(freq_vals), v_mean(f_out,N_points), v_std(f_out,N_points)))

        # Show results (just to check simulation)
        if show :
            # Effective plotting
            if not ax: fig, ax = plt.subplots(1,1)
            ax.errorbar(unique(freq_vals),v_mean(f_out,N_points),yerr=v_std(f_out,N_points),fmt=fmt,color=color)
            ax.set_xscale('log')
            ax.set_xlabel('$\\nu_{0}$ (${1}$)'.format('{pre}',sympy.latex(Hz)))
            # ax.set_ylim([0.0,1.0])
            # ax.set_yticks(arange(0.0,1.1,0.5))
            ax.set_ylabel('$\\nu_{0}$ (${1}$)'.format('{post}',sympy.latex(Hz)))

            plt.show()

        return data

def threshold_crossing(time,signal, threshold):
    if size(threshold)<2:
        # Simple threshold
        index = signal>=threshold
    else:
        # Two thresholds
        index = (signal>=threshold[0]) & (signal<threshold[1])

    if sum(index)>0:
        # print index
        index = where(diff(index))[0]
        return index, diff(time[index])[::2]
    else:
        return [], []

def time_above_threshold(time,signal,thresholds={'Theta_d': 0.0, 'Theta_p': 0.1}):
    '''
    Compute the average time above the two thresholds. This works correctly only in the case of Theta_d<Theta_p.

    Input parameters:
    - time   : time vector
    - signal : calcium signal
    - thresholds : thresholds for LTP and LTD

    Return:
    - Average time above the thresholds (dictionary)
    '''
    tat = {}
    for thr in list(threhsolds.keys()):
        index = signal>=thr
        if sum(index)>0:
            if index[-1]: index[-1] = False
            index = where(diff(index))[0]
            tat[thr] = sum(diff(time[index])[::2])/time[-1]
    return tat

def threshold_regions(time, signal, theta_d, theta_p,
                      y0=0.01, ax=None):


    lw = 1.5
    colors = {'Theta_d': '#0000ff',
              'Theta_p': '#ff6600',
              'ltd'    : '#8080ff',
              'ltp'    : '#ffc8a3'}

    # Retrieve indexes (assuming last index not coincident w/ the last element of the solution)
    idxd = (signal>=theta_d) & (signal<theta_p)
    idxp = signal>=theta_p

    # Create a new figure in case no axis object is provided
    if not ax: fig, ax = plt.subplots(1,1)

    # patches = []
    if sum(idxd)>0:
        index = where(diff(idxd))[0]
        for i in range(0,size(index),2):
            xpts = concatenate(([time[index[i]]], time[index[i]:index[i+1]], [time[index[i+1]]]))
            ypts = concatenate(([y0], signal[index[i]:index[i+1]], [y0]))
            # patches.append(ptc.Polygon(zip(*vstack((xpts,ypts))), True, ec='none', fc=colors['ltd']))
            ax.add_artist(ptc.Polygon(list(zip(*vstack((xpts,ypts)))), True, ec='none', fc=colors['ltd']))
            # patches.append(ptc.Polygon(vstack((xpts,ypts)), ec='none', fc=colors['ltd']))

    if sum(idxp)>0:
        index = where(diff(idxp))[0]
        for i in range(0,size(index),2):
            xpts = concatenate(([time[index[i]]], time[index[i]:index[i+1]], [time[index[i+1]]]))
            ypts = concatenate(([y0], signal[index[i]:index[i+1]], [y0]))
            # patches.append(ptc.Polygon(zip(*vstack((xpts,ypts))), True, ec='none', fc=colors['ltp']))
            ax.add_artist(ptc.Polygon(list(zip(*vstack((xpts,ypts)))), True, ec='none', fc=colors['ltp']))

    # if size(patches)>0: ax.add_collection(PatchCollection(patches))

    # Add thresholds
    ax.plot((time[0],time[-1]), (theta_d,theta_d), ls='--', c=colors['Theta_d'], lw=lw)
    ax.plot((time[0],time[-1]), (theta_p,theta_p), ls='--', c=colors['Theta_p'], lw=lw)



#-----------------------------------------------------------------------------------------------------------------------
# Network analysis
#-----------------------------------------------------------------------------------------------------------------------
def plot_connections(model_group=None,connections=None,
                     x_coords=None,y_coords=None,ax=None,
                     connections_i=None,connections_j=None,
                     linestyle=':',color='y',linewidth=1.5,markersize=30):
    '''
    Quick plotting of a network

    model_group                 : Group
     Model group
    connections                 : Group
     Synapse group that specifies connections
    x_coords, y_coords          : Quantity
     Arrays of x,y coordinates (w/out units)
    connections_i,connections_j : Quantity
     Arrays of connections between i,j nodes
    ls                          : String
     Line Style for connections
    c                           : String or Quantity
     Color
    '''

    if model_group is not None:
        # This mode will show network connectivity in a standalone figure
        plt.figure()
        # Draw edges
        plt.plot((model_group.x_[connections.i], model_group.x_[connections.j]),
                 (model_group.y_[connections.i], model_group.y_[connections.j]),ls=linestyle,c=color,lw=linewidth)

        # Add the nodes as a scatter
        # TODO: currently nodes are not opaque but edges cross them: I don't know how to set matplotlib to have nice plotting...
        node_trace = plt.scatter(x=[], y=[], c=color, s=markersize, marker='o', facecolor='none', edgecolor=color)
        for node in range(size(model_group.x)):
            x, y = model_group.x_[node], model_group.y_[node]
            node_trace.set_offsets(append(node_trace.get_offsets(),[x,y]))
    else:
        # x_coords and y_coords are specified instead (this mode will show connections superimposed on pseudo_df images
        # Draw edges
        ax.plot((x_coords[connections_i], x_coords[connections_j]),
                (y_coords[connections_i], y_coords[connections_j]),ls=linestyle,c=color,lw=linewidth)

        # Add the nodes as a scatter
        # TODO: currently nodes are not opaque but edges cross them: I don't know how to set matplotlib to have nice plotting...
        node_trace = ax.scatter(x=[], y=[], c=color, s=markersize, marker='o', facecolor='none', edgecolor=color)
        for node in range(size(x_coords)):
            x, y = x_coords[node], y_coords[node]
            node_trace.set_offsets(append(node_trace.get_offsets(),[x,y]))

    # plt.tight_layout()

def plot_traces(t,signals,var_name):
    '''
    t        : Quantity
      time (no units)
    signals  : Quantity
      variable (no units)
    var_name : String
      variable name
    '''

    # Initialize plot
    lineprops = dict(linewidth=1, color='black', linestyle='-')
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ticklocs = []
    for i, s in enumerate(signals):
        offset = i+0.5
        # Normalize traces
        s = (s - amin(s))/(amax(s)-amin(s))
        ax.plot(t,s+offset,**lineprops)
        ticklocs.append(i+1)

    # Labelling and Refine
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(['${0}$'.format(var_name)+'%d'%(i+1) for i in range(shape(signals)[0])])
    ax.set_ylim([-0.1,shape(signals)[0]+0.6])
    ax.set_xlabel('Time (s)')

def gauss_kern(sizex, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(sizex)
    if not sizey:
        sizey = sizex
    else:
        sizey = int(sizey)
    #print size, sizey
    x, y = mgrid[-sizex:sizex+1, -sizey:sizey+1]
    g = exp(-(x**2/float(sizex)+y**2/float(sizey)))
    return g / g.sum()

def plot_pseudo_df_f(t,signals,
                     x_coords,y_coords,
                     connections_i,connections_j,
                     time_samples=None,
                     img_size=None):
    '''
    Plot of astrocyte network signalling as a pseudo-fluorescence image

    t             : Quantity (w/out units)
     Time
    signals       : Quantity (w/out units)
     Signals to plot
    x_coords      : Quantity (w/ units)
    y_coords      : Quantity (w/ units)
     Cells' X,Y-coordinates
    connections_i : Quantity
    connections_j : Quantity
     Connections between nodes i,j
    time_samples  : Quantity
     Array of time instants at which to get an image
    img_size      : Qantity (w/ units)
     Max size of the image (if >x_size or y_size)
    '''

    # Retrieve index of element in array nearest to value
    find_nearest = lambda values,array : [abs(array-value).argmin() for value in values]

    # Default parameters
    cell_size = 60*umeter
    # img_margin = 2*cell_size / meter
    img_margin = cell_size / meter

    N_pixels = 100
    time_step = 3

    # If time_samples for extraction of images are not specified it automatically gives the 0,1/step,...,end Figures
    if time_samples is None:
        t_index = append(arange(0,size(t),size(t)/time_step,dtype=int),-1)
    else:
        t_index = find_nearest(time_samples,t)
    # time_samples = t[t_index]

    # Build the coordinates (x,y) of the final figure starting from the given coordinates
    # First make sure that data are in the x,y>0 region
    x = x_coords - amin(x_coords)
    y = y_coords - amin(y_coords)
    # Then add left margin
    x = x + img_margin
    y = y + img_margin
    # And compute max size of the figure (including right margin)
    x_size = amax(x) + img_margin
    y_size = amax(y) + img_margin

    # A basic interface to plot with fixed figure size (useful to compare different data sets)
    if img_size is not None:
        img_size /= meter
        if img_size<=x_size:
            print("img_size is <= max coordinate and will be ignored")
        else:
            x_size = img_size
            y_size = img_size
            x_offset = img_size/2.0 - mean(x)
            y_offset = img_size/2.0 - mean(y)
            x = x + x_offset
            y = y + y_offset

    # Bin coordinates into available pixels
    metrics = arange(0.0,x_size,x_size/N_pixels)
    x_index = find_nearest(x,metrics)
    y_index = find_nearest(y,metrics)

    # Normalize signals
    signals_norm = (signals-amin(signals))/(amax(signals)-amin(signals))

    # Gaussian filter
    filter = gauss_kern(floor(N_pixels*cell_size/x_size / meter), sizey=None)

    # Build images
    fig, ax = plt.subplots(1,size(t_index), figsize=(15,3.5), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.00, top=0.99, hspace=0.1, wspace=0.05)
    ax = ax.ravel()
    for n,sample in enumerate(t_index):
        space = zeros((N_pixels,N_pixels))
        for cell,pixel in enumerate(zip(x_index,y_index)):
            space[pixel] = signals_norm[cell,sample]
        # Perform convolution
        img_pseudo_df = convolve2d(space, filter, mode='same')
        ax[n].imshow(img_pseudo_df.T, cmap=plt.cm.gray, interpolation='hermite',extent=(0.0,N_pixels,0.0,N_pixels),origin='lower')
        ax[n].set_title('t='+str(round(t[sample]))+' s')
        # Superimpose network connections (and translate data point by 1/2 pixel to center them w.r.t. the fluorescence signal)
        plot_connections(ax=ax[n],x_coords=array(x_index)+.5,y_coords=array(y_index)+.5,
                         connections_i=connections_i,connections_j=connections_j)
    #plt.subplot_tool()

#-----------------------------------------------------------------------------------------------------------------------
# COBA Network analysis
#-----------------------------------------------------------------------------------------------------------------------
def plot_raster(t_spk, i_n, N_e=1, ax=None, c_e='k', c_i='r'):
    """
    Raster plot for COBA network showing inhibitory and excitatory neurons in different colors.

    t_spk : Quantity
     The spike time vector as retrieved by SpikeMonitor
    i_n   : Quantity
     The indexes of the spikes as retrived by SpikeMonitor
    N_e   : Integer
     The number of excitatory neurons in the network
    ax    : Axis handle
    c_e   : String
     Color of excitatory neurons
    c_i   : String
     Color of inhibitory neurons
    """

    # Retrieve index of excitatory neurons
    idx = i_n<N_e

    if not ax: fig, ax = plt.subplots(1,1)
    # Excitatory neurons
    ax.plot(t_spk[idx], i_n[idx], '.', color=c_e)
    # Inhibitory neurons
    ax.plot(t_spk[~idx], i_n[~idx], '.', color=c_i)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron index')