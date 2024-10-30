"""
tripartite_figures.py

Generate Figures and simulations for the paper

De Pitta and Brunel, "Modulation of synaptic plasticity by glutamatergic gliotransmission: A model study. Neural Plasticity, 2016: 7607924

v2.0 Maurizio De Pitta, Basque Center for Mathematics, 2020
Translated and Debugged to run in Python 3.x

v1.0 Maurizio De Pitta, The University of Chicago, 2015
"""
from brian2 import *
from brian2.units.allunits import mole, umole, mmole
import logging
logging.basicConfig(level=logging.DEBUG)
import scipy.signal as sg
from math import sqrt

import os
import numpy as np
# Optional settings for faster compilations and execution by Brian
prefs.codegen.cpp.extra_compile_args = ['-Ofast', '-march=native']

# User-defined modules
import astrocyte_models as asmod
import analysis
import figures_template as figtem
import graupner_model as camod
import brian_utils as bu
import save_utils as svu
from svg_converter import svg_to_png
import Geometry as geom
import sympy

import numpy.random as random
import matplotlib
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline

matplotlib.use('Agg') 

#LECTURE
from scipy.signal import argrelmax
import numpy as np


import numpy as np

def interx(l, l0):
    """
    Find approximate intersection points of two curves provided as 2D arrays.
    
    Parameters:
    l, l0: 2D arrays where the first row represents x-values and the second row represents y-values.

    Returns:
    intersections: A list of (x, y) tuples representing intersection points, or [(np.nan, np.nan)] if none are found.
    """
    # Separate x and y values for each input
    x1, y1 = l[0], l[1]
    x2, y2 = l0[0], l0[1]

    # Ensure all arrays are 1D
    x1, x2, y1, y2 = np.ravel(x1), np.ravel(x2), np.ravel(y1), np.ravel(y2)
    
    # Interpolate y-values for x1 on the second curve
    y2_interp = np.interp(x1, x2, y2)

    # Find where y1 and y2 intersect by checking for sign changes
    idx = np.argwhere(np.diff(np.sign(y1 - y2_interp))).flatten()

    # Collect intersection points or return [(np.nan, np.nan)] if none found
    intersections = [(x1[i], y1[i]) for i in idx]
    return intersections if intersections else [(np.nan, np.nan)]



def save_figures(label='tmp', filename=['figure'], format='svg', dpi=600):
    '''
    General routine to save multiple Figures with personalized names within each simulation
    :param label:
    :param filename:
    :param format:
    :param dpi:
    :return:
    '''

    if filename is None:
        filename = [f'figure_{i}' for i in range(len(plt.get_fignums()))]
    
    for i, n in enumerate(plt.get_fignums()):
        figure(n)
        plt.savefig(label+filename[i]+'.'+format, format=format, dpi=dpi)


def simulate(sim_id):
    # Basic interface to select different simulations (does not allow to modify simulations)
    return {'io'              : tripartite_io,
            'stp'             : pre_regulation,
            'filter'          : syn_filtering,
            'sic'             : sic,
            'stdp_pre'        : stdp_pre,
            'stdp_sic'        : stdp_sic
            }.get(sim_id)()


def plotyy(x, y1, y2, y3=None,
           ax=None,
           ls='-', lw=1.5, colors = {'y1' : 'k', 'y2' : 'r', 'y3': 'g'},
           spine_position = 5, spine_position_y3=1.0):
    '''

    :param x:
    :param y1:
    :param y2:
    :param colors:
    :param spine_position:
    '''

    if not ax:
        # Create a new figure in case no axis object is provided with two panels: spikes (top) and var (bottom)
        fig, ax = plt.subplots(1,1)

    # Plot y1
    ax.plot(x, y1, ls=ls, lw=lw, color=colors['y1'])
    figtem.adjust_spines(ax, ['left'], position=spine_position)

    # Plot y2
    # Create twin y-axis
    ax_aux = ax.twinx()
    ax_aux.plot(x, y2, ls=ls, lw=lw, color=colors['y2'])
    # First make patch and spines of ax_aux invisible
    figtem.make_patch_spines_invisible(ax_aux)
    # # Then show only the right spine of ax_aux
    ax_aux.spines['right'].set_visible(True)
    ax_aux.spines['right'].set_position(('outward', spine_position))  # outward by 10 points

    # Plot y3 (optional)
    if y3 is not None:
        ax_aux = [ax_aux, ax.twinx()]
        ax_aux[1].plot(x, y3, ls=ls, lw=lw, color=colors['y3'])
        ax_aux[1].spines['right'].set_position(('axes', spine_position_y3))  # outward by fraction of the x-axis
        figtem.make_patch_spines_invisible(ax_aux[1])
        ax_aux[1].spines['right'].set_visible(True)

    # Set y-axis
    tkw = dict(size=4, width=1.5)
    if type(ax_aux)==type(list()):
        ax_list = [ax] + ax_aux
    else:
        ax_list = [ax, ax_aux]
    for i,a in enumerate(ax_list):
        a.yaxis.label.set_color(colors['y'+str(i+1)])
        a.tick_params(axis='y', colors=colors['y'+str(i+1)], **tkw)

    return ax, ax_aux

def set_axlw(ax, lw=1.0):
    '''
    Adjust axis line width
    '''
    for axis in list(ax.spines.keys()):
        ax.spines[axis].set_linewidth(lw)

def set_axfs(ax, fs=14):
    '''
    Adjust axis label font size
    '''
    ax.xaxis.set_tick_params(labelsize=fs)
    ax.yaxis.set_tick_params(labelsize=fs)

# Simple lambda function to format colors appropriately
chex = lambda rgb : '#%02x%02x%02x' % rgb
# Define some global lambdas
peak_normalize = lambda peak, taur, taud : peak*(1./taud - 1./taur)/( (taur/taud)**(taud/(taud-taur))-(taur/taud)**(taur/(taud-taur)) )

#-----------------------------------------------------------------------------------------------------------------------
# Figure 1
#-----------------------------------------------------------------------------------------------------------------------
def tripartite_syn(sim=False, syn_type='depressing', format='eps', data_dir='../data/', fig_dir='../Figures/'):
    '''
    Show I/O characteristics in terms of released neurotransmitter resources of a synapse w/out w/ gliotransmission
    '''

    # Plotting defaults
    spine_position = 5
    lw = 1.5
    alw = 1.0
    afs = 14
    lfs = 16

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
    f_in = 5*Hz

    # Testing synapses
    offset = 10*msecond
    transient = 16.2*second
    spikes = concatenate((arange(0.0,200,50.0),arange(300.0,390,10.0),(450.0,)))*msecond
    spikes += transient + offset

    if sim :
        # Simple TM synapse (sample dynamics)
        # Even if test_out will not be used, to properly create the model, it must be explicitly assigned and made available
        # w/in the local scope
        test_in, test_out, syn_test = asmod.synapse_model(params, N_syn=1, connect='i==j',
                                                          stimulus='test',
                                                          spikes = spikes,
                                                          name='testing_synapse*')

        # Stimulated astrocyte
        neuron_stim, neuron_out, syn_stim, astro, ecs_syn = asmod.astro_with_syn_model(params, N_syn=1, N_astro=1,
                                                                                       stimulus='poisson', ics=None)
        neuron_stim.f_in = f_in
        astro.C = 0.0 * umole
        astro.I = 0.0 * umole
        astro.h = 0.99

        # Gliotransmission
        gliot = asmod.gliotransmitter_release(astro, params)

        # Create monitors
        S = StateMonitor(syn_test, variables=['u_S','x_S','r_S','Y_S'], record=True)
        A0 = StateMonitor(syn_stim, variables=['Y_S'], record=True)
        A1 = StateMonitor(astro, variables=['Gamma_A'], record=True)
        A2 = StateMonitor(astro, variables=['C', 'h', 'I'], record=True, dt=0.05*ms)
        G = StateMonitor(gliot, variables=['x_A','G_A'], record=True)

        duration = 30*second
        run(duration,namespace={},report='text')

        # Convert monitors to dictionaries for saving
        S_mon = bu.monitor_to_dict(S)
        A0_mon = bu.monitor_to_dict(A0)
        A1_mon = bu.monitor_to_dict(A1)
        A2_mon = bu.monitor_to_dict(A2)
        G_mon = bu.monitor_to_dict(G)

        svu.savedata([S_mon,A0_mon,A1_mon,A2_mon,G_mon],data_dir+'io_syn_'+syn_type[:3]+'.pkl')

    #-------------------------------------------------------------------------------------------------------------------
    # Building Figures
    #-------------------------------------------------------------------------------------------------------------------
    [S,A0,A1,A2,G] = svu.loaddata(data_dir+'io_syn_'+syn_type[:3]+'.pkl')

    plt.close('all')
    #-------------------------------------------------------------------------------------------------------------------
    # Synapse
    #-------------------------------------------------------------------------------------------------------------------
    fig0, ax = figtem.generate_figure('3x1',figsize=(6.5,5.5),left=0.21,bottom=0.14,right=0.12)
    tlims = array([16.0,17])

    # Plot spikes
    analysis.plot_release(S['t'], S['r_S'][0], var='spk',  ax=ax[0], redraw=True)
    ax[0].set_xlim(tlims)

    # Plot u,x
    # Retrieve u,x traces
    colors = {'y1': chex((0,107,164)), 'y2': chex((255,126,14))}
    u = analysis.build_uxsyn(S['t'], S['u_S'][0], 1.0/params['Omega_f'], 'u')
    x = analysis.build_uxsyn(S['t'], S['x_S'][0], 1.0/params['Omega_d'], 'x')
    ax[1],ax_aux = plotyy(S['t'], u, x, ax=ax[1], colors=colors)

    # Format x-axis
    ax[1].set_xlim(tlims)
    ax[1].set_xlabel('')

    # Format y-axis
    ax[1].set_ylim([-0.01,1.01])
    ax[1].set_yticks([0.0,0.5,1.0])
    # ax[1].set_ylabel('$u_S$', fontsize=lfs)
    ax[1].set_ylabel('Docking Pr.', fontsize=lfs, va='center')
    ax[1].yaxis.set_label_coords(-.2,0.5)
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    ax_aux.set_ylim([-0.01,1.01])
    ax_aux.set_yticks([0.0,0.5,1.0])
    # ax_aux.set_ylabel('$x_S$', fontsize=lfs)
    ax_aux.set_ylabel('Avail. Nt. Pr.', fontsize=lfs)
    # Adjust labels
    set_axlw(ax_aux, lw=alw)
    set_axfs(ax_aux, fs=afs)

    # Add Y_S
    cf = 1*umole / mole
    analysis.plot_release(S['t']-tlims[0], S['Y_S'][0] / cf, var='y', var_units=umole,
                          time_units=ms, ax=ax[2], color='k', redraw=True, spine_position=5)
    ax[2].set_xlim(tlims-tlims[0])
    ax[2].set_xticks(arange(0.0,1.1,0.2))
    ax[2].set_xticklabels([str(int(t*1e3)) for t in ax[2].get_xticks()])
    ax[2].set_ylim([-6.0,1300])
    ax[2].set_yticks(arange(0.0,1201,400))
    ax[2].set_ylabel('Released Nt.\n(${0}$)'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    ax[2].yaxis.set_label_coords(-.13,0.5)
    # Adjust labels
    set_axlw(ax[2], lw=alw)
    set_axfs(ax[2], fs=afs)

    #-------------------------------------------------------------------------------------------------------------------
    # Astrocyte
    #-------------------------------------------------------------------------------------------------------------------
    fig1, ax = figtem.generate_figure('3x1',figsize=(7.3,5.5), axref_size=(6.5,5.5),
                                 left=0.21,bottom=0.14,right=0.12)
    tlims = array([0,duration])

    # Synaptic release (Y_S)
    cf = 1*umole / mole
    analysis.plot_release(A0['t'], A0['Y_S'][0] / cf, var='y', ax=ax[0], color='k', linewidth=lw)
    ax[0].set_xlim(tlims)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlabel('')
    ax[0].set_ylim([-5.0,1000])
    ax[0].set_yticks(arange(0.0,1000,300))
    # ax[0].set_ylabel('$Y_S$ (${0}$)'.format(sympy.latex(umole)), fontsize=lfs)
    ax[0].set_ylabel('Released Nt.\n(${0}$)'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    ax[0].yaxis.set_label_coords(-.15,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Astrocyte receptors
    ax[1].plot(A1['t'], A1['Gamma_A'][0], lw=lw, color='k')
    ax[1].set_xlim(tlims)
    figtem.adjust_spines(ax[1], ['left'], position=spine_position)
    ax[1].set_ylim([-0.01,1.01])
    ax[1].set_yticks(arange(0.0,1.01,0.5))
    # ax[1].set_ylabel('$\gamma_A$')
    ax[1].set_ylabel('Bound\nAst. Rec.', fontsize=lfs, multialignment='center')
    ax[1].yaxis.set_tick_params(labelsize=afs)
    ax[1].yaxis.set_label_coords(-.15,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # Astrocyte dynamics (ChI model)
    colors = {'y1': 'k', 'y2': chex((227,119,194)), 'y3': chex((188,189,34))}
    ax[2], ax_aux = plotyy(A2['t'], A2['C'][0] / cf, A2['I'][0] / cf, y3=A2['h'][0],
                           ax=ax[2], spine_position_y3=1.18, colors=colors)

    # Format x-axis
    ax[2].spines['bottom'].set_color('k') # Make bottom axis visible
    figtem.adjust_spines(ax[2], ['left','bottom'], position=spine_position)
    ax[2].set_xlim(tlims)
    ax[2].set_xticks(arange(0.0,31.0,10.0))
    ax[2].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=lfs)
    set_axlw(ax[2], lw=alw)
    set_axfs(ax[2], fs=afs)

    # Format y-axis
    Clims = [-0.02,1.6]
    Ilims = [-0.02,2.1]
    hlims = [-0.01,1.01]
    ax[2].set_ylim(Clims)
    ax[2].set_yticks(arange(0.0,Clims[1],0.5))
    # ax[2].set_ylabel('$C$ (${0}$)'.format(sympy.latex(umole)), fontsize=lfs)
    ax[2].set_ylabel('Ast. Calcium\n(${0}$)'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    ax[2].yaxis.set_label_coords(-.15,0.5)
    set_axlw(ax[2], lw=alw)
    set_axfs(ax[2], fs=afs)

    ax_aux[0].set_ylim(Ilims)
    ax_aux[0].set_yticks(arange(0.0,Ilims[1],0.6))
    # ax_aux[0].set_ylabel('$I$ (${0}$)'.format(sympy.latex(umole)), fontsize=lfs)
    ax_aux[0].set_ylabel('IP$_3$ (${0}$)'.format(sympy.latex(umole)), fontsize=lfs)
    ax_aux[1].set_ylim(hlims)
    ax_aux[1].set_yticks(arange(0.0,hlims[1],0.5))
    # ax_aux[1].set_ylabel('$h$', fontsize=lfs)
    ax_aux[1].set_ylabel('Deinact. IP$_3$Rs', fontsize=lfs)
    for axis in ax_aux:
        set_axlw(axis, lw=alw)
        set_axfs(axis, fs=afs)

    #-------------------------------------------------------------------------------------------------------------------
    # Gliotransmission
    #-------------------------------------------------------------------------------------------------------------------
    fig2, ax = figtem.generate_figure('3x1',figsize=(6.5,5.5),left=0.21,bottom=0.14,right=0.12)
    tlims = array([0.0,6.0])

    # Astrocyte Ca2+
    ax[0].plot(tlims, params['C_Theta']/umole*ones((2,1)), lw=lw, color='c', ls='--')
    ax[0].plot(A2['t'], A2['C'][0] / cf, lw=lw, color='k')
    ax[0].set_xlim(tlims)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    Clims = [-0.02,1.6]
    ax[0].set_ylim(Clims)
    ax[0].set_yticks(arange(0.0,Clims[1],0.5))
    # ax[0].set_ylabel('$C$ (${0}$)'.format(sympy.latex(umole)), fontsize=lfs)
    ax[0].set_ylabel('Ast. Calcium\n(${0}$)'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    ax[0].yaxis.set_label_coords(-.15,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Gliotransmitter variable (x_A)
    analysis.plot_release(G['t'], G['x_A'][0], var='x', tau=1/params['Omega_A'], ax=ax[1])
    figtem.adjust_spines(ax[1], ['left'], position=spine_position)
    ax[1].set_xlim(tlims)
    ax[1].set_xlabel('')
    ax[1].set_ylim([-0.01,1.01])
    ax[1].set_yticks([0.0,0.5,1.0])
    # ax[1].set_ylabel('$x_A$', fontsize=lfs)
    ax[1].set_ylabel('Avail. Gt. Pr.', fontsize=lfs, multialignment='center')
    ax[1].yaxis.set_label_coords(-.18,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # Gliotransmitter release (G_A)
    analysis.plot_release(G['t'], G['G_A'][0] / cf, var='y', ax=ax[2], color='k', linewidth=lw)
    ax[2].set_xlim(tlims)
    figtem.adjust_spines(ax[2], ['left','bottom'], position=spine_position)
    ax[2].set_xlim(tlims)
    ax[2].set_xticks(arange(0.0,6.01,2.0))
    ax[2].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=lfs)
    ax[2].set_ylim([-1.0,80])
    ax[2].set_yticks(arange(0.0,80,25))
    # ax[2].set_ylabel('$G_A$ (${0}$)'.format(sympy.latex(umole)), fontsize=lfs)
    ax[2].set_ylabel('Released Gt.\n(${0}$)'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    ax[2].yaxis.set_label_coords(-.15,0.5)
    # Adjust labels
    set_axlw(ax[2], lw=alw)
    set_axfs(ax[2], fs=afs)

    # Save Figures
    plt.figure(1)
    # plt.savefig('io_syn.svg', format='svg', dpi=600)
    plt.savefig(fig_dir+'fig1_io_syn.'+format, format=format, dpi=600)
    # plt.close(fig0)

    plt.figure(2)
    # plt.savefig('io_ast.svg', format='svg', dpi=600)
    plt.savefig(fig_dir+'fig1_io_ast.'+format, format=format, dpi=600)
    # plt.close(fig1)

    plt.figure(3)
    # plt.savefig('io_glt.svg', format='svg', dpi=600)
    plt.savefig(fig_dir+'fig1_io_glt.'+format, format=format, dpi=600)
    # plt.close(fig2)

    plt.show()

def synapse_io(format='svg', dir='../Figures/'):
    """
    (Lecture-friendly)
    Only plot depressing and facilitating synapses from tripartite_syn together
    """

    # Plotting defaults
    spine_position = 5
    lw = 1.5
    alw = 1.0
    afs = 14
    lfs = 16

    #-------------------------------------------------------------------------------------------------------------------
    # Building Figures
    #-------------------------------------------------------------------------------------------------------------------
    syn_type = ['dep','fac']
    [Sd,_,_,_,_] = svu.loaddata('io_syn_dep.pkl')
    pard = asmod.get_parameters('FM',synapse='depressing')
    [Sf,_,_,_,_] = svu.loaddata('io_syn_fac.pkl')
    parf = asmod.get_parameters('FM',synapse='facilitating')

    plt.close('all')
    #-------------------------------------------------------------------------------------------------------------------
    # Synapse
    #-------------------------------------------------------------------------------------------------------------------
    fig0, ax = figtem.generate_figure('3x1',figsize=(6.5,5.5),left=0.21,bottom=0.14,right=0.12)
    tlims = array([16.0,17])

    # Plot spikes
    analysis.plot_release(Sd['t'], Sd['r_S'][0], var='spk',  ax=ax[0], redraw=True)
    ax[0].set_xlim(tlims)

    # Plot u,x
    # Retrieve u,x traces
    colors_d = {'y1': chex((0,107,164)), 'y2': chex((255,126,14))}
    colors_f = {'y1': 'b', 'y2': chex((200,82,0))}
    u = analysis.build_uxsyn(Sd['t'], Sd['u_S'][0], 1.0/pard['Omega_f'], 'u')
    x = analysis.build_uxsyn(Sd['t'], Sd['x_S'][0], 1.0/pard['Omega_d'], 'x')
    ax[1],ax_aux = plotyy(Sd['t'], u, x, ax=ax[1], colors=colors_d, ls=':')

    # # Format x-axis
    # ax[1].set_xlim(tlims)
    # ax[1].set_xlabel('')
    #
    # # Format y-axis
    # ax[1].set_ylim([-0.01,1.01])
    # ax[1].set_yticks([0.0,0.5,1.0])
    # # ax[1].set_ylabel('$u_S$', fontsize=lfs)
    # ax[1].set_ylabel('Nt. Rel. Pr.', fontsize=lfs, va='center')
    # ax[1].yaxis.set_label_coords(-.2,0.5)
    # set_axlw(ax[1], lw=alw)
    # set_axfs(ax[1], fs=afs)

    ax_aux.set_ylim([-0.01,1.01])
    ax_aux.set_yticks([])
    # # ax_aux.set_ylabel('$x_S$', fontsize=lfs)
    # ax_aux.set_ylabel('Avail. Nt. Pr.', fontsize=lfs)
    # Adjust labels
    set_axlw(ax_aux, lw=alw)
    set_axfs(ax_aux, fs=afs)

    # Add facilitating synapse
    u = analysis.build_uxsyn(Sf['t'], Sf['u_S'][0], 1.0/parf['Omega_f'], 'u')
    x = analysis.build_uxsyn(Sf['t'], Sf['x_S'][0], 1.0/parf['Omega_d'], 'x')
    ax[1],ax_aux = plotyy(Sf['t'], u, x, ax=ax[1], colors=colors_f,ls='-')
    # Format x-axis
    ax[1].set_xlim(tlims)
    ax[1].set_xlabel('')

    # Format y-axis
    ax[1].set_ylim([-0.01,1.01])
    ax[1].set_yticks([0.0,0.5,1.0])
    # ax[1].set_ylabel('$u_S$', fontsize=lfs)
    ax[1].set_ylabel('Nt. Rel. Pr.', fontsize=lfs, va='center')
    ax[1].yaxis.set_label_coords(-.2,0.5)
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    ax_aux.set_ylim([-0.01,1.01])
    ax_aux.set_yticks([0.0,0.5,1.0])
    # ax_aux.set_ylabel('$x_S$', fontsize=lfs)
    ax_aux.set_ylabel('Avail. Nt. Pr.', fontsize=lfs)
    # Adjust labels
    set_axlw(ax_aux, lw=alw)
    set_axfs(ax_aux, fs=afs)

    # Add Y_S
    cf = 1*umole / mole
    analysis.plot_release(Sd['t']-tlims[0], Sd['Y_S'][0] / cf, var='y', var_units=umole,
                          time_units=ms, ax=ax[2], color='k', redraw=True, spine_position=5, linestyle=':')
    analysis.plot_release(Sf['t']-tlims[0], Sf['Y_S'][0] / cf, var='y', var_units=umole,
                          time_units=ms, ax=ax[2], color=chex((48,147,67)), redraw=True, spine_position=5, linestyle='-')
    ax[2].set_xlim(tlims-tlims[0])
    ax[2].set_xticks(arange(0.0,1.1,0.2))
    ax[2].set_xticklabels([str(int(t*1e3)) for t in ax[2].get_xticks()])
    ax[2].set_ylim([-6.0,1300])
    ax[2].set_yticks(arange(0.0,1201,400))
    ax[2].set_ylabel('Released Nt.\n(${0}$)'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    ax[2].yaxis.set_label_coords(-.13,0.5)
    # Adjust labels
    set_axlw(ax[2], lw=alw)
    set_axfs(ax[2], fs=afs)

    plt.savefig(dir+'TM_synaptic_dynamics_2.'+format, format=format, dpi=600)

    plt.show()

def tripartite_io(sim=True, format='svg', fig_dir='../Figures/'):
# def tripartite_io(sim=False, format='svg', fig_dir='../Figures/'):
    tripartite_syn(sim=sim, format=format, fig_dir=fig_dir)
    # synapse_io(format=format, fig_dir=fig_dir)

#-----------------------------------------------------------------------------------------------------------------------
# Figure 2
#-----------------------------------------------------------------------------------------------------------------------
def stp_regulation(sim=False, synapse='facilitating', format='eps', data_dir='../data/', fig_dir='../Figures/'):
# def stp_regulation(sim=False,synapse='depressing', format='eps'):
    '''
    Build figure considering the presynaptic effect
    '''

    # Plotting defaults
    spine_position = 5
    lw = 1.5
    alw = 1.0
    afs = 14
    lfs = 16
    # color = {'facilitating': 'g', 'depressing': 'r', 'intermediate': chex((143,135,130))}
    color = {'facilitating': chex((44,160,44)),
             'depressing'  : chex((214,39,40)),
             'intermediate': chex((143,135,130))}

    # sim = True

    if synapse=='facilitating' :
        params = asmod.get_parameters(synapse=synapse)
        alpha = [1.0,(1-params['U_0__star'])/2.0]
    else :
        params = asmod.get_parameters(synapse=synapse)
        alpha = [0.0,params['U_0__star']/2.0]

    # Adjust parameters for the simulations
    params['f_c'] = 0.2*Hz
    params['rho_e'] = 1.0e-4
    params['O_G'] = 0.6/umole/second
    params['Omega_G'] = 1.0/(30*second)
    params['t_off'] = 21*second
    params['G_e'] = 3*mV

    # Add synaptic weight conversion to interface with postsynaptic LIF
    # params['we'] = (60*0.27/10)*mV / (params['rho_c']*params['Y_T']) # excitatory synaptic weight (voltage)
    params['we'] = (60*0.27/10) / (params['rho_c']*params['Y_T'])

    # General parameters for simulation also used in the analysis
    duration = 90

    # Stimulation (used in Figures too)
    offset = 0.05
    isi = 0.1
    stim = offset+arange(0.0,duration,2.0)
    spikes = sort(concatenate((stim, stim+isi)))*second

    if sim :
        # Synapses (2 with different xi-values)
        source_group,target_group,synapses = asmod.synapse_model(params, N_syn=2, connect='i==j',
                                                                 linear=False,
                                                                 post=True,
                                                                 stimulus='test',
                                                                 spikes=spikes)
        # Postysnaptic neuron settings
        target_group.v = params['E_L']
        # Synapses' settings
        synapses.alpha = alpha

        # Gliotransmitter release
        gliot = asmod.gliotransmitter_release(None, params, standalone=True, N_astro=1)
        gliot.f_c = params['f_c']

        # Connection to synapses
        ecs = asmod.extracellular_astro_to_syn(gliot, synapses, params, connect=True)

        # Set monitors
        pre = StateMonitor(synapses, ['Gamma_S','ge'], record=True, dt=0.02*second)
        post = StateMonitor(target_group, ['v'], record=True)
        glt = StateMonitor(gliot, ['G_A'], record=True, dt=0.02*second)

        # Run
        run(duration*second,namespace={},report='text')

        # Convert monitors to dictionaries and save them for analysis
        pre_mon = bu.monitor_to_dict(pre)
        post_mon = bu.monitor_to_dict(post)
        glt_mon = bu.monitor_to_dict(glt)

        # Save data
        svu.savedata([pre_mon,post_mon,glt_mon],data_dir+'stp_regulation_'+synapse[:3]+'.pkl')

    #-------------------------------------------------------------------------------------------------------------------
    # Data Analysis
    #-------------------------------------------------------------------------------------------------------------------
    # Load data
    [pre,post,glt] = svu.loaddata(data_dir+'stp_regulation_'+synapse[:3]+'.pkl')

    # Define some lambdas
    u0_sol = lambda gamma_S, u0_star, xi : (1.0-gamma_S)*u0_star + xi*gamma_S

    # # Generate Figures
    fig1, ax = figtem.generate_figure('2x1',figsize=(6.0,5.5),left=0.18,bottom=0.15,right=0.05,top=0.08)
    tlims = [0.0, duration]

    # Plot G_A (gliotransmitter release)
    cf = 1*umole / mole
    analysis.plot_release(glt['t'], glt['G_A'][0] / cf, var='y', ax=ax[0], color='k', linewidth=lw)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(tlims)
    ax[0].set_xlabel('')
    ax[0].set_ylim([-1.0,15])
    ax[0].set_yticks(arange(0.0,15.1,5))
    # ax[0].set_ylabel('$G_A$ (${0}$)'.format(sympy.latex(umole)), fontsize=lfs)
    ax[0].set_ylabel('Released Gt.\n(${0}$)'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    ax[0].yaxis.set_label_coords(-0.09,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Plot Gamma_S (presynaptic receptors)
    ax[1].plot(pre['t'], pre['Gamma_S'][0], color='k', linewidth=lw)
    figtem.adjust_spines(ax[1], ['left','bottom'], position=spine_position)
    ax[1].set_xlim(tlims)
    ax[1].set_xticks(arange(0.0,91.0,30.0))
    ax[1].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=lfs)
    ax[1].set_ylim([-0.01,1.01])
    ax[1].set_yticks([0.0,0.5,1.0])
    # ax[1].set_ylabel('$\gamma_S$', fontsize=lfs)
    ax[1].set_ylabel('Bound\nPresyn. Rec.', fontsize=lfs, multialignment='center')
    ax[1].yaxis.set_label_coords(-0.1,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    fig2, ax = figtem.generate_figure('2x1_1L2S',figsize=(6.0,5.5),left=0.18,bottom=0.15,right=0.05,top=0.08,vs=[0.15],hs=[0.15])
    # Plot U_0
    ax[0].plot(pre['t'], u0_sol(pre['Gamma_S'][0],params['U_0__star'],alpha[0]), color=color[synapse], linewidth=lw)
    ax[0].plot(pre['t'], u0_sol(pre['Gamma_S'][1],params['U_0__star'],alpha[1]), color=color['intermediate'], linewidth=lw)
    ax[0].plot(tlims, params['U_0__star']*ones((2,1)), ls='--', color='k', linewidth=lw)
    figtem.adjust_spines(ax[0], ['left', 'bottom'], position=spine_position)
    ax[0].set_xlim(tlims)
    ax[0].set_xticks(arange(0.0,91.0,30.0))
    ax[0].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=lfs)
    ax[0].set_ylim([-0.01,1.01])
    ax[0].set_yticks([0.0,0.5,1.0])
    # ax[0].set_ylabel('$U_0$', fontsize=lfs)
    ax[0].set_ylabel('Synaptic\n Release Pr.', fontsize=lfs, multialignment='center')
    ax[0].yaxis.set_label_coords(-0.10,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Plot PPR(t)
    # First retrieve PSC peaks
    post['v'] -= params['E_L']/volt
    i0 = sg.argrelextrema(post['v'][0], np.greater)[0]
    i1 = sg.argrelextrema(post['v'][1], np.greater)[0]
    peaks_0 = post['v'][0][i0]
    peaks_1 = post['v'][1][i1]

    # Compute PPRs
    ppr_ref = peaks_0[1]/peaks_0[0]
    ppr_0 = (peaks_0[1::2]/peaks_0[::2])/ppr_ref*100
    ppr_1 = (peaks_1[1::2]/peaks_1[::2])/ppr_ref*100

    # Effective plotting
    time = post['t'][i0[1::2]]
    ax[1].plot(time, ppr_0, color=color[synapse], linewidth=lw, ls='none', marker='o')
    ax[1].plot(time, ppr_1-2.5, color=color['intermediate'], linewidth=lw, ls='none', marker='o')
    figtem.adjust_spines(ax[1], ['left', 'bottom'], position=spine_position)
    ax[1].set_xlim(tlims)
    ax[1].set_xticks(arange(0.0,91.0,30.0))
    ax[1].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=lfs)
    if synapse=='facilitating':
        ax[1].set_ylim([0.0,125])
        ax[1].set_yticks(arange(0.0,121,20))
    else:
        ax[1].set_ylim([79,250])
        ax[1].set_yticks(arange(100,251,50))
    ax[1].set_ylabel('PPR (%PPR$_0$)', fontsize=lfs)
    ax[1].yaxis.set_tick_params(labelsize=afs)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # Plot PSC
    R_soma = 60 * Mohm
    cf = 1*pamp / amp
    indices = where((post['t']>offset-0.02) & (post['t']<offset+isi+0.05))
    time_ref = post['t'][indices] * 1e3  # convert to ms
    time_ref -= time_ref[0]
    psc_ref = post['v'][0][indices]
    psc_ref /= R_soma * cf

    t_off = 10.0
    indices = where((post['t'] >= offset-0.02+t_off) & (post['t']< offset+isi+0.05+t_off))
    time_mod = post['t'][indices] * 1e3  # convert to ms
    time_mod -= time_mod[0]
    psc_0 = post['v'][0][indices]
    psc_0 /= R_soma * cf

    tlims = [0.0,150]
    ax[2].plot(time_ref+5, -psc_ref, color='k', linewidth=lw)
    ax[2].plot(time_mod, -psc_0, color=color[synapse], linewidth=lw, ls='-')
    figtem.adjust_spines(ax[2], ['left', 'bottom'], position=spine_position)
    ax[2].set_xlim(tlims)
    ax[2].set_xticks(arange(0.0,151,50))
    ax[2].set_xlabel('Time (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[2].set_ylim([-31,0.1])
    ax[2].set_yticks(arange(-30,0.1,10))
    ax[2].set_ylabel('PSC (pA)', fontsize=lfs)
    ax[2].yaxis.set_tick_params(labelsize=afs)
    # Adjust labels
    set_axlw(ax[2], lw=alw)
    set_axfs(ax[2], fs=afs)

    # Save Figures
    plt.figure(1)
    plt.savefig(fig_dir+'fig2_stpreg_glt.'+format, format=format, dpi=600)
    # plt.close(fig0)

    plt.figure(2)
    plt.savefig(fig_dir+'fig2_stpreg_rel_'+synapse[:3]+'.'+format, format=format, dpi=600)
    # plt.close(fig1)

    plt.show()

# LECTURE friendly
def regulation_check(sim=False, synapse='facilitating', format='eps', data_dir='../data/', fig_dir='../Figures/'):
    '''
    Build figure considering the presynaptic effect -- Lecture Friendly
    '''

    # Plotting defaults
    spine_position = 5
    lw = 1.5
    alw = 1.0
    afs = 14
    lfs = 16
    # color = {'facilitating': 'g', 'depressing': 'r', 'intermediate': chex((143,135,130))}
    color = {'facilitating': chex((44,160,44)),
             'depressing'  : chex((214,39,40)),
             'intermediate': chex((143,135,130))}

    # sim = True

    if synapse=='facilitating' :
        params = asmod.get_parameters(synapse=synapse)
        alpha = [params['U_0__star'],1.0]
    else :
        params = asmod.get_parameters(synapse=synapse)
        alpha = [params['U_0__star'],0.0]

    # General parameters for simulation also used in the analysis
    duration = 60*second

    # Adjust parameters for the simulations
    params['f_c'] = 1./duration
    params['rho_e'] = 1.0e-4
    params['O_G'] = 0.6/umole/second
    params['Omega_G'] = 1.0/(8*second)
    params['t_off'] = 21*second

    # Add synaptic weight conversion to interface with postsynaptic LIF
    params['we'] = (60*0.27/10)*mV / (params['rho_c']*params['Y_T']) # excitatory synaptic weight (voltage)

    # Stimulation (used in Figures too)
    # offset = 0.2
    # isi = 0.1
    # stim = offset+arange(0.0,duration,2.0)
    # spikes = sort(concatenate((stim, stim+isi)))*second

    params['f_in'] = 1.0*Hz
    N_syn = 50
    # spikes = spg.spk_poisson(T=duration/second, N=1, rate=f_in, trp=params['tau_r']/second)[0]*second

    # Lecture version
    if sim :
        keys = ['ctrl','glt']
        # keys = ['glt','ctrl']
        P, G = {}, {}
        for i in range(size(alpha)):
        # for i in xrange(1):
            # N_syn and repeat for two xi values
            source_group,target_group,synapses = asmod.synapse_model(params, N_syn=N_syn, connect='i==j',
                                                                     linear=False,
                                                                     post=True,
                                                                     stimulus='poisson')
            source_group.f_in = params['f_in']
            synapses.alpha = alpha[i]

            # Gliotransmitter release
            gliot = asmod.gliotransmitter_release(None, params, standalone=True, N_astro=1)
            gliot.f_c = params['f_c']
            gliot.v_A = 0.99

            # Connection to synapses
            ecs = asmod.extracellular_astro_to_syn(gliot, synapses, params, connect=True)

            # Set monitors
            post = StateMonitor(target_group, ['g_e'], record=True, dt=2*ms)

            # Run
            run(duration,namespace={},report='text')

            # Convert monitors to dictionaries and save them for analysis
            P[keys[i]] = bu.monitor_to_dict(post)

        # Save data
        # svu.savedata([P],data_dir+'consistency_check_'+synapse[:3]+'.pkl')
        # svu.savedata([G,P],data_dir+'consistency_check_'+synapse[:3]+'.pkl')

    #-------------------------------------------------------------------------------------------------------------------
    # Data Analysis
    #-------------------------------------------------------------------------------------------------------------------
    # Load data
    P = svu.loaddata(data_dir+'consistency_check_'+synapse[:3]+'.pkl')[0]
    G,P = svu.loaddata(data_dir+'consistency_check_'+synapse[:3]+'.pkl')

    cf = .9
    before = cf*mean(ravel(P['ctrl']['g_e'])[argrelmax(ravel(P['ctrl']['g_e']))[0]])
    psc = ravel(P['ctrl']['g_e'])
    argpeaks = argrelmax(ravel(P['ctrl']['g_e']))[0]
    nbins = 4
    bins = r_[arange(0,size(P['glt']['t']),size(P['glt']['t'])/nbins),size(P['glt']['t'])]
    m,s = [0]*nbins,[0]*nbins
    for i in range(0,size(bins)-1):
        m[i] = mean(psc[argpeaks[((argpeaks>=bins[i])&(argpeaks<bins[i+1]))]])/before*100
        s[i] = std(psc[argpeaks[((argpeaks>=bins[i])&(argpeaks<bins[i+1]))]])/before*100

    # Fake consistency check
    nbins = 5
    m = [1.,1.9,2.5,2.3,1.5]
    s = vstack((zeros((1,nbins)),[0.,.4,.25,.21,.3]))
    colors = ['0.99','0.8','0.6','0.7','0.99']
    binLabel = ['Before','15 s','30 s','40 s','60 s']

    # Increase of synaptic release
    width = 0.5
    offset = 0.2
    fig1, ax = figtem.generate_figure('1x1',figsize=(6.0,4.5),left=0.15,bottom=0.15,right=0.05,top=0.08)
    rects = ax[0].bar(offset+arange(nbins),m,width,yerr=s,
                   color=colors,
                   error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2))
    figtem.adjust_spines(ax[0], ['left','bottom'], position=spine_position)
    ax[0].set_xticks(offset+width/2+arange(nbins))
    ax[0].set_xticklabels(binLabel,fontsize=lfs)
    ax[0].set_yticks(arange(0.0,3.1,1.))
    ax[0].set_ylabel('$\Delta$% Before Frequency'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    # # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # PPR
    # Fake consistency check
    nbins = 2
    m = [143,75.8]
    s = vstack((zeros((1,nbins)),[6.8,8.2]))
    colors = ['0.','0.99']
    binLabel = ['eEPSC\n amplitude','PPR']

    # Increase of synaptic release
    width = 0.3
    offset = 0.3
    fig1, ax = figtem.generate_figure('1x1',figsize=(6.0,4.5),left=0.15,bottom=0.15,right=0.05,top=0.08)
    ax[0].plot([0.,offset*2+width*5],[100.,100.],'k--',lw=lw)
    rects = ax[0].bar(offset+arange(nbins),m,width,yerr=s,
                   color=colors,
                   error_kw=dict(ecolor='black', lw=2, capsize=5, capthick=2))
    figtem.adjust_spines(ax[0], ['left','bottom'], position=spine_position)
    ax[0].set_xlim([0.,offset*2+width*5])
    ax[0].set_xticks(offset+width/2+arange(nbins))
    ax[0].set_xticklabels(binLabel,fontsize=lfs)
    ax[0].set_yticks(arange(0.0,150.1,50.))
    ax[0].set_ylabel('$\Delta$% Before'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    # # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Save Figures
    plt.figure(1)
    plt.savefig(fig_dir+'cons_check_Fiacco.'+format, format=format, dpi=600)
    # plt.close(fig0)

    plt.figure(2)
    plt.savefig(fig_dir+'cons_check_Jourdain.'+format, format=format, dpi=600)
    # plt.close(fig1)

    plt.show()

def u0_fun(oma,omg,ua,ja,xi,u0,fc):
    return (omg*oma*u0+(omg*u0+xi*ja*oma)*ua*fc)/(omg*oma+(ja*oma+omg)*ua*fc)

def u_thr(omd,omf):
    return omd/(omd+omf)

def fc_thr(oma,omg,ua,ja,xi,uthr,u0):
    return omg*oma*(uthr-u0)/(ja*oma*ua*(xi-uthr)-omg*ua*uthr+omg*ua*u0)

def glt_threshold(format='svg', fig_dir='../Figures/'):

    hill = lambda x, k, n : (x**n)/(x**n + k**n)
    noise = lambda a, b, N: a + (b-a)*rand(N)

    NP = 15
    n = 5
    x = logspace(log10(0.003),log10(0.7),NP)*Hz

    # Plotting defaults
    spine_position = 5
    lw = 1.5
    alw = 1.0
    afs = 14
    lfs = 16
    fmt = 'o'
    ms = 8
    colors = {'depressing' : chex((255,102,0)),
              'facilitating' : chex((0,128,0)),
              'uthr': 'b',
              'u0'  : chex((210,0,0)),
              'fct' : 'm'}

    params = {'fac': asmod.get_parameters(synapse='facilitating'),
              'dep': asmod.get_parameters(synapse='depressing')}
    # params['fac']['U_0__star'] = 0.35
    params['fac']['rho_e'] = 1.0e-4
    params['fac']['Omega_G'] = 1.0/(15*second)
    params['dep']['rho_e'] = 1.0e-4
    params['dep']['Omega_G'] = 1.0/(15*second)

    NPTS = 50
    yLim = [0.,1.]
    fcLim = [0.001,1.]
    fc = logspace(log10(fcLim[0]),log10(fcLim[1]),NPTS)*Hz

    u0 = {'fac': u0_fun(params['fac']['Omega_A'],params['fac']['Omega_G'],params['fac']['U_A'],
                        params['fac']['rho_e']*params['fac']['O_G']*params['fac']['G_T']/params['fac']['Omega_e'],
                        params['fac']['alpha'],params['fac']['U_0__star'],fc),
          'dep': u0_fun(params['dep']['Omega_A'],params['dep']['Omega_G'],params['dep']['U_A'],
                        params['dep']['rho_e']*params['dep']['O_G']*params['dep']['G_T']/params['dep']['Omega_e'],
                        params['dep']['alpha'],params['dep']['U_0__star'],fc)}

    uthr = {'fac': u_thr(params['fac']['Omega_d'],params['fac']['Omega_f']),
            'dep': u_thr(params['dep']['Omega_d'],params['dep']['Omega_f'])}

    fct = {'fac': fc_thr(params['fac']['Omega_A'],params['fac']['Omega_G'],params['fac']['U_A'],
                         params['fac']['rho_e']*params['fac']['O_G']*params['fac']['G_T']/params['fac']['Omega_e'],
                         params['fac']['alpha'],uthr['fac'],params['fac']['U_0__star']),
           'dep': fc_thr(params['dep']['Omega_A'],params['dep']['Omega_G'],params['dep']['U_A'],
                         params['dep']['rho_e']*params['dep']['O_G']*params['dep']['G_T']/params['dep']['Omega_e'],
                         params['dep']['alpha'],uthr['dep'],params['dep']['U_0__star'])}

    # Fake data
    data = {'fac': u0_fun(params['fac']['Omega_A'],params['fac']['Omega_G'],params['fac']['U_A'],
                        params['fac']['rho_e']*params['fac']['O_G']*params['fac']['G_T']/params['fac']['Omega_e'],
                        params['fac']['alpha'],params['fac']['U_0__star'],x)+noise(0,0.05,x.size),
          'dep': u0_fun(params['dep']['Omega_A'],params['dep']['Omega_G'],params['dep']['U_A'],
                        params['dep']['rho_e']*params['dep']['O_G']*params['dep']['G_T']/params['dep']['Omega_e'],
                        params['dep']['alpha'],params['dep']['U_0__star'],x)+noise(0,0.05,x.size)}
    cf = .3
    b = cf*(1.-hill(x/Hz,0.9,1)) #x/Hz*exp(-cf*x/Hz)
    sdata = {'fac': noise(0.,b,x.size),
             'dep': noise(0.,b,x.size)}

    fig1, ax = figtem.generate_figure('1x1',figsize=(6.0,4.5),left=0.18,bottom=0.18,right=0.05,top=0.08)
    yLim = [.0,.65]
    ax[0].add_artist(mpatches.Polygon(list(zip(*vstack(([fcLim[0],fct['dep'],fct['dep'],fcLim[0]],[uthr['dep'],uthr['dep'],yLim[1],yLim[1]])))), ec='none', fc=colors['depressing']))
    ax[0].add_artist(mpatches.Polygon(list(zip(*vstack(([fcLim[1],fct['dep'],fct['dep'],fcLim[1]],[uthr['dep'],uthr['dep'],yLim[0],yLim[0]])))), ec='none', fc=colors['facilitating']))
    ax[0].plot(fcLim,[uthr['dep'],uthr['dep']],'-',color=colors['uthr'],lw=lw)
    ax[0].errorbar(x/Hz,data['dep'],yerr=sdata['dep'],fmt=fmt,color='k',markersize=ms)
    print(x,data['dep'],sdata['dep'])
    ax[0].plot(fc,u0['dep'],'-',color=colors['u0'],lw=lw*1.5)
    ax[0].plot([fct['dep'],fct['dep']],yLim,':',color=colors['fct'],lw=lw*2)
    # Adjust axes
    figtem.adjust_spines(ax[0], ['left','bottom'], position=spine_position)
    ax[0].set_xscale('log')
    ax[0].set_xlim(fcLim)
    ax[0].set_xticklabels([str(xt) for xt in ax[0].get_xticks()])
    ax[0].set_ylim(yLim)
    ax[0].set_yticks(arange(0.,0.61,0.2))
    ax[0].set_xlabel(r'Gt. Release Rate, $\nu_A$ (${0}$)'.format(sympy.latex(Hz)), fontsize=lfs, multialignment='center')
    ax[0].set_ylabel('Syn. Rel. Pr.', fontsize=lfs, multialignment='center')
    ax[0].yaxis.set_label_coords(-.13,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    fig2, ax = figtem.generate_figure('1x1',figsize=(6.0,4.5),left=0.18,bottom=0.18,right=0.05,top=0.08)
    yLim = [.15,1.]
    ax[0].add_artist(mpatches.Polygon(list(zip(*vstack(([fcLim[0],fct['fac'],fct['fac'],fcLim[0]],[uthr['fac'],uthr['fac'],yLim[0],yLim[0]])))), ec='none', fc=colors['facilitating']))
    ax[0].add_artist(mpatches.Polygon(list(zip(*vstack(([fcLim[1],fct['fac'],fct['fac'],fcLim[1]],[uthr['fac'],uthr['fac'],yLim[1],yLim[1]])))), ec='none', fc=colors['depressing']))
    ax[0].plot(fcLim,[uthr['fac'],uthr['fac']],'-',color=colors['uthr'],lw=lw)
    ax[0].errorbar(x/Hz,data['fac'],yerr=sdata['fac'],fmt=fmt,color='k',markersize=ms)
    ax[0].plot(fc,u0['fac'],'-',color=colors['u0'],lw=lw*1.5)
    ax[0].plot([fct['fac'],fct['fac']],yLim,':',color=colors['fct'],lw=lw*2)
    ax[0].set_xscale('log')
    # Adjust axes
    figtem.adjust_spines(ax[0], ['left','bottom'], position=spine_position)
    ax[0].set_xscale('log')
    ax[0].set_xlim(fcLim)
    ax[0].set_xticklabels([str(xt) for xt in ax[0].get_xticks()])
    ax[0].set_ylim(yLim)
    ax[0].set_yticks(arange(0.15,1.1,0.2))
    ax[0].set_xlabel(r'Gt. Release Rate, $\nu_A$ (${0}$)'.format(sympy.latex(Hz)), fontsize=lfs, multialignment='center')
    ax[0].set_ylabel('Syn. Rel. Pr.', fontsize=lfs, multialignment='center')
    ax[0].yaxis.set_label_coords(-.13,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Save Figures
    plt.figure(1)
    plt.savefig(fig_dir+'stp_switch_dep.'+format, format=format, dpi=600)
    # plt.close(fig0)

    plt.figure(2)
    plt.savefig(fig_dir+'stp_switch_fac.'+format, format=format, dpi=600)
    # plt.close(fig1)

    plt.show()

def pre_regulation(sim=True, format='svg', fig_dir='../Figures/'):
# def pre_regulation(sim=False, format='svg', fig_dir='../Figures/'):
    stp_regulation(sim=sim, synapse='facilitating', format=format, fig_dir=fig_dir)
    stp_regulation(sim=sim, synapse='depressing', format=format, fig_dir=fig_dir)
    # regulation_check(sim=sim, synapse='facilitating', format=format, fig_dir=fig_dir)
    # glt_threshold(format=format, fig_dir=fig_dir)

#-----------------------------------------------------------------------------------------------------------------------
# Figure 3
#-----------------------------------------------------------------------------------------------------------------------
def syn_filtering(sim=True, format='svg', data_dir='../data/', fig_dir='../Figures/'):
# def syn_filtering(sim=False, format='svg', data_dir='../data/', fig_dir='../Figures/'):
    # Global variable
    synapses = ['facilitating','depressing']
    # synapses = ['facilitating']
    # synapses = ['depressing']
    #---------------------------------------------------------------------------------------------------------------
    # Simulation /1 : Synaptic filter (handling of simulation is internal in analysis module)
    #---------------------------------------------------------------------------------------------------------------
    # Model size
    N_points = 30
    N_trials = 10
    duration = 60*second

    if sim:
        #---------------------------------------------------------------------------------------------------------------
        # Simulation /1 : I/O synapse and equivalent change in the release probability
        #---------------------------------------------------------------------------------------------------------------
        freq_range = [0.1,100.0]
        R = {}
        for i, syn in enumerate(synapses):
            params = asmod.get_parameters(synapse=syn)
            mon,_ = analysis.rr_mean(params, N_points, N_trials, freq_range=[0.1,100], gliot=False,
                                     duration=duration, plot_results=False)
            R['nog'] = mon
            mon,freq_vals = analysis.rr_mean(params, N_points, N_trials, freq_range=[0.1,100], gliot=True,
                                             duration=duration, plot_results=False)
            R['glt_open'] = mon
            mon,freq_vals = analysis.rr_mean(params, N_points, N_trials, freq_range=[0.1,100], gliot=True, ics=0.0,
                                             duration=duration, plot_results=False)
            R['glt_closed'] = mon

            # Save data
            svu.savedata([R, freq_vals], data_dir+'syn_filter_'+syn[:3]+'.pkl')

        #---------------------------------------------------------------------------------------------------------------
        # Simulation /2 : Stepped stimulation and processing
        #---------------------------------------------------------------------------------------------------------------
        f_in = {'facilitating' : [0.0, 1.0, 8.0, 32.0], # in Hz
                'depressing'   : [0.0, 1.0, 8.0, 32.0]}
        t_step = 2.0 # second
        duration = 8 * second

        for syn in synapses:
            # Retrieve parameters
            params = asmod.get_parameters(synapse=syn)
            params['f_c'] = 1.0/t_step * Hz
            # params['we'] = (60*0.27/10)*mV / (params['rho_c']*params['Y_T']) # excitatory synaptic weight (voltage)
            params['we'] = (60*0.27/10)/(params['rho_c']*params['Y_T'])  # excitatory synaptic weight (voltage)
            stim_opts = {'f_in' : f_in[syn], 't_step' : t_step}

            if syn=='depressing':
                params['alpha'] = params['U_0__star']/8
            # Simulation parameters
            N_syn = 1000
            alpha = [params['U_0__star'], params['alpha']]
            neuron_in,neuron_out,syns,gliot,es_astro2syn = asmod.openloop_model(params, N_syn=N_syn*size(alpha), N_astro=1, ics=None,
                                                                                linear=False, post=True, sic=None, stdp=False,
                                                                                stimulus_syn='steps', stim_opts=stim_opts,
                                                                                stimulus_glt='periodic')
            syns.alpha = tile(alpha, (1,N_syn))[0]
            gliot.f_c = params['f_c']
            gliot.v_A = '1.0-f_c*second'     # In this fashion the first spike will occur at t=t_step

            # Monitors
            # N = SpikeMonitor(neuron_in, record=True)
            N = StateMonitor(neuron_in, variables=['f_in'], record=0, dt=5*ms)
            PSC = StateMonitor(neuron_out, variables=['g_e'], record=True, dt=1*ms)

            # Run simulation
            run(duration, namespace={}, report='text')

            # Save data
            N_mon = bu.monitor_to_dict(N)
            PSC_mon = bu.monitor_to_dict(PSC)
            svu.savedata([N_mon,PSC_mon], data_dir+'syn_comp_'+syn[:3]+'.pkl')

    #-------------------------------------------------------------------------------------------------------------------
    # Plotting
    #-------------------------------------------------------------------------------------------------------------------
    # Plotting defaults
    spine_position = 5
    lw = 1.8
    alw = 1.0
    afs = 14
    lfs = 16
    legfs = 12
    fmt = 'o'
    colors = {'facilitating' : chex((44,160,44)),
              'depressing'   : chex((214,39,40)),
              'closed_loop'  : chex((23,190,207))}

    # Define some lambdas
    # Functions to compute mean and std of released resources
    r_mean = lambda rv, npts : [mean(rv[i::npts]) for i in range(npts)] # mean r_S per freq value
    r_std  = lambda rv, npts : [std(rv[i::npts]) for i in range(npts)]  # std on r_S per freq value

    for i,syn in enumerate(synapses):
        #---------------------------------------------------------------------------------------------------------------
        # Simulation /1 : Synaptic filtering
        #---------------------------------------------------------------------------------------------------------------
        # Generate figure
        _, ax = figtem.generate_figure('2x1', figsize=(5.0,5.0), left=0.20, right=0.08, bottom=0.15, top=0.05, vs=[0.05])

        # Load data
        [R, freq_vals] = svu.loaddata(data_dir+'syn_filter_'+syn[:3]+'.pkl')

        # Plot
        xticks = [0.1,1.0,10,100]
        ylims = [-0.01,1.0]
        yticks = arange(0.0,1.01,0.2)
        # color = ['k', colors[syn], colors['closed_loop']]
        color = {'nog': 'k', 'glt_open': colors[syn], 'glt_closed': colors['closed_loop']}
        label = {'nog': 'no gliot.', 'glt_open': 'heterosyn. gliot.', 'glt_closed': 'homosyn. gliot.'}

        # Add legend as a proxy artist
        keys_legend = ['nog','glt_open','glt_closed']
        patches_legend, labels_legend = [], []
        for k in keys_legend:
            # Add artist for legend
            patches_legend.append(mpatches.Patch(color=color[k]))
            labels_legend.append(label[k])
        # Add Legend
        ax[1].legend(patches_legend, labels_legend,
                     fontsize=legfs, frameon=False, loc=1)

        # Plot data
        for _,k in enumerate(R.keys()):
            ax[0].errorbar(freq_vals,r_mean(R[k]['U_0'],N_points),yerr=r_std(R[k]['U_0'],N_points),fmt=fmt,color=color[k])
            ax[1].errorbar(freq_vals,r_mean(R[k]['r_S'],N_points),yerr=r_std(R[k]['r_S'],N_points),fmt=fmt,color=color[k])
            for j in range(2): ax[j].set_xscale('log')

        figtem.adjust_spines(ax[0], ['left'], position=spine_position)
        figtem.adjust_spines(ax[1], ['left','bottom'], position=spine_position)
        # ax[1].set_xlabel('$f_{0}$ (${1}$)'.format('{pre}',sympy.latex(Hz)))
        ax[1].set_xlabel('Presynaptic Firing (${1}$)'.format('{pre}',sympy.latex(Hz)), fontsize=lfs)
        # ax[0].set_ylabel(r'$\langle U_0\rangle$')
        # ax[1].set_ylabel(r'$\langle r_S\rangle$')
        ax[0].set_ylabel('Synaptic\nRelease Pr.', fontsize=lfs, multialignment='center')
        ax[1].set_ylabel('Released\nNt. Resources', fontsize=lfs, multialignment='center')

        for i in range(2):
            ax[i].set_xscale('log')
            ax[i].set_ylim(ylims)
            ax[i].set_yticks(yticks)
            ax[i].yaxis.set_tick_params(labelsize=afs)
            # Adjust labels
            set_axlw(ax[i], lw=alw)
            set_axfs(ax[i], fs=afs)

        ax[0].xaxis.set_visible(False)
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels([str(xt) for xt in xticks])
        ax[1].xaxis.set_tick_params(labelsize=afs)

        # Save figure
        figure(plt.get_fignums()[-1])
        plt.savefig(fig_dir+'fig3_syn_filter_'+syn[:3]+'.'+format, format=format, dpi=600)

        #---------------------------------------------------------------------------------------------------------------
        # Simulation /2 : Synaptic processing
        #---------------------------------------------------------------------------------------------------------------
        # Generate figure
        _, ax = figtem.generate_figure('2x1_custom', figsize=(4.5,5.0), left=0.20, right=0.08, bottom=0.15, top=0.05, vs=[0.05])

        # Load data
        N,PSC = svu.loaddata(data_dir+'syn_comp_'+syn[:3]+'.pkl')

        # Additional data (for rescaling PSCs)
        R_soma = 60 * Mohm
        cf = pamp / (50*amp)

        # Plot frequency stimulation
        tlims = [0.0, 8.0]
        xticks = arange(0.0,8.01,2)
        ylims = [-0.5,33]
        # yticks = arange(0.0,30.1,10)
        ax[0].plot(N['t'],N['f_in'][0],c='k', lw=lw)
        figtem.adjust_spines(ax[0], [], position=spine_position)
        ax[0].set_xlim(tlims)
        ax[0].set_ylim(ylims)
        # ax[0].set_yticks(yticks)
        # ax[0].set_ylabel('$\\nu_{0}$ (${1}$)'.format('{pre}',sympy.latex(Hz)), fontsize=lfs)
        # ax[0].yaxis.set_tick_params(labelsize=afs)
        # # Adjust labels
        # set_axlw(ax[0], lw=alw)
        # set_axfs(ax[0], fs=afs)

        # Plot PSCs
        ylims = [-1.0, 60.0]
        yticks = arange(0,60.1,20)
        ax[1].plot(PSC['t'], mean(PSC['g_e'][::2]/(R_soma*cf),axis=0), c='k', lw=lw, zorder=9)    # original PSC (w/out gliotransmission)
        ax[1].plot(PSC['t'], mean(PSC['g_e'][1::2]/(R_soma*cf),axis=0), c=colors[syn], lw=lw, zorder=10, alpha=0.6) # w/ gliotransmission
        figtem.adjust_spines(ax[1], ['left', 'bottom'], position=spine_position)
        ax[1].set_xlim(tlims)
        ax[1].set_xticks(xticks)
        ax[1].xaxis.set_tick_params(labelsize=afs)
        ax[1].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=lfs)
        # ax[1].set_ylim(ylims)
        # ax[1].set_yticks(yticks)
        ax[1].set_ylabel('PSC (${0}$)'.format(sympy.latex(pA)), fontsize=lfs)
        # Adjust labels
        set_axlw(ax[1], lw=alw)
        set_axfs(ax[1], fs=afs)

        # Save figure
        figure(plt.get_fignums()[-1])
        plt.savefig(fig_dir+'fig3_syn_processing_'+syn[:3]+'.'+format, format=format, dpi=600)

    plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Figure 4
#-----------------------------------------------------------------------------------------------------------------------
def sic_mechanism(sim=False, format='eps', data_dir='../data/', fig_dir='../Figures/'):
    '''
    Build figure for the effect of SIC currents (single synapse)
    '''

    # Plotting defaults
    spine_position = 5
    lw = 1.5
    alw = 1.0
    afs = 14
    lfs = 16

    # Retrieve Model parameters
    params = asmod.get_parameters()

    # Adjust parameters for the simulations
    params['f_c'] = 0.2*Hz
    params['rho_e'] = 1.0e-4
    params['tau_m'] = 40*ms
    # Add synaptic weight conversion to interface with postsynaptic LIF
    we = 0.0
    params['we'] = we/second / (params['rho_c']*params['Y_T'])/params['tau_e'] # excitatory synaptic weight (voltage)
    wa = 51.0
    params['wa'] = wa/second/ (params['rho_e']*params['G_T'])/params['tau_sic']
    params['tau_sic_r'] = 10*ms
    params['Omega_e'] = 50.0/second

    # General parameters for simulation also used in the analysis
    duration = 3*second

    if sim :
        #---------------------------------------------------------------------------------------------------------------
        # Show SIC mechanism
        #---------------------------------------------------------------------------------------------------------------
        # First show the single synapse and the effect of SIC
        source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params, N_syn=1, N_astro=1,
                                                                                     linear=True,
                                                                                     post='double-exp',
                                                                                     sic='double-exp',
                                                                                     stdp=None)

        source_group.f_in = 0*Hz # No stimulus
        target_group.v = params['E_L']

        # Gliotransmitter release
        gliot.f_c = params['f_c']
        gliot.v_A = '1.0 - f_c*0.2*second' # Start firing at 0.2*1/f_in seconds

        # Set monitors
        post = StateMonitor(target_group, ['g_sic','v'], record=True)
        glt = StateMonitor(gliot, ['G_A'], record=True, dt=0.01*second)

        # Run
        run(duration,namespace={},report='text')

        # Convert monitors to dictionaries and save them for analysis
        post_mon = bu.monitor_to_dict(post)
        glt_mon = bu.monitor_to_dict(glt)

        # Save data
        svu.savedata([post_mon,glt_mon],data_dir+'sic_mechanism.pkl')

    # Plot SIC mechanism
    [post, glt] = svu.loaddata(data_dir+'sic_mechanism.pkl')

    fig0, ax = figtem.generate_figure('3x1',figsize=(4.8,5.5),left=0.23,right=0.05,bottom=0.12,top=0.05,vs=[0.04])
    tlims = array([0,duration])

    # Plot gliotransmitter release
    cf = 1*umole / mole
    ax[0].plot(glt['t'], glt['G_A'][0] / cf, color='k', linewidth=lw)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(tlims)
    ax[0].set_ylim([-1.0,15.1])
    ax[0].set_yticks(arange(0.0,15.1,5))
    # ax[0].set_ylabel('$G_A$ (${0}$)'.format(sympy.latex(umole)), fontsize=lfs)
    ax[0].set_ylabel('Released Gt.\n(${0}$)'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    ax[0].yaxis.set_label_coords(-0.15,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Plot SIC
    R_soma = 150.0e6 # in Ohm
    cf = 1*pamp / amp
    ax[1].plot(post['t'], -post['g_sic'][0] / R_soma / cf, color='k', linewidth=lw)
    # ax[1].plot(post['t'], -post['g_sic'][0] / mV, color='k', linewidth=lw)
    # Format x-axis
    figtem.adjust_spines(ax[1], ['left'], position=spine_position)
    ax[1].set_xlim(tlims)
    ax[1].set_ylim([-30,0.5])
    ax[1].set_yticks(arange(-30,1.0,10))
    ax[1].set_ylabel('PSC (${0}$)'.format(sympy.latex(pamp)), fontsize=lfs)
    ax[1].yaxis.set_label_coords(-0.2,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # Plot PSP
    cf = 1*mV / volt
    ax[2].plot(post['t'], post['v'][0] / cf, color='k', linewidth=lw)
    # Format x-axis
    figtem.adjust_spines(ax[2], ['left','bottom'], position=spine_position)
    ax[2].set_xlim(tlims)
    ax[2].set_xticks(arange(0.0,3.1,1.0))
    ax[2].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=lfs)
    ax[2].set_ylim([-60.1,-53.9])
    ax[2].set_yticks(arange(-60,-53.9,2.0))
    # ax[2].set_ylabel('v (${0}$)'.format(sympy.latex(mV)), fontsize=lfs)
    ax[2].set_ylabel('PSP (${0}$)'.format(sympy.latex(mV)), fontsize=lfs)
    ax[2].yaxis.set_label_coords(-0.2,0.5)
    # Adjust labels
    set_axlw(ax[2], lw=alw)
    set_axfs(ax[2], fs=afs)

    # Save Figures
    figure(plt.get_fignums()[-1])
    plt.savefig(fig_dir+'fig4_sic_mechanism.'+format, format=format, dpi=600)
    # plt.close('all')

    # plt.show()

def post_firing(sim=False, format='eps', data_dir='../data/',fig_dir='../Figures/'):
    '''
    Build figure for the effect of SIC currents (single synapse)
    '''

    # Plotting defaults
    spine_position = 5
    lw = 1.5
    alw = 1.0
    afs = 14
    lfs = 16
    legfs = 12
    colors = {'nog': 'k', 'sic': 'c'}

    # Retrieve Model parameters
    params = asmod.get_parameters()

    # Adjust parameters for the simulations
    params['f_c'] = 0.2*Hz
    params['rho_e'] = 1.0e-4
    params['tau_m'] = 40*ms
    params['tau_e'] = 10*ms
    params['tau_sic_r'] = 10*ms
    params['Omega_e'] = 50.0/second
    params['G_e'] = 2*mV
    # Add synaptic weight conversion to interface with postsynaptic LIF
    we = 3.2
    params['we'] = we/second / (params['rho_c']*params['Y_T'])/params['tau_e'] # excitatory synaptic weight (voltage)
    wa = 51.0
    params['wa'] = wa/second/ (params['rho_e']*params['G_T'])/params['tau_sic']

    # General parameters for simulation also used in the analysis
    duration = 1.0*second

    # # Stimulation (used in Figures too)
    offset = 0.08
    isi = 0.03
    spikes = arange(offset,0.5,isi)*second
    # spikes = array([offset])*second

    if sim :
        #---------------------------------------------------------------------------------------------------------------
        # Show SIC effect on firing
        #---------------------------------------------------------------------------------------------------------------
        # First show the single synapse and the effect of SIC
        source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params, N_syn=2, N_astro=1,
                                                                                     dt=0.1*ms,
                                                                                     connect='i==j',
                                                                                     linear=True,
                                                                                     post='double-exp',
                                                                                     sic='double-exp',
                                                                                     stimulus_syn ='test',
                                                                                     spikes_pre=spikes)
        target_group.v = params['E_L']

        # Gliotransmitter release
        gliot.f_c = params['f_c']
        gliot.v_A = '1.0 - f_c*0.2*second' # Start firing at 0.2*1/f_in seconds

        # Set monitors
        post = StateMonitor(target_group, ['g_e','g_sic','v'], record=True)
        # glt = StateMonitor(gliot, ['G_A'], record=True, dt=0.01*second)

        # Run
        run(duration,namespace={},report='text')

        # Convert monitors to dictionaries and save them for analysis
        post_mon = bu.monitor_to_dict(post)
        # glt_mon = bu.monitor_to_dict(glt)
        svu.savedata([post_mon],data_dir+'sic_firing.pkl')

    #-------------------------------------------------------------------------------------------------------------------
    # Plot results
    #-------------------------------------------------------------------------------------------------------------------
    # Load simulation
    post = svu.loaddata(data_dir+'sic_firing.pkl')[0]

    fig1, ax = figtem.generate_figure('3x1',figsize=(4.8,5.5),left=0.23,right=0.05,bottom=0.12,top=0.05,vs=[0.04])
    tlims = array([0,duration])

    # Plot PRE spikes
    l, m, b = ax[0].stem(spikes, ones((len(spikes),1)), linefmt='k', markerfmt='', basefmt='')
    plt.setp(m, 'linewidth', lw)
    plt.setp(l, 'linestyle', 'none')
    plt.setp(b, 'linestyle', 'none')
    figtem.adjust_spines(ax[0], [])
    ax[0].set_xlim(tlims)
    ax[0].set_ylim([0.0,1.0])

    # Plot PSCs
    # First plot legend
    # Add legend as a proxy artist
    label = {'nog': 'no gliot.', 'sic': 'w/ gliot.'}
    keys_legend = ['nog','sic']
    patches_legend, labels_legend = [], []
    for k in keys_legend:
        # Add artist for legend
        patches_legend.append(mpatches.Patch(color=colors[k]))
        labels_legend.append(label[k])
    # Add Legend
    ax[1].legend(patches_legend, labels_legend,
                 fontsize=legfs, frameon=False, loc=4)

    # Plot data
    R_soma = 150.0e6 # in Ohm
    cf = 1.0*pamp / amp
    # w/ SIC
    ax[1].plot(post['t'], -(post['g_e'][0] + post['g_sic'][0]) / R_soma / cf, color=colors['sic'], linewidth=lw)
    # ax[1].plot(post['t'], -(post['g_e'][0] + post['g_sic'][0]), color=colors['sic'], linewidth=lw)
    # w/out SIC
    ax[1].plot(post['t']+0.005, -post['g_e'][1] / R_soma / cf, color=colors['nog'], linewidth=lw)
    # ax[1].plot(post['t']+0.005, post['v'][1]/1.e-3, color=colors['nog'], linewidth=lw)
    # Format x-axis
    figtem.adjust_spines(ax[1], ['left'], position=spine_position)
    ax[1].set_xlim(tlims)
    ax[1].set_ylim([-80,1.0])
    ax[1].set_yticks(arange(-80,1.0,20.0))
    ax[1].set_ylabel('PSC (${0}$)'.format(sympy.latex(pamp)), fontsize=lfs)
    ax[1].yaxis.set_label_coords(-0.18,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # Plot PSP
    cf = 1*mV / volt
    # Add spikes (cut)
    V_peak = -45
    post['v'] =  bu.show_spikes(post['v']/cf, params['V_r']/mV, V_peak)
    # w/ SIC
    ax[2].plot(post['t'], post['v'][0], color=colors['sic'], linewidth=lw)
    # w/out SIC
    ax[2].plot(post['t'], post['v'][1], color=colors['nog'], linewidth=lw)
    # Format x-axis
    figtem.adjust_spines(ax[2], ['left','bottom'], position=spine_position)
    ax[2].set_xlim(tlims)
    ax[2].set_xticks(arange(0.0,1.01,0.2))
    ax[2].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=lfs)
    ax[2].set_ylim([-60.1,-45.0])
    ax[2].set_yticks(arange(-60,-44.9,5.0))
    ax[2].set_ylabel('PSP (${0}$)'.format(sympy.latex(mV)), fontsize=lfs)
    ax[2].yaxis.set_label_coords(-0.18,0.5)
    # Adjust labels
    set_axlw(ax[2], lw=alw)
    set_axfs(ax[2], fs=afs)

    # Save Figures
    figure(plt.get_fignums()[-1])
    plt.savefig(fig_dir+'fig4_sic_postfiring.'+format, format=format, dpi=600)
    # plt.close('all')

    # plt.show()

def sic_io(sim=False, format='eps', data_dir='../data/',fig_dir='../Figures/'):
    '''
    Compute I/O f_out vs. f_in (f_c) of postsynaptic firing
    :return:
    '''

    # Plotting defaults
    spine_position = 5
    lw = 1.5
    alw = 1.0
    afs = 14
    lfs = 16
    legfs = 10
    fmt = '.'

    # Get parameters
    params = asmod.get_parameters()

    # Adjust parameters for the simulations
    params['f_c'] = 0.2*Hz
    params['rho_e'] = 1.0e-4
    params['tau_m'] = 40*ms
    params['tau_e'] = 10*ms
    params['tau_sic_r'] = 10*ms
    params['Omega_e'] = 50.0/second
    params['G_e'] = 2*mV
    # Add synaptic weight conversion to interface with postsynaptic LIF
    we = 3.2
    params['we'] = we/second / (params['rho_c']*params['Y_T'])/params['tau_e'] # excitatory synaptic weight (voltage)
    wa = 51.0
    params['wa'] = wa/second/ (params['rho_e']*params['G_T'])/params['tau_sic']

    # Global parameters
    duration = 60*second

    if sim :
        # Output data dictionary
        io_data = {}

        N_points = 50
        N_trials = 50

        # fout vs. fin w/out gliotransmission (SIC)
        # Simulation parameters
        fin_range = [1.0,1e3]
        fc = [0,1.0,1.0]*Hz
        G_sic_0 = params['G_sic']
        mf = 1.5
        G_sic = [0.0*mV, params['G_sic'], mf*params['G_sic']]
        for i in range(size(fc)):
            params['G_sic'] = G_sic[i]
            io_data['fin_'+str(i)] = analysis.freq_out(params, sim=sim,
                                                       duration=duration,
                                                       N_points=N_points, N_trials=N_trials, freq_range=fin_range, freq_rel=fc[i],
                                                       gliot=False, show=False)

        # fout vs. fc w/ gliotransmission (SIC)
        # Simulation parameters
        fin = [0,1.0,0,1.0]*Hz
        fc_range = [0.01,10]
        G_sic = [G_sic_0, G_sic_0, mf*G_sic_0, mf*G_sic_0]
        for i in range(size(fin)):
            params['G_sic'] = G_sic[i]
            io_data['fc_'+str(i)] = analysis.freq_out(params, sim=sim,
                                                      duration=duration,
                                                      N_points=N_points, N_trials=N_trials, freq_range=fc_range, freq_rel=fin[i],
                                                      gliot=True, show=False)

        # Save data
        svu.savedata([io_data],data_dir+'sic_io.pkl')

    # Load data
    io_data = svu.loaddata(data_dir+'sic_io.pkl')[0]

    # # Generate figure template
    fig2, ax = figtem.generate_figure('2x1', figsize=(5.0,6.0), left=0.23, right=0.08, bottom=0.12, top=0.05, vs=[0.15])
    tlims = array([0,duration])

    fc_colors = ['k','c',chex((0,107,164))]
    fin_colors = ['k','b',chex((89,89,89)),chex((95,158,209))]

    # f_out vs. f_in
    # First plot legend
    # Add legend as a proxy artist
    label = ['no gliot.', r'1 Hz 30 pA SICs', r'1 Hz 45 pA SICs']
    patches_legend, labels_legend = [], []
    for k,_ in enumerate(fc_colors):
        # Add artist for legend
        patches_legend.append(mpatches.Patch(color=fc_colors[k]))
        labels_legend.append(label[k])
    # Add Legend
    ax[0].legend(patches_legend, labels_legend,
                 fontsize=legfs, frameon=False, loc=2)

    # Plot data
    XTicks = [1.0,10.0,100,1000]
    ax[0].errorbar(io_data['fin_0'][0],io_data['fin_0'][1],yerr=io_data['fin_0'][2],fmt=fmt,color=fc_colors[0],zorder=3)
    ax[0].errorbar(io_data['fin_1'][0],io_data['fin_1'][1],yerr=io_data['fin_1'][2],fmt=fmt,color=fc_colors[1],zorder=5)
    ax[0].errorbar(io_data['fin_2'][0],io_data['fin_2'][1],yerr=io_data['fin_2'][2],fmt=fmt,color=fc_colors[2],zorder=4)
    figtem.adjust_spines(ax[0], ['left','bottom'], position=spine_position)
    ax[0].set_xlim([1.0,1000])
    ax[0].set_xscale('log')
    ax[0].set_xticks(XTicks)
    ax[0].set_xticklabels([str(xt) for xt in XTicks])
    # ax[0].set_xlabel('$\\nu_{0}$ (${1}$)'.format('{pre}',sympy.latex(Hz)), fontsize=lfs)
    ax[0].set_xlabel('Presynaptic Firing (${0}$)'.format(sympy.latex(Hz)), fontsize=lfs)
    ax[0].set_ylim([-1.0,180])
    ax[0].set_yticks(arange(0.0,181,60.0))
    # ax[0].set_ylabel('$\\nu_{0}$ (${1}$)'.format('{post}',sympy.latex(Hz)), fontsize=lfs)
    ax[0].set_ylabel('Postsynaptic Firing\n(${0}$)'.format(sympy.latex(Hz)), fontsize=lfs, multialignment='center')
    ax[0].yaxis.set_label_coords(-0.15,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # f_out vs. f_c
    label = [r'no ntr.; 30 pA SICs',
             r'1 Hz ntr.; 30 pA SICs',
             r'no ntr.; 45 pA SICs',
             r'1 Hz ntr.; 45 pA SICs']
    patches_legend, labels_legend = [], []
    for k,_ in enumerate(fin_colors):
        # Add artist for legend
        patches_legend.append(mpatches.Patch(color=fin_colors[k]))
        labels_legend.append(label[k])
    # Add Legend
    ax[1].legend(patches_legend, labels_legend,
                 fontsize=legfs, frameon=False, loc=2)

    # Plot data
    XTicks = [0.01,0.1,1.0,10]
    ax[1].errorbar(io_data['fc_0'][0],io_data['fc_0'][1],yerr=io_data['fc_0'][2],fmt=fmt,color=fin_colors[0])
    ax[1].errorbar(io_data['fc_1'][0]-5.5e-3,io_data['fc_1'][1],yerr=io_data['fc_1'][2],fmt=fmt,color=fin_colors[1]) # Shifted by -5.5e-3 for visualization purposes
    ax[1].errorbar(io_data['fc_2'][0],io_data['fc_2'][1],yerr=io_data['fc_2'][2],fmt=fmt,color=fin_colors[2])
    ax[1].errorbar(io_data['fc_3'][0]-5.5e-3,io_data['fc_3'][1],yerr=io_data['fc_3'][2],fmt=fmt,color=fin_colors[3]) # Shifted by -5.5e-3 for visualization purposes
    figtem.adjust_spines(ax[1], ['left','bottom'], position=spine_position)
    ax[1].set_xlim([0.1,10.0])
    ax[1].set_xscale('log')
    ax[1].set_xticks(XTicks)
    ax[1].set_xticklabels([str(xt) for xt in XTicks])
    # ax[1].set_xlabel('$\\nu_{0}$ (${1}$)'.format('{glt}',sympy.latex(Hz)), fontsize=lfs)
    ax[1].set_xlabel('Gliotransmitter Release (${0}$)'.format(sympy.latex(Hz)), fontsize=lfs)
    ax[1].set_ylim([-0.1,3.0])
    ax[1].set_yticks(arange(0.0,3.1,1.0))
    # ax[1].set_ylabel('$\\nu_{0}$ (${1}$)'.format('{post}',sympy.latex(Hz)), fontsize=lfs)
    ax[1].set_ylabel('Postsynaptic Firing\n(${0}$)'.format(sympy.latex(Hz)), fontsize=lfs, multialignment='center')
    ax[1].yaxis.set_label_coords(-0.15,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # Save figure
    figure(plt.get_fignums()[-1])
    plt.savefig(fig_dir+'fig4_sic_io.'+format, format=format, dpi=600)
    # plt.savefig('fig4_sic_io.pdf', format='pdf', dpi=600)

    # plt.show()

def sic(sim=True, format='svg', fig_dir='../Figures/'):
# def sic(sim=False, format='svg', fig_dir='../Figures/'):
    # Generate figure that show mechanism of SICs
    sic_mechanism(sim=sim, format=format, fig_dir=fig_dir)
    post_firing(sim=sim, format=format, fig_dir=fig_dir)
    sic_io(sim=sim, format=format, fig_dir=fig_dir)
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Figure 5
#-----------------------------------------------------------------------------------------------------------------------
def plot_stdp(S, P, alpha, params, color='k'):

    # Defaults
    opts = {'spine_position' : 5,
            'lw'  : 1.5,
            'lw2' : 2.0,
            'alw' : 1.0,
            'afs' : 14,
            'lfs' : 16,
            'cf'  : 0.1,   # correction Factor for plotting \rho(t) over reference line
            }

    # Fixed parameters
    dt = diff(S['t'])[0]
    tlims = [S['t'][0],S['t'][-1]+dt]
    xticks = arange(0.0,1.1,0.2)
    colors = {'u0' : ['b', chex((214,39,40)), chex((44,160,44))],
              'w'  : chex((148,103,189)),
              'w0' : chex((143,135,130))}

    r0 = max(unique(S['r_S'][0]))  # Reference value for r_S

    # Define some lambdas
    u0 = lambda g, U_0_star, xi : (1-g)*U_0_star + xi*g
    normalize = lambda x, x0 : (x-x0)/x0*100

    for i in range(size(alpha)):
        # Generate figure
        fig, ax = figtem.generate_figure('4x1_1L3S', figsize=(7.0,6.0), left=0.14, right=0.1, bottom=0.15, top=0.05)
        #-------------------------------------------------------------------------------------------------------------------
        # Top plot / 1
        #-------------------------------------------------------------------------------------------------------------------
        # Plot u0
        # Create twin y-axis
        ax_aux = ax[0].twinx()
        # First plot u0(t)
        ax_aux.plot(S['t'], u0(S['Gamma_S'][0], params['U_0__star'], alpha[0]), color=colors['u0'][0], linewidth=opts['lw'], linestyle=':')
        if i>0:
            ax_aux.plot(S['t'], u0(S['Gamma_S'][i], params['U_0__star'], alpha[i]), color=colors['u0'][i], linewidth=opts['lw'], linestyle='-')
        # # i = 0
        # # Plot
        # ax_aux.plot(S['t'], u0(S['Gamma_S'][i], params['U_0__star'], alpha[i]), color=opts['u0'], linewidth=opts['lw'], linestyle='-')
        # First make patch and spines of ax_aux invisible
        figtem.make_patch_spines_invisible(ax_aux)
        ax_aux.patch.set_visible(True)
        # # Then show only the right spine of ax_aux
        ax_aux.spines['right'].set_visible(True)
        ax_aux.spines['right'].set_position(('outward', opts['spine_position']))  # outward by 10 points

        # Plot Cpre
        r, ispk = unique(S['r_S'][i], return_index=True)
        if np.ndim(P['t'][i]) == 0:  # Check if it's a scalar
            l, m, b = ax[1].stem(P['t'][0], ones((1, 1)), linefmt='k', markerfmt='', basefmt='')  # Use 1 instead of len()
        else:
            l, m, b = ax[1].stem(P['t'][0], ones((len(P['t'][i]), 1)), linefmt='k', markerfmt='', basefmt='')

        plt.setp(m, 'linewidth', opts['lw'])
        plt.setp(l, 'linestyle', 'none')
        plt.setp(b, 'linestyle', 'none')
        figtem.adjust_spines_uneven(ax[0], va='left', ha='bottom', position=opts['spine_position'])
        ax[0].set_xlim(tlims)
        ax[0].set_xticks(xticks)
        ax[0].set_xticklabels('')

        ax[0].set_zorder(ax_aux.get_zorder()+1) # put ax[0] stem plot in front of ax_aux
        ax[0].patch.set_visible(False)          # hide the 'canvas'

        # Adjust axes (Cpre)
        ax[0].set_xlim(tlims)
        ax[0].set_ylim([0.0,2.0])
        ax[0].set_yticks(arange(0.0,2.1,1.0))
        set_axlw(ax[0], lw=opts['alw'])
        set_axfs(ax[0], fs=opts['afs'])

        # Adjust axes (u0)
        ax_aux.set_xlim(tlims)
        ax_aux.set_ylim([-0.01,1.01])
        ax_aux.set_yticks(arange(0.0,1.1,0.5))
        ax_aux.yaxis.set_tick_params(labelsize=opts['afs'])
        set_axlw(ax_aux, lw=opts['alw'])
        # Colorize
        tkw = dict(size=4, width=1.5)
        ax_aux.yaxis.label.set_color(colors['u0'][i])
        ax_aux.tick_params(axis='y', colors=colors['u0'][i], **tkw)

        # Add Label
        ax[0].set_ylabel('Pre', fontsize=opts['lfs'])
        ax[0].yaxis.set_label_coords(-0.09,0.5)
        ax_aux.set_ylabel('S.R.P.', fontsize=opts['lfs'])

        #-------------------------------------------------------------------------------------------------------------------
        # Top plot / 2
        #-------------------------------------------------------------------------------------------------------------------
        # Cpost
        if np.ndim(P['t'][i]) == 0:  # Scalar case
            l, m, b = ax[1].stem(P['t'][0], np.ones((1, 1)), linefmt='k', markerfmt='', basefmt='')  # Use 1 instead of len()
        else:  # Array case
            l, m, b = ax[1].stem(P['t'][0], np.ones((len(P['t'][i]), 1)), linefmt='k', markerfmt='', basefmt='')
        plt.setp(m, 'linewidth', opts['lw'])
        plt.setp(l, 'linestyle', 'none')
        plt.setp(b, 'linestyle', 'none')
        figtem.adjust_spines_uneven(ax[1], va='left', ha='bottom', position=opts['spine_position'])
        ax[1].set_xlim(tlims)
        ax[1].set_xticks(xticks)
        ax[1].set_xticklabels('')
        ax[1].set_ylim([0.0,2.0])
        ax[1].set_yticks(arange(0.0,2.1,1.0))
        # ax[1].set_ylabel('$C_{post}$', fontsize=opts['lfs'])
        ax[1].set_ylabel('Post', fontsize=opts['lfs'])
        ax[1].yaxis.set_label_coords(-0.09,0.5)
        set_axlw(ax[1], lw=opts['alw'])
        set_axfs(ax[1], fs=opts['afs'])

        #-------------------------------------------------------------------------------------------------------------------
        # Central plot
        #-------------------------------------------------------------------------------------------------------------------
        # Calcium
        ylim = [-0.05,3.0]
        analysis.threshold_regions(S['t'], S['C_syn'][i], params['Theta_d'], params['Theta_p'], ax=ax[2], y0=-0.05)
        # Plot Ca2+ trace
        ax[2].plot(S['t'], S['C_syn'][i], color=color, lw=opts['lw2'])
        figtem.adjust_spines_uneven(ax[2], va='left', ha='bottom', position=opts['spine_position'])
        ax[2].set_xlim(tlims)
        ax[2].set_xticks(xticks)
        ax[2].set_xticklabels('')
        ax[2].set_ylim(ylim)
        ax[2].set_yticks(arange(0.0,3.01,1.0))
        ax[2].set_ylabel('Postsynaptic\n Calcium', fontsize=opts['lfs'],multialignment='center')
        ax[2].yaxis.set_label_coords(-0.08,0.5)
        # Adjust labels
        set_axlw(ax[2], lw=opts['alw'])
        set_axfs(ax[2], fs=opts['afs'])

        #-------------------------------------------------------------------------------------------------------------------
        # Bottom plot
        #-------------------------------------------------------------------------------------------------------------------
        # Synaptic Weight
        if S['W'][i][-1]>=0:
            cf = opts['cf']
        else:
            cf = -opts['cf']
        # Plot reference line
        ax[3].plot(tlims, zeros((2,1)), color=colors['w0'], lw=opts['lw'])
        # Plot rho
        # A correction factor is added to avoid superposition of reference line and effective \rho(t)
        ax[3].plot(S['t'], cf+normalize(S['W'][i], params['W_0']), color=colors['w'], lw=opts['lw'])
        figtem.adjust_spines(ax[3], ['left', 'bottom'], position=opts['spine_position'])
        ax[3].set_xlim(tlims)
        ax[3].set_xticks(xticks)
        ax[3].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=opts['lfs'])
        ax[3].set_ylim([-5.0,5.0])
        ax[3].set_yticks(arange(-4.0,4.1,2.0))
        ax[3].set_ylabel('%$\Delta$ Syn.\n Str.', fontsize=opts['lfs'], multialignment='center')
        ax[3].yaxis.set_label_coords(-0.08,0.5)

        # ax[3].yaxis.set_tick_params(labelsize=opts['afs'])
        # Adjust labels
        set_axlw(ax[3], lw=opts['alw'])
        set_axfs(ax[3], fs=opts['afs'])

def stdp_mechanism(sim=False, format='eps', data_dir='../data/', fig_dir='../Figures/'):
# def stdp_mechanism(sim=True):

    # It is likely that you may tuned not on the refractory but rather on the ICs
    params = asmod.get_parameters(synapse='neutral', stdp='nonlinear')
    params['f_in'] = 1.5*Hz

    # Adjust gliotransmission modulation parameters
    params['rho_e'] = 1.0e-4
    params['Omega_G'] = 0.2 / second # Makes gliotransmitter effect decay fast to better show mechanism
    params['Omega_c'] = 1000./second
    # params['Cpost'] = 0.
    # params['Cpre'] = 0.

    # Add synaptic weight conversion to interface with postsynaptic LIF
    we = 59.0
    # we = 35.0
    params['we'] = we/second / (params['rho_c']*params['Y_T'])/params['tau_pre_d'] # excitatory synaptic weight (voltage)

    # Global simulation parameters
    alpha = [params['U_0__star'], 0.0, 1.0]
    t_on = 0.2*second
    # Dt = {'ltp': -30*ms, 'ltd': 30*ms}
    Dt = {'ltp': 25*ms, 'ltd': 25*ms}
    Dt['ltp'] = -25 * ms

    t_glt = t_on-0.1*second
    duration =  0.8*second
    #t_on = t_on.rescale(second)  # Ensure t_on is in seconds
   # Dt['ltp'] = Dt['ltp'].rescale(second)  # Ensure Dt['ltp'] is in seconds
    #Dt['ltd'] = Dt['ltd'].rescale(second)  # Ensure Dt['ltd'] is in seconds
    print(f"Dt['ltp']: {Dt['ltp']}, units: {Dt['ltp'].dim}")
    #Dt['ltp'] = sqrt(abs(Dt['ltp'])) * ms  # Ensure Dt['ltp'] is in milliseconds
    print(f"Dt['ltp']: {Dt['ltp']}, units: {Dt['ltp'].dim}")




    # Build stimulation protocol
    spikes_pre = {'ltd': t_on + arange(0.0, duration, 1/params['f_in']) - Dt['ltp'],
                'ltp': t_on + arange(0.0, duration, 1/params['f_in'])}

    spikes_post ={'ltd': t_on+arange(0.0,duration,1/params['f_in']),
                  'ltp': t_on+arange(0.0,duration,1/params['f_in'])+Dt['ltd']}

    # Dt = {'ltd': 25*ms}
    # spikes_pre = {'ltd': t_on+arange(0.0,duration,1/params['f_in'])*second-Dt['ltd']}
    # spikes_post ={'ltd': t_on+arange(0.0,duration,1/params['f_in'])*second}
    # Update with
    duration += t_on

    if sim:
        S_mon, P_mon = {}, {}
        for k,v in Dt.items():
            # Generate model
            source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params, N_syn=3, N_astro=1,
                                                                                         dt=0.1*ms,
                                                                                         connect=True,
                                                                                         stdp='nonlinear',
                                                                                         stdp_protocol='test',
                                                                                         spikes_pre=spikes_pre[k],
                                                                                         spikes_post=spikes_post[k])

            # Initialize other variables
            synapses.A = 0.0
            synapses.P = 0.0
            synapses.alpha = alpha
            # synapses.alpha = params['U_0__star']
            gliot.f_c = 1./duration  # dummy value to assure that you have only one relerase w/in the simulation
            gliot.v_A = '1.0 - f_c*t_glt'

            # The modulation by gliotransmitter must be initialized and start before the protocol
            S = StateMonitor(synapses, variables=['r_S','Gamma_S','W','C_syn','Y_S','A','P'], record=True, dt=0.2*ms)
            P = SpikeMonitor(target_group, record=True)

            run(duration,namespace={},report='text')

            # Convert monitors to dictionaries
            S_mon[k] = bu.monitor_to_dict(S)
            P_mon[k] = bu.monitor_to_dict(P, monitor_type='spike', fields=['spk'])

        # Save data
        svu.savedata([S_mon, P_mon], data_dir+'stdp_mechanism.pkl')

    # Load data
    [S, P] = svu.loaddata(data_dir+'stdp_mechanism.pkl')

    # Plotting options
    colors = {'ltd' : '#003366', 'ltp': '#FF0066'}

    # tLTD protocol
    plot_stdp(S['ltd'], P['ltd'], alpha, params, color=colors['ltd'])

    # tLTP protocol
    plot_stdp(S['ltp'], P['ltp'], alpha, params, color=colors['ltp'])

    # Save figure
    protocol = ['ltd','ltp']
    label = ['u0_star','xi0','xi1']
    for i,n in enumerate(plt.get_fignums()):
        figure(n)
        if n<=3:
            p = protocol[0]
        else:
            p = protocol[1]
        plt.savefig(fig_dir+'fig5_'+p+'_'+label[i%3]+'.'+format, format=format, dpi=600)

    plt.show()

def reshape_monitor(monitor):
    '''
    Reshape serial monitor to parallel

    Needed for handling saved data from serial simulations when the chaining params is not defined as variable and
    requires simulations to be run for each of its values and saved as monitor[N_sim][keys][values].

    Returns:

    reshaped_monitor[keys][values] where values are [values[sim_0][0], values[0][sim_1],...,
                                                     values[1][sim_0],values[1][sim_1],...,
                                                     ...
                                                     values[n-1][sim_0],..., values[n-1][N_sim]]
    '''
    if type(monitor) != type({}):
        rmon = {}
        for k,data in monitor[0].items():
            rmon[k] = data
            if size(monitor)>0:
                # if more than one simulation then do unfold otherwise just a standard dictionary
                # if k=='t':
                    # if time, just vstack all time vectors
                #     for val in monitor[1:]:
                #         rmon[k] = vstack((rmon[k], val[k]))
                # else:
                if k!='t':
                    # Create an empty array to stack on it (same number of columns as the data to stack)
                    rmon[k] = array([], dtype=int64).reshape(0,shape(data)[1])
                    for n in range(shape(data)[0]):
                        rmon[k] = vstack((rmon[k], data[n]))
                        # rmon[k] = data[n]   # get the 'n' entry of data in the first simulation of the monitor array
                        for val in monitor[1:]:
                            # Get other 'n' entries in the other simulations
                            rmon[k] = vstack((rmon[k], val[k][n]))
        return rmon

def stdp_curves_mean(Dt, C, params, duration, N_sim=1):

    # Define some lambdas
    rhoBar = lambda Gd, Gp : Gp/(Gd+Gp)
    sigmaRhoSquared = lambda sigmaSquared, ad, ap, gd, gp : sigmaSquared*(ap+ad)/(gp*ap + gd*ad)
    tauEff = lambda tauw, Gd, Gp : tauw/(Gp + Gd)

    # Retrieve data set size
    N_syn = size(Dt)

    # Debug parameters
    # npts = 3
    # # s = logspace(-2,log10(2),npts)
    # b = linspace(2,10,npts)
    # # beta = linspace(0.1,0.9,npts)

    d, p = zeros((N_syn,1)), zeros((N_syn,1))
    dw = []
    for i in range(N_sim):
    # i = 0
    # for v in b:
        # params['sigma'] = v
        # params['b'] = v
        # params['beta'] = v
        for n,signal in enumerate(C['C_syn'][i::N_sim]):
            idx, frac = analysis.threshold_crossing(C['t'],signal,params['Theta_p'])
            p[n] = sum(frac)/duration
            # idx, frac = analysis.threshold_crossing(C[i]['t'],signal,(params['Theta_d'],params['Theta_p']))
            idx, frac = analysis.threshold_crossing(C['t'],signal,params['Theta_d'])
            d[n] = sum(frac)/duration

        N = 61
        interval = 1./params['f_in']
        p_UP = camod.transitionProbability(N,interval,params['W_0'],0.,
                                           rhoBar(params['gamma_d']*d[:,0],params['gamma_p']*p[:,0]),
                                           sigmaRhoSquared(params['sigma']**2, d[:,0], p[:,0], params['gamma_d'], params['gamma_p']),
                                           tauEff(params['tau_w'],params['gamma_d']*d[:,0],params['gamma_p']*p[:,0]))
        p_DOWN = camod.transitionProbability(N,interval,params['W_0'],1.,
                                             rhoBar(params['gamma_d']*d[:,0],params['gamma_p']*p[:,0]),
                                             sigmaRhoSquared(params['sigma']**2, d[:,0], p[:,0], params['gamma_d'], params['gamma_p']),
                                             tauEff(params['tau_w'],params['gamma_d']*d[:,0],params['gamma_p']*p[:,0]))
        # subtracted value (so as to center around 0.)
        dw.append(camod.changeSynapticStrength(params['beta'],p_UP,p_DOWN,params['b'])-1.)

    # return dw
    return array(dw)

def compute_stdp_curves(Dt, C, W, params, duration, N_sim=1, mean_field=False):
    '''
    Compute STDP curves and Ca2+ fraction curves

    :param Dt:
    :param C:
    :param W:
    :param params:
    :param duration:
    :param ax:
    :return:
    '''

    # Define some lambdas
    normalize = lambda x,x0 : (x-x0)/x0*100

    # Plot Calcium fraction curves
    N_syn = shape(C['C_syn'][::N_sim])[0]

    # Allocate memory
    d, p = zeros((N_sim,shape(C['C_syn'])[0]//N_sim)), zeros((N_sim,shape(C['C_syn'])[0]//N_sim))
    # Compute Ca2+ time fractions
    for i in range(N_sim):
        for n,signal in enumerate(C['C_syn'][i::N_sim]):
            idx, frac = analysis.threshold_crossing(C['t'],signal,params['Theta_p'])
            p[i][n] = sum(frac)/duration
            idx, frac = analysis.threshold_crossing(C['t'],signal,params['Theta_d'])
            d[i][n] = sum(frac)/duration

    # Compute STDP curve
    if not mean_field:
        dw = zeros((N_sim,shape(W['W'])[0]/N_sim))
        for i in range(N_sim):
            dw[i] = normalize(W['W'][i::N_sim][:,-1],params['W_0'])
    else:
        dw = stdp_curves_mean(Dt, C, params, duration, N_sim=N_sim)
        for i in range(N_sim): dw[i] *= 100

    return d,p,dw

def plot_stdp_curves(Dt, C, W, params, duration, N_sim=1, mean_field=False,
                     ax=None, color=['k', 'lime', 'm'], zorder=None, alpha=1.):
    '''
    Plot STDP curves and Ca2+ fraction curves

    :param Dt:
    :param C:
    :param W:
    :param params:
    :param duration:
    :param ax:
    :return:
    '''

    lw0 = 1.5
    lw = 2.0
    ls = {'ltd': '-', 'ltp': '--'}
    color0 = '0.5'

    # Define some lambdas
    normalize = lambda x,x0 : (x-x0)/x0*100

    # Generate figure template
    if not ax: fig0, ax = figtem.generate_figure('2x1', figsize=(4.5,6.0), left=0.20, right=0.08, bottom=0.12, top=0.05, vs=[0.15])

    # Compatibility for C,W saved as lists of dict (ADDITION)
    if type(C)==type(list()):
        for i in range(size(C)):
            ax = plot_stdp_curves(Dt, C[i], W[i], params, duration, N_sim=1, mean_field=mean_field,
                                  ax=ax, color=[color[i]], zorder=[zorder[i]], alpha=[alpha[i]])
    else:
        N_syn = shape(C['C_syn'][::N_sim])[0]
        # Generate zorder
        if not zorder: zorder = arange(10,10+2*N_sim,2)
        # Generate alpha map
        if size(alpha)==size(1): alpha = alpha*ones((1,N_sim))[0]

        d, p = zeros((N_syn,1)), zeros((N_syn,1))
        # Treatment of scalar alpha
        for i in range(N_sim):
            for n,signal in enumerate(C['C_syn'][i::N_sim]):
                idx, frac = analysis.threshold_crossing(C['t'],signal,params['Theta_p'])
                p[n] = sum(frac)/duration
                # idx, frac = analysis.threshold_crossing(C[i]['t'],signal,(params['Theta_d'],params['Theta_p']))
                idx, frac = analysis.threshold_crossing(C['t'],signal,params['Theta_d'])
                d[n] = sum(frac)/duration
            ax[0].plot(Dt, p*100, ls=ls['ltp'], color=color[i], lw=lw, zorder=zorder[i], alpha=alpha[i])
            ax[0].plot(Dt, d*100, ls=ls['ltd'], color=color[i], lw=lw, zorder=zorder[i]+1, alpha=alpha[i])
            # ax[0].plot(Dt, p*100, ls=ls['ltp'], color='k', lw=lw, zorder=zorder[i], alpha=0.5)
            # ax[0].plot(Dt, d*100, ls=ls['ltd'], color='k', lw=lw)

            #
        # Plot STDP curve
        if not mean_field:
            ax[1].plot((Dt[0], Dt[-1]), (0.0,0.0), c=color0, ls='-', lw=lw0)
            ax[1].plot((0.0, 0.0), (-15,15), c=color0, ls='-', lw=lw)
            for i in range(N_sim):
                ax[1].plot(Dt, normalize(W['W'][i::N_sim][:,-1],params['W_0']), marker='o', color=colors[i], lw=lw0, zorder=zorder[i], alpha=alpha[i])
        else:
            dw = stdp_curves_mean(Dt, C, params, duration, N_sim=N_sim)
            ax[1].plot((Dt[0], Dt[-1]), (0.0,0.0), c=color0, ls='-', lw=lw0)
            ax[1].plot((0.0, 0.0), (-60,60), c=color0, ls='-', lw=lw)
            for i in range(N_sim):
                ax[1].plot(Dt, dw[i]*100., marker='o', color=color[i], lw=lw0, zorder=zorder[i], alpha=alpha[i])
    return ax

def stdp_curve_parameters(dt, C, N_sim, params, duration=61*second):
    # Retrieve ltp/ltd threshold, max/min and area ratio
    # N_sim = size(alpha) # Equivalent to N_astro in the simulation
    thr, wmin, wmax, aratio = [], [], [], []
    thr2 = []   # second crossing threshold from LTP to LTD

    # Define some lambdas
    # normalize = lambda x,x0 : (x-x0)/x0*100.0
    inbetween = lambda x,curve : (x>=x[argmin(curve)]) & (x<=x[argmax(curve)])

    # Compute mean-field curves
    stdp_curve = stdp_curves_mean(dt, C, params, duration=duration, N_sim=N_sim)

    for i in range(N_sim):
        # Find crossing point
        # First identify first/last point above/beyond zero
        indices = ((inbetween(dt,stdp_curve[i])&(stdp_curve[i]<0)).nonzero()[0][-1],
                   (inbetween(dt,stdp_curve[i])&(stdp_curve[i]>0)).nonzero()[0][0])
        l = vstack((dt.take(indices),stdp_curve[i].take(indices)))
        l0 = vstack((dt.take(indices),[0,0]))
        # Find intersection
        result = interx(l, l0)
        if result[0][0] is not np.nan:
            thr.append(result[0][0])
        # Max and Min
        wmin.append(abs(amin(stdp_curve[i])))
        wmax.append(amax(stdp_curve[i]))
        # Area ratio
        # Also look for additional threhsold from LTP to LTD at \Delta t>0
        indices = ((dt>=thr[i])&(stdp_curve[i]<0)).nonzero()[0]
        if size(indices)>0:
            indices = (indices[0]-1,indices[0])
            l = vstack((dt.take(indices),stdp_curve[i].take(indices)))
            l0 = vstack((dt.take(indices[:2]),[0,0]))
            result = interx(l, l0)
            if result[0][0] is not np.nan:
                thr2.append(result[0][0])
                aratio.append(abs(sum(stdp_curve[i].take((dt>=thr[i]).nonzero())) / sum(stdp_curve[i].take((dt<=thr[i]).nonzero()))))
            else:
                aratio.append(0)  # Or handle this case in another way if necessary
        else:
            thr2.append(Inf)
            aratio.append(abs(sum(stdp_curve[i].take((dt>=thr[i]).nonzero()))/sum(stdp_curve[i].take((dt<=thr[i]).nonzero()))))

    return array(thr), array(wmin), array(wmax), array(aratio), array(thr2)

def stdp_curves_stp(params, Dt_min, Dt, duration=61*second,
                    dt_solv=0.1*ms, N_pts=50, stdp='nonlinear', gliot=False,
                    data_dir='../data/'):
    '''
    Show effect of short-term plasticity on STDP curves
    '''
    print(f"Function stdp_curves_stp called with gliot={gliot} and stdp={stdp}")

    # Define some lambdas
    ns = lambda Nsyn, Nastro, Npts  : [Nsyn,Nastro,Npts]

    # Define dt
    dt = arange(Dt_min, Dt_min+Dt, Dt/N_pts)

    #---------------------------------------------------------------------------------------------------------------
    # Simulation /1: Show effect of short-term plasticity on STDP curves
    #---------------------------------------------------------------------------------------------------------------
    if stdp=='linear':
        U_0__star = [0.5, 0.05, 0.9]
        lbl = 'lin'
    else:
        U_0__star = [0.5, 0.1, 0.8]
        lbl = 'nonlin'
    # Debug
    # U_0__star = [0.5]
    C_mon = []
    W_mon = []

    print(" got to  stdp_curves_stp glio is", gliot)
    if not gliot:

        print("Running simulation without gliot.")
        for i,_ in enumerate(U_0__star):
            # No gliotransmitter modulation
            params['U_0__star'] = U_0__star[i]
            source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params,
                                                                                         dt=dt_solv,
                                                                                         N_pts=N_pts, N_syn=1, N_astro=1,
                                                                                         linear=False,
                                                                                         connect=True,
                                                                                         stdp=stdp,
                                                                                         stdp_protocol='pairs',
                                                                                         Dt_min=Dt_min, Dt=Dt)

            # Initialize other variables
            synapses.alpha = U_0__star[i]
            # synapses.alpha = U_0__star[0]
            gliot.f_c = 1/duration  # dummy value to assure that you have no release w/in the simulation
            gliot.v_A = 0.0

            # The 'before_thresholds' option assures that also the initial condition on the synaptic Ca2+ is recorded reliably
            C = StateMonitor(synapses, variables=['C_syn'], record=True, dt=1*ms, when='before_thresholds')
            W = StateMonitor(synapses, variables=['W'], record=True, dt=0.1*second)

            run(duration,namespace={},report='text')

            # Convert monitors to dictionaries
            C_mon.append(bu.monitor_to_dict(C))
            W_mon.append(bu.monitor_to_dict(W))
       # logging.debug(f"Saving data for 'nonlin' with U_0__star: {U_0__star}")
       # logging.debug(f"C_mon length: {len(C_mon)} | W_mon length: {len(W_mon)} | dt shape: {dt.shape} | ns: {ns(1, 1, N_pts)}")
        # Save data
            try:
                svu.savedata([C_mon, W_mon, dt, U_0__star, ns(1,1,N_pts)], data_dir+'stdp_curves_0_'+lbl+'.pkl')
                print("Data saved successfully")
            except Exception as e:
                    print(f"Error while saving data: {e}")

        #print("Data saved successfully. Exiting...")
        

    else:
    #---------------------------------------------------------------------------------------------------------------
    # Simulation /2 : show effect of gliotransmission modulation on synaptic plasticity
    #---------------------------------------------------------------------------------------------------------------
        print("Running simulation with gliot.")
        params['U_0__star'] = U_0__star[0]
        alpha = [params['U_0__star'], 0.0, 1.0]
        N_syn = size(alpha)

        source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params,
                                                                                     dt=dt_solv,
                                                                                     N_pts=N_pts, N_syn=N_syn, N_astro=1,
                                                                                     linear=False,
                                                                                     connect=True,
                                                                                     stdp=stdp,
                                                                                     stdp_protocol='pairs',
                                                                                     Dt_min=Dt_min, Dt=Dt)

        # Initialize other variables
        synapses.alpha = tile(array(alpha), (1, N_pts))[0]
        gliot.f_c = 1/duration  # dummy value to assure that you have only one relerase w/in the simulation
        gliot.v_A = 1.0

        # The 'before_thresholds' option assures that also the initial condition on the synaptic Ca2+ is recorded reliably
        C = StateMonitor(synapses, variables=['C_syn'], record=True, dt=1*ms, when='before_thresholds')
        W = StateMonitor(synapses, variables=['W'], record=True, dt=0.1*second)

        run(duration,namespace={},report='text')

        # Convert monitors to dictionaries
        C_mon = bu.monitor_to_dict(C)
        W_mon = bu.monitor_to_dict(W)
        # You will need to slice data as [::N_astro]
        logging.debug(f"Saving data for 'gliot' with alpha: {alpha}")
        #logging.debug(f"C_mon length: {len(C_mon)} | W_mon length: {len(W_mon)} | dt shape: {dt.shape} | ns: {ns(N_syn, 1, N_pts)}")
        
        # Save data
        svu.savedata([C_mon, W_mon, dt, alpha, ns(N_syn,1,N_pts)], data_dir+'stdp_curves_glt_'+lbl+'.pkl')
        filename = data_dir + 'stdp_curves_0_' + lbl + '.pkl'
       # raise SystemExit("Cutting the code early after saving the data.")


def stdp_curves_parameters_sim_pre(params, Dt_min, Dt, dt_solv, N_pts, N_syn, stdp='nonlinear', data_dir='../data/'):
    '''
    Compute how threshold and LTP/LTD area ratio change for different gliotransmitter modulations
    '''

    if stdp=='linear':
        lbl = 'lin'
    else:
        lbl = 'nonlin'

    # Define some lambdas
    ns = lambda Nsyn, Nastro, Npts  : [Nsyn,Nastro,Npts]

    # Duration of each stimulus protocol
    duration = 61*second

    # Extend range of simulation wrt above (needed to compute finer areas under curve)
    dt = arange(Dt_min, Dt_min+Dt, Dt/N_pts)
    # Data sets
    alpha = concatenate((
        linspace(0.0, params['U_0__star'] * (1 - 1.0 / N_syn), int(np.round(N_syn / 2))),
        linspace(params['U_0__star'], 1.0, int(np.round(N_syn / 2)))
    ))
    

    source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params,
                                                                                 dt=dt_solv,
                                                                                 N_pts=N_pts, N_syn=N_syn, N_astro=1,
                                                                                 connect='default',
                                                                                 stdp=stdp,
                                                                                 stdp_protocol='pairs',
                                                                                 Dt_min=Dt_min, Dt=Dt)

    # Initialize other variables
    synapses.alpha = tile(alpha, (1, N_pts))[0]
    gliot.f_c = 1/duration  # dummy value to assure that you have only one release w/in the simulation
    gliot.v_A = 1.0

    # The 'before_thresholds' option assures that also the initial condition on the synaptic Ca2+ is recorded reliably
    C = StateMonitor(synapses, variables=['C_syn'], record=True, dt=10*ms, when='before_thresholds')
    # W = StateMonitor(synapses, variables=['W'], record=True, dt=1.0*second)

    run(duration,namespace={},report='text')

    # Convert monitors to dictionaries
    C_mon = bu.monitor_to_dict(C)
    # W_mon = bu.monitor_to_dict(W)
    # You will need to slice data as [::N_astro]

    # Save data
    svu.savedata([C_mon, dt, alpha, ns(N_syn,1,N_pts)], data_dir+'stdp_curves_xi_'+lbl+'.pkl')

def stdp_curves(sim=False, stdp='nonlinear', format='eps', data_dir='../data/', fig_dir='../Figures/'):

    # Get model parameters
    params = asmod.get_parameters(synapse='neutral', stdp=stdp)
    if stdp=='linear':
        params['f_in'] = 1.0*Hz
        params['D'] = 11.85*ms

        # Adjust gliotransmission modulation parameters
        params['rho_e'] = 1.0e-4
        params['O_G'] = 0.6/umole/second

        # File labels
        lbl='lin'
    else:
        params['f_in'] = 1.0*Hz

        # Adjust gliotransmission modulation parameters
        params['rho_e'] = 1.0e-4
        params['Omega_c'] = 1000./second
        params['Omega_G'] = (1./30.)*Hz
        # DP standard at regular synapse
        we = 59.0
        # DP standard at strong dep. synapse
        # we = 35.0
        params['we'] = we/second / (params['rho_c']*params['Y_T'])/params['tau_pre_d'] # excitatory synaptic weight (voltage)

        # File labels
        lbl='nonlin'

    # Define some lambdas
    ns = lambda Nsyn, Nastro, Npts  : [Nsyn,Nastro,Npts]

    # Stimulus duration
    duration =  61*second

    # Global simulation parameters
    Dt_min = -100*ms
    dt_step = 0.1*ms
    N_pts = 100
    print("the sim value is ",sim)
    if sim:
        #---------------------------------------------------------------------------------------------------------------
        # Simulations /1: Show effect of short-term plasticity on STDP curves w/out
        #---------------------------------------------------------------------------------------------------------------
        print("made it to sim 1")
        stdp_curves_stp(params, Dt_min=Dt_min, Dt=abs(Dt_min)*2, N_pts=N_pts,
                        duration=duration, dt_solv=dt_step, stdp=stdp, gliot=False, data_dir=data_dir)
        # ---------------------------------------------------------------------------------------------------------------
        # Simulations /2: Show effect of short-term plasticity on STDP curves w/ gliotransmission
        # ---------------------------------------------------------------------------------------------------------------
        print("made it to sim 2")
        stdp_curves_stp(params, Dt_min=Dt_min, Dt=abs(Dt_min)*2, N_pts=N_pts,
                        duration=duration, dt_solv=dt_step, stdp=stdp, gliot=True, data_dir=data_dir)
        # ---------------------------------------------------------------------------------------------------------------
        # Simulation /3 : show effect of gliotransmission for finer xi on threshold and peaks / areas of different plasticity
        #---------------------------------------------------------------------------------------------------------------
        print("made it to sim 3")
        stdp_curves_parameters_sim_pre(params, Dt_min=Dt_min, Dt=abs(Dt_min)*2, dt_solv=dt_step, N_pts=200, N_syn=50, data_dir=data_dir)

    #---------------------------------------------------------------------------------------------------------------
    # Plot results
    #---------------------------------------------------------------------------------------------------------------
    # Plotting defaults
    spine_position = 5
    lw = [1.5,2.0]
    alw = 1.0
    afs = 14
    lfs = 16
    legfs = 10
    fmt = '.'
    color = {'zero' : chex((143,135,130)),
             'thr'  : ['c', chex((204,153,0))],
             'U_0__star' : 'b',
             'ltp'  : chex((230,92,0)),
             'ltd'  : chex((0,51,102)),
             'ratio' : 'm',
             'facilitating' : chex((44,160,44)),
             'depressing'   : chex((214,39,40))
             }

    # ---------------------------------------------------------------------------------------------------------------
    # Simulation /1
    # ---------------------------------------------------------------------------------------------------------------
    [C, W, dt, u0, _] = svu.loaddata(data_dir+'stdp_curves_0_'+lbl+'.pkl')

    # C = reshape_monitor(C)
    # W = reshape_monitor(W)

    # Plot results
    p_colors = ['k','m','lime']
    _, ax = figtem.generate_figure('2x1', figsize=(7.0,8.0), left=0.18, right=0.1, bottom=0.08, top=0.02, vs=[0.05])
    ax = plot_stdp_curves(dt/1e-3, C, W, params, duration, N_sim=3, mean_field=True,
                          ax=ax, zorder=[10,6,8], alpha=[0.6,1.0,1.0], color=p_colors)

    # Add legend / 0
    labels_legend = ['LTD threshold',
                     'LTP threshold']
    linestyle_legend = ['-','--']
    obj_legend  = []
    for i,_ in enumerate(labels_legend):
        # Add artist for legend
        obj_legend.append(ax[1].plot([],[],color='k', lw=lw[1], ls=linestyle_legend[i], label=labels_legend[i])[0])
    # Add Legend
    ax[0].legend(obj_legend, labels_legend,
                 fontsize=legfs, frameon=False, loc=2)

    # Add legend / 1
    labels_legend = ['no s.t.p.',
                     's.t. fac.',
                     's.t. dep.']
    obj_legend  = []
    for i,_ in enumerate(labels_legend):
        # Add artist for legend
        obj_legend.append(ax[1].plot([],[], color=p_colors[i], marker='o', lw=lw[1], label=labels_legend[i])[0])
    # Add Legend
    ax[1].legend(obj_legend, labels_legend, numpoints=1,
                 fontsize=legfs, frameon=False, loc=2)

    # Adjust axes
    xlim = (-100,100.1)
    ylim = (0.0,8.0)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_yticks(arange(0.0,8.01,2.))
    ax[0].yaxis.set_label_coords(-.11,0.5)
    ax[0].set_ylabel('% Time above thr.', fontsize=lfs)
    ax[0].yaxis.set_tick_params(labelsize=afs)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    ylim = (-60,60.01)
    figtem.adjust_spines(ax[1], ['left', 'bottom'], position=spine_position)
    ax[1].set_xlim(xlim)
    ax[1].set_xticks(arange(-100,100.1,50))
    ax[1].set_xlabel(r'$\Delta t$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[1].set_ylim(ylim)
    ax[1].set_yticks(arange(-60,60.1,30))
    ax[1].yaxis.set_label_coords(-.11,0.5)
    ax[1].set_ylabel(r'%$\Delta$ Syn. Strength', fontsize=lfs)
    ax[1].xaxis.set_tick_params(labelsize=afs)
    ax[1].yaxis.set_tick_params(labelsize=afs)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    #---------------------------------------------------------------------------------------------------------------
    # Simulation /2
    #---------------------------------------------------------------------------------------------------------------
    [C, W, dt, _, _] = svu.loaddata(data_dir+'stdp_curves_glt_'+lbl+'.pkl')

    # Plot results
    p_colors = ['k',color['depressing'], color['facilitating']]
    _, ax = figtem.generate_figure('2x1', figsize=(7.0,8.0), left=0.18, right=0.1, bottom=0.08, top=0.02, vs=[0.05])
    ax = plot_stdp_curves(dt/1e-3, C, W, params, duration, N_sim=3, mean_field=True,
                          ax=ax, zorder=[10,6,8], alpha=[0.5,1,0.8], color=p_colors)

    # Adjust axes
    xlim = (-100,100.1)
    ylim = (0.0,8.0)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_yticks(arange(0.0,8.01,2.))
    ax[0].yaxis.set_label_coords(-.11,0.5)
    ax[0].set_ylabel('% Time above thr.', fontsize=lfs)
    ax[0].yaxis.set_tick_params(labelsize=afs)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Add legend / 1
    labels_legend = ['no gliot.',
                     'R.I. gliot.',
                     'R.D. gliot.']
    obj_legend  = []
    for i,_ in enumerate(labels_legend):
        # Add artist for legend
        obj_legend.append(ax[1].plot([],[], color=p_colors[i], marker='o', lw=lw[1], label=labels_legend[i])[0])
    # Add Legend
    ax[1].legend(obj_legend, labels_legend, numpoints=1,
                 fontsize=legfs, frameon=False, loc=2)


    ylim = (-60,60.01)
    figtem.adjust_spines(ax[1], ['left', 'bottom'], position=spine_position)
    ax[1].set_xlim(xlim)
    ax[1].set_xticks(arange(-100,100.1,50))
    ax[1].set_xlabel(r'$\Delta t$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[1].set_ylim(ylim)
    ax[1].set_yticks(arange(-60,60.1,30))
    ax[1].yaxis.set_label_coords(-.11,0.5)
    ax[1].set_ylabel(r'%$\Delta$ Syn. Strength', fontsize=lfs)
    ax[1].xaxis.set_tick_params(labelsize=afs)
    ax[1].yaxis.set_tick_params(labelsize=afs)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    #---------------------------------------------------------------------------------------------------------------
    # Simulation /3
    #---------------------------------------------------------------------------------------------------------------
    [C, dt, alpha, _] = svu.loaddata(data_dir+'stdp_curves_xi_'+lbl+'.pkl')

    # Compute curve parameters
    thr, wmin, wmax, aratio, thr2 = stdp_curve_parameters(dt/second, C, size(alpha), params)

    # Define some lambdas for post-processing
    thr_expanded = np.tile(thr, (len(alpha), 1)).T if thr.ndim == 1 else thr
    altthr = lambda thr, alpha: (thr_expanded < np.inf) & (alpha > params['U_0__star'])



    # Plot results
    _, ax = figtem.generate_figure('2x1', figsize=(7.0,8.0), left=0.18, right=0.1, bottom=0.08, top=0.02, vs=[0.03])

    # Plot threshold
    # Split axis for better plotting (only in the case of two thresholds)
    ax_aux = ax[0].twinx()
    pos = ax[0].get_position()
    sp = 0.03
    pos_aux = [pos.x0, pos.y0+pos.height*(.5+sp), pos.width, pos.height*(.5-sp)]
    pos = [pos.x0, pos.y0, pos.width, pos.height*(.5-sp)]
    ax[0].set_position(pos)
    ax_aux.set_position(pos_aux)

    # Lower part of the graph
    xlim = [0.0,1.01]
    ylim = [-3.0,1.7]
    transparency = 0.4
    print("Alpha:", alpha.shape)
    print("Indices or Mask:", altthr(thr2, alpha))

    # Add patches
    ax[0].add_artist(mpatches.Polygon(list(zip(*vstack(([params['U_0__star'],params['U_0__star'],xlim[0],xlim[0]],ylim+ylim[::-1])))), ec='none', fc=color['depressing'],alpha=transparency))
    ax[0].add_artist(mpatches.Polygon(list(zip(*vstack(([params['U_0__star'],params['U_0__star'],xlim[-1],xlim[-1]],ylim+ylim[::-1])))), ec='none', fc=color['facilitating'], alpha=transparency))
    
    ax[0].add_artist(
        mpatches.Polygon(
            list(
                zip(
                    *vstack(
                        (
                            [alpha[altthr(thr2, alpha)], alpha[altthr(thr2, alpha)], xlim[-1], xlim[-1]],
                            [ylim[0], 1.1 * ylim[-1], 1.1 * ylim[-1], ylim[0]],
                        )
                    )
                )
            )
        ),
        ec='k',
        ls='dotted',
        fill=False,
        hatch='/'
    )

    # Zero crossing
    ax[0].plot(xlim, [0.0,0.0], ls='-', c=color['zero'], lw=lw[0])
    # U_0__star
    ax[0].plot([params['U_0__star'],params['U_0__star']], ylim, ls='--', c=color['U_0__star'], lw=lw[0])
    # Effective data
    ax[0].plot(alpha, thr, ls='-', marker='o', color=color['thr'][0], lw=lw[1])
    # Adjust axes
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_yticks(arange(-3.0,1.01,1.0))
    ax[0].yaxis.set_tick_params(labelsize=afs)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Upper part of the graph
    ylim = [32,90.0]
    # Add patches
    ax_aux.add_artist(mpatches.Polygon(list(zip(*vstack(([params['U_0__star'],params['U_0__star'],xlim[0],xlim[0]],ylim+ylim[::-1])))), ec='none', fc=color['depressing'], alpha=transparency))
    ax_aux.add_artist(mpatches.Polygon(list(zip(*vstack(([params['U_0__star'],params['U_0__star'],xlim[-1],xlim[-1]],ylim+ylim[::-1])))), ec='none', fc=color['facilitating'], alpha=transparency))
    ax_aux.add_artist(mpatches.Polygon(list(zip(*vstack(([alpha[altthr(thr2,alpha)][0],alpha[altthr(thr2,alpha)][0],xlim[-1],xlim[-1]],[ylim[0],1.1*ylim[-1],1.1*ylim[-1],ylim[0]])))), ec='k', ls='dotted', fill=False, hatch='/'))
    # U_0__star
    ax_aux.plot([params['U_0__star'],params['U_0__star']], ylim, ls='--', c=color['U_0__star'], lw=lw[0])
    ax_aux.plot(alpha[altthr(thr2,alpha)], thr2[altthr(thr2,alpha)], ls='-', marker='o', color=color['thr'][1], lw=lw[1])
    # Adjust axes
    figtem.adjust_spines(ax_aux, ['left'], position=spine_position)
    ax_aux.set_xlim(xlim)
    ax_aux.set_ylim(ylim)
    ax_aux.set_yticks(arange(40,90.01,20.0))
    ax_aux.yaxis.set_tick_params(labelsize=afs)
    ax_aux.set_ylabel(r'Threshold $\Delta t$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs, verticalalignment='bottom')
    ax_aux.yaxis.set_label_coords(-0.11,-0.1)
    # Add cut-out diagonal lines (from matplotlib broken_axis.py example)
    o = .02
    d = .015 # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax[0].transAxes, color='k', clip_on=False)
    ax[0].plot((-d-o,+d-o),(1-d,1+d), **kwargs)  # top-left diagonal
    kwargs.update(transform=ax_aux.transAxes)  # switch to the bottom axes
    ax_aux.plot((-d-o,+d-o),(-d,+d), **kwargs)   # bottom-left diagonal

    # Plot other curve parameters
    # Add legend to new axis
    labels_legend = [r'D$\rightarrow$P threshold',
                     r'P$\rightarrow$D threshold',
                     'Max. LTP',
                     'Max. LTD']
    obj_legend  = []
    color_legend = [color['thr'][0],color['thr'][1],color['ltp'],color['ltd']]
    for i,_ in enumerate(labels_legend):
        # Add artist for legend
        obj_legend.append(ax[1].plot([],[],color=color_legend[i], lw=lw[1], marker='o', label=labels_legend[i])[0])
    # Add Legend
    ax[1].legend(obj_legend, labels_legend,
                 fontsize=legfs, frameon=True, numpoints=1,
                 loc=7)

    # Effective plotting
    ylim = ([9.0,60.01],[-0.2,15.])
    # Zero crossing
    ax[1].plot(xlim, [0.0,0.0], ls='-', color=color['zero'], lw=lw[0])
    # U_0__star
    ax[1].plot([params['U_0__star'],params['U_0__star']], ylim[0], ls='--', color=color['U_0__star'], lw=lw[0])
    # Peaks
    ax[1].plot(alpha, wmin*100., ls='-', marker='o', color=color['ltd'], lw=lw[1])
    ax[1].plot(alpha, wmax*100., ls='-', marker='o', color=color['ltp'], lw=lw[1])

    # Area ratio
    ax_aux = ax[1].twinx()
    # # Add patches
    ax_aux.add_artist(mpatches.Polygon(list(zip(*vstack(([params['U_0__star'],params['U_0__star'],xlim[0],xlim[0]],ylim[1]+ylim[1][::-1])))), ec='none', fc=color['depressing'],alpha=0.5))
    ax_aux.add_artist(mpatches.Polygon(list(zip(*vstack(([params['U_0__star'],params['U_0__star'],xlim[-1],xlim[-1]],ylim[1]+ylim[1][::-1])))), ec='none', fc=color['facilitating'], alpha=0.5))
    ax_aux.add_artist(mpatches.Polygon(list(zip(*vstack(([alpha[altthr(thr2,alpha)][0],alpha[altthr(thr2,alpha)][0],xlim[-1],xlim[-1]],[ylim[1][0],1.1*ylim[1][-1],1.1*ylim[1][-1],ylim[1][0]])))), ec='k', ls='dotted', fill=False, hatch='/'))
    # # Extrapolate the first data points due to limited Dt exploration (might be commented for Dt range sufficiently large)
    # if sum(aratio[~altthr(thr2,alpha)]==0)>0:
    #     index = where(aratio[~altthr(thr2,alpha)]==0)[0][-1] + 1
    #     extrapolator = UnivariateSpline(alpha[~altthr(thr2,alpha)][index::], aratio[~altthr(thr2,alpha)][2::], k=3)
    #     aratio[0:index] = extrapolator(alpha[0:index])
    # # ax_aux.plot(alpha[~altthr(thr2,alpha)], aratio[~altthr(thr2,alpha)], ls='-', marker='^', color=color['ratio'], lw=lw[1])
    ax_aux.plot(alpha,aratio,ls='-',marker='^',color=color['ratio'],lw=lw[1])
    # First make patch and spines of ax_aux invisible
    figtem.make_patch_spines_invisible(ax_aux)
    # # Then show only the right spine of ax_aux
    ax_aux.spines['right'].set_visible(True)
    ax_aux.spines['right'].set_position(('outward', spine_position))

    # Bring original axes in front
    ax[1].set_zorder(ax_aux.get_zorder()+1) # put ax[0] stem plot in front of ax_aux
    ax[1].patch.set_visible(False)          # hide the 'canvas'

    # Adjust axes
    figtem.adjust_spines(ax[1], ['left','bottom'], position=spine_position)
    ax[1].set_xlim(xlim)
    ax[1].set_xticks(arange(0.0,1.01,0.2))
    ax[1].set_xlabel('Gliotransmission Type', fontsize=lfs)
    ax[1].set_ylim(ylim[0])
    ax[1].set_yticks(arange(10.0,60.1,10.0))
    ax[1].yaxis.set_tick_params(labelsize=afs)
    ax[1].set_ylabel('Peak %$\Delta$ \n Syn. Strength', fontsize=lfs, multialignment='center')
    ax[1].yaxis.set_label_coords(-0.11,0.5)
    ax_aux.set_ylim(ylim[1])
    ax_aux.set_yticks(arange(0.,15.01,3.))
    ax_aux.set_ylabel('LTP/LTD Area Ratio', fontsize=lfs)
    ax_aux.yaxis.set_tick_params(labelsize=afs)

    # Colorize
    tkw = dict(size=4, width=1.5)
    ax_aux.yaxis.label.set_color(color['ratio'])
    ax_aux.tick_params(axis='y', colors=color['ratio'], **tkw)

    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    #-------------------------------------------------------------------------------------------------------------------
    # # Save Figures
    #-------------------------------------------------------------------------------------------------------------------
    save_figures(label=fig_dir+'fig5_stdp_pre_',filename=['stp','xi','curves'], format=format, dpi=600)

    plt.show()

def test_currents(post=True):
    #TODO: READ THIS COMMENT FOR TODO WORK UPON REVIVING THE PROJECT
    #- Figure out the 2-exp nonlinear calcium model and run Figures 6-7 accordingly
    #- Once understood the nonlinear Ca2+ model also implement SIC currents and postsynaptic currents as a 2-exp currents
    #- The current version of SIC-mechanism (Figure 5) is correct, but I am not sure the model used is correct (refer to Barbier et al. JN 2014)


    # It is likely that you may tuned not on the refractory but rather on the ICs
    params = asmod.get_parameters(synapse='neutral', stdp='nonlinear')
    params['f_in'] = 0.1*Hz
    # params['Omega_c'] = 500.0/second

    # Adjust gliotransmission modulation parameters
    params['rho_e'] = 1.0e-4
    params['O_G'] = 0.6/umole/second
    # params['Omega_G'] = 0.2 / second # Makes gliotransmitter effect decay fast to better show mechanism
    params['Omega_c'] = 100./second
    # Make Ca2+ decay time exaggeratedly large for visualization purposes
    params['tau_ca'] = 200*ms

    # we = 10000
    # params['we'] = we*params['tau_pre_d']/0.00125*mole/second**2
    # Global simulation parameters
    alpha = [params['U_0__star'], 0.0, 1.0]
    t_on = 0.2*second
    Dt = {'ltp': -30*ms, 'ltd': 30*ms}
    t_glt = t_on-0.1*second
    duration =  0.8*second

    # Build stimulation protocol
    spikes_pre = {'ltd': t_on+arange(0.0,duration,1/params['f_in'])*second-Dt['ltp'],
                  'ltp': t_on+arange(0.0,duration,1/params['f_in'])*second}
    # spikes_pre = {'ltd': t_on+arange(0.0,duration,1/params['f_in'])*second-Dt['ltp'],
    #               'ltp': t_on+arange(duration,duration,1/params['f_in'])*second}
    spikes_post ={'ltd': t_on+arange(0.0,duration,1/params['f_in'])*second,
                  'ltp': t_on+arange(0.0,duration,1/params['f_in'])*second+Dt['ltd']}
    # Update with
    duration += t_on
    dt=0.1*ms
    N_syn=1

    indices = arange(N_syn).repeat(len(spikes_pre['ltp']))
    spikes  = spikes_pre['ltp'][None,:].repeat(N_syn,axis=0).flatten() # Repeat the spike train as many as N_syn
    source_group = SpikeGeneratorGroup(N_syn,
                                       indices=indices,
                                       times=spikes,
                                       dt=dt)
    # eqs_target = '''
    # dv/dt = (g_e+(E_L - v))/tau_m : volt (unless refractory)
    # g_e : volt
    # '''
    # Post
    indices_post = arange(N_syn).repeat(len(spikes_post['ltp']))
    spikes_post  = spikes_post['ltp'][None,:].repeat(N_syn,axis=0).flatten() # Repeat the spike train as many as N_syn
    target_group = SpikeGeneratorGroup(N_syn,
                                       indices=indices_post,
                                       times=spikes_post,
                                       name='neu_post*',
                                       dt=dt)


    eqs_syn = '''
    dA/dt = -A/tau_pre_r + npre*B : 1
    dB/dt = -B/tau_pre_d + Y_S*we : Hz
    dY_S/dt = -Omega_c * Y_S : mole
    # g_e_post = A*G_e : volt
    dRe/dt = -Re/tau_post_r + npost*F : 1
    dF/dt = -F/tau_post_d : Hz
    '''
    pre  = '''
    Y_S += rho_c * Y_T
    '''
    # pre = None
    # post  = '''
    # F += (1.0+eta*A)/second
    # '''
    post = None

    we = 1.85
    params['we'] = we/second / (params['rho_c']*params['Y_T'])/params['tau_pre_d'] # excitatory synaptic weight (voltage)
    params['npre'] = peak_normalize(params['Cpre'],params['tau_pre_r'],params['tau_pre_d'])
    params['npost'] = peak_normalize(params['Cpost'],params['tau_post_r'],params['tau_post_d'])

    synapses = Synapses(source_group, target_group, eqs_syn,
                    pre=pre,
                    post=post,
                    connect='i==j',
                    namespace=params,
                    name='synapse',
                    dt=dt)

    synapses.A = 0
    synapses.B = 0
    synapses.Re = 0
    synapses.F = 0

    # Monitors
    S = StateMonitor(synapses,['Y_S','A','B','Re','F'], record=True)

    run(duration,namespace={},report='text')

    fig,ax = plt.subplots(2,2)
    print(shape(ax))
    # ax[0].plot(S.t_,S[0].Y_S_,'k-')
    ax[0][0].plot(S.t_,S[0].Y_S_,'r-')
    ax[0][0].set_ylabel('B')
    ax[1][0].plot(S.t_,S[0].A_,'g-')
    ax[1][0].set_ylabel('A')
    ax[0][1].plot(S.t_,S[0].F_,'k-')
    ax[0][1].set_ylabel('F')
    ax[1][1].plot(S.t_,S[0].Re_,'b-')
    ax[1][1].set_ylabel('E')
        # ax[1].plot(S.t_,peak_normalize(params['tau_post_r'],params['tau_post_d'])*S[0].E_,'b-')


    # plt.show()

# def stdp_pre(sim=True, format='eps', fig_dir='../Figures/'):
def stdp_pre(sim=True, format='svg', fig_dir='../Figures/'):
    stdp_mechanism(sim=sim, format=format, fig_dir=fig_dir)
    print(f"sim: {sim}, format: {format}, fig_dir: {fig_dir}")
    stdp_curves(sim=sim, format=format, fig_dir=fig_dir)
#     test_currents()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Figure 6
#-----------------------------------------------------------------------------------------------------------------------
def plot_stdp_sic(S, N, P, G, Csic, params, N_syn=1, N_astro=1, color='k'):

    # Defaults
    opts = {'spine_position' : 5,
            'lw'  : 1.5,
            'lw2' : 2.0,
            'alw' : 1.0,
            'afs' : 14,
            'lfs' : 16,
            'cf'  : 0.1,   # correction Factor for plotting \rho(t) over reference line
            }

    # Fixed parameters
    dt = diff(S['t'])[0]
    tlims = [S['t'][0],S['t'][-1]+dt]
    xticks = arange(0.0,1.1,0.2)
    w_col = '#660066'
    colors = {'sic' : 'c',
              'w'  : chex((148,103,189)),
              'w0' : chex((143,135,130))}

    # Define some lambdas
    normalize = lambda x, x0 : (x-x0)/x0*100

    # for i in xrange(size(Csic[::N_astro])):
    for i in range(N_syn):
        for j in range(N_astro):
            # Generate figure
            fig, ax = figtem.generate_figure('4x1_1L3S', figsize=(6.0,6.0), left=0.16, right=0.1, bottom=0.15, top=0.05)
            ylims = (0.0, params['Cpost']+0.1)
            yticks = arange(0.0,2.51,1.)
            #-------------------------------------------------------------------------------------------------------------------
            # Top plot / 1
            #-------------------------------------------------------------------------------------------------------------------
            idx = np.where(N['i']==i+j*N_syn)[0]
            l, m, b = ax[0].stem(N['t'][idx], ones(len(idx)), linefmt='k', markerfmt='', basefmt='')
            plt.setp(m, 'linewidth', opts['lw'])
            plt.setp(l, 'linestyle', 'none')
            plt.setp(b, 'linestyle', 'none')

            figtem.adjust_spines_uneven(ax[0], va='left', ha='bottom', position=opts['spine_position'])
            ax[0].set_xlim(tlims)
            ax[0].set_xticks(xticks)
            ax[0].set_xticklabels('')
            ax[0].set_ylim(ylims)
            ax[0].set_yticks(yticks)
            # ax[0].set_ylabel('$C_{pre}$', fontsize=opts['lfs'])
            ax[0].set_ylabel('Pre', fontsize=opts['lfs'])
            ax[0].yaxis.set_label_coords(-0.09,0.5)
            # Adjust axes (Cpre)
            set_axlw(ax[0], lw=opts['alw'])
            set_axfs(ax[0], fs=opts['afs'])

            #-------------------------------------------------------------------------------------------------------------------
            # Top plot / 2
            #-------------------------------------------------------------------------------------------------------------------
            # Cpost
            idx = np.where(P['i']==i+j*N_syn)[0]
            l, m, b = ax[1].stem(P['t'][idx], params['Cpost']*ones(len(idx)), linefmt='k', markerfmt='', basefmt='')
            plt.setp(m, 'linewidth', opts['lw'])
            plt.setp(l, 'linestyle', 'none')
            plt.setp(b, 'linestyle', 'none')

            # Add SIC
            ax_aux = ax[1].twinx()
            # cf = 3.5 # correction factor to show currrent in mV (hard-coded) as in the SIC
            figtem.adjust_spines_uneven(ax_aux, va='right', ha='bottom', position=opts['spine_position'])
            # Same axes (obsolete)
            lg, mg, bg =  ax_aux.stem([G['t'][j]], [Csic[i]], linefmt=colors['sic'], markerfmt='', basefmt='')
            plt.setp(mg, 'linewidth', opts['lw'])
            plt.setp(lg, 'linestyle', 'none')
            plt.setp(bg, 'linestyle', 'none')
            ax_aux.patch.set_alpha(0.)

            # Auxiliary axis (the additional axis does not contain any object is added for readiblity)
            ax_aux.set_xlim(tlims)
            ax_aux.set_ylim(ylims)
            ax_aux.set_yticks(yticks)
            ax_aux.yaxis.set_tick_params(labelsize=opts['afs'])
            ax_aux.set_ylabel('SIC', fontsize=opts['lfs'])
            # Colorize
            tkw = dict(size=4, width=1.5)
            ax_aux.yaxis.label.set_color(colors['sic'])
            ax_aux.tick_params(axis='y', colors=colors['sic'], **tkw)
            # Adjust axes
            set_axlw(ax_aux, lw=opts['alw'])
            set_axfs(ax_aux, fs=opts['afs'])

            # Adjust axis
            figtem.adjust_spines_uneven(ax[1], va='left', ha='bottom', position=opts['spine_position'])
            ax[1].set_xlim(tlims)
            ax[1].set_xticks(xticks)
            ax[1].set_xticklabels('')
            ax[1].set_ylim(ylims)
            ax[1].set_yticks(yticks)
            ax[1].yaxis.set_tick_params(labelsize=opts['afs'])
            # ax[1].set_ylabel('$C_{post}$', fontsize=opts['lfs'])
            ax[1].set_ylabel('Post', fontsize=opts['lfs'])
            ax[1].yaxis.set_label_coords(-0.09,0.5)

            #-------------------------------------------------------------------------------------------------------------------
            # Central plot
            #-------------------------------------------------------------------------------------------------------------------
            ylim = [-0.05,3.0]
            # Calcium
            analysis.threshold_regions(S['t'], S['C_syn'][i+j*N_syn], params['Theta_d'], params['Theta_p'], ax=ax[2], y0=-0.05)
            # Plot Ca2+ trace
            ax[2].plot(S['t'], S['C_syn'][i+j*N_syn], color=color, lw=opts['lw2'])
            figtem.adjust_spines_uneven(ax[2], va='left', ha='bottom', position=opts['spine_position'])
            ax[2].set_xlim(tlims)
            ax[2].set_xticklabels('')
            ax[2].set_ylim(ylim)
            ax[2].set_yticks(arange(0.0,3.1,1.0))
            ax[2].set_ylabel('Postsynaptic\n Calcium', fontsize=opts['lfs'], multialignment='center')
            ax[2].yaxis.set_label_coords(-0.08,0.5)
            # Adjust labels
            set_axlw(ax[2], lw=opts['alw'])
            set_axfs(ax[2], fs=opts['afs'])

            #-------------------------------------------------------------------------------------------------------------------
            # Bottom plot
            #-------------------------------------------------------------------------------------------------------------------
            # Synaptic Weight
            if S['W'][i][-1]>=0:
                cf = opts['cf']
            else:
                cf = -opts['cf']
            # Plot reference line
            ax[3].plot(tlims, zeros((2,1)), color=colors['w0'], lw=opts['lw'])
            # Plot rho
            # A correction factor is added to avoid superposition of reference line and effective \rho(t)
            ax[3].plot(S['t'], cf+normalize(S['W'][i+j*N_syn], params['W_0']), color=colors['w'], lw=opts['lw'])
            figtem.adjust_spines(ax[3], ['left', 'bottom'], position=opts['spine_position'])
            ax[3].set_xlim(tlims)
            ax[3].set_xticks(xticks)
            ax[3].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=opts['lfs'])
            ax[3].set_ylim([-2.6,2.6])
            ax[3].set_yticks(arange(-2.0,2.1,1.0))
            ax[3].set_ylabel('%$\Delta$ Syn.\n Str.', fontsize=opts['lfs'], multialignment='center')
            ax[3].yaxis.set_label_coords(-0.08,0.5)
            # Adjust labels
            set_axlw(ax[3], lw=opts['alw'])
            set_axfs(ax[3], fs=opts['afs'])

def stdp_sic_mechanism(sim=False, format='eps', data_dir='../data/', fig_dir='../Figures/'):

    # It is likely that you may tuned not on the refractory but rather on the ICs
    params = asmod.get_parameters(stdp='nonlinear')
    params['f_in'] = 1.5*Hz

    # Adjust gliotransmission modulation parameters
    params['rho_e'] = 1.0e-4
    params['Omega_c'] = 1000./second
    params['tau_sic_r'] = 5.*ms
    params['tau_sic'] = 100.*ms
    params['Omega_e'] = 50./second
    # params['Cpost'] = 0.
    # params['Cpre'] = 0.

    # Add synaptic weight conversion to interface with postsynaptic LIF
    we = 29.8
    params['we'] = we/second / (params['rho_c']*params['Y_T'])/params['tau_pre_d'] # excitatory synaptic weight (voltage)
    # SIC contribution
    # wa = 10.68
    wa = 10.6
    params['wa'] = wa/second / (params['rho_e']*params['G_T'])/params['tau_sic'] # SIC/SOC synaptic weight (voltage)

    # Global simulation parameters
    # Csic = [0.0, 1.0]
    Csic = params['Cpre']*array([1.])
    t_on = 0.2*second # start of testing protocol
    Dt = {'ltp': -25*ms, 'ltd': 25*ms}
    t_glt = [.1,.4]*second # Time of gliot. release (absolute)
    N_syn = size(Csic)
    N_astro = size(t_glt)
    # Prepare parameters for simulation
    Csic = tile(Csic,(N_astro,1)).reshape((1,N_syn*N_astro),order='F')[0]

    # Simulation time
    duration =  0.8*second
    # Build stimulation protocol
    spikes_pre = {'ltd': t_on+arange(0.0,duration,1/params['f_in'])-Dt['ltp'],
                  'ltp': t_on+arange(0.0,duration,1/params['f_in'])}
    spikes_post ={'ltd': t_on+arange(0.0,duration,1/params['f_in']),
                  'ltp': t_on+arange(0.0,duration,1/params['f_in'])+Dt['ltd']}

    # # # Debug
    # Dt = {'ltd': 25*ms}
    # spikes_pre = {'ltd': t_on+arange(0.0,duration,1/params['f_in'])*second-Dt['ltd']}
    # spikes_post ={'ltd': t_on+arange(0.0,duration,1/params['f_in'])*second}
    # Update with
    duration += t_on
    dt_solv = 0.1*ms


    if sim:
        S_mon, N_mon, P_mon, G_mon = {}, {}, {}, {}
        for k,v in Dt.items():
            # Generate model
            source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params, N_syn=N_syn, N_astro=N_astro,
                                                                                         dt=dt_solv,
                                                                                         linear=True,
                                                                                         connect='default',
                                                                                         stdp='nonlinear',
                                                                                         sic='double-exp',
                                                                                         stdp_protocol='test',
                                                                                         spikes_pre=spikes_pre[k],
                                                                                         spikes_post=spikes_post[k])

            # Initialize other variables
            synapses.Csic = Csic
            gliot.f_c = 1/duration  # dummy value to assure that you have only one release w/in the simulation
            gliot.v_A = 1. - params['f_c']*t_glt

            # The modulation by gliotransmitter must be initialized and start before the protocol
            # NOTE: a way to check that you are simulating g_sic correctly is to set Csic=1 and see C_syn, w/out recording g_sic
            S = StateMonitor(synapses, variables=['W','C_syn'], record=True, dt=1*ms)
            N = SpikeMonitor(source_group, record=True)
            P = SpikeMonitor(target_group, record=True)
            G = SpikeMonitor(gliot, record=True)

            run(duration,namespace={},report='text')

            # Convert monitors to dictionaries
            S_mon[k] = bu.monitor_to_dict(S)
            N_mon[k] = bu.monitor_to_dict(N, monitor_type='spike', fields=['spk'])
            P_mon[k] = bu.monitor_to_dict(P, monitor_type='spike', fields=['spk'])
            G_mon[k] = bu.monitor_to_dict(G, monitor_type='spike', fields=['spk'])

        # Save data
        svu.savedata([S_mon, N_mon, P_mon, G_mon], data_dir+'stdp_sic_mechanism.pkl')

    # Load data
    [S, N, P, G] = svu.loaddata(data_dir+'stdp_sic_mechanism.pkl')

    # # Plotting options
    colors = {'ltd' : '#003366', 'ltp': '#FF0066'}
    # tLTD protocol
    plot_stdp_sic(S['ltd'], N['ltd'], P['ltd'], G['ltd'], Csic, params, N_syn=N_syn, N_astro=N_astro, color=colors['ltd'])

    # tLTP protocol
    plot_stdp_sic(S['ltp'], N['ltp'], P['ltp'], G['ltp'], Csic, params, N_syn=N_syn, N_astro=N_astro, color=colors['ltp'])
    #
    # Save Figures
    protocol = ['ltd','ltp']
    label = ['nosic','sic']
    for i,n in enumerate(plt.get_fignums()):
        figure(n)
        if n<=2:
            p = protocol[0]
        else:
            p = protocol[1]
        plt.savefig(fig_dir+'fig6_'+p+'_'+label[i%2]+'.'+format, format=format, dpi=600)

    plt.show()

def concatenate_sim_timing(params, duration, lbl, mode='default', dt=-75*ms, tag='stdp', data_dir='../data/'):
    """
    Extract solutions from different data sets, lump and save them in a single list. Currently used to suitably handle
    simulations that regard the timing of SICs w.r.t. pairs.

    :param params:
    :param duration:
    :param lbl:
    :param mode:
    :param dt:
    :param tag:
    :return:
    """
    dataset = ['','_tau_sic_r','_tau_sic']
    filestamp = 'stdp_sic_curves_timing_'+lbl

    cad,cap,dW = [],[],[]
    for _,dset in enumerate(dataset):
        [C, W, dt_gliot, dt_syn, N] = svu.loaddata(data_dir+filestamp+dset+'.pkl')
        dt_gliot = np.asarray(dt_gliot)
        dt_syn = np.asarray(dt_syn)
        thd,thp,dw = compute_stdp_curves(dt_syn, C, W, params, duration, N_sim=N[1], mean_field=True)
        if mode=='sic':
            index = (dt_syn>=dt/second).nonzero()[0][0]
            cad.append(thd.T[index])
            cap.append(thp.T[index])
            dW.append(dw.T[index])
        else:
            index = (dt_gliot>=dt/second).nonzero()[0][0]
            cad.append(thd[index])
            cap.append(thp[index])
            dW.append(dw[index])

    print(shape(cad),shape(cap),shape(dw))

    # Save final data
    svu.savedata([cad,cap,dW,dt_gliot,dt_syn], data_dir+'stdp_sic_curves_timing_'+lbl+'_tau_sic_all_'+tag+'.pkl')

def stdp_curves_sic_sim(params, Dt_min, Dt, duration=61*second, dt_solv=0.1*ms, N_pts=50, stdp='nonlinear', data_dir='../data/'):
    if stdp=='linear':
        # File label
        lbl = 'lin'
    else:
        # File label
        lbl = 'nonlin'

    # Define some useful lambdas
    ns = lambda Nsyn, Nastro, Npts  : [Nsyn,Nastro,Npts] # produce list of 'N's useful to handle data
    synpar_reshape = lambda p, Nsyn, Nastro, Npts : tile(tile(p, (Nastro,1)).reshape((1,Nsyn*Nastro),order='F'),(1,Npts))[0] # make parameters suitable for parallel computation

    # define dt_range of investigation
    dt = arange(Dt_min, Dt_min+Dt, Dt/N_pts)
    #---------------------------------------------------------------------------------------------------------------
    # First set of simulations: show effect of SICs at two different frequencies on STDP curves
    #---------------------------------------------------------------------------------------------------------------
    f_c = [0.0, 0.1, 0.5]*Hz
    N_astro = size(f_c)
    source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params,
                                                                                 dt=dt_solv,
                                                                                 N_pts=N_pts, N_syn=1, N_astro=N_astro,
                                                                                 linear=True,
                                                                                 connect='default',
                                                                                 stdp='nonlinear',
                                                                                 sic='double-exp',
                                                                                 stdp_protocol='pairs',
                                                                                 Dt_min=Dt_min, Dt=Dt)

    # Initialize other variables
    synapses.Csic = 1.0*params['Cpre']
    gliot.f_c = f_c  # dummy value to assure that you have no release w/in the simulation
    gliot.v_A = 1.0

    # The 'before_thresholds' option assures that also the initial condition on the synaptic Ca2+ is recorded reliably
    C = StateMonitor(synapses, variables=['C_syn'], record=True, dt=1*ms, when='before_thresholds')
    W = StateMonitor(synapses, variables=['W'], record=True, dt=0.1*second)

    run(duration,namespace={},report='text')

    # Convert monitors to dictionaries
    C_mon = bu.monitor_to_dict(C)
    W_mon =bu.monitor_to_dict(W)

    # Save data
    svu.savedata([C_mon, W_mon, dt, f_c/Hz, ns(1, N_astro, N_pts)], data_dir+'stdp_sic_curves_fc_'+lbl+'.pkl')

    #---------------------------------------------------------------------------------------------------------------
    # Simulation /2 : show SICs effect on synaptic plasticity for three different values of Csic
    #---------------------------------------------------------------------------------------------------------------
    params['f_c'] = 0.1*Hz
    Csic = params['Cpre']*array([1.0,0.5,1.5])
    N_syn = size(Csic)

    source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params,
                                                                                 dt=dt_solv,
                                                                                 N_pts=N_pts, N_syn=N_syn, N_astro=1,
                                                                                 linear=True,
                                                                                 connect=True,
                                                                                 stdp='nonlinear',
                                                                                 sic='double-exp',
                                                                                 stdp_protocol='pairs',
                                                                                 Dt_min=Dt_min, Dt=Dt)

    # Initialize other variables
    synapses.Csic = tile(array(Csic), (1, N_pts))[0]
    gliot.f_c = params['f_c']  # dummy value to assure that you have only one release w/in the simulation
    gliot.v_A = 1.0

    # The 'before_thresholds' option assures that also the initial condition on the synaptic Ca2+ is recorded reliably
    C = StateMonitor(synapses, variables=['C_syn'], record=True, dt=1*ms, when='before_thresholds')
    W = StateMonitor(synapses, variables=['W'], record=True, dt=0.1*second)

    run(duration,namespace={},report='text')

    # Convert monitors to dictionaries
    C_mon = bu.monitor_to_dict(C)
    W_mon = bu.monitor_to_dict(W)
    # You will need to slice data as [::N_astro]

    # Save data
    svu.savedata([C_mon, W_mon, dt, Csic, ns(N_syn, 1, N_pts)], data_dir+'stdp_sic_curves_Csic_'+lbl+'.pkl')

    #---------------------------------------------------------------------------------------------------------------
    # Simulation /3 : show SICs effect on synaptic plasticity for different values of \Delta\varsigma
    #---------------------------------------------------------------------------------------------------------------
    params['f_in'] = 1.*Hz
    params['f_c'] = 0.2*Hz
    Csic = params['Cpre']*array([1.0])
    N_pts = 40
    N_astro = 30
    # Synaptic Dt
    Dt_min_syn = -100.*ms
    Dt_syn = abs(Dt_min_syn)*2.1
    dt_syn = arange(Dt_min_syn, Dt_min_syn+Dt_syn, Dt_syn/N_pts)
    # Astrocytic / SIC Dt
    Dt_min_gliot = -500.*ms
    Dt_gliot = abs(Dt_min_gliot)*2.1
    dt_gliot = arange(Dt_min_gliot, Dt_min_gliot+Dt_gliot, Dt_gliot/N_astro)

    # Build model
    source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params,
                                                                                 dt=dt_solv,
                                                                                 N_pts=N_pts, N_syn=1, N_astro=N_astro,
                                                                                 linear=True,
                                                                                 connect='default',
                                                                                 stdp='nonlinear',
                                                                                 sic='double-exp',
                                                                                 stdp_protocol='pairs',
                                                                                 Dt_min=Dt_min_syn,Dt=Dt_syn)
    # Initialize
    # Gliotransmission
    gliot.f_c = params['f_c']
    gliot.v_A = array([int( (Dt_min_gliot + i/float(N_astro) * Dt_gliot)<0*second ) + params['f_c']*( Dt_min_gliot + i/float(N_astro) * Dt_gliot) for i in range(N_astro)])
    # SIC
    synapses.Csic = params['Cpre']
    # Neural firing
    source_group.f_in = params['f_in']
    target_group.f_in = params['f_in']

    # Monitors
    C = StateMonitor(synapses, variables=['C_syn'], record=True, dt=1*ms, when='before_thresholds')
    W = StateMonitor(synapses, variables=['W'], record=True, dt=0.1*second)
    # Run
    run(duration,namespace={},report='text')

    # Transform to dictionary
    C_mon = bu.monitor_to_dict(C)
    W_mon =bu.monitor_to_dict(W)

    # Save data
    svu.savedata([C_mon, W_mon, dt_gliot, dt_syn, ns(1,N_astro,N_pts)], data_dir+'stdp_sic_curves_timing_'+lbl+'.pkl')

    #---------------------------------------------------------------------------------------------------------------
    # Simulation /4 : show SICs effect on synaptic plasticity for different values of \Delta\varsigma w/ longer SIC rise
    #---------------------------------------------------------------------------------------------------------------
    params['tau_sic_r'] *= 1.5
    # Build model
    source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params,
                                                                                 dt=dt_solv,
                                                                                 N_pts=N_pts, N_syn=1, N_astro=N_astro,
                                                                                 linear=True,
                                                                                 connect='default',
                                                                                 stdp='nonlinear',
                                                                                 sic='double-exp',
                                                                                 stdp_protocol='pairs',
                                                                                 Dt_min=Dt_min_syn,Dt=Dt_syn)
    # Initialize
    # Gliotransmission
    gliot.f_c = params['f_c']
    gliot.v_A = array([int( (Dt_min_gliot + i/float(N_astro) * Dt_gliot)<0*second ) + params['f_c']*( Dt_min_gliot + i/float(N_astro) * Dt_gliot) for i in range(N_astro)])
    # SIC
    synapses.Csic = params['Cpre']
    # Neural firing
    source_group.f_in = params['f_in']
    target_group.f_in = params['f_in']

    # Monitors
    C = StateMonitor(synapses, variables=['C_syn'], record=True, dt=1*ms, when='before_thresholds')
    W = StateMonitor(synapses, variables=['W'], record=True, dt=0.1*second)
    # Run
    run(duration,namespace={},report='text')

    # Transform to dictionary
    C_mon = bu.monitor_to_dict(C)
    W_mon =bu.monitor_to_dict(W)

    # Save data
    svu.savedata([C_mon, W_mon, dt_gliot, dt_syn, ns(1,N_astro,N_pts)], data_dir+'stdp_sic_curves_timing_'+lbl+'_tau_sic_r.pkl')

    #---------------------------------------------------------------------------------------------------------------
    # Simulation /5 : show SICs effect on synaptic plasticity for different values of \Delta\varsigma w/ longer SIC rise
    #---------------------------------------------------------------------------------------------------------------
    params['tau_sic_r'] /= 1.5
    params['tau_sic'] *= 1.5
    # Build model
    source_group,target_group,synapses,gliot,es_astro2syn = asmod.openloop_model(params,
                                                                                 dt=dt_solv,
                                                                                 N_pts=N_pts, N_syn=1, N_astro=N_astro,
                                                                                 linear=True,
                                                                                 connect='default',
                                                                                 stdp='nonlinear',
                                                                                 sic='double-exp',
                                                                                 stdp_protocol='pairs',
                                                                                 Dt_min=Dt_min_syn,Dt=Dt_syn)
    # Initialize
    # Gliotransmission
    gliot.f_c = params['f_c']
    gliot.v_A = array([int( (Dt_min_gliot + i/float(N_astro) * Dt_gliot)<0*second ) + params['f_c']*( Dt_min_gliot + i/float(N_astro) * Dt_gliot) for i in range(N_astro)])
    # SIC
    synapses.Csic = params['Cpre']
    # Neural firing
    source_group.f_in = params['f_in']
    target_group.f_in = params['f_in']

    # Monitors
    C = StateMonitor(synapses, variables=['C_syn'], record=True, dt=1*ms, when='before_thresholds')
    W = StateMonitor(synapses, variables=['W'], record=True, dt=0.1*second)
    # Run
    run(duration,namespace={},report='text')

    # Transform to dictionary
    C_mon = bu.monitor_to_dict(C)
    W_mon =bu.monitor_to_dict(W)

    # Save data
    svu.savedata([C_mon, W_mon, dt_gliot, dt_syn, ns(1,N_astro,N_pts)], data_dir+'stdp_sic_curves_timing_'+lbl+'_tau_sic.pkl')

    #---------------------------------------------------------------------------------------------------------------
    # Build Data from Simulations 4 and 5
    #---------------------------------------------------------------------------------------------------------------
    # The computation of the dw and Ca2+ time fractions are based on the same model parameters that are not touched
    concatenate_sim_timing(params, duration, lbl, mode='sic', dt=-50*ms, tag='ltd', data_dir=data_dir)
    concatenate_sim_timing(params, duration, lbl, mode='sic', dt=60*ms, tag='ltp', data_dir=data_dir)
    concatenate_sim_timing(params, duration, lbl, mode='default', dt=-75*ms, tag='stdp', data_dir=data_dir)

def stdp_sic_curves(sim=False, stdp='nonlinear', format='eps', data_dir='../data/', fig_dir='../Figures/'):

    params = asmod.get_parameters(synapse='neutral',stdp=stdp)
    if stdp=='linear':
        params['f_in'] = 1.0*Hz
        params['D'] = 13.7*ms

        # Adjust gliotransmission modulation parameters
        params['rho_e'] = 1.0e-4
        params['O_G'] = 0.6/umole/second

        # SIC parameters
        params['wa'] = 1.0 / (params['rho_e']*params['G_T']) # SIC/SOC synaptic weight (voltage)
        params['Csic'] = 2.0    # Should be the same of stdp_sic_mechanism

        lbl = 'lin'
    else:
        params['rho_e'] = 1.0e-4
        params['Omega_c'] = 1000./second
        params['tau_sic_r'] = 5.*ms
        params['tau_sic'] = 100.*ms
        params['Omega_e'] = 50./second
        # Add synaptic weight conversion to interface with postsynaptic LIF
        we = 29.8
        params['we'] = we/second / (params['rho_c']*params['Y_T'])/params['tau_pre_d'] # excitatory synaptic weight (voltage)
        # SIC contribution
        wa = 10.6
        params['wa'] = wa/second / (params['rho_e']*params['G_T'])/params['tau_sic'] # SIC/SOC synaptic weight (voltage)

        lbl = 'nonlin'
    # Global simulation parameters
    N_pts = 100
    Dt_min = -100*ms
    Dt = abs(Dt_min)*2.

    # Stimulus duration
    duration =  61*second

    if sim:
        stdp_curves_sic_sim(params, Dt_min, Dt, duration=duration, dt_solv=0.1*ms, N_pts=N_pts, stdp='nonlinear', data_dir=data_dir)

    # #---------------------------------------------------------------------------------------------------------------
    # # Plot results
    # #---------------------------------------------------------------------------------------------------------------
    # Plotting defaults
    spine_position = 5
    lw = [1.5,2.0]
    alw = 1.0
    afs = 14
    lfs = 16
    legfs = 10
    colors = {'ltp'  : '#FF0066',
              'ltd'  : '#003366',
              'Csic' : ('k','c','b'),
              'f_c'  : ('k',chex((230,128,128)),chex((163,0,0)))}

    # ---------------------------------------------------------------------------------------------------------------
    # Simulation /1 / f_c
    # ---------------------------------------------------------------------------------------------------------------
    [C, W, dt, f_c, N] = svu.loaddata(data_dir+'stdp_sic_curves_fc_'+lbl+'.pkl')

    N_astro = N[1]
    # Plot results
    _, ax = figtem.generate_figure('2x1', figsize=(6.0,6.0), left=0.16, right=0.1, bottom=0.12, top=0.05, vs=[0.05])
    ax = plot_stdp_curves(dt/1e-3, C, W, params, duration, N_sim=N_astro, mean_field=True,
                          ax=ax, zorder=[10,6,8], alpha=[0.6,1.0,1.0], color=colors['f_c'])

    # Adjust axes
    xlims = (-100,100.1)
    xticks = arange(xlims[0],xlims[1],50)
    ylims = (0.0,6.0)
    yticks = arange(0.0,6.01,2.0)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].set_yticks(yticks)
    ax[0].set_ylabel('% Time above thr.', fontsize=lfs)
    ax[0].yaxis.set_label_coords(-.14,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    ylims = (-60,60.01)
    figtem.adjust_spines(ax[1], ['left', 'bottom'], position=spine_position)
    ax[1].set_xlim(xlims)
    ax[1].set_xticks(xticks)
    ax[1].set_xlabel('$\Delta t$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[1].set_ylim(ylims)
    ax[1].set_yticks(arange(-60,60.1,30.))
    ax[1].set_ylabel(r'%$\Delta$ Syn. Strength', fontsize=lfs)
    ax[1].yaxis.set_label_coords(-.14,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # Add legend / 1
    labels_legend = ['no gliot.',
                     'SICs @ 0.1 Hz  ',
                     'SICs @ 1.0 Hz']
    obj_legend  = []
    for i,_ in enumerate(labels_legend):
        # Add artist for legend
        obj_legend.append(ax[1].plot([],[], color=colors['f_c'][i], marker='o', lw=lw[1], label=labels_legend[i])[0])
    # Add Legend
    ax[1].legend(obj_legend, labels_legend, numpoints=1,
                 fontsize=legfs, frameon=False, loc=4)

    # ---------------------------------------------------------------------------------------------------------------
    # Simulation /2 / Csic
    # ---------------------------------------------------------------------------------------------------------------
    [C, W, dt, Csic, N] = svu.loaddata(data_dir+'stdp_sic_curves_Csic_'+lbl+'.pkl')

    N_syn = N[0]
    # Plot results
    p_colors = ['k','m','lime']
    _, ax = figtem.generate_figure('2x1', figsize=(6.0,6.0), left=0.16, right=0.1, bottom=0.12, top=0.05, vs=[0.05])
    ax = plot_stdp_curves(dt/1e-3, C, W, params, duration, N_sim=N_syn, mean_field=True,
                          ax=ax, zorder=[6,10,8], alpha=[1.0,0.5,1.0], color=colors['Csic'])

    # Adjust axes
    xlims = (-100,100.1)
    xticks = arange(xlims[0],xlims[1],50)
    ylims = (0.0,6.0)
    yticks = arange(0.0,6.01,2.0)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].set_yticks(yticks)
    ax[0].set_ylabel('% Time above thr.', fontsize=lfs)
    ax[0].yaxis.set_label_coords(-.14,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    ylims = (-60,60.01)
    figtem.adjust_spines(ax[1], ['left', 'bottom'], position=spine_position)
    ax[1].set_xlim(xlims)
    ax[1].set_xticks(xticks)
    ax[1].set_xlabel('$\Delta t$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[1].set_ylim(ylims)
    ax[1].set_yticks(arange(-60,60.1,30.))
    ax[1].set_ylabel(r'%$\Delta$ Syn. Strength', fontsize=lfs)
    ax[1].yaxis.set_label_coords(-.14,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # Add legend / 0
    labels_legend = ['LTD threshold',
                     'LTP threshold']
    linestyle_legend = ['-','--']
    obj_legend  = []
    for i,_ in enumerate(labels_legend):
        # Add artist for legend
        obj_legend.append(ax[1].plot([],[],color='k', lw=lw[1], ls=linestyle_legend[i], label=labels_legend[i])[0])
    # Add Legend
    ax[0].legend(obj_legend, labels_legend,
                 fontsize=legfs, frameon=False, loc=4)

    # Add legend / 1
    labels_legend = ['SICs = EPSCs',
                     'SICs = 0.5$\cdot$EPSCs',
                     'SICs = 1.5$\cdot$EPSCs']
    obj_legend  = []
    for i,_ in enumerate(labels_legend):
        # Add artist for legend
        obj_legend.append(ax[1].plot([],[], color=colors['Csic'][i], marker='o', lw=lw[1], label=labels_legend[i])[0])
    # Add Legend
    ax[1].legend(obj_legend, labels_legend, numpoints=1,
                 fontsize=legfs, frameon=False, loc=4)

    #-------------------------------------------------------------------------------------------------------------------
    # Save Figures
    #-------------------------------------------------------------------------------------------------------------------
    save_figures(label=fig_dir+'fig6_stdp_sic_',filename=['fc','Csic'], format=format, dpi=600)

    # # ---------------------------------------------------------------------------------------------------------------
    # # Simulation /3 / Timing
    # # ---------------------------------------------------------------------------------------------------------------
    [C, W, dt_gliot, dt_syn, N] = svu.loaddata(data_dir+'stdp_sic_curves_timing_'+lbl+'.pkl')

    # Extract data
    N_astro = N[1]
    thd,thp,dw = compute_stdp_curves(dt_syn, C, W, params, duration, N_sim=N_astro, mean_field=True)

    # Redefine axis parameters
    lw = [1.8,2.3]
    apt = 3
    alw += 1
    afs += apt
    lfs += apt
    legfs += 2

    # Prepare x-y axis
    dt, dsic = meshgrid(dt_syn/1.e-3,dt_gliot/1.e-3)
    xlims = (-100.,100.1)
    xticks = arange(-100,100.1,50.)
    ylims = (-500.,500.1)
    yticks = arange(ylims[0],ylims[1],250)

    # Fraction Ca2+ above \theta_d
    vlims = (0.,5.)
    vticks = arange(0.,5.1,1.)
    _, ax = figtem.generate_figure('1x1', figsize=(5.5,4.0), left=0.2, right=0.01, bottom=0.18, top=0.03)
    p = ax[0].pcolormesh(dt,dsic,thd*100., cmap='Greys', shading='gouraud')
    figtem.adjust_spines(ax[0], ['left','bottom'], position=spine_position)
    ax[0].set_xlim(xlims)
    ax[0].set_xticks(xticks)
    ax[0].set_xlabel('$\Delta t$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[0].set_ylim(ylims)
    ax[0].set_yticks(yticks)
    # ax[0].set_ylabel('SIC-pair Delay (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[0].set_ylabel(r'$\Delta \varsigma$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[0].yaxis.set_label_coords(-.22,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)
    # Colorbar
    p.set_clim(vlims)
    cax = make_axes_locatable(ax[0]).append_axes('right', size='10%', pad=0.65)
    cbar = plt.colorbar(p, cax=cax, ticks=vticks)
    cbar.ax.yaxis.tick_left()
    cbar.ax.set_ylabel('% Time above LTD thr.', size=lfs)
    cbar.ax.yaxis.set_label_position('left')
    set_axlw(cbar.ax, lw=alw)
    set_axfs(cbar.ax, fs=afs)

    # Fraction Ca2+ above \theta_p
    _, ax = figtem.generate_figure('1x1', figsize=(5.5,4.0), left=0.2, right=0.01, bottom=0.18, top=0.03)
    p = ax[0].pcolormesh(dt,dsic,thp*100., cmap='Greys', shading='gouraud')
    figtem.adjust_spines(ax[0], ['left','bottom'], position=spine_position)
    ax[0].set_xlim(xlims)
    ax[0].set_xticks(xticks)
    ax[0].set_xlabel('$\Delta t$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[0].set_ylim(ylims)
    ax[0].set_yticks(yticks)
    # ax[0].set_ylabel('SIC-pair Delay (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[0].set_ylabel(r'$\Delta \varsigma$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[0].yaxis.set_label_coords(-.22,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)
    # Add colorbar
    p.set_clim(vlims)
    cax = make_axes_locatable(ax[0]).append_axes('right', size='10%', pad=0.65)
    cbar = plt.colorbar(p, cax=cax, ticks=vticks)
    cbar.ax.set_ylabel('% Time above LTP thr.', size=lfs)
    cbar.ax.yaxis.tick_left()
    cbar.ax.yaxis.set_label_position('left')
    set_axlw(cbar.ax, lw=alw)
    set_axfs(cbar.ax, fs=afs)

    # W
    _, ax = figtem.generate_figure('1x1', figsize=(5.8,4.0), left=0.16, right=0.01, bottom=0.18, top=0.03)
    p = ax[0].pcolormesh(dt,dsic,dw, cmap='seismic', shading='gouraud')
    # Adjust axis
    figtem.adjust_spines(ax[0], ['left','bottom'], position=spine_position)
    ax[0].set_xlim(xlims)
    ax[0].set_xticks(xticks)
    ax[0].set_xlabel('$\Delta t$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[0].set_ylim(ylims)
    ax[0].set_yticks(yticks)
    # ax[0].set_ylabel('SIC-pair Delay (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[0].set_ylabel(r'$\Delta \varsigma$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[0].yaxis.set_label_coords(-.18,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    # Creates a new axis, cax, located 0.05 inches to the right of ax, whose width is 15% of ax
    # cax is used to plot a colorbar for each subplot
    vlims = (-50.,50)
    vticks = arange(-50.,50.1,25.)
    cax = make_axes_locatable(ax[0]).append_axes('right', size='10%', pad=0.95)
    p.set_clim(vlims)
    cbar = plt.colorbar(p, cax=cax, ticks=vticks)
    cbar.ax.set_ylabel('%$\Delta$ Syn. Strength', size=lfs)
    cbar.ax.yaxis.tick_left()
    cbar.ax.yaxis.set_label_position('left')
    set_axlw(cbar.ax, lw=alw)
    set_axfs(cbar.ax, fs=afs)

    # PNG format is necessary for proper handling
    label = ['ca_d','ca_p','w']
    for i,n in enumerate(plt.get_fignums()):
        figure(n)
        if i < 3:
            plt.savefig(fig_dir+'fig6_'+label[i]+'.'+'png', format='png', dpi=1200)

    # # ---------------------------------------------------------------------------------------------------------------
    # # Simulation /4,5 / Different SIC time constants
    # # ---------------------------------------------------------------------------------------------------------------
    lbl = ['stdp','ltd','ltp']
    filespec = 'stdp_sic_curves_timing_nonlin_tau_sic_all'
    ctau = [chex((255,128,14)),chex((0,107,164))]
    color0 = '0.5'

    # ----------------------------------
    # STDP curve
    # ----------------------------------
    # Define plot defaults
    colors = ['y']
    colors.extend(ctau)
    # Load data
    [cad, cap, dw, _, dt] = svu.loaddata(data_dir+filespec+'_'+lbl[0]+'.pkl')
    # Rescale data
    dt /= 1.e-3
    # Plot
    _, ax = figtem.generate_figure('2x1', figsize=(4.5,6.0), left=0.2, right=0.05, bottom=0.12, top=0.05, vs=[0.05])
    ax[1].plot((dt[0], dt[-1]), (0.0,0.0), c=color0, ls='-', lw=lw[0])
    ax[1].plot((0.0, 0.0), (-60,60), c=color0, ls='-', lw=lw[0])
    for i in range(shape(cad)[0]):
        ax[0].plot(dt, cad[i]*100., ls='-', color=colors[i], lw=lw[1])
        ax[0].plot(dt, cap[i]*100., ls='--', color=colors[i], lw=lw[1])
        ax[1].plot(dt, dw[i], ls='-', color=colors[i], lw=lw[1], marker='o')

    # Adjust axes
    xlims = (-100,100.1)
    xticks = arange(xlims[0],xlims[1],50)
    ylims = (0.0,6.0)
    yticks = arange(0.0,6.01,2.0)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].set_yticks(yticks)
    ax[0].set_ylabel('% Time above thr.', fontsize=lfs)
    ax[0].yaxis.set_label_coords(-.18,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    ylims = (-60,60.01)
    figtem.adjust_spines(ax[1], ['left', 'bottom'], position=spine_position)
    ax[1].set_xlim(xlims)
    ax[1].set_xticks(xticks)
    ax[1].set_xlabel('$\Delta t$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[1].set_ylim(ylims)
    ax[1].set_yticks(arange(-60,60.1,30.))
    ax[1].set_ylabel(r'%$\Delta$ Syn. Strength', fontsize=lfs)
    ax[1].yaxis.set_label_coords(-.18,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # Add legend / 1
    labels_legend = ['',
                     r'$\tau_s^r$ = 1.5$\cdot \bar{\tau}_s^{\,r}$',
                     r'$\tau_s$ = 1.5$\cdot \bar{\tau}_s$']
    obj_legend  = []
    for i,_ in enumerate(labels_legend):
        # Add artist for legend
        if i>-1: obj_legend.append(ax[1].plot([],[], color=colors[i], marker='o', lw=lw[1], label=labels_legend[i])[0])
    # Add Legend
    ax[1].legend(obj_legend[1:], labels_legend[1:], numpoints=1,
                 fontsize=legfs, frameon=False, loc=4)

    # ----------------------------------
    # LTD
    # ----------------------------------
    # Define plot defaults
    colors = [chex((23,190,207))]
    colors.extend(ctau)
    # Load data
    [cad, cap, dw, dt, _] = svu.loaddata(data_dir+filespec+'_'+lbl[1]+'.pkl')
    # Rescale data
    dt /= 1.e-3
    # Plot
    _, ax = figtem.generate_figure('2x1', figsize=(4.5,6.0), left=0.2, right=0.05, bottom=0.12, top=0.05, vs=[0.05])
    ax[1].plot((dt[0], dt[-1]), (0.0,0.0), c=color0, ls='-', lw=lw[0])
    ax[1].plot((0.0, 0.0), (-60,60), c=color0, ls='-', lw=lw[0])
    for i in range(shape(cad)[0]):
        ax[0].plot(dt, cad[i]*100., ls='-', color=colors[i], lw=lw[1])
        ax[0].plot(dt, cap[i]*100., ls='--', color=colors[i], lw=lw[1])
        ax[1].plot(dt, dw[i], ls='-', color=colors[i], lw=lw[1], marker='o')
    xlims = (-500,500.1)
    xticks = arange(xlims[0],xlims[1],250)
    ylims = (0.0,4.3)
    yticks = arange(0.0,4.3,2.0)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].set_yticks(yticks)
    ax[0].set_ylabel('% Time above thr.', fontsize=lfs)
    ax[0].yaxis.set_label_coords(-.18,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    ylims = (-35,3.0)
    figtem.adjust_spines(ax[1], ['left', 'bottom'], position=spine_position)
    ax[1].set_xlim(xlims)
    ax[1].set_xticks(xticks)
    ax[1].set_xlabel(r'$\Delta \varsigma$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[1].set_ylim(ylims)
    ax[1].set_yticks(arange(-30,3.,10.))
    ax[1].set_ylabel(r'%$\Delta$ Syn. Strength', fontsize=lfs)
    ax[1].yaxis.set_label_coords(-.18,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    # ----------------------------------
    # LTP
    # ----------------------------------
    # Define plot defaults
    colors = [chex((123,102,210))]
    colors.extend(ctau)
    # Load data
    [cad, cap, dw, dt, _] = svu.loaddata(data_dir+filespec+'_'+lbl[2]+'.pkl')
    # Rescale data
    dt /= 1.e-3
    # Plot
    _, ax = figtem.generate_figure('2x1', figsize=(4.5,6.0), left=0.2, right=0.05, bottom=0.12, top=0.05, vs=[0.05])
    ax[1].plot((dt[0], dt[-1]), (0.0,0.0), c=color0, ls='-', lw=lw[0])
    ax[1].plot((0.0, 0.0), (-60,60), c=color0, ls='-', lw=lw[0])
    for i in range(shape(cad)[0]):
        ax[0].plot(dt, cad[i]*100., ls='-', color=colors[i], lw=lw[1])
        ax[0].plot(dt, cap[i]*100., ls='--', color=colors[i], lw=lw[1])
        ax[1].plot(dt, dw[i], ls='-', color=colors[i], lw=lw[1], marker='o')

    # Adjust axes
    xlims = (-500,500.1)
    xticks = arange(xlims[0],xlims[1],250)
    ylims = (0.,4.3)
    yticks = arange(0.,4.3,2.0)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].set_yticks(yticks)
    ax[0].set_ylabel('% Time above thr.', fontsize=lfs)
    ax[0].yaxis.set_label_coords(-.18,0.5)
    # Adjust labels
    set_axlw(ax[0], lw=alw)
    set_axfs(ax[0], fs=afs)

    ylims = (-5,30.)
    figtem.adjust_spines(ax[1], ['left', 'bottom'], position=spine_position)
    ax[1].set_xlim(xlims)
    ax[1].set_xticks(xticks)
    ax[1].set_xlabel(r'$\Delta \varsigma$ (${0}$)'.format(sympy.latex(ms)), fontsize=lfs)
    ax[1].set_ylim(ylims)
    ax[1].set_yticks(arange(0.,30.01,10.))
    ax[1].set_ylabel(r'%$\Delta$ Syn. Strength', fontsize=lfs)
    ax[1].yaxis.set_label_coords(-.18,0.5)
    # Adjust labels
    set_axlw(ax[1], lw=alw)
    set_axfs(ax[1], fs=afs)

    for i,n in enumerate(plt.get_fignums()):
        figure(n)
        if i < 3:
            plt.savefig(fig_dir+'fig6_tau_sic_'+lbl[i]+'.'+format, format=format, dpi=600)

    plt.show()

def stdp_sic(sim=True, format='svg', fig_dir='../Figures/'):
# def stdp_sic(sim=False, format='svg', fig_dir='../Figures/'):
    stdp_sic_mechanism(sim=sim, format=format, fig_dir=fig_dir)
    stdp_sic_curves(sim=sim, format=format, fig_dir=fig_dir)

#input_folder = "/c/Users/tommy/OneDrive/Documents/GitHub/NeuralPlasticity2016/Figures"

# Set the output folder (where you want to save PNG files)
#output_folder = "/c/Users/tommy/OneDrive/Documents/GitHub/NeuralPlasticity2016/PNGFigures"

# Call the function to convert SVG to PNG
#svg_to_png(input_folder, output_folder)
if __name__ == '__main__':
   # simulate('io')
    #simulate('stp')
    #simulate('filter')
    simulate('sic')
    simulate('stdp_pre')
    simulate('stdp_sic')
    #svg_to_png(input_folder, output_folder)
   # print profiling_summary()




