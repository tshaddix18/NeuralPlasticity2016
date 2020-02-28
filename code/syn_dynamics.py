"""
syn_dynamics.py

Generate Figures for synaptic dynamics to be used in lecture slides.
"""
from brian2 import *
from brian2.units.allunits import mole, umole, mmole

import scipy.signal as sg

# Optional settings for faster compilations and execution by Brian
prefs.codegen.cpp.extra_compile_args = ['-Ofast', '-march=native']

# User-defined modules
import astrocyte_models as asmod
import analysis
import figures_template as figtem
import graupner_model as camod

import os, sys
sys.path.append(os.path.join(os.path.expanduser('~'), 'Dropbox/Ongoing.Projects/pycustommodules'))
import brian_utils as bu
import save_utils as svu
import general_utils as gu
from graphics_utils import plot_utils as pu
from graphics_utils import customplots as cpt

import geometry as geom

import numpy.random as random
import matplotlib
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline

def tm_model(sim=False, syn_type='depressing', format='svg', data_dir='../data/', fig_dir='./', num_spk=1):
    # Plotting defaults
    spine_position = 5
    lw = 1.5
    alw = 1.0
    afs = 14
    lfs = 16

    colors = {'u': pu.hexc((128,0,128)),
              'x': pu.hexc((255,126,14)),
              'r': pu.hexc((0,107,164))}

    if syn_type=='depressing':
        params = asmod.get_parameters('FM',synapse='depressing')
        # params['xi'] = 0
    else:
        params = asmod.get_parameters('FM',synapse='facilitating')
        # params['xi'] = 1

    if num_spk == 1:
        # single spike
        spikes = np.asarray([50.0])*msecond
    elif num_spk == 2:
        # two spikes
        spikes = np.asarray([50.0,75.])*msecond
    else:
        # three spikes
        spikes = np.asarray([50.0,75.,125.])*msecond


    if sim:
        # Simple TM synapse (sample dynamics)
        # Even if test_out will not be used, to properly create the model, it must be explicitly assigned and made available
        # w/in the local scope
        test_in, test_out, syn_test = asmod.synapse_model(params, N_syn=1, connect='i==j',
                                                          stimulus='test',
                                                          spikes = spikes,
                                                          name='testing_synapse*')

        # Create monitors
        S = StateMonitor(syn_test, variables=['u_S','x_S','r_S'], record=True,dt=0.1*msecond)

        duration = 0.15*second
        run(duration,namespace={},report='text')

        # Convert monitors to dictionaries for saving
        S_mon = bu.monitor_to_dict(S)
        svu.savedata([S_mon],data_dir+'tm_model'+'.pkl')

    # Effective plotting
    #-------------------------------------------------------------------------------------------------------------------
    # Synapse
    #-------------------------------------------------------------------------------------------------------------------
    S = svu.loaddata(data_dir+'tm_model'+'.pkl')[0]
    fig0, ax = figtem.generate_figure('4x1',figsize=(6.5,5.5),left=0.21,bottom=0.14,right=0.12,vs=[0.04])
    tlims = array([0,0.15])

    # Plot spikes
    analysis.plot_release(S['t'], S['r_S'][0], var='spk',  ax=ax[0], redraw=True)
    ax[0].set_xlim(tlims)

    # Plot u,x
    # Retrieve u,x traces
    u = analysis.build_uxsyn(S['t'], S['u_S'][0], 1.0/params['Omega_f'], 'u')
    x = analysis.build_uxsyn(S['t'], S['x_S'][0], 1.0/params['Omega_d'], 'x')
    ax[1].plot(S['t'],u,lw=lw,color=colors['u'])
    figtem.adjust_spines(ax[1], ['left'], position=spine_position)
    ax[1].set_xlim(tlims)
    # Format y-axis
    ax[1].set_ylim([-0.01,1.01])
    ax[1].set_yticks([0.0,0.5,1.0])
    # ax[1].set_ylabel('$u_S$', fontsize=lfs)
    ax[1].set_ylabel('$u_S$', fontsize=lfs, va='center')
    ax[1].yaxis.label.set_color(colors['u'])
    ax[1].tick_params(axis='y',color=colors['u'])
    ax[1].yaxis.set_label_coords(-.2,0.5)
    pu.set_axlw(ax[1], lw=alw)
    pu.set_axfs(ax[1], fs=afs)

    ax[2].plot(S['t'],x,lw=lw,color=colors['x'])
    figtem.adjust_spines(ax[2], ['left'], position=spine_position)
    ax[2].set_xlim(tlims)
    ax[2].set_ylim([-0.01,1.01])
    ax[2].set_yticks([0.0,0.5,1.0])
    ax[2].set_ylabel('$x_S$', fontsize=lfs)
    ax[2].yaxis.label.set_color(colors['x'])
    ax[2].tick_params(axis='y',color=colors['x'])
    ax[2].yaxis.set_label_coords(-.2,0.5)
    # Adjust labels
    pu.set_axlw(ax[2], lw=alw)
    pu.set_axfs(ax[2], fs=afs)

    # Plot released resources
    idx = np.in1d(S['t'],spikes/second)
    l, m, b = ax[3].stem(spikes/msecond, S['r_S'][0][1+np.where(idx)[0]], linefmt=colors['r'], markerfmt='', basefmt='')
    plt.setp(m, 'linewidth', lw)
    plt.setp(l, 'linestyle', 'none')
    plt.setp(b, 'linestyle', 'none')
    figtem.adjust_spines(ax[3], ['left','bottom'], position=spine_position)
    ax[3].set_xlim(tlims*1000)
    ax[3].set_ylim([-0.01,1.01])
    ax[3].set_yticks([0.0,0.5,1.0])
    ax[3].set_xlabel('Time (ms)',fontsize=lfs,ha='center')
    ax[3].set_ylabel('$r_S$', fontsize=lfs, va='center')
    ax[3].yaxis.set_label_coords(-.2,0.5)
    ax[3].yaxis.label.set_color(colors['r'])
    ax[3].tick_params(axis='y',color=colors['r'])
    pu.set_axlw(ax[3], lw=alw)
    pu.set_axfs(ax[3], fs=afs)

    plt.savefig(fig_dir+'tm_model_'+str(num_spk)+'.'+format, format=format, dpi=600)
    plt.show()


def tm_synapse_io(sim=False, syn_type='depressing', format='svg', dir='./'):
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

    spikes = concatenate((arange(50.0,250,50.0),arange(350.0,450,10.0)))*msecond

    if sim:
        # Simple TM synapse (sample dynamics)
        # Even if test_out will not be used, to properly create the model, it must be explicitly assigned and made available
        # w/in the local scope
        test_in, test_out, syn_test = asmod.synapse_model(params, N_syn=1, connect='i==j',
                                                          stimulus='test',
                                                          spikes = spikes,
                                                          name='testing_synapse*')

        # Create monitors
        S = StateMonitor(syn_test, variables=['u_S','x_S','r_S'], record=True)

        duration = 0.5*second
        run(duration,namespace={},report='text')

        # Convert monitors to dictionaries for saving
        S_mon = bu.monitor_to_dict(S)
        svu.savedata([S_mon],'tm_io_syn_'+syn_type[:3]+'.pkl')

    # Effective plotting
    #-------------------------------------------------------------------------------------------------------------------
    # Synapse
    #-------------------------------------------------------------------------------------------------------------------
    S = svu.loaddata('tm_io_syn_'+syn_type[:3]+'.pkl')[0]
    fig0, ax = figtem.generate_figure('3x1',figsize=(6.5,5.5),left=0.21,bottom=0.14,right=0.12)
    tlims = array([0,0.5])

    # Plot spikes
    analysis.plot_release(S['t'], S['r_S'][0], var='spk',  ax=ax[0], redraw=True)
    ax[0].set_xlim(tlims)

    # Plot u,x
    # Retrieve u,x traces
    colors = {'y1': pu.hexc((128,0,128)), 'y2': pu.hexc((255,126,14))}
    u = analysis.build_uxsyn(S['t'], S['u_S'][0], 1.0/params['Omega_f'], 'u')
    x = analysis.build_uxsyn(S['t'], S['x_S'][0], 1.0/params['Omega_d'], 'x')
    ax[1],ax_aux = cpt.plotyy(S['t'], u, x, ax=ax[1], colors=colors)

    # Format x-axis
    ax[1].set_xlim(tlims)
    ax[1].set_xlabel('')

    # Format y-axis
    ax[1].set_ylim([-0.01,1.01])
    ax[1].set_yticks([0.0,0.5,1.0])
    # ax[1].set_ylabel('$u_S$', fontsize=lfs)
    ax[1].set_ylabel('$u_S$', fontsize=lfs, va='center')
    ax[1].yaxis.set_label_coords(-.2,0.5)
    pu.set_axlw(ax[1], lw=alw)
    pu.set_axfs(ax[1], fs=afs)

    ax_aux.set_ylim([-0.01,1.01])
    ax_aux.set_yticks([0.0,0.5,1.0])
    # ax_aux.set_ylabel('$x_S$', fontsize=lfs)
    ax_aux.set_ylabel('$x_S$', fontsize=lfs)
    # Adjust labels
    pu.set_axlw(ax_aux, lw=alw)
    pu.set_axfs(ax_aux, fs=afs)

    # Plot released resources
    idx = np.in1d(S['t'],spikes/second)
    l, m, b = ax[2].stem(spikes/msecond, S['r_S'][0][1+np.where(idx)[0]], linefmt=pu.hexc((0,107,164)), markerfmt='', basefmt='')
    plt.setp(m, 'linewidth', lw)
    plt.setp(l, 'linestyle', 'none')
    plt.setp(b, 'linestyle', 'none')
    figtem.adjust_spines(ax[2], ['left','bottom'], position=spine_position)
    ax[2].set_xlim(tlims*1000)
    ax[2].set_ylim([-0.01,1.01])
    ax[2].set_yticks([0.0,0.5,1.0])
    ax[2].set_xlabel('Time (ms)',fontsize=lfs,ha='center')
    ax[2].set_ylabel('$r_S$', fontsize=lfs, va='center')
    ax[2].yaxis.set_label_coords(-.2,0.5)
    ax[2].yaxis.label.set_color(pu.hexc((0,107,164)))
    ax[2].tick_params(axis='y',colors=pu.hexc((0,107,164)))
    pu.set_axlw(ax[2], lw=alw)
    pu.set_axfs(ax[2], fs=afs)

    plt.savefig(dir+'tm_io_syn_'+syn_type[:3]+'.'+format, format=format, dpi=600)
    plt.show()

def stp_regulation(sim=False, format='eps', dir='./'):
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
    color = {'rd'  : pu.hexc((210,0,0)), # Reddish
             'ri'  : pu.hexc((0,128,0)), # DarkGreen
             'maroon'  : pu.hexc((128,0,0)),
             'yellow'  : pu.hexc((170,136,0)),
             'magenta' : pu.hexc((255,0,143)),
             'intermediate': pu.hexc((143,135,130))}

    # sim = True
    params = asmod.get_parameters(synapse='facilitating')
    params['U_0__star'] = 0.5
    alpha = [0.0,1.0]

    # Adjust parameters for the simulations
    #params['f_c'] = 0.2*Hz # 4 pulses
    params['f_c'] = 0.05*Hz # 4 pulses
    params['rho_e'] = 1.0e-4
    params['O_G'] = 0.6/umole/second
    params['Omega_G'] = 1.0/(30*second)
    params['t_off'] = 21*second

    # Add synaptic weight conversion to interface with postsynaptic LIF
    params['we'] = (60*0.27/10)*mV / (params['rho_c']*params['Y_T']) # excitatory synaptic weight (voltage)

    # General parameters for simulation also used in the analysis
    duration = 90*second

    # Stimulation (used in Figures too)
    offset = 0.2
    isi = 0.1
    stim = offset+arange(0.0,duration/second,2.0)
    spikes = sort(concatenate((stim, stim+isi)))*second

    if sim :
        # Synapses (2 with different xi-values)
        source_group,target_group,synapses = asmod.synapse_model(params, N_syn=2, connect='i==j',
                                                                 linear=False,
                                                                 post=None,
                                                                 stimulus=None)
                                                                 # stimulus='test',
                                                                 #spikes=spikes)
        synapses.alpha = alpha

        # Gliotransmitter release
        gliot = asmod.gliotransmitter_release(None, params, standalone=True, N_astro=1)
        gliot.f_c = params['f_c']

        # Connection to synapses
        ecs = asmod.extracellular_astro_to_syn(gliot, synapses, params, connect=True)

        # Set monitors
        pre = StateMonitor(synapses, ['Gamma_S'], record=True, dt=0.02*second)
        # post = StateMonitor(target_group, ['g'], record=True)
        glt = StateMonitor(gliot, ['G_A'], record=True, dt=0.02*second)

        # Run
        run(duration,namespace={},report='text')

        # Convert monitors to dictionaries and save them for analysis
        pre_mon = bu.monitor_to_dict(pre)
        # post_mon = bu.monitor_to_dict(post)
        glt_mon = bu.monitor_to_dict(glt)

        # Save data
        # svu.savedata([pre_mon,glt_mon],dir+'stp_regulation'+'.pkl')
        svu.savedata([pre_mon,glt_mon],dir+'stp_regulation_1'+'.pkl')

    #-------------------------------------------------------------------------------------------------------------------
    # Data Analysis
    #-------------------------------------------------------------------------------------------------------------------
    # Load data
    [pre,glt] = svu.loaddata('stp_regulation_1'+'.pkl')

    # Define some lambdas
    u0_sol = lambda gamma_S, u0_star, xi : (1.0-gamma_S)*u0_star + xi*gamma_S

    # Generate Figures
    fig1, ax = figtem.generate_figure('3x1',figsize=(6.0,5.5),left=0.18,bottom=0.15,right=0.05,top=0.08)
    tlims = np.array([0.0, duration/second])

    # Plot G_A (gliotransmitter release)
    cf = 1*umole / mole
    analysis.plot_release(glt['t'], glt['G_A'][0] / cf, var='y', ax=ax[0], color=color['maroon'], linewidth=lw)
    figtem.adjust_spines(ax[0], ['left'], position=spine_position)
    ax[0].set_xlim(tlims)
    ax[0].set_xlabel('')
    ax[0].set_ylim([-1.0,15])
    ax[0].set_yticks(arange(0.0,15.1,5))
    ax[0].set_ylabel('$G_A$ (${0}$)'.format(sympy.latex(umole)), fontsize=lfs)
    # ax[0].set_ylabel('Released Gt.\n(${0}$)'.format(sympy.latex(umole)), fontsize=lfs, multialignment='center')
    ax[0].yaxis.set_label_coords(-.13,0.5)
    ax[0].yaxis.label.set_color(color['maroon'])
    ax[0].tick_params(axis='y',color=color['maroon'])
    pu.set_axlw(ax[0], lw=alw)
    pu.set_axfs(ax[0], fs=afs)

    # Plot Gamma_S (presynaptic receptors)
    ax[1].plot(pre['t'], pre['Gamma_S'][0], color=color['yellow'], linewidth=lw)
    figtem.adjust_spines(ax[1], ['left'], position=spine_position)
    ax[1].set_xlim(tlims)
    ax[1].set_ylim([-0.01,1.01])
    ax[1].set_yticks([0.0,0.5,1.0])
    ax[1].set_ylabel('$\gamma_S$', fontsize=lfs)
    # ax[1].set_ylabel('Bound\nPresyn. Rec.', fontsize=lfs, multialignment='center')
    ax[1].yaxis.set_label_coords(-.13,0.5)
    ax[1].yaxis.label.set_color(color['yellow'])
    ax[1].tick_params(axis='y',color=color['yellow'])
    # Adjust labels
    pu.set_axlw(ax[1], lw=alw)
    pu.set_axfs(ax[1], fs=afs)

    # Plot U_0
    ax[2].plot(pre['t'], u0_sol(pre['Gamma_S'][0],params['U_0__star'],alpha[0]), color=color['rd'], linewidth=lw)
    ax[2].plot(pre['t'],u0_sol(pre['Gamma_S'][1],params['U_0__star'],alpha[1]),color=color['ri'],linewidth=lw)
    ax[2].plot(tlims, params['U_0__star']*ones((2,1)), ls='--', color=color['magenta'], linewidth=lw)
    figtem.adjust_spines(ax[2], ['left', 'bottom'], position=spine_position)
    ax[2].set_xlim(tlims)
    ax[2].set_xticks(arange(0.0,91.0,30.0))
    ax[2].set_xlabel('Time (${0}$)'.format(sympy.latex(second)), fontsize=lfs)
    ax[2].set_ylim([-0.01,1.01])
    ax[2].set_yticks([0.0,0.5,1.0])
    ax[2].set_ylabel('$U_0$', fontsize=lfs)
    # ax[0].set_ylabel('Synaptic\n Release Pr.', fontsize=lfs, multialignment='center')
    ax[2].yaxis.set_label_coords(-.13,0.5)
    ax[2].yaxis.label.set_color(color['magenta'])
    ax[2].tick_params(axis='y',color=color['magenta'])
    # Adjust labels
    pu.set_axlw(ax[2], lw=alw)
    pu.set_axfs(ax[2], fs=afs)

    # Save Figures
    plt.figure(1)
    # plt.savefig(dir+'glt_release_4.'+format, format=format, dpi=600)
    plt.savefig(dir+'glt_release_1.'+format, format=format, dpi=600)
    # plt.close(fig0)

    plt.show()

def sic_mechanism(sim=False, format='eps', dir='./'):
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
        svu.savedata([post_mon,glt_mon],'sic_mechanism.pkl')

    # Plot SIC mechanism
    [post, glt] = svu.loaddata('sic_mechanism.pkl')

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
    pu.set_axlw(ax[0], lw=alw)
    pu.set_axfs(ax[0], fs=afs)

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
    pu.set_axlw(ax[1], lw=alw)
    pu.set_axfs(ax[1], fs=afs)

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
    pu.set_axlw(ax[2], lw=alw)
    pu.set_axfs(ax[2], fs=afs)

    # Save Figures
    figure(plt.get_fignums()[-1])
    # plt.savefig(dir+'sic_mechanism_'+str(num_gre)+'.'+format, format=format, dpi=600)
    # plt.close('all')

    plt.show()

if __name__ == '__main__':
    # Explains TM synaptic model
    # tm_model(sim=True,syn_type='depressing',format='svg',dir='./',num_spk=1)
    tm_model(sim=True,syn_type='depressing',format='svg',dir='./',num_spk=2)
    # tm_model(sim=False,syn_type='depressing',format='svg',dir='./')

    # # Produce TM dep vs/ fac cases
    # tm_synapse_io(sim=True,syn_type='depressing')
    # tm_synapse_io(sim=True,syn_type='facilitating')
    # # tm_synapse_io(sim=False,syn_type='depressing')
    # # tm_synapse_io(sim=False,syn_type='facilitating')

    # # Gliotransmitter release
    # stp_regulation(sim=True,format='svg',dir='./')

    # SIC mechanism
    # sic_mechanism(sim=True, format='svg', dir='./')