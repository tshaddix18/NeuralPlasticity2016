'''
brian_utils.py

Contains methods to handle Brian2 data
'''

import numpy as np
from scipy import *

# Import Brian units
from brian2.units import *

# import os, sys
# sys.path.append(os.path.join(os.path.expanduser('~'), 'Dropbox/Ongoing.Projects/pycustommodules'))
# import decorator_utils as dcu

def show_spikes(v_signal, v_reset, v_peak):
    '''
    Add spikes to v_signal, given reset value (v_reset). Each spike is of v_peak value.
    '''
    for i in range(np.shape(v_signal)[0]):
        indices = np.where((v_signal[i]==v_reset) & (np.insert(np.diff(v_signal[i]),0,[0]) < 0))[0]-1
        v_signal[i][indices] = v_peak
    return v_signal


def monitor_to_dict(monitor, monitor_type='state', var_units={}, fields=['spk'],**kwargs):
    '''
    Convert Brian monitor to dictionary

    Inputs:
    - monitor      : Brian monitor
    - monitor_type : {'state'} (for StateMonitor) | 'spike' (for SpikeMonitor) | 'poprate' (for PopulationRateMonitor)
    - var_units    : Dictionary (for StateMonitor)
     Dictionary with entries {'var_name' : units, ...} that will be used to save the variable in the desired scaled values
    - fields       : List of Strings (for SpikeMonitor)
      {'spk'} : retrieve spike_trains as a list of arrays
      'count' : spike count

    Return
    - mon : dictionary with 't' and recorded_variables (w/out units)
    '''
    mon = {}
    if monitor_type=='state':
        # Add time
        mon['t'] = np.asarray(monitor.t_[:])
        for key in list(monitor.recorded_variables.keys()):
            if key in list(var_units.keys()):
                mon[key] = getattr(monitor, key)/var_units[key]
            else:
                mon[key] = getattr(monitor, key+'_')
    elif monitor_type=='spike':
        for k in fields:
            if k=='count':
                mon[k] = monitor.count[:]
            if k=='spk':
                mon['t'] = np.asarray(monitor.t[:]/second)
                mon['i'] = monitor.i[:]
    elif monitor_type=='poprate':
        mon['t'] = np.asarray(monitor.t_[:])
        mon['rate'] = monitor.smooth_rate(**kwargs)/Hz
    return mon