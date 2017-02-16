"""
figures_template.py

Provides axis grids and figure templates to build complex figures for publication.
"""

from numpy import *
from scipy import *
import sys
# sys.path.append(r'/home/maurizio/Documents/PyCustomModules')

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#-----------------------------------------------------------------------------------------------------------------------
# Utility methods
#-----------------------------------------------------------------------------------------------------------------------
def varargin(pars,**kwargs):
    """
    varargin-like option for user-defined parameters in any function/module
    Use:
    pars = varargin(pars,**kwargs)

    Input:
    - pars     : the dictionary of parameters of the calling function
    - **kwargs : a dictionary of user-defined parameters

    Output:
    - pars     : modified dictionary of parameters to be used inside the calling
                 (parent) function
    """
    for key,val in kwargs.iteritems():
        if key in pars:
            pars[key] = val
    return pars

#-----------------------------------------------------------------------------------------------------------------------
# Plot methods
#-----------------------------------------------------------------------------------------------------------------------
def fignum():
    '''
    Assign figure number

    Return :
    new figure number : Integer
    '''
    if not size(plt.get_fignums()):
        return 1
    else:
        return plt.get_fignums()[-1]+1

def adjust_spines(ax, spines, position=0, smart_bounds=False):
    """
    Set custom visibility and position of axes

    ax       : Axes
     Axes handle
    spines   : List
     String list of 'left', 'bottom', 'right', 'top' spines to show
    position : Integer
     Number of points for position of axis
    """
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', position))  # outward by 10 points
            spine.set_smart_bounds(smart_bounds)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    elif 'top' in spines:
        ax.xaxis.set_ticks_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def adjust_spines_uneven(ax, va='left', ha='bottom', position=5):
    '''
    Basic adjustment of spines: one spine on horizontal (v) {'left'} | 'right' on outer position and another one on
    horizontal position at standard location {'bottom} | 'top

    Inputs:
    - ax   :  Axes object
    - va   :  vertical axis {'left'} | 'right'
    - ha   :  horizontal axis {'bottom'} | 'top'
    - position : outer position in points of va
    '''
    make_patch_spines_invisible(ax)
    ax.patch.set_visible(True)
    # Vertical axis
    ax.spines[va].set_visible(True)
    ax.spines[va].set_position(('outward', position))  # outward by 5 points
    ax.yaxis.set_ticks_position(va)
    # Horizontal axis
    ax.spines[ha].set_visible(True)
    ax.xaxis.set_ticks_position(ha)

def make_patch_spines_invisible(ax):
    '''
    Used in the creation of a twin axis. Activate the twin frame but make the patch and spines invisible

    ax : twin axis handle
    '''
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.itervalues():
        sp.set_visible(False)

def freeze_canvas(opts, AxisWidth, AxisHeight):
    '''
    Resize axis to ref_size (keeping left and bottom margins fixed)
    Useful to plot figures in a larger or smaller figure box but with canvas of the same canvas

    Inputs :

    opts : Dictionary
     Options for figure plotting. Must contain keywords: 'left', 'bottom', 'hs', 'vs', 'figsize', 'axref_size'.
    AxisWidth : Value / List
    AxisHeight: Value / List

    Return:
    opts, AxisWidth, AxisHeight
    '''
    # Basic function to proper resize
    resize = lambda obj, dx : [val+dx*val for val in obj] if type(obj)==type(list()) else obj+dx*obj
    if opts['axref_size'] != None :
        # Compute size differences
        dw = (opts['axref_size'][0]-opts['figsize'][0])/opts['figsize'][0]
        dh = (opts['axref_size'][1]-opts['figsize'][1])/opts['figsize'][1]
        for k,v in opts.iteritems():
            if k=='left' or k=='hs' : opts[k] = resize(v,dw)
            if k=='bottom' or k=='vs' : opts[k] = resize(v,dh)
        AxisWidth = resize(AxisWidth,dw)
        AxisHeight = resize(AxisHeight,dh)
    return opts, AxisWidth, AxisHeight

#-----------------------------------------------------------------------------------------------------------------------
# Figure templates
#-----------------------------------------------------------------------------------------------------------------------
def generate_figure(template, **kwargs):
    # Basic interface to select different figure templates
    if template=='1x1':
        return figure_1x1(**kwargs)
    if template=='2x1':
        return figure_2x1(**kwargs)
    if template=='2x1_custom':
        return figure_2x1_custom(**kwargs)
    if template=='2x1_1L2S':
        return figure_2x1_1L2S(**kwargs)
    if template=='3x1':
        return figure_3x1(**kwargs)
    if template=='4x1':
        return figure_4x1(**kwargs)
    if template=='4x1_1L3S':
        return figure_4x1_1L3S(**kwargs)
    if template=='5x1_4S1L':
        return figure_5x1_4S1L(**kwargs)
    # Horizontally-developed figures
    if template=='1x3':
        return figure_1x3(**kwargs)

def figure_1x1(**kwargs):
    '''
    figure_1x1(**kwargs) : Single figure with option of freeze_canvas

    Input:
    - left,right,top,bottom
    - vs      : [val] vertical space between plots
    - figsize : (width,height) 2x1 tuple for figure size

    Output:
    - fig     : fig handler
    - ax      : a list of ax handles (top-to-bottom)

    '''

    opts = {'left'   : 0.1,
            'right'  : 0.05,
            'bottom' : 0.1,
            'top'    : 0.02,
            'figsize': (6.5,5.5),
            'axref_size': None}

    # User-defined parameters
    opts = varargin(opts,**kwargs)

    # Axis specification
    AxisWidth = 1.0-opts['left']-opts['right']
    AxisHeight = 1.0-opts['bottom']-opts['top']

    # Optional resizing
    opts, AxisWidth, AxisHeight = freeze_canvas(opts, AxisWidth, AxisHeight)

    # Axis Grid
    # x position
    xpos = opts['left']
    # y position
    ypos = opts['bottom']

    # Axis boxes (bottom-up)
    axBoxes = list()
    axBoxes.append([xpos,ypos,AxisWidth,AxisHeight])

    fig = plt.figure(fignum(), figsize=opts['figsize'])

    ax = list()
    for i in xrange(shape(axBoxes)[0]-1,-1,-1):
        ax_aux = fig.add_axes(axBoxes[i])
        ax.append(ax_aux)

    return fig, ax

def figure_2x1(**kwargs):
    '''
    figure_2x1(**kwargs) : Create grid plot for a 1-column complex figure composed of (top-to-bottom):
                         2 equal plots | custom space

    Input:
    - left,right,top,bottom
    - vs      : [val] vertical space between plots
    - figsize : (width,height) 2x1 tuple for figure size

    Output:
    - fig     : fig handler
    - ax      : a list of ax handles (top-to-bottom)

    '''

    opts = {'left'   : 0.1,
            'right'  : 0.05,
            'bottom' : 0.1,
            'top'    : 0.02,
            'vs'     : [0.03],
            'figsize': (6.5,5.5),
            'axref_size': None}

    # User-defined parameters
    opts = varargin(opts,**kwargs)

    ncol = 1
    nrow = 2

    # Axis specification
    AxisWidth = 1.0-opts['left']-opts['right']
    AxisHeight = [(1.0-opts['bottom']-opts['top']-opts['vs'][0])/nrow]

    # Optional resizing
    opts, AxisWidth, AxisHeight = freeze_canvas(opts, AxisWidth, AxisHeight)

    # Axis Grid
    # x position
    xpos = opts['left']
    # y position
    ypos = list()
    ypos.append(opts['bottom'])
    ypos.append(ypos[0]+AxisHeight[0]+opts['vs'][0])

    # Axis boxes (bottom-up)
    axBoxes = list()
    axBoxes.append([xpos,ypos[0],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[1],AxisWidth,AxisHeight[0]])

    fig = plt.figure(fignum(), figsize=opts['figsize'])

    ax = list()
    for i in xrange(shape(axBoxes)[0]-1,-1,-1):
        ax_aux = fig.add_axes(axBoxes[i])
        ax.append(ax_aux)

    return fig, ax

def figure_2x1_custom(**kwargs):
    '''
    figure_2x1(**kwargs) : Create grid plot for a 1-column complex figure composed of (top-to-bottom):
                         2 equal plots | custom space

    Input:
    - left,right,top,bottom
    - vs      : [val] vertical space between plots
    - figsize : (width,height) 2x1 tuple for figure size

    Output:
    - fig     : fig handler
    - ax      : a list of ax handles (top-to-bottom)

    '''

    opts = {'left'   : 0.1,
            'right'  : 0.05,
            'bottom' : 0.1,
            'top'    : 0.02,
            'vs'     : [0.03],
            'vratio' : 0.25,
            'figsize': (6.5,5.5),
            'axref_size': None}

    # User-defined parameters
    opts = varargin(opts,**kwargs)

    ncol = 1
    nrow = 2

    # Axis specification
    AxisWidth = 1.0-opts['left']-opts['right']
    AxisHeight = [0,0]
    AxisHeight[0] = (1.0-opts['bottom']-opts['top']-opts['vs'][0])*opts['vratio']
    AxisHeight[1] = (1-opts['vratio'])*AxisHeight[0]/opts['vratio']

    # Optional resizing
    opts, AxisWidth, AxisHeight = freeze_canvas(opts, AxisWidth, AxisHeight)

    # Axis Grid
    # x position
    xpos = opts['left']
    # y position
    ypos = list()
    ypos.append(opts['bottom'])
    ypos.append(ypos[0]+AxisHeight[1]+opts['vs'][0])


    # Axis boxes (bottom-up)
    axBoxes = list()
    axBoxes.append([xpos,ypos[0],AxisWidth,AxisHeight[1]])
    axBoxes.append([xpos,ypos[1],AxisWidth,AxisHeight[0]])

    fig = plt.figure(fignum(), figsize=opts['figsize'])

    ax = list()
    for i in xrange(shape(axBoxes)[0]-1,-1,-1):
        ax_aux = fig.add_axes(axBoxes[i])
        ax.append(ax_aux)

    return fig, ax

def figure_2x1_1L2S(**kwargs):
    '''
    figure_2x1(**kwargs) : Create grid plot for a 1-column complex figure composed of (top-to-bottom):
                         1 long plots
                         1 small plot | custom space | small/medium plot

    Input:
    - left,right,top,bottom
    - vs      : [val] vertical space between plots
    - hs      : [val] horizontal space between plots on the second row
    - ratio   : width ratio for the 2 subplots in the same row
    - figsize : (width,height) 2x1 tuple for figure size
    - axref_size : reference for axis freezing

    Output:
    - fig     : fig handler
    - ax      : a list of ax handles (top-to-bottom)

    '''

    opts = {'left'   : 0.1,
            'right'  : 0.05,
            'bottom' : 0.1,
            'top'    : 0.02,
            'vs'     : [0.03],
            'hs'     : [0.1],
            'figsize': (6.5,5.5),
            'axref_size': None}

    # User-defined parameters
    opts = varargin(opts,**kwargs)

    ncol = 1
    nrow = 2

    # Axis specification
    AxisWidth = [0,0,0]
    AxisWidth[0] = 1.0-opts['left']-opts['right']
    AxisWidth[1] = (AxisWidth[0]-opts['hs'][0])/3.0
    AxisWidth[2] = 2*AxisWidth[1]
    AxisHeight = [(1.0-opts['bottom']-opts['top']-opts['vs'][0])/nrow]

    # Optional resizing
    opts, AxisWidth, AxisHeight = freeze_canvas(opts, AxisWidth, AxisHeight)

    # Axis Grid
    # x position
    xpos = list()
    xpos.append(opts['left'])
    xpos.append(xpos[0]+AxisWidth[1]+opts['hs'][0])
    # y position
    ypos = list()
    ypos.append(opts['bottom'])
    ypos.append(ypos[0]+AxisHeight[0]+opts['vs'][0])


    # Axis boxes (bottom-up)
    axBoxes = list()
    axBoxes.append([xpos[0],ypos[0],AxisWidth[1],AxisHeight[0]])
    axBoxes.append([xpos[1],ypos[0],AxisWidth[2],AxisHeight[0]])
    axBoxes.append([xpos[0],ypos[1],AxisWidth[0],AxisHeight[0]])

    fig = plt.figure(fignum(), figsize=opts['figsize'])

    ax = list()
    for i in xrange(shape(axBoxes)[0]-1,-1,-1):
        ax_aux = fig.add_axes(axBoxes[i])
        ax.append(ax_aux)

    return fig, ax

def figure_3x1(**kwargs):
    '''
    figure_3x1(**kwargs) : Create grid plot for a 1-column complex figure composed of (top-to-bottom):
                         3 equal plots | custom space

    Input:
    - left,right,top,bottom
    - vs      : [val] vertical space between plots
    - figsize : (width,height) 2x1 tuple for figure size

    Output:
    - fig     : fig handler
    - ax      : a list of ax handles (top-to-bottom)

    '''

    opts = {'left'   : 0.1,
            'right'  : 0.05,
            'bottom' : 0.1,
            'top'    : 0.02,
            'vs'     : [0.03],
            'figsize': (6.5,5.5),
            'axref_size': None}

    # User-defined parameters
    opts = varargin(opts,**kwargs)

    ncol = 1
    nrow = 3

    # Axis specification
    AxisWidth = 1.0-opts['left']-opts['right']
    AxisHeight = [(1.0-opts['bottom']-opts['top']-2*opts['vs'][0])/nrow]

    # Optional resizing
    opts, AxisWidth, AxisHeight = freeze_canvas(opts, AxisWidth, AxisHeight)

    # Axis Grid
    # x position
    xpos = opts['left']
    # y position
    ypos = list()
    ypos.append(opts['bottom'])
    ypos.append(ypos[0]+AxisHeight[0]+opts['vs'][0])
    ypos.append(ypos[1]+AxisHeight[0]+opts['vs'][0])

    # Axis boxes (bottom-up)
    axBoxes = list()
    axBoxes.append([xpos,ypos[0],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[1],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[2],AxisWidth,AxisHeight[0]])

    fig = plt.figure(fignum(), figsize=opts['figsize'])

    ax = list()
    for i in xrange(shape(axBoxes)[0]-1,-1,-1):
        ax_aux = fig.add_axes(axBoxes[i])
        ax.append(ax_aux)

    return fig, ax

def figure_4x1(**kwargs):
    '''
    figure_4x1(**kwargs) : Create grid plot for a 1-column complex figure composed of (top-to-bottom):
                         4 equal plots | custom space

    Input:
    - left,right,top,bottom
    - vs      : [val] vertical space between plots
    - figsize : (width,height) 2x1 tuple for figure size

    Output:
    - fig     : fig handler
    - ax      : a list of ax handles (top-to-bottom)

    '''

    opts = {'left'   : 0.1,
            'right'  : 0.05,
            'bottom' : 0.08,
            'top'    : 0.02,
            'vs'     : [0.02],
            'figsize': (5.5,8.0)}

    # User-defined parameters
    opts = varargin(opts,**kwargs)

    ncol = 1
    nrow = 4

    # Axis specification
    AxisWidth = 1.0-opts['left']-opts['right']
    AxisHeight = [(1.0-opts['bottom']-opts['top']-3*opts['vs'][0])/nrow]

    # Axis Grid
    # x position
    xpos = opts['left']
    # y position
    ypos = list()
    ypos.append(opts['bottom'])
    ypos.append(ypos[0]+AxisHeight[0]+opts['vs'][0])
    ypos.append(ypos[1]+AxisHeight[0]+opts['vs'][0])
    ypos.append(ypos[2]+AxisHeight[0]+opts['vs'][0])

    # Axis boxes (bottom-up)
    axBoxes = list()
    axBoxes.append([xpos,ypos[0],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[1],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[2],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[3],AxisWidth,AxisHeight[0]])

    fig = plt.figure(fignum(), figsize=opts['figsize'])

    ax = list()
    for i in xrange(shape(axBoxes)[0]-1,-1,-1):
        ax_aux = fig.add_axes(axBoxes[i])
        ax.append(ax_aux)

    return fig, ax

def figure_4x1_1L3S(**kwargs):
    '''
    figure_4x1(**kwargs) : Create grid plot for a 1-column complex figure composed of (top-to-bottom):
                         2 small plots (for 1/3 of Height) | custom space
                         1 large plot (for 2/3 of 2/3 of Height) | custom space
                         1 small plot (for 1/3 of 2/3 of height)

    Input:
    - left,right,top,bottom
    - vs      : [val] vertical space between plots
    - figsize : (width,height) 2x1 tuple for figure size

    Output:
    - fig     : fig handler
    - ax      : a list of ax handles (top-to-bottom)

    '''

    opts = {'left'   : 0.1,
            'right'  : 0.05,
            'bottom' : 0.08,
            'top'    : 0.02,
            'vs'     : [0.04,0.03,0.04],
            'figsize': (5.0,7.0),
            'axref_size': None}

    # User-defined parameters
    opts = varargin(opts,**kwargs)

    ncol = 1
    nrow = [3, 3]


    # Axis specification
    AxisWidth = 1.0-opts['left']-opts['right']
    AxisHeight = [0,0,0]
    AxisHeight[0] = (1.0-opts['bottom']-opts['top']-sum(opts['vs']))/nrow[0]/2.0 # Small top plots
    AxisHeight[2] = 2.0*(2.0*AxisHeight[0])/nrow[1] # Small bottom plot
    AxisHeight[1] = (nrow[1]-1)*AxisHeight[2]         # Large central plot

    # Optional resizing
    opts, AxisWidth, AxisHeight = freeze_canvas(opts, AxisWidth, AxisHeight)

    # Axis Grid
    # x position
    xpos = opts['left']
    # y position
    ypos = list()
    ypos.append(opts['bottom'])
    ypos.append(ypos[0]+AxisHeight[2]+opts['vs'][2])
    ypos.append(ypos[1]+AxisHeight[1]+opts['vs'][1])
    ypos.append(ypos[2]+AxisHeight[0]+opts['vs'][0])

    # Axis boxes (bottom-up)
    axBoxes = list()
    axBoxes.append([xpos,ypos[0],AxisWidth,AxisHeight[2]])
    axBoxes.append([xpos,ypos[1],AxisWidth,AxisHeight[1]])
    axBoxes.append([xpos,ypos[2],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[3],AxisWidth,AxisHeight[0]])

    fig = plt.figure(fignum(), figsize=opts['figsize'])

    ax = list()
    for i in xrange(shape(axBoxes)[0]-1,-1,-1):
        ax_aux = fig.add_axes(axBoxes[i])
        ax.append(ax_aux)

    return fig, ax

def figure_5x1_4S1L(**kwargs):
    '''
    figure_5x1_4S1(**kwargs) : Create grid plot for a 1-column complex figure composed of (top-to-bottom):
                               1 small plot | medium space |
                               2 small plots| medium space |
                               1 small plot | large space  |
                               1 large plot

    Input:
    - left,right,top,bottom
    - vs  : [Small,Medium,Large] 3x1 list of Vertical space between plots
    - figsize : (width,height) 2x1 tuple for figure size

    Output:
    - fig     : fig handler
    - ax      : a list of ax handles (top-to-bottom)

    '''

    opts = {'left'   : 0.1,
            'right'  : 0.05,
            'bottom' : 0.08,
            'top'    : 0.02,
            'vs'     : [0.015,0.025,0.10],
            'figsize': (5.5,7.5)}

    # User-defined parameters
    opts = varargin(opts,**kwargs)

    ncol = 1
    nrow = 3

    # Axis specification
    AxisWidth = 1.0-opts['left']-opts['right']
    AxisHeight = (1.0-opts['bottom']-opts['top']-opts['vs'][0]-2*opts['vs'][1]-opts['vs'][2])/nrow
    AxisHeight = [AxisHeight/2,AxisHeight]

    # Axis Grid
    # x position
    xpos = opts['left']
    # y position
    ypos = list()
    ypos.append(opts['bottom'])
    ypos.append(ypos[0]+AxisHeight[1]+opts['vs'][2])
    ypos.append(ypos[1]+AxisHeight[0]+opts['vs'][0])
    ypos.append(ypos[2]+AxisHeight[0]+opts['vs'][0])
    ypos.append(ypos[3]+AxisHeight[0]+opts['vs'][1])

    # Axis boxes (bottom-up)
    axBoxes = list()
    axBoxes.append([xpos,ypos[0],AxisWidth,AxisHeight[1]])
    axBoxes.append([xpos,ypos[1],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[2],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[3],AxisWidth,AxisHeight[0]])
    axBoxes.append([xpos,ypos[4],AxisWidth,AxisHeight[0]])

    fig = plt.figure(fignum(), figsize=opts['figsize'])

    ax = list()
    for i in xrange(shape(axBoxes)[0]-1,-1,-1):
        ax_aux = fig.add_axes(axBoxes[i])
        ax.append(ax_aux)

    return fig, ax

#-----------------------------------------------------------------------------------------------------------------------
# Horizontally-developing figures
#-----------------------------------------------------------------------------------------------------------------------
def figure_1x3(**kwargs):
    '''
    figure_1x3(**kwargs) : Create grid plot for a 1-row complex figure composed of (left-to-right):
                         3 equal plots | custom space

    Input:
    - left,right,top,bottom
    - hs      : [val] horizontal space between plots
    - figsize : (width,height) 2x1 tuple for figure size

    Output:
    - fig     : fig handler
    - ax      : a list of ax handles (top-to-bottom)

    '''

    opts = {'left'   : 0.1,
            'right'  : 0.03,
            'bottom' : 0.1,
            'top'    : 0.1,
            'hs'     : [0.1],
            'figsize': (15,5.0),
            'axref_size': None}

    # User-defined parameters
    opts = varargin(opts,**kwargs)

    ncol = 3
    nrow = 1

    # Axis specification
    AxisWidth = [(1.0-opts['left']-opts['right']-2*opts['hs'][0])/ncol]
    AxisHeight = [(1.0-opts['bottom']-opts['top'])/nrow]

    # Optional resizing
    opts, AxisWidth, AxisHeight = freeze_canvas(opts, AxisWidth, AxisHeight)

    # Axis Grid
    # x position
    xpos = list()
    xpos.append(opts['left'])
    xpos.append(xpos[0]+AxisWidth[0]+opts['hs'][0])
    xpos.append(xpos[1]+AxisWidth[0]+opts['hs'][0])

    # y position
    ypos = opts['bottom']

    # Axis boxes (left-right)
    axBoxes = list()
    axBoxes.append([xpos[0],ypos,AxisWidth[0],AxisHeight[0]])
    axBoxes.append([xpos[1],ypos,AxisWidth[0],AxisHeight[0]])
    axBoxes.append([xpos[2],ypos,AxisWidth[0],AxisHeight[0]])

    fig = plt.figure(fignum(), figsize=opts['figsize'])

    ax = list()
    for i in xrange(shape(axBoxes)[0]):
        ax_aux = fig.add_axes(axBoxes[i])
        ax.append(ax_aux)

    return fig, ax

# Temporary testing (to comment at final stage)
if __name__ == "__main__":
    # fig, ax = figure_2x1()
    fig, ax = figure_2x1_custom()
    # figure_2x1_1L2S()
    # fig, ax = figure_3x1()
    # fig, ax = figure_4x1()
    # figure_4x1_1L3S()
    # fig,ax = figure_5x1_4S1L()
    #--------------------------------
    # figure_1x3()
    plt.show()
    # print ax