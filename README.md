# NeuralPlasticity2016
This repository contains software used to generate figures by De Pitta' & Brunel, Neural Plasticity 2016, Article ID: 7607924 (DOI: 10.1155/2016/7607924).

The software uses the python-based simulator [Brian 2.x][1].

To build figures, enter the `code` folder and launch from command line

`cd code && python3 tripatite_figures.py`

Data and figures produced by the simulations in are respectively saved in the `data` and `Figures` folders.

The actual `Brian 2` methods used to model tripartite synapses are in `astocyte_models.py`

Question and inquiries: maurizio.depitta --at-- gmail.com.

# Requirements
- [Brian 2.x][1] simulator 
- dill
- sympy

# Versions tracking
v2.0
Maurizio De Pitta', Basque Cente of Applied Mathematics, Bilbao, Spain, Feb 27, 2020

v1.0
Maurizio De Pitta', The University of Chicago, 25 March 2016.

[1] https://brian2.readthedocs.io/en/stable/
