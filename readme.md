# Code for MPC Algorithm

This repository contains a reference implementation of the algorithms for the paper:

Hongchao Qin, Rong-Hua Li, Guoren Wang, Lu Qin, Yurong Cheng, Ye Yuan. Mining Periodic Cliques in Temporal Networks. ICDE 2019: 1130-1141

## Environment Setup

Codes run on Python 2.7 or later. [PyPy](http://pypy.org/) compiler is recommended because it can make the computations quicker without change the codes.

You may use Git to clone the repository from
GitHub and run it manually like this:

    git clone https://github.com/VeryLargeGraph/MPC.git
    cd MPC
    pip install click
    python run.py  

## Dateset description
We focus on mining the temporal network so each edge is associated with a timestamp. Temporal edges are stored at the raw data in which each line is one temporal edge.
 
| from_id | \t  | to_id    | \t  |  timestamps  |
| :----:  |:----: | :----:   |:----:   | :----: |

Note that, the function _readGraph_ in _mpc.py_ can make each snapshot G_i (see Fig 1.c in the paper) to be a simple graph.  

## Running example
You can type in dataset name, parameters theta, k and method name to control the program:

    Dataset name(str): chess_year
    theta(int): 3
    k(int): 3
    Type one number to chose the algorithm: [1]MPCKC; [2]MPCWC; [3]MPCSC. (int): 1
    loading...
    number of nodes:7301
    number of edges:55899
    number of temporal edges:62385
    kCore time:0:00:00.013000
    kCore #nodes:5472
    #new nodes:8416
    #new edges:681
    #mpc:3
    All time:0:00:00.199000
    New nodes stored in file "NEWNODES.json" : {new_node_id: [raw_node_id, [starttime, interval]]}
    if theta = 4, one node [1, [2003, 1]] means node id 1 in raw data is periodic at 2003, 2004, 2005, 2006
    if theta = 3, one node [2, [2003, 2]] means node id 2 in raw data is periodic at 2003, 2005, 2007
    MPCliques (New nodes):
    [set([302, 263, 742, 762]), set([1840, 1848, 3448, 3520]), set([3872, 4151, 4132, 7479])]

