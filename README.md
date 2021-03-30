# Source code for TEPX Beam Induced Background (BIB) timing studies.

The workflow uses data simulated with CMSSW version 11.2.0.pre6. <br>
Simulation outputs are saved in hdf5 files and no ROOT files are involved in the process. <br>
The hdf5 files have the following data structure: <br>

- group: Tof_q <br>
- datasets: dXrY, where X is the TEPX disc ID ((-4)-4) and Y is the TEPX ring ID (1-5) <br>

The hdf5 files together with the standard ROOT simulation outputs are located on lxplus EOS at: <br>
`/eos/cms/store/group/dpg_bril/comm_bril/phase2-sim`

BIB simulation outputs are located at: <br>
`/eos/cms/store/group/dpg_bril/comm_bril/phase2-sim/bib_simulations_fullgeo/hdf5`

The directory `full_stat_simhit` contains the main simulation results with the following number of events: <br>

- beam halo: 500.000 (500.000)
- beam gas carbon: 165.000 (200.000)
- beam gas oxygen: 190.000 (200.000)
- beam gas hydrogen: 185.000 (195.000)

The number of events launched are indicated in brackets. <br>
Pileup events are simulated in a similar way and the corresponding hdf5 outputs are located at:
`/eos/cms/store/group/dpg_bril/comm_bril/phase2-sim/pu_simulations_fullgeo/hdf5`
