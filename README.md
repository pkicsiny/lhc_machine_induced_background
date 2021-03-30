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

Pileup events are simulated in a similar way and the corresponding hdf5 outputs are located at:
`/eos/cms/store/group/dpg_bril/comm_bril/phase2-sim/pu_simulations_fullgeo/hdf5`

The directory `full_stat_simhit` contains the main simulation results with the following number of events: <br>

- PU 200: 930 (1000)

The number of events launched are indicated in brackets. Simulations are always split into multiple jobs (10 events/job for PU and 5000 events for BIB) and submitted to lxbatch. <br>

For BIB the setting beam 1 was used. This means that the BIB particles come from the +Z side from the interface plane between the LHC long straight section and the CMS cavern, at Z=2260 cm. Shower will be produced on the -Z side due to interactions with the CMS detector material, therefore TEPX discs on the negative side will have a larger number of hits. We are interested in the incoming BIB not the shower, so take disc 4 that is on the +Z side and is reached first by the BIB particles and therefore has negative time of flight values (~(-8.5) ns). For PU samples both disc 4 and -4 have a similar number of hits as the CMS geometry is symmetric. <br>
<p align="center">
<img src="images/LHC.png" width="500" align="center">
</p>
([source](https://sviret.web.cern.ch/sviret/Images/CMS/MIB/MIB/Welcome.php?n=Work.Gen))


