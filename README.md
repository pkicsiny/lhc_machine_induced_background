# Source code for TEPX Beam Induced Background (BIB) timing studies.

The workflow uses data simulated with CMSSW version 11.2.0.pre6. <br>
Simulation outputs are saved in hdf5 files and no ROOT files are involved in the process. <br>
the hdf5 files have the following dada structure: <br>
- group: Tof_q <br>
- datasets: dXrY, where X is the TEPX disc ID ((-4)-4) and Y is the TEPX ring ID (1-5) <br>
