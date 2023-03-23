## Outputs folder

By default, this output folder is where `run_sim.py` dumps out plots and files associated to the run.

+ `plots/` is expected to store any output plots, including the followings. Please see the `UserManual.pdf` for more information on the plots themselves.
    + Patient image time-flow plots for the first 200 patient images in both with and without CADt scenarios. 
    + Distributions of number of patient images in system right before the arrival of a new patient.
    + Distributions of waiting time per patient (both absolute waiting time in with / without CADt scenarios and wait-time saving)
* `stats/` is expected to store the output pickled dictionaries. Default filename is `stats.p`. However, a user may want to change it using `--statsFile` flag or `statsFile` variable in `inputs/config.dat` such that each simulation run has an output associated to the clinical parameters. Please refer to the `UserManual.pdf` for more information on the pickled dictionary outputs.