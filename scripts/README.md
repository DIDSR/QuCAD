## Scripts folder

This scripts folder contains all the python scripts to take user input values, run simulations, calculate predictions, and generate plots. Please refer to `tools/` sub folder for the details on how simulation and calculation is done. This folder contains two main files.

`requirements.txt` is the text file listed out all the packages needed for this software. Please refer to the `UserManual.pdf` for more information on how to use this file to set up a virtual environment.

`run_sim.py` is the main python script to simulate radiology reading workflow at a specific clinical setting with a CADt diagnostic performance. This simulation software handles a simplified scenario with 1 AI that is trained to identify 1 disease condition from 1 modality and anatomy. Patients in the reading queue either have the disease condition or not.

User can either specify all parameters via argument flags ...

```
$ python run_sim.py --traffic 0.8 --TPFThresh 1.0 --prevalence 0.1
                    --nRadiologists 1 --fractionED 0.0 
                    --meanServiceTimeDiseasedMin 10
                    --meanServiceTimeNonDiseasedMin 10
                    --meanServiceTimeInterruptingMin 5
                    --statsFile /path/to/outputs/stats/stats.p
                    --nTrials 10 --nPatientsTarget 1000 
                    (--FPFThresh 0.1) # if using a Se, Sp threshold point
                    (--rocFile /path/to/inputs/exampleROC.dat) # if using a parameterized ROC curve
                    (--runtimeFile /path/to/outputs/runTime.txt) # if print out runtime performance
                    (--plotPath /path/to/outputs/plots/) # if generate plots
                    (--verbose) # if print out progress 
```

or via an input file `config.dat`.

```
$ python run_sim.py --configFile ../inputs/config.dat
```

For more information on input parameters, please refer to the `UserManual.pdf`.