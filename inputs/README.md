## Inputs folder

By default, this input folder stores a `config.dat` file and an `exampleROC.dat`. 

* `config.dat` is an example user input text file that can be fed directly to `run_sim.py`. This is an alternative way of specifying user values (as opposed to using argument flags). Please refer to the example `config.dat` the explanation of  each parameter.
* `exampleROC.dat` is an example ROC curve of the CADt device. First column is False-Positive Fraction (i.e. 1 - AI specificity), and second column is True-Positive Fraction (i.e. AI sensitivity). These values will be used to parameterize an ROC curve assuming a bi-normal distribution.