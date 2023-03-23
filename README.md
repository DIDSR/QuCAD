QuCADt is a software tool for evaluating wait-time savings for Computer-Aided Triage and Notification (CADt) devices.

# Purpose
In the past decade, Artificial Intelligence (AI) algorithms have made promising impacts to transform health-care in all aspects. One application is to triage patients’ radiological medical images based on the algorithm’s binary outputs. Such AI-based prioritization software is known as computer-aided triage and notification (CADt). Their main benefit is to speed up radiological review of images with time-sensitive findings. However, as CADt devices become more common in clinical workflows, there is still a lack of quantitative methods to evaluate a device’s effectiveness in saving patients’ waiting times.

This software tool is developed to simulate clinical workflow of image review/interpretation. Included in this tool, we also provide a mathematical framework based on queueing theory to calculate the average waiting time per patient image before and after a CADt device is used. For more complex workflow model with multiple priority classes and radiologists, an approximation method known as the Recursive Dimensionality Reduction technique proposed by [Harchol-Balter et al (2005)](https://www.cs.cmu.edu/~harchol/Papers/questa.pdf) is applied. We define a performance metric to measure the device’s time-saving effectiveness. Simulated and theoretical average time-saving is comparable, and the simulation is used to provide confidence intervals of the performance metric we defined.

# Software Requirements
This software package was developed using Python 3.9.4 with the following extra packages.
* numpy
* pandas
* scipy
* matplotlib
* statsmodels

`scripts/requirements.txt` contains a list of packages required to build a virtual enviornment to run this software.

# Software Usage
`scripts/run_sim.py` is the main script to run this software. There are two ways to accept user input values that specify clinical settings, CADt AI diagnostic performance, and software preferences. By default, outputs will be dumped in outputs/ automatically including plots and a pickled python dictionary that contains all simulation information. Please refer to  the `UserManual.pdf` and `scripts/README.md` for more information

# Relevant Publications
* [arXiv pre-print (2023) - To Be Published]()
* [SPIE Medical Imaging (2022)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12035/0000/Wait-time-saving-analysis-and-clinical-effectiveness-of-Computer-Aided/10.1117/12.2603184.full?SSO=1)
* [FDA Science Forum (2021)](https://www.fda.gov/media/148986/download)

# Contact
For any questions/suggestions/collaborations, please contact Elim Thompson either via this GitHub repo or via email (yeelamelim.thompson@fda.hhs.gov).
