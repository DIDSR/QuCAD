QuCADt is a software tool for evaluating wait-time savings for Computer-Aided Triage and Notification (CADt) devices.

# Purpose
In the past decade, Artificial Intelligence (AI) algorithms have made promising impacts to transform health-care in all aspects. One application is to triage patients’ radiological medical images based on the algorithm’s binary outputs. Such AI-based prioritization software is known as computer-aided triage and notification (CADt). Their main benefit is to speed up radiological review of images with time-sensitive findings. However, as CADt devices become more common in clinical workflows, there is still a lack of quantitative methods to evaluate a device’s effectiveness in saving patients’ waiting times.

This software tool is developed to simulate clinical workflow of image review/interpretation. Included in this tool, we also provide a mathematical framework based on queueing theory to calculate the average waiting time per patient image before and after a CADt device is used. For more complex workflow model with multiple priority classes and radiologists, an approximation method known as the Recursive Dimensionality Reduction technique proposed by [Harchol-Balter et al (2005)](https://www.cs.cmu.edu/~harchol/Papers/questa.pdf) is applied. We define a performance metric to measure the device’s time-saving effectiveness. Simulated and theoretical average time-saving is comparable, and the simulation is used to provide confidence intervals of the performance metric we defined.

# Tool Reference
* RST Reference Number: RST24AI01.01
* Date of Publication: 09/24/2023
* Recommended Citation: U.S. Food and Drug Administration. (2023). QuCAD: Software to Evaluate Wait-Time-Saving Benefits of CADt Devices (RST24AI01.01). https://cdrh-rst.fda.gov/qucad-software-evaluate-wait-time-saving-benefits-cadt-devices

# Disclaimer
## About the Catalog of Regulatory Science Tools
The enclosed tool is part of the Catalog of Regulatory Science Tools, which provides a peer- reviewed resource for stakeholders to use where standards and qualified Medical Device Development Tools (MDDTs) do not yet exist. These tools do not replace FDA-recognized standards or MDDTs. This catalog collates a variety of regulatory science tools that the FDA's Center for Devices and Radiological Health's (CDRH) Office of Science and Engineering Labs (OSEL) developed. These tools use the most innovative science to support medical device development and patient access to safe and effective medical devices. If you are considering using a tool from this catalog in your marketing submissions, note that these tools have not been qualified as [Medical Device Development Tools](https://www.fda.gov/medical-devices/medical-device-development-tools-mddt) and the FDA has not evaluated the suitability of these tools within any specific context of use. You may [request feedback or meetings for medical device submissions](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/requests-feedback-and-meetings-medical-device-submissions-q-submission-program) as part of the Q-Submission Program.
For more information about the Catalog of Regulatory Science Tools, email OSEL_CDRH@fda.hhs.gov.

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
* [Yee Lam Elim Thompson, Gary M. Levine, Weijie Chen, Berkman Sahiner, Qin Li, Nicholas Petrick, Jana G. Delfino, Miguel A. Lago, Qian Cao, and Frank W. Samuelson. "Evaluation of wait time saving effectiveness of triage algorithms." arXiv preprint arXiv:2303.07050 (2023).](https://arxiv.org/abs/2303.07050)
* [Yee Lam Elim Thompson, Gary Levine, Weijie Chen, Berkman Sahiner, Qin Li, Nicholas Petrick, Frank Samuelson, "Wait-time-saving analysis and clinical effectiveness of computer-aided triage and notification (CADt) devices based on queueing theory," Proc. SPIE 12035, Medical Imaging 2022: Image Perception, Observer Performance, and Technology Assessment, 120350Q (4 April 2022); https://doi.org/10.1117/12.2603184](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12035/0000/Wait-time-saving-analysis-and-clinical-effectiveness-of-Computer-Aided/10.1117/12.2603184.full?SSO=1)
* [FDA Science Forum (2021)](https://www.fda.gov/media/148986/download)

# Citation for this tool
Yee Lam Elim Thompson, Jixin Audrey Zheng, Frank W. Samuelson. (2023) QuCAD [Source Code] https://github.com/DIDSR/QuCAD/.

# Contact
For any questions/suggestions/collaborations, please contact Elim Thompson either via this GitHub repo or via email (yeelamelim.thompson@fda.hhs.gov).
