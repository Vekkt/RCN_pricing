######################################

######################################

File Descirptions
###########

-Main.ipynb is the main notebook file where all the pricing is coded with the respective graphs. 

-calibration.py is the file where we callibrate our model, calculate parameters of r, delta, u and d.
Calibration file is called inside the Main.ipynb which returns aforemented params.

-rnc.py provides direct implementation of both (non)callable (B)RCNs. It returns class instance that
will be used for pricing callable versions in Main.ipynb.

-binomial.py provides indirect implementation of non-callable versions of (B)RCNs via pricing the
replicating portfolio using bond and put. This model also returns class instance that is used for 
pricing the non-callable notes in the Main.ipynb 

###################

#######################
Instructions

-Simply open the Main.ipynb and run the cells. All the functions will me loaded automatically. 
No need to run other files separately. 

###########################################

###########################################
Modules Needed
-numpy
-statsmodel
-scipy
-matplotlib
-tqdm