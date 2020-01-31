#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 21:51:11 2020

author: Jon K. Golla <jgolla2@illinois.edu>
function: Read-in and process output from Metrohm Ion Chromatograph of 
    Hydrogeochemistry Lab, Department of Geology,
    University of Illinois at Urbana-Champaign
"""

## Import necessary libraries and functions
import os
import glob
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#%%
## Data Processing

# Compile all IC output files (e.g., .txt) into a list without the extension
def OutputList(ext=""):
    "Returns a list of output file names and a list of sample names"
    files = [f for f in glob.glob(f"*{ext}")]
    samples = [os.path.splitext(val)[0] for val in files]
    return files, samples

files, samples = OutputList(".txt")

# Create a list of pandas data frames of samples
data = []
for file in files:
    data.append(pd.read_csv(file, delim_whitespace=True, skiprows=4))
    
# Some housekeeping...
for n in data:
    n.iloc[:,0] = pd.to_numeric(n.iloc[:,0].map(lambda x: x.lstrip(':')))
    n.rename(columns={ n.columns[0]: "SpCond" }, inplace=True)
    n.index = n.index/60 # 60 cycles in 1 min

# Plot chromatograms and save to png files
for i, sample in itertools.zip_longest(data, samples):
    i.plot(legend=False, title=sample);
    plt.ylabel('Specific Conductance ($\mu$S/cm)');
    plt.xlabel('Time (min)');
    plt.xlim(0,25);
    plt.tight_layout();
    plt.savefig(sample, dpi=300)

# Known standard concentrations
stds_known = np.array((2.5, 5, 10, 20))

# Find position/indeces of calibration standards in sample list
stds_indeces = np.array((samples.index("std1"), samples.index("std2"), 
                 samples.index("std3"), samples.index("std4")))

# subset calibration standard data
stds_data = [data[i] for i in stds_indeces]

#%%
## Peak analysis of calibration standards

# Make blank lists for data
cl_cal = []
no3_cal = []
so4_cal = []

# This for loop finds peak heights for anions of interest within each respective
# elution time range and appends to previously created blank lists
for std in stds_data:
    cl_cal.append(np.max(np.asarray(std.loc[5:7, :])))
    no3_cal.append(np.max(np.asarray(std.loc[9:11, :])))
    so4_cal.append(np.max(np.asarray(std.loc[12.5:15, :])))

cl_cal2D = np.asarray(cl_cal).reshape(-1, 1)
no3_cal2D = np.asarray(no3_cal).reshape(-1, 1)
so4_cal2D = np.asarray(so4_cal).reshape(-1, 1)

# Construct calibration (signal-concentration) curves    
CalibCurve_cl = LinearRegression()  
CalibCurve_cl.fit(cl_cal2D, stds_known)

CalibCurve_no3 = LinearRegression()  
CalibCurve_no3.fit(no3_cal2D, stds_known)

CalibCurve_so4 = LinearRegression()  
CalibCurve_so4.fit(so4_cal2D, stds_known)

#%%
## Peak analysis of unknown samples

# Make blank lists for data
cl_unk = []
no3_unk = []
so4_unk = []

# This for loop finds peak heights for anions of interest within each respective
# elution time range and appends to previously created blank lists
for i in data:
    cl_unk.append(np.max(np.asarray(i.loc[5:7, :])))
    no3_unk.append(np.max(np.asarray(i.loc[9:11, :])))
    so4_unk.append(np.max(np.asarray(i.loc[12.5:15, :])))

# Reshape lists into 2-D arrays (necessary for fitting to sklearn models)
cl_unk2D = np.asarray(cl_unk).reshape(-1, 1)
no3_unk2D = np.asarray(no3_unk).reshape(-1, 1)
so4_unk2D = np.asarray(so4_unk).reshape(-1, 1)

# Calculate unknown concentrations using standard calibration curves
cl_conc = CalibCurve_cl.predict(cl_unk2D)
no3_conc = CalibCurve_no3.predict(no3_unk2D)
so4_conc = CalibCurve_so4.predict(so4_unk2D)

# Make blank lists for comments on data
cl_comm = []
no3_comm = []
so4_comm = []

# These for loops check and return comments if calculated concentrations are
# below detection of first calibration standard concentration
for i in cl_conc:
    if i < 2.5:
        cl_comm.append('below std1')
    else:
        cl_comm.append('') # no comment

for i in no3_conc:
    if i < 2.5:
        no3_comm.append('below std1')
    else:
        no3_comm.append('')
        
for i in so4_conc:
    if i < 2.5:
        so4_comm.append('below std1')
    else:
        so4_comm.append('')

#%% 
## Export output

# Aggregate results into one pandas data frame        
results = pd.DataFrame({'Cl_peak': cl_unk, 'NO3_peak': no3_unk, 'SO4_peak': so4_unk,
                        'Cl_ppm': cl_conc, 'NO3_ppm': no3_conc, 'SO4_ppm': so4_conc,
                        'Cl_mmol': cl_conc/35.453, 'NO3_mmol': no3_conc/62.0049, 'SO4_mmol': so4_conc/96.06,
                        'Comments (Cl)': cl_comm, 'Comments (NO3)': no3_comm, 'Comments (SO4)': so4_comm},
    index = samples)

# Sort results by index alphabetically
results.sort_index(inplace=True)

# Get base name of present working directory
folder = os.path.basename(os.getcwd())

# Write results to an Excel file
results.to_excel(folder+'.xlsx')