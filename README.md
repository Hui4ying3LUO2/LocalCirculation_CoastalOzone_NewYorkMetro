# LocalCirculation_CoastalOzone_NewYorkMetro
Open Resources for "Investigating Impacts of Local Circulation on Coastal Ozone Pollution in the New York Metropolitan Area: Evidence from Multi-year Observations" submitted to Journal of Geophysical Research: Atmospheres


******************Figures*********************

Figure 12d: 20180630mapHRRRO3tend.png 

Figure 12b: 20180630mapHRRRcloudsO3.png 

******************Key scripts and corresponding results*********************

Note: We do not own the original data to support the running of the scripts shared below. If you would like to reproduce our results or adapt our method, please download/acquire and prepare data accordingly. Our data sources are listed in the Acknowledgements.

Temperature clustering: 

Tcluster.py 

TCLusterAssignQUEEkmeans_norm0__24_3.txt with 0-2 representing 'Hot', 'Moderate', and 'Cool' days.


Wind clustering at QUEE: 

WindCluster.py 

QUEEkmeans_norm1__6_4CLusterAssign.txt with 0-3 representing 'SeaBreeze', 'Oscillation', 'Southerly', and 'Westerly' days.

Hot W wind subset clustering at WANT: 

SubQUEEW.py

SubQUEEW_WANTCLusterOrig.txt with  0-1 representing 'W_w' and 'W_sb' days.

All results correspond to date listed in TCLusterDateQUEEkmeans_norm0__24_3.txt
