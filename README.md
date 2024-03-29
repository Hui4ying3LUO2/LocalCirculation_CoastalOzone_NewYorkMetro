# LocalCirculation_CoastalOzone_NewYorkMetro
Open Resources for "Investigating Impacts of Local Circulation on Coastal Ozone Pollution in the New York Metropolitan Area: Evidence from Multi-year Observations" submitted to Journal of Geophysical Research: Atmospheres


******************Key scripts and corresponding results*********************

Note: We do not own the original data to support the running of the scripts shared below. If you would like to reproduce our results or adapt our method, please download/acquire and prepare data accordingly. Our data sources are listed in the Acknowledgements.

1 Temperature clustering: 

    Tcluster.py 

    TCLusterAssignQUEEkmeans_norm0__24_3.txt with 0-2 representing 'Hot', 'Moderate', and 'Cool' days.


2 Wind clustering at QUEE: 

    WindCluster.py 

    QUEEkmeans_norm1__6_4CLusterAssign.txt with 0-3 representing 'SeaBreeze', 'Oscillation', 'Southerly', and 'Westerly' days.

3 Hot W wind subset clustering at WANT: 

    SubQUEEW.py

    SubQUEEW_WANTCLusterOrig.txt with  0-1 representing 'W_w' and 'W_sb' days.

4 All results correspond to the dates listed in TCLusterDateQUEEkmeans_norm0__24_3.txt
