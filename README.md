# Calculate GHG emissions from forest fires

## Credit
- Chiara Aquino
- Manuela Balzarolo
- Maria Vincenza Chiriacò

## Overview
This repository provides the functions to calculate GHG emissions from forest fires, given a burnt area. 
GHG emissions are calculated using the model from Chiriacò et al.(2013):

    GHG emissions = A x B x C x D x 10-6 

where 
<br>
GHG emissions = amount of GHGs released as a result of fire [kgton of GHG]; 
<br>
A = burnt area [ha], provided by input data
<br>
B = mass of available fuels, in [kg dry matter ha-1]
<br>
C = combustion factor, portion of biomass combusted [dimensionless]
<br>
D = emission factor [g GHG kg–1] for each GHG compound. 

**References**
<br>
Chiriacò, M.V., Perugini, L., Cimini, D., D’Amato, E., Valentini, R., Bovio, G., Corona, P. and Barbati, A., 2013. Comparison of approaches for reporting forest fire-related biomass loss and greenhouse gas emissions in southern Europe. _International Journal of Wildland Fire_, 22(6), pp.730-738.
