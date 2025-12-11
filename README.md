# FIRE-TRACE - Tier 3 Monitoring of Greenhouse Gas Emissions from Forest Fires 
## An Earth Observation-Based Framework in support of the UN Climate Goals
Chiara Aquino, Valentina Bacciu, Maria Vincenza Chiriacò, Alessandro D’Anca, Jishnu Jeevan, Sergio Noce, Adriana Torelli, Manuela Balzarolo

## Abstract
Greenhouse gas (GHG) emissions from forest fires are increasingly contributing to atmospheric concentrations of climate-altering gases and represent a growing and poorly constrained component of global carbon budgets, particularly under more frequent droughts and heatwaves. Closing this gap is critical for achieving the United Nations Sustainable Development Goal (SDG) 13, specifically indicator 13.2.2, “Total GHG emissions per year”, urging accurate quantification and monitoring for timely mitigation strategies. Here we introduce FIRE-TRACE, an Earth Observation-based framework integrating satellite data with field-based biomass measurements to improve estimates of forest fuel load and fire emissions. Unlike existing national inventories and global satellite products, FIRE-TRACE resolves spatial variation in burned area and fire severity at the scale of individual fires, enabling emission disaggregation across ecological and administrative units. This work advances SDG 13 by providing a standardised approach, demonstrated for Italy over the period 2018-2023 yet transferable to other countries and time periods, thereby strengthening national reporting capacities and advancing global climate monitoring.

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

