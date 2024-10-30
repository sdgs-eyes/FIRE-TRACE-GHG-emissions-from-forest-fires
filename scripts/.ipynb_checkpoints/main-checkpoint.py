# -*- coding: utf-8 -*-
"""
@author: Chiara Aquino

@date: 12 January 2024

This script calculates GHG emissions using the functions defined in
ghg_fire_emissions_functions.py

"""

############################################################################
########################### IMPORT LIBRARIES ################################
############################################################################
from ghg_fire_emissions_functions import *
import numpy as np

##############################################################################
##### LOCATION OF INPUT TABLES ########
##############################################################################
#EFFIS data on burnt area
effis_table = '../data/effis_shapefile_downloaded_6Dec23/modis.ba.poly.shp'
#BIOMASS
biomass_table = "../data/INFC2015_EFFIS_merged.csv"
#EFFIS_BOVIO class
effis_bovio_conversion_table = "../data/eFFIS_bovio_2007_forest_classes.csv"
#FIRE DAMAGE
fire_damage_table = "../data/scorch_height_bovio_2007.csv"
#EMISSION FACTORS
emission_factors_table = "../data/ghg_emission_factors.csv"

##############################################################################
#################### PARAMETERS OF CHOICE ####################################
##############################################################################
fire_id = None
year = 2021
country = "IT" 
region = "Calabria"
province = "Cosenza" 
commune = None
scorch_height = None

forest_types = ["BROADLEA","CONIFER","SCLEROPH","TRANSIT"]


##############################################################################
#################### GHG CALCULATION ####################################
##############################################################################

# STEP 1. Get total burnt area (A) for each vegetation type
# In this example, we are choosing EFFIS data. First, we need to filter EFFIS by selecting columns corresponding to our chosen parameters
burnt_area_data = get_effis_data(effis_table,ID=fire_id,YEAR=year,COUNTRY=country,PROVINCE=province,COMMUNE=commune)
# get total burnt area for each fire event, organised by forest classes
A = get_total_burnt_area(burnt_area_data,forest_types)

# STEP 2. Get biomass of available fuel (B) for each vegetation type
B = get_biomass(biomass_table,region)

# STEP 3. Get combustion factor (C) for each vegetation type
C = get_combustion_factor(effis_bovio_conversion_table,fire_damage_table,scorch_height)

# STEP 4. Calculate total GHG emissions from emission factors
ghg = get_total_annual_GHG_emissions(A,B,C,emission_factors_table,forest_types)

#check Total GHG per year
print("total GHG emissions for the year " + str(year)  +" in " + str(province) +" province: " + str(np.round(ghg,2))+ " Kton CO2eq")








