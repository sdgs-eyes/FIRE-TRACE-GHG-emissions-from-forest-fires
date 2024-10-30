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
##### LOCATION OF INPUT TABLES ########
#######################################
#EFFIS data on burnt area
effis_shapefile = '../data/shapefiles/effis_shapefile_downloaded_6Dec23/modis.ba.poly.dbf'
#LANDCOVER
landcover_legend_table = "../data/tables/FOREST_LANDCOVER_LEGEND.csv"
clc18_landcover = "../data/shapefiles/CORINE_LC18/CLC18_IVLIV_IT_FOREST_CLASSES_fixed.dbf"
#BIOMASS
INFC15_lookup_table = "../data/tables/INFC15_FOREST_CLASSES_LOOKUP.csv"
biomass_table = "../data/tables/INFC15_AGB_PER_REGION.csv"
#COMBUSTION FACTOR
c_factor_bovio_conversion_table = "../data/tables/C_FACTOR_BOVIO_FOREST_CLASSES_LOOKUP.csv"
#FIRE DAMAGE
fire_damage_table = "../data/tables/C_FACTOR_BOVIO_SCORCH_HEIGHT.csv"
#EMISSION FACTORS
emission_factors_table = "../data/tables/GHG_EMISSION_FACTORS.csv"

# ADDITIONAL TABLES
#REGIONS SHAPEFILE
it_regions_shapefile = '../data/shapefiles/ItalianRegions/Reg01012023_WGS84.dbf'

#OUTPUT TABLES
ghg_output_table="../outputs/_"+region+"_"+landcover+"_scorch_height_"+str(scorch_height)+"_GHG_EMISSIONS.csv"
ghg_output_plot="../outputs/_"+region+"_"+landcover+"_scorch_height_"+str(scorch_height)+"_ghg_by_forest_type.shp"

##############################################################################
#################### PARAMETERS OF CHOICE ####################################
##############################################################################
landcover="CLC18"

fire_id = None
year = 2022
country = "IT" 
region = "Lazio"
province = None
commune = None
scorch_height = None
language = "ITALIAN"

crs = "epsg:4326"

##############################################################################
#################### GHG CALCULATION ####################################
##############################################################################

forest_classes, forest_labels, forest_colors = get_landcover_classes(landcover,landcover_legend_table,language)
burnt_shape = import_data(effis_shapefile,it_regions_shapefile,crs,region,COMMUNE=commune,PROVINCE=province,COUNTRY=country,YEAR=year,ID=fire_id)

# STEP 1. Get total burnt area (A) for each vegetation type within burnt shape
if landcover == "EFFIS":
    A,_ = get_total_burnt_area(landcover,clc18_landcover,burnt_shape,forest_classes, crs)
else:
    A,A_by_event = get_total_burnt_area(landcover,clc18_landcover,burnt_shape,forest_classes, crs)

# STEP 2. Get biomass of available fuel (B) for each vegetation type
B = get_biomass(INFC15_lookup_table, biomass_table, landcover, region)

# STEP 3. Get combustion factor (C) for each vegetation type
C = get_combustion_factor(c_factor_bovio_conversion_table,fire_damage_table,landcover,scorch_height)

# STEP 4. Get emission factors (D) for each GHG compound
D = get_emission_factors(emission_factors_table)

# STEP 5. Calculate total GHG emissions from emission factors
ghg, ghg_std, ghg_all_events = get_total_ghg_emissions(A,B,C,D,landcover)

#check Total GHG per year
print("total GHG emissions for the year " + str(year)  +" in " + str(region) +" region in "
      + str(np.round(ghg,2))+ " +/- " + str(np.round(ghg_std,2))+ " ktonnes CO2eq")


plot_burnt_area_and_forest_classes(landcover_legend_table,clc18_landcover,it_regions_shapefile,region,language,burnt_shape,ghg_all_events,crs,
                                   plot_region=True)

# save_ghg_emissions(ghg, A, ghg_output_table,year,country,region,province,commune)







