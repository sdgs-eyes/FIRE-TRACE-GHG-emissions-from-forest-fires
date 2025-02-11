def import_data(path_to_data_location, year,fire_id):
    """
    Retrieve burnt area polygon from the shapefile and filter it based on selected columns and values.

    Parameters:
    - path_to_data_location (string) : location of the EFFIS shapefile
    - path_to_italian_regions_shapefile (string) : location of the shapefile containing Italian regions
    - crs (string) : coordinate reference system of choice
    - region (string) : name of Italian region
    - *kwargs: Optional arguments for column name and corresponding values.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
     #open shapefile as DataFrame
    df = geopandas.read_file(path_to_data_location)
    
    #Copy the original DataFrame
    filtered_df = df.copy()
       
    #extract year from EFFIS date column so that the DataFrame can be filtered by year only
    filtered_df['FIREDATE'] = pd.to_datetime(filtered_df['FIREDATE'], format="mixed")
    filtered_df['YEAR'] = filtered_df['FIREDATE'].dt.year
    filtered_df = filtered_df.rename(columns={"FIREDATE": "DATE"})
    filtered_df = filtered_df[filtered_df['YEAR'] == year]

    if fire_id:
        filtered_df = filtered_df[filtered_df['id'] == fire_id]
    
    return filtered_df    



def crop_shapefile(path_to_country_shapefile, path_to_region_shapefile, path_to_province_shapefile,
                   path_to_commune_shapefile, filtered_df, country=None, region=None, province=None, commune=None):
    """
    Crop the shapefile based on the specified geographical hierarchy:
    country > region > province > commune.
    
    Parameters:
    - path_to_country_shapefile: Path to the country shapefile.
    - path_to_region_shapefile: Path to the region shapefile.
    - path_to_province_shapefile: Path to the province shapefile.
    - path_to_commune_shapefile: Path to the commune shapefile.
    - filtered_df: The GeoDataFrame to crop.
    - country: Optional; name of the country.
    - region: Optional; name of the region.
    - province: Optional; name of the province.
    - commune: Optional; name of the commune.
    
    Returns:
    - A cropped GeoDataFrame.
    """

    import geopandas as gpd
    
    # Load country shapefile
    # Ensure both GeoDataFrames have the same CRS

    if country and not region and not province and not commune:
        countries = gpd.read_file(path_to_country_shapefile)
        filtered_df = filtered_df.to_crs(countries.crs)
        cut_geometry = countries
        # Filter for the specified country
        #countries = countries[countries['COUNTRY_COLUMN'] == country] 
        # Clip the filtered DataFrame with the country
        filtered_df = gpd.clip(filtered_df, countries)

    if region and not province and not commune:
        # Load region shapefile
        regions = gpd.read_file(path_to_region_shapefile)
        regions = regions[regions['DEN_REG'] == region] 
        cut_geometry = regions
        # Ensure CRS matches
        filtered_df = filtered_df.to_crs(regions.crs)
        filtered_df = gpd.clip(filtered_df, regions)

    if province and not commune:
        # Load province shapefile
        provinces = gpd.read_file(path_to_province_shapefile)
        provinces = provinces[provinces['DEN_PROV'] == province] 
        cut_geometry = provinces
        # Ensure CRS matches
        filtered_df = filtered_df.to_crs(provinces.crs)
        filtered_df = gpd.clip(filtered_df, provinces)

    if commune:
        # Load commune shapefile
        communes = gpd.read_file(path_to_commune_shapefile)
        communes = communes[communes['COMUNE'] == commune]  # Adjust 'COMMUNE_COLUMN' to your shapefile's column name
        cut_geometry = communes
        # Ensure CRS matches
        filtered_df = filtered_df.to_crs(communes.crs)
        filtered_df = gpd.clip(filtered_df, communes) 

    return filtered_df, cut_geometry

def get_landcover_classes(landcover, path_to_landcover_legend, language):
    """
    Retrieve landcover classes from the landcover shapefile and filter based on selected columns and values.
    
    Parameters:
    - landcover (string): name of landcover type
    - path_to_landcover_legend (string): location of landcover legend table
    - language (string): 'English' or 'Italian'
    
    Returns:
    - classes, names, colors: Lists of landcover codes, names, and colors.
    """
    landcover_legend = pd.read_csv(path_to_landcover_legend)
    language_column = f"{landcover}_NAME_{language}"
    
    valid_rows = landcover_legend[[f"{landcover}_CODE", language_column, f"{landcover}_COLOR"]].dropna()
    classes = valid_rows[f"{landcover}_CODE"].astype(str).tolist()
    names = valid_rows[language_column].tolist()
    colors = valid_rows[f"{landcover}_COLOR"].tolist()
    
    return classes, names, colors

def intersect_vectors(path_to_poly_to_overlay, poly_underneath, crs):
    """
    Filter Corine (CLC18) landcover classes by clipping Corine landcover with burnt area shapefile.

    Parameters:
    - path_to_clc18_shapefile (string): location of Corine shapefile
    - df_burnt_area_shape (pd.DataFrame): burnt area DataFrame
    - forest_classes (list of str): forest classes
    - crs (string): chosen coordinate system, e.g., "epsg:4326"

    Returns:
    - geopandas.GeoDataFrame: Corine landcover classes in burnt area.
    """
    top = geopandas.read_file(path_to_poly_to_overlay).to_crs(crs)
    underneath = poly_underneath.to_crs(crs)
    
    return geopandas.overlay(top, underneath, how="intersection")
    
    


def read_biomass_table(path):
    """
    Helper function to read and clean the biomass tables.
    """
    biomass = pd.read_csv(path)
    biomass = biomass.iloc[1:]

    #Ensure all columns after the first are float numbers
    biomass.iloc[:, 1:] = biomass.iloc[:, 1:].astype(float)

    #Transform all 0s in NANs 
    biomass = biomass.replace(0, np.nan)

    # Tidy up INFC biomass data so we can merge with lookup table
    biomass = biomass.T
    biomass.columns = biomass.iloc[0]
    biomass = biomass.drop(biomass.index[0])
    # Make the index a regular column and rename columns
    biomass = biomass.reset_index().rename(columns={'index': 'INFC_ID'})
    return biomass


def get_biomass(path_to_lookup_table,
                path_to_biomass_live_table, path_to_biomass_dead_table, 
                path_to_biomass_fwd_table, path_to_biomass_litter_table, 
                landcover, region=None):
    """
    Retrieve pre disturbance biomass for each vegetation type
    Data is derived from average standing volume estimates from National Forest Inventory 2015 (INFC2015). INFC2015 classes have been 
    averaged for each of the 20 Italian administrative regions to match EFFIS or CLC18 vegetation classes

    Parameters:
    - path_to_lookup_table (str) : location of lookup tables for different landcover classes -> INFC2015 classes
    - path_to_biomass_live_table (str) : location of National Forest Inventory 2015 biomass values for live biomass
    - path_to_biomass_dead_table (str) : location of National Forest Inventory 2015 biomass values for dead biomass
    - path_to_biomass_fwd_table (str) : location of National Forest Inventory 2005 biomass values for forward biomass
    - path_to_biomass_litter_table (str) : location of National Forest Inventory 2005 carbon values for litter
    - landcover (str) : landcover type to use (CLC18 or EFFIS)
    - region (str): Italian region of interest. Default region is None. 
    
    Returns:
    - pd.DataFrame: Processed DataFrame with average biomass values and standard deviation per selected region per vegetation class
    """
    # Read the biomass tables
    biomass_live = read_biomass_table(path_to_biomass_live_table)
    biomass_dead = read_biomass_table(path_to_biomass_dead_table)
    biomass_fwd = read_biomass_table(path_to_biomass_fwd_table)
    biomass_litter = read_biomass_table(path_to_biomass_litter_table)
    
    # Merge the biomass tables
    biomass_merged = pd.merge(biomass_live, biomass_dead, on='INFC_ID', suffixes=('_live', '_dead'))
    biomass_merged = pd.merge(biomass_merged, biomass_fwd, on='INFC_ID', suffixes=('', '_fwd'))
    biomass_merged = pd.merge(biomass_merged, biomass_litter, on='INFC_ID', suffixes=('_fwd', '_litter'))
    
    # Read the lookup table for the landcover classes
    lookup = pd.read_csv(path_to_forest_classes_table)
    lookup = lookup[[landcover + "_CLASS", "INFC_ID"]]
    
    lookup['INFC_ID'] = lookup['INFC_ID'].astype(int)
    biomass_merged['INFC_ID'] = biomass_merged['INFC_ID'].astype(int)
    
    biomass_by_landcover = pd.merge(lookup, biomass_merged, on='INFC_ID')
    # Convert landcover class to string
    biomass_by_landcover[landcover + "_CLASS"] = biomass_by_landcover[landcover + "_CLASS"].astype(str)

    
    # Apply division by 0.45 to litter biomass columns
    for col in biomass_by_landcover.columns:
        if "_litter" in col:
            biomass_by_landcover[col] = biomass_by_landcover[col]/ 0.45  # Divide instead of multiply
    

    if region is not None:
        region_columns = [col for col in biomass_by_landcover.columns if col.startswith(region)]
    else:
        region_columns = [col for col in biomass_by_landcover.columns if col.startswith("Italia")]
    
    # Compute mean and uncertainty for each landcover class
    grouped = biomass_by_landcover.groupby([landcover + "_CLASS"])[region_columns].mean()
    grouped_uncertainty = biomass_by_landcover.groupby([landcover + "_CLASS"])[region_columns].sem()
    
    # Convert to DataFrame
    df_biomass = grouped.reset_index()
    df_biomass_uncertainty = grouped_uncertainty.reset_index()
    
    # Rename columns
    df_biomass = df_biomass.rename(columns={df_biomass.columns[0]: landcover})
    df_biomass_uncertainty = df_biomass_uncertainty.rename(columns={df_biomass_uncertainty.columns[0]: landcover})
    
    # Replace NaNs with 0 in uncertainty
    df_biomass_uncertainty = df_biomass_uncertainty.fillna(0)
    
    # Merge biomass and uncertainty tables
    df_biomass = pd.merge(df_biomass, df_biomass_uncertainty, on=landcover, suffixes=('', '_STD'))
    
    # Convert landcover class to integer
    df_biomass[landcover] = df_biomass[landcover].astype(int)
    
    
    ### TOTALS
    df_totals = df_biomass.copy()
    # Identify biomass and standard deviation columns
    biomass_cols = [col for col in df_totals.columns if col.endswith(("_live", "_dead", "_fwd", "_litter"))]
    std_cols = [col for col in df_totals.columns if col.endswith(("_live_STD", "_dead_STD", "_fwd_STD", "_litter_STD"))]
    
    # Sum biomass values normally
    df_totals["total_biomass"] = df_totals[biomass_cols].sum(axis=1)
    
    # Apply error propagation for standard deviation columns
    df_totals["total_biomass_STD"] = np.sqrt((df_totals[std_cols] ** 2).sum(axis=1))
    
    # Keep only CLC18, total biomass, and its standard deviation
    df_totals = df_totals[["CLC18", "total_biomass", "total_biomass_STD"]]
    
    # Rename columns
    df_totals = df_totals.rename(columns={"total_biomass": "BIOMASS", "total_biomass_STD": "BIOMASS_STD"})


    return df_totals

def get_combustion_factor(path_to_fire_severity_shapefile, forest_in_burnt_area, 
                          path_to_forest_classes, path_to_C_factor_table, landcover, crs, language):


    """
    Retrieve combustion factor for each forest type in each burn area.
    These value are retrieved from an empirical combustion model by Aquino et al. (202X)

    Parameters:
    - path_to_fire_severity_shapefile (str) : location of the fire severity shapefile (converted from raster from the original by EFFIS)
    - forest_in_burnt_area (pd.DataFrame) : geodataframe with the landcover forest classes in burnt area
    - path_to_forest_classes (str) : location of the lookup tables between CLC18 landcover and forest types
    - path_to_C_factor_table: location of the combustion factor table per forest type
    - landcover (str) : landcover type to use (eg CLC18)
    - crs (str) : coordinate reference system

    Returns:
    - pd.DataFrame: Processed DataFrame with combustion factor values per vegetation type AND area 
    """
    
    fire_severity = intersect_vectors(path_to_fire_severity_shapefile, forest_in_burnt_area, crs)
    
    fire_severity.rename(columns={'value': 'fire_severity'}, inplace=True)
    # remove empty values of fire severity
    fire_severity = fire_severity.dropna(subset=['fire_severity'])
    # remove values of fire severity == 1 (unburnt)
    fire_severity = fire_severity [fire_severity ['fire_severity'] != 1]
    fire_severity = fire_severity [fire_severity ['fire_severity'] != 0]

    forest_lookup = pd.read_csv(path_to_forest_classes)
    C_model = pd.read_csv(path_to_C_factor_table)

    forest_lookup[landcover+'_CLASS'] = forest_lookup[landcover+'_CLASS'].astype(int)
    language_column = f"{landcover}_NAME_{language}"
    forest_lookup = forest_lookup[[landcover+'_CLASS', language_column, 'FOREST_TYPE']] 
    fire_severity[landcover] = fire_severity[landcover].astype(int)

    forest_lookup = forest_lookup.drop_duplicates()
    merged = fire_severity.merge(forest_lookup, left_on=landcover, right_on=landcover+'_CLASS', how='left')

    merged_C = pd.merge(merged,C_model, left_on=['FOREST_TYPE', 'fire_severity'], right_on=['Forest_Type', 'EFFIS_Severity_Class'], how='left')
    merged_C = merged_C.drop(columns=['Forest_Type', 'EFFIS_Severity_Class', 'Severity_Class'])

    #get area
    merged_C = merged_C.to_crs({'proj':'cea'})
    merged_C["AREA_HA"] = merged_C.geometry.area / 10000
    
    return merged_C

def get_emission_factors(path_to_emission_factor_table):

    """
    Retrieve emission factor table containing each GHG compound and its uncertainity

    Parameters:
    - path_to_emission_factor_table (str) : location of emission factor table (IPCC,2006)

    Returns:
    - pd.DataFrame: DataFrame with emission factor values per GHG compound
    """

    df_emission_factors = pd.read_csv(path_to_emission_factor_table)
    return df_emission_factors
    
    
def get_total_ghg_emissions(AC,B,D,landcover):
    """

    This function puts together all the previous steps of the model and calculates final GHG emissions. 
    
    Parameters: 
    - AC (pd.DataFrame) : burnt area and combustion factor for each vegetation type, as retrieved by function get_combustion_factor() 
    - B (pd.DataFrame) : pre disturbance biomass for each vegetation type, as retrived by function get_biomass()
    - D (pd.DataFrame) : emission factor for each GHG, as retrieved by function get_emission_factors()
    - landcover (str) : landcover type to use (CLC18 or EFFIS)

    Returns:
    - Float: total_ghg_kton (total GHG emissions in kton)
    - Float: total_ghg_kton_std (standard deviation of total GHG emissions in kton)
    - pd.Dataframe: ABCD (GHG emissions by forest class)

    """
    
    #Merge A,B,C and D DataFrames on the 'landcover' column
    ABC = AC.merge(B, on=landcover, how='inner')
    ABCD = ABC.merge(D, how='cross')

    #for the biomass values replace nan with 0
    #ABCD = ABCD.dropna(subset=['BIOMASS'])
    ABCD = ABCD.fillna(0)
    
    gases = ['CO2','CH4','CO','N2O','NOx', 'PM2.5']

    # calculate emissions and standard deviation for each gas, according to the model formula
    for gas in gases:
        ABCD[gas+'_MG'] = (1e-3 * ABCD['AREA_HA'] * ABCD['BIOMASS'] * ABCD['COMBUSTION_FACTOR']* ABCD[gas]).astype(float)
    
        ABCD[gas+'_STD_MG'] = (1e-3 * ABCD[gas+'_MG'] * ((ABCD['BIOMASS_STD'] / ABCD['BIOMASS'] if ['BIOMASS'] != 0 else 0)**2 
                                                         + (ABCD['COMBUSTION_FACTOR_STD'] / ABCD['COMBUSTION_FACTOR'] if ['COMBUSTION_FACTOR'] != 0 else 0)**2
                                                        + (ABCD[gas+'_STD'] / ABCD[gas])**2)**0.5).astype(float)
    
    
    # apply conversion factor for N2O e CH4, to convert into CO2eq
    ABCD['N2OEQ_MG'] = ABCD['N2O_MG'] * 273
    ABCD['N2OEQ_STD_MG'] = 273 * ABCD['N2OEQ_MG'] * (ABCD['N2O_STD_MG'] / ABCD['N2O_MG'])
    
    ABCD['CH4EQ_MG'] = ABCD['CH4_MG'] * 28
    ABCD['CH4EQ_STD_MG'] = 28 * ABCD['CH4EQ_MG'] * (ABCD['CH4_STD_MG'] / ABCD['CH4_MG'])
    
    ABCD['TOTEQ_MG'] = ABCD['CO2_MG'] + ABCD['CH4EQ_MG'] + ABCD['N2OEQ_MG']
    ABCD['TOTEQ_STD_MG'] = (ABCD['CO2_STD_MG'] ** 2 + ABCD['CH4EQ_STD_MG'] ** 2 + ABCD['N2OEQ_STD_MG'] ** 2) ** 0.5
    
    #get total emissions in kton
    total_ghg_kton = ABCD['TOTEQ_MG'].sum() /1000
    
    #Calculate error over the sum using error propagation
    error_squared = (ABCD['TOTEQ_STD_MG'] ** 2).sum()
    error_over_sum = np.sqrt(error_squared) 
    total_ghg_kton_std = error_over_sum/1000
        
    return total_ghg_kton,total_ghg_kton_std, ABCD



    #Save total GHGs
def save_total_ghg_emissions(ghg_kton,ghg_kton_std, ABCD, year, country,
                          region,province,commune, tot_events):
        
    #get total of single gases
    CO2_kton = ABCD['CO2_MG'].sum() /1000
    CH4_kton = ABCD['CH4_MG'].sum()/1000
    N2O_kton = ABCD['N2O_MG'].sum()/1000
    PM_kton = ABCD['PM2.5_MG'].sum()/1000
    CO_kton = ABCD['CO_MG'].sum() /1000
    NOx_kton = ABCD['NOx_MG'].sum() /1000

    #get error over the sum
    #Calculate error over the sum using error propagation
    CO2_kton_std = np.sqrt((ABCD['CO2_STD_MG'] ** 2).sum())/1000
    CH4_kton_std = np.sqrt((ABCD['CH4_STD_MG'] ** 2).sum())/1000
    N2O_kton_std = np.sqrt((ABCD['N2O_STD_MG'] ** 2).sum())/1000
    PM_kton_std = np.sqrt((ABCD['PM2.5_STD_MG'] ** 2).sum())/1000
    CO_kton_std = np.sqrt((ABCD['CO_STD_MG'] ** 2).sum())/1000
    NOx_kton_std = np.sqrt((ABCD['NOx_STD_MG'] ** 2).sum())/1000

    area = ABCD['AREA_HA'].sum()

    results = ({
    "Year": year,
    "Country": country,
    "Region": region if region else "N/A",
    "Province": province if province else "N/A",
    "Municipality": commune if commune else "N/A",
    "Burn events in year":tot_events,
    "Burn area (ha)": area,
    "GHG (kt)": ghg_kton,
    "GHG Std Dev (kt)": ghg_kton_std,
    "CO2 Emissions (kt)": CO2_kton,
    "CO2 Std Dev (kt)": CO2_kton_std,
    "CH4 (kt)": CH4_kton,
    "CH4 Std Dev (kt)": CH4_kton_std,
    "N2O (kt)": N2O_kton,
    "N2O Std Dev (kt)": N2O_kton_std,
    "PM2.5 (kt)": PM_kton,
    "PM2.5 Std Dev (kt)": PM_kton_std,
    "CO (kt)": CO_kton,
    "CO Std Dev (kt)": CO_kton_std,
    "NOx (kt)": NOx_kton,
    "NOx Std Dev (kt)": NOx_kton_std
})
    
    return results

        #Save data by forest class
def save_total_ghg_emissions_by_forest_type(ABCD, year, country,
                      region,province,commune, tot_events, landcover, language):

    
    # convert std to variance so that they can be summed
    ABCD['TOTEQ_VAR_MG'] = ABCD['TOTEQ_STD_MG'] ** 2
    ABCD['CO2_VAR_MG'] = ABCD['CO2_STD_MG'] ** 2
    ABCD['CH4_VAR_MG'] = ABCD['CH4_STD_MG'] ** 2
    ABCD['N2O_VAR_MG'] = ABCD['N2O_STD_MG'] ** 2
    ABCD['PM2.5_VAR_MG'] = ABCD['PM2.5_MG']  ** 2
    ABCD['NOx_VAR_MG'] = ABCD['NOx_MG']  ** 2
    ABCD['CO_VAR_MG'] = ABCD['CO_MG']  ** 2
    
    
    lc = ABCD.groupby([landcover]).agg({'AREA_HA' : 'sum', landcover+"_NAME_"+language : "first",
    'TOTEQ_MG': 'sum',  'TOTEQ_VAR_MG': 'sum', 'CO2_MG' :'sum', 'CO2_VAR_MG':'sum',
    'CH4_MG':'sum', 'CH4_VAR_MG':'sum', 'N2O_MG'  :'sum', 'N2O_VAR_MG'  :'sum', 
    'PM2.5_MG'  :'sum', 'PM2.5_VAR_MG'  :'sum', 
    'NOx_MG'  :'sum', 'NOx_VAR_MG'  :'sum',
    'CO_MG'  :'sum', 'CO_VAR_MG'  :'sum'
                                       }).reset_index()

      # convert variance back to standard deviation
    lc['TOTEQ_STD_MG'] = np.sqrt(lc['TOTEQ_VAR_MG'])
    lc['CO2_STD_MG'] = np.sqrt(lc['CO2_VAR_MG'])
    lc['CH4_STD_MG'] = np.sqrt(lc['CH4_VAR_MG'])
    lc['N2O_STD_MG'] = np.sqrt(lc['N2O_VAR_MG'])
    lc['PM2.5_STD_MG'] = np.sqrt(lc['PM2.5_VAR_MG'])
    lc['NOx_STD_MG'] = np.sqrt(lc['NOx_VAR_MG'])
    lc['CO_STD_MG'] = np.sqrt(lc['CO_VAR_MG'])

    ghg_kton = lc['TOTEQ_MG']/1000
    CO2_kton = lc['CO2_MG']/1000
    CH4_kton = lc['CH4_MG']/1000
    N2O_kton = lc['N2O_MG']/1000
    PM_kton = lc['PM2.5_MG']/1000
    NOx_kton = lc['NOx_MG']/1000
    CO_kton = lc['CO_MG']/1000
    
    ghg_kton_std = lc['TOTEQ_STD_MG']/1000
    CO2_kton_std = lc['CO2_STD_MG']/1000
    CH4_kton_std = lc['CH4_STD_MG']/1000
    N2O_kton_std = lc['N2O_STD_MG']/1000
    PM_kton_std = lc['PM2.5_STD_MG']/1000
    NOx_kton_std = lc['NOx_STD_MG']/1000
    CO_kton_std = lc['CO_STD_MG']/1000
    
    area = lc['AREA_HA']

    forest_class = lc[landcover]
    forest_label = lc[landcover+"_NAME_"+ language]
    
    results = []
    for _, row in lc.iterrows():
        result = {
            "Year": year,
            "Country": country,
            "Region": region if region else "N/A",
            #"Region": region[_] if region else "N/A",
            #"Province": province[_] if province else "N/A",
            "Municipality": commune[_] if commune else "N/A",
            "Forest Class": row[landcover], 
            "Forest Label": row[landcover + "_NAME_" + language], 
            "Burn events in year": tot_events,
            "Burn area (ha)": area[_],
            "GHG (kt)": ghg_kton[_],
            "GHG Std Dev (kt)": ghg_kton_std[_],
            "CO2 Emissions (kt)": CO2_kton[_],
            "CO2 Std Dev (kt)": CO2_kton_std[_],
            "CH4 (kt)": CH4_kton[_],
            "CH4 Std Dev (kt)": CH4_kton_std[_],
            "N2O (kt)": N2O_kton[_],
            "N2O Std Dev (kt)": N2O_kton_std[_],
            "PM2.5 (kt)": PM_kton[_],
            "PM2.5 Std Dev (kt)": PM_kton_std[_],
            "NOx (kt)": NOx_kton[_],
            "NOx Std Dev (kt)": NOx_kton_std[_],
             "CO (kt)": CO_kton[_],
            "CO Std Dev (kt)": CO_kton_std[_]
        }
        results.append(result)
    
    return results


def plot_map(forest_classes, forest_labels, forest_colors, border,
                                       language,df_burnt_shape,df,crs,
                                       path_to_shapefile_output=None,plot_border=False):
    """
    This function creates: 1) plot of the burnt area and forest classes within it
                           2) bar plot of GHG emissions for each forest class
                           3) table with total GHG emissions per class (to save)
                           4) shapefile with perimeters of forest classes and emissions (to save)
    
    Parameters: 
    - forest_classes, forest_label, forest_colors(list) : list of names of all forest classes, labels and colors
    - path_to_italian_region_shapefile (str) : location of italian region shapefile
    - region (str) : name of region
    - language (string) : if 'English' or 'Italian' 
    - df_burnt_shape (pd.DataFrame): input DataFrame containing burnt area for the fire event
    - df (pd.DataFrame) : GHG emissions by each forest type, as outputted by get_total_ghg_emissions() function
    - crs (str) : coordinate reference system
    - path_to_table_output (str): path to location where to save csv table of land cover classes with GHG emissions (Default is None)
    - path_to_shapefile_output (str): path to location where to save shapefile of land cover classes with GHG emissions (Default is None)
    - plot_region (bool): if plotting region shapefile (Default is False)
    

    Returns:
    - None
    """
    
    # create dataframe for plotting
    df_plot = pd.DataFrame(
    {'_CLASS': forest_classes,
     '_LABEL': forest_labels,
     '_COLOR': forest_colors
    })
    
    df= df[['CLC18', "geometry",
           'AREA_HA', 'FOREST_TYPE', 'COMBUSTION_FACTOR', 'COMBUSTION_FACTOR_STD',
           'BIOMASS', 'BIOMASS_STD', 'CO2_MG', 'CO2_STD_MG', 'CH4_MG', 'CH4_STD_MG', 
            'N2O_MG', 'N2O_STD_MG', 'TOTEQ_MG','TOTEQ_STD_MG']]
    
    
    # convert std to variance so that they can be summed
    df['TOTEQ_VAR_MG'] = df['TOTEQ_STD_MG'] ** 2
    df['CO2_VAR_MG'] = df['CO2_STD_MG'] ** 2
    df['CH4_VAR_MG'] = df['CH4_STD_MG'] ** 2
    df['N2O_VAR_MG'] = df['N2O_STD_MG'] ** 2
    
    # dissolve geometry on landcover classes
    lc = df.dissolve(by=landcover, aggfunc={'AREA_HA' : 'sum',
        'TOTEQ_MG': 'sum',  'TOTEQ_VAR_MG': 'sum', 'CO2_MG' :'sum', 'CO2_VAR_MG':'sum',
        'CH4_MG':'sum', 'CH4_VAR_MG':'sum', 'N2O_MG'  :'sum', 'N2O_VAR_MG'  :'sum'      
    })

    #lc= df
    
    # convert variance back to standard deviation
    lc['TOTEQ_STD_MG'] = np.sqrt(lc['TOTEQ_VAR_MG'])
    lc['CO2_STD_MG'] = np.sqrt(lc['CO2_VAR_MG'])
    lc['CH4_STD_MG'] = np.sqrt(lc['CH4_VAR_MG'])
    lc['N2O_STD_MG'] = np.sqrt(lc['N2O_VAR_MG'])
    
    lc = lc.reset_index()
    
    #make sure landcover is in string format in both tables
    df_plot['_CLASS'] = df_plot['_CLASS'].astype(str)
    lc[landcover] = lc[landcover].astype(str)
    
    #make sure landcover and burnt area are in the same CRS
    lc = lc.to_crs(crs)
    df_burnt_shape = df_burnt_shape.to_crs(crs)
    
    ghg_in_lc_plot = pd.merge(lc,df_plot, left_on=landcover,right_on="_CLASS", how="inner")
    ghg_in_lc_plot["CLASS_LABEL"] = ghg_in_lc_plot["_CLASS"] + " - " + ghg_in_lc_plot["_LABEL"]

    # PLOT1: BURNT AREA
    if df_burnt_shape.empty or ghg_in_lc_plot.empty:
        print("Cannote plot empty shapefile.")
    else:
        if plot_border:
            fig, axs = plt.subplots(figsize=(8, 5))
            border = border.to_crs(crs)
            border.plot(ax=axs, facecolor='none', edgecolor='k', linewidth=1)  
            #plot burnt shape
            #df_burnt_shape.plot(ax=axs, facecolor='none', edgecolor='red', linewidth=1, ls="--")
            #plot land cover classes
            ghg_in_lc_plot.plot(ax=axs,facecolor='none', edgecolor='red', linewidth=1)
            ghg_in_lc_plot.plot(ax=axs, label=ghg_in_lc_plot._LABEL, color=ghg_in_lc_plot._COLOR,legend=True)
        else:
            fig, axs = plt.subplots(figsize=(8, 5))
            #plot burnt shape
            #df_burnt_shape.plot(ax=axs, facecolor='none', edgecolor='red', linewidth=1, ls="--")  
            #plot land cover classes
            ghg_in_lc_plot.plot(ax=axs,facecolor='none', edgecolor='red', linewidth=2)
            ghg_in_lc_plot.plot(ax=axs, label=ghg_in_lc_plot._LABEL, color=ghg_in_lc_plot._COLOR,legend=True)
    
        shapes = []
        lab =[]
            
        #Create legend patches
        for color, label, ghg, ghg_std, fclass in zip(ghg_in_lc_plot._COLOR,ghg_in_lc_plot.CLASS_LABEL,
                                                      ghg_in_lc_plot.TOTEQ_MG, ghg_in_lc_plot.TOTEQ_STD_MG, ghg_in_lc_plot._CLASS):
             p = matplotlib.patches.Patch(facecolor=color, edgecolor='gray')
             l = label +" - " + str(np.round(ghg/1000,2)) + " ktonnes CO2eq"
             shapes.append(p)
             lab.append(l)
            
        #create legend, by first getting the already present handles, labels
        handles, labels = plt.gca().get_legend_handles_labels()
        
        #and then adding the new ones
        handles.extend(shapes)
        labels.extend(lab)
        
        by_label = dict(zip(labels, handles))
    
        #plot legend
        axs.legend(by_label.values(), by_label.keys(), framealpha=1.,fontsize="medium",bbox_to_anchor=(0.5, -0.4), loc='center', ncol=1)
    
        plt.show()
    
    if path_to_shapefile_output:
            ghg_in_lc_plot.to_file(path_to_shapefile_output)
            print(f"GHG per forest type shapefile save to '{path_to_shapefile_output}'")

    return ghg_in_lc_plot


def plot_barplot(ghg_in_lc_plot):
    import matplotlib.cm as cm
    # PLOT2: BARPLOT
    num_classes = len(ghg_in_lc_plot["_LABEL"])
    colors = cm.turbo(np.linspace(0, 1, num_classes))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    plt.bar(ghg_in_lc_plot["_LABEL"], ghg_in_lc_plot['TOTEQ_MG']/1000, color=colors)
    #plt.xlabel('CLC18 Class')
    plt.ylabel('Total GHG (ktonnes)')
    plt.title('Total GHG emissions per forest type')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_stacked(df,region):
    df = databs[databs['Region'] == region]

    # Sum 'GHG (kt)' by 'YEAR' and 'FOREST LABEL'
    df_sum = df.groupby(['Year', 'Forest Label'])['GHG (kt)'].sum().unstack(fill_value=0)
    
    # Plotting the stacked bar plot
    ax = df_sum.plot(kind='bar', stacked=True, figsize=(8, 6), colormap='tab20b',edgecolor='k')
    
    # Adding labels and title
    ax.set_title('Total GHG Emissions (2018-2023) for ' +region, fontsize=14)
    ax.set_xlabel('', fontsize=14)
    ax.set_ylabel('GHG Emissions (ktCO2e)', fontsize=14)
    
    # Adding a grid and frame for better aesthetics
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=14)
    #plt.tight_layout()
    plt.show()


def make_all_fires_shp(df):
    

    df= df[['id','CLC18', "geometry",
           'AREA_HA', 'FOREST_TYPE', 'COMBUSTION_FACTOR', 'COMBUSTION_FACTOR_STD',
           'BIOMASS', 'BIOMASS_STD', 'CO2_MG', 'CO2_STD_MG', 'CH4_MG', 'CH4_STD_MG', 
            'N2O_MG', 'N2O_STD_MG', 'TOTEQ_MG','TOTEQ_STD_MG']]
    
    # convert std to variance so that they can be summed
    df['TOTEQ_VAR_MG'] = df['TOTEQ_STD_MG'] ** 2
    df['CO2_VAR_MG'] = df['CO2_STD_MG'] ** 2
    df['CH4_VAR_MG'] = df['CH4_STD_MG'] ** 2
    df['N2O_VAR_MG'] = df['N2O_STD_MG'] ** 2
    
    # dissolve geometry on landcover classes
    lc = df.dissolve(by=['id','CLC18'], aggfunc={'AREA_HA' : 'sum',
        'TOTEQ_MG': 'sum',  'TOTEQ_VAR_MG': 'sum', 'CO2_MG' :'sum', 'CO2_VAR_MG':'sum',
        'CH4_MG':'sum', 'CH4_VAR_MG':'sum', 'N2O_MG'  :'sum', 'N2O_VAR_MG'  :'sum'      
    })

    
    # convert variance back to standard deviation
    lc['TOTEQ_STD_MG'] = np.sqrt(lc['TOTEQ_VAR_MG'])
    lc['CO2_STD_MG'] = np.sqrt(lc['CO2_VAR_MG'])
    lc['CH4_STD_MG'] = np.sqrt(lc['CH4_VAR_MG'])
    lc['N2O_STD_MG'] = np.sqrt(lc['N2O_VAR_MG'])
    
    lc = lc.reset_index()

    return lc


# Define a function to get the category based on the region
def get_category(region):
    # If the region is None, return an empty string
    if region is None:
        return ""
    
    # Define lookups for each region
    categories = {
        "_NW": ['piemonte', 'valle d\'aosta', 'liguria', 'lombardia'],
        "_NE": ['alto adige', 'trentino', 'friuli venezia giulia', 'emilia-romagna', 'veneto'],
        "_CE": ['toscana', 'umbria', 'marche', 'lazio'],
        "_SE": ['abruzzo', 'puglia', 'molise', 'basilicata'],
        "_SW": ['campania', 'calabria'],
        "_SIC": ['sicilia'],
        "_SAR": ['sardegna']
    }

    # Loop through the categories and match the region
    for category, regions in categories.items():
        if region.lower() in [r.lower() for r in regions]:
            return category
    
    # If the region is not found in any category, raise an error
    raise ValueError(f"Region '{region}' not found in any category")

