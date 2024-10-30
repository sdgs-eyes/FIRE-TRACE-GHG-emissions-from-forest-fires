"""
@Author: Chiara Aquino
@Date : 02 February 2024

Functions to calculate GHG emissions using model from Chiriacò et al.(2013)

"""

def import_data(dataname,path_to_data_location,**kwargs):

    """
    Retrieve burnt area polygon from the shapefile and filter it based on selected columns and values.

    Parameters:
    - dataname (string) : name of the input data (e.g., if EFFIS, SERCO)
    - path_to_data_location (string) : location of the shapefile, 
        if Effis, downloaded from https://effis.jrc.ec.europa.eu/applications/data-and-services 
    - *kwargs: Optional arguments for column name and corresponding values.

    Returns:
    - pd.DataFrame: Filtered DataFrame with selected burnt area.
    """


    import geopandas
    import pandas as pd

    #open shapefile as DataFrame
    df = geopandas.read_file(path_to_data_location)
    
    #Copy the original DataFrame
    filtered_df = df.copy()
    
    if dataname == "EFFIS":
        
        #extract year from EFFIS date column so that the DataFrame can be filtered by year only
        filtered_df['FIREDATE'] = pd.to_datetime(filtered_df['FIREDATE'], format="mixed")
        filtered_df['YEAR'] = filtered_df['FIREDATE'].dt.year
        
        #rename id column to avoid confusion
        filtered_df.rename(columns={'id': 'ID'}, inplace=True)
        
        #filter by selected columns specified in optional arguments
        for column, value in kwargs.items():
            
            # filter only for values that have been declared in function
            if value is not None:
                filtered_df = filtered_df[filtered_df[column] == value]
                
                #Check if the specified value exists in the filtered DataFrame
                if filtered_df.empty:
                    raise ValueError(f"No data found where '{column}' is '{value}'.")
    
    elif dataname == "SERCO":

        #filter by selected columns specified in optional arguments
        for column, value in kwargs.items():
            
            # filter only for values that have been declared in function
            if value is not None:
                filtered_df = filtered_df[filtered_df[column] == value]
                
                #Check if the specified value exists in the filtered DataFrame
                if filtered_df.empty:
                    raise ValueError(f"No data found where '{column}' is '{value}'.")
        
    else:
    
        print("Unrecognised data name")
    
    return filtered_df
        

def get_landcover_classes(landcover,path_to_landcover_legend,language):

    """
    Retrieve landcover classes from the shapefile and filter it based on selected columns and values.

    Parameters:
    - landcover (string): name of landcover type to use (e.g., if Corine or EFFIS)
    - path_to_data_location (string) : location of landcover legend table
    - language (string) : if 'English' or 'Italian'

    Returns:
    - classes (list of str)
    - names (list of str) 
    - colors (list of int) 
    """


    import geopandas
    import pandas as pd

    # get all classes, labels and associated colors
    landcover_legend = pd.read_csv(path_to_landcover_legend)
    classes = landcover_legend[landcover+"_CODE"].values.T.tolist()
    names = landcover_legend[landcover+"_NAME_"+language].values.T.tolist()
    colors = landcover_legend[landcover+"_COLOR"].values.T.tolist()

    # drop Nan in the list
    classes = [x for x in classes if not pd.isna(x)]
    names = [x for x in names if not pd.isna(x)]
    colors = [x for x in colors if not pd.isna(x)]
    
    # Transform integers to strings
    classes = [str(i) for i in classes]
  
    return classes, names, colors  

def get_clc18_forest_types_in_burnt_shape(path_to_clc18_shapefile, df_burnt_area_shape, forest_classes,crs):

    """
    Filter corine (clc18) landcover classes by clipping corine landcover with the burnt area shapefile

    Parameters:
    - path_to_clc18_shapefile (string):  location of corine shapefile 
    - df_burnt_area_shape (pd.DataFrame): input DataFrame containing burnt area for the fire event 
    - forest_classes (list of str): forest classes 
    - crs (str) : chosen coordinate system, e.g "epsg:4326" 

    Returns:
    - CLC18_burnt_area : Filtered geopandas DataFrame with corine land cover classes in burnt area
    """

    import geopandas

    # Read corine shapefile
    CLC18 = geopandas.read_file(path_to_clc18_shapefile)

    #make sure both corine and the burnt area shapefile are in the same crs
    CLC18 = CLC18.to_crs(crs)
    df_burnt_area_shape= df_burnt_area_shape.to_crs(crs)

    #clip Corine with the burnt area shapefile
    CLC18_burnt_area = geopandas.clip(CLC18, df_burnt_area_shape)

    return CLC18_burnt_area


def get_clc18_areas(path_to_clc18_shapefile, df_burnt_area_shape, forest_classes,crs):

    """
    Calculate areas of each corine landcover class within burnt area

    Parameters:
    - path_to_clc18_shapefile (string):  location of corine shapefile 
    - df_burnt_area_shape (pd.DataFrame): input DataFrame containing burnt area for the fire event 
    - forest_classes (list of str): forest classes 
    - crs (str) : chosen coordinate system, e.g "epsg:4326" 

    Returns:
    - clc18_sum_areas_by_forest_class : DataFrame containing areas of each corine land cover class in burnt area
    """

    CLC18_burnt_area_forest_only = get_clc18_forest_types_in_burnt_shape(path_to_clc18_shapefile, df_burnt_area_shape, forest_classes,crs)
    #groupby clc18 forest class, sum and convert from m2 to HA
    df_sum_areas_by_forest_class = CLC18_burnt_area_forest_only.groupby('CLC18')['Shape_Area'].sum() / 10000

    # rename index so it corresponds to EFFIS area table
    df_sum_areas_by_forest_class =  df_sum_areas_by_forest_class.reset_index()
    df_sum_areas_by_forest_class['CLC18'] = df_sum_areas_by_forest_class['CLC18'] + "_AREA_HA"

    #get all the other forest classes from Corine and order them in a table that can be joined to the previous one
    df_forest_classes = pd.DataFrame(columns=['CLC18', 'Shape_Area'])
    df_forest_classes['CLC18'] = forest_classes 
    df_forest_classes['CLC18'] = df_forest_classes['CLC18'] + "_AREA_HA"

    #merge the two tables so we have ALL forest classes, with area=0 for the classes that do not exist in burnt area
    df_all_forest_classes = pd.merge(df_sum_areas_by_forest_class, df_forest_classes, how='outer')
    # eliminate the duplicates, but only for those that have area > 0 
    df_all_forest_classes = df_all_forest_classes.drop_duplicates(subset=['CLC18'], keep='first') 
    df_all_forest_classes = df_all_forest_classes.fillna(0) #convert NANs to 0

    #organise the table so that it matches the same structure of EFFIS table
    df_all_forest_classes = df_all_forest_classes.set_index('CLC18')
    df_all_forest_classes =  df_all_forest_classes.stack(level=0)
    clc18_sum_areas_by_forest_class = df_all_forest_classes.droplevel(1)

    return clc18_sum_areas_by_forest_class

def get_effis_areas(df_burnt_area_shape, forest_classes):

    """
    Calculate areas of each EFFIS landcover class within burnt area

    Parameters:
    - df_burnt_area_shape (pd.DataFrame): input DataFrame containing burnt area for the fire event 
    - forest_classes (list of str): forest classes 

    Returns:
    - effis_sum_areas_by_forest_class : DataFrame containing areas of each EFFIS land cover class in burnt area
    """
    
    for forest_class in forest_classes: 

        # Burnt areas in EFFIS for each forest class are reported in percentage units
        # To get areas for each burnt forest type in Ha, we need to 
        # multiply the proportion of each burnt forest type by total burnt area in Ha
        df_burnt_area_shape[forest_class+'_AREA_HA'] = df_burnt_area_shape[forest_class].astype(float)/ 100 * df_burnt_area_shape['AREA_HA'].astype(float) 
        
        # Retain in DataFrame only columns with burnt areas in hectares for each forest type
        df_areas_by_forest_class = df_burnt_area_shape.filter(regex='_AREA_HA')
    
        # Get TOTAL burnt areas in hectares for each forest type by summing over each column
        effis_sum_areas_by_forest_class = df_areas_by_forest_class.astype(float).sum(axis=0)

    return effis_sum_areas_by_forest_class

def get_total_burnt_area(landcover,path_to_clc18_shapefile, df_burnt_area_shape, forest_classes,crs):
    """
    Retrieve total burnt area for each forest class deciding if input is Corine or Effis

    Parameters:
    - landcover (string): name of landcover type to use (e.g., if Corine or EFFIS)
    - path_to_clc18_shapefile (string):  location of corine shapefile 
    - df_burnt_area_shape (pd.DataFrame): input DataFrame containing burnt area for the fire event 
    - forest_classes (list of str): forest classes 
    - crs (str) : chosen coordinate system, e.g "epsg:4326" 

    Returns:
    -  pd.DataFrame: Processed DataFrame containing total burnt area in Ha for each forest class
    """

    #if landcover is effis, retrieve function to calculate area in effis landcover classes
    if input_data == "EFFIS":
        if landcover=="EFFIS":
            df_sum_areas_by_forest_class = get_effis_areas(df_burnt_area_shape, forest_classes)
            
        #if landcover is corine, retrieve function to calculate area in corine landcover classes
        elif landcover=="CLC18":
            df_sum_areas_by_forest_class = get_clc18_areas(path_to_clc18_shapefile, df_burnt_area_shape, forest_classes,crs)
        
    else:
        df_sum_areas_by_forest_class = get_clc18_areas(path_to_clc18_shapefile, df_burnt_area_shape, forest_classes,crs)
            
    return df_sum_areas_by_forest_class
    
def get_biomass(path_to_lookup_table, path_to_biomass_table, landcover, region=None,):
    """
    Retrieve pre disturbance biomass for each vegetation type
    Data is derived from average standing volume estimates from National Forest Inventory 2015 (INFC2015). INFC2015 classes have been 
    averaged for each of the 20 Italian administrative regions to match EFFIS or CLC18 vegetation classes

    Parameters:
    - path_to_lookup_table (str) : location of lookup tables for different landcover classes -> INFC2015 classes
    - path_to_biomass_table (str) : location of National Forest Inventory 2015 biomass values, per vegetation type, per each Italian region 
    - landcover (str) : landcover type to use (CLC18 or EFFIS)
    - region (str): Italian region of interest. Default region is None. 
    
    Returns:
    - pd.DataFrame: Processed DataFrame with average biomass values per selected region per vegetation class
    """
    import pandas as pd

    lookup = pd.read_csv(path_to_lookup_table)
    biomass = pd.read_csv(path_to_biomass_table)

    # Drop the first row from the biomass table, as this contains forest types as strings (not needed)
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

    # Make sure that the INFC codes are integer columns, so we can correctly match them from the two tables
    lookup['INFC_ID'] = lookup['INFC_ID'].astype(int)
    biomass['INFC_ID'] = biomass['INFC_ID'].astype(int)

    # From the lookup table, select only data for the landcover type we are using
    lookup = lookup[[landcover+"_CLASS","INFC_ID"]]

    # get biomass calculation by landcover type
    # 1) merge the two dataframes on common "INFC_Code" (ie, vegetation classes from INFC)
    biomass_by_landcover = pd.merge(lookup, biomass, on='INFC_ID')

    # Drop duplicate rows and sort table by landcover class and vegetation ID
    biomass_by_landcover = biomass_by_landcover.drop_duplicates().sort_values(by=[landcover+"_CLASS","INFC_ID"])

    #make sure the code is a string, so that we can use it later
    biomass_by_landcover[landcover+"_CLASS"] = biomass_by_landcover[landcover+"_CLASS"].astype(str)

    # 2) take average of values for each region grouped by landcover class
    # pandas skips Nan values by default
    grouped = biomass_by_landcover.groupby([landcover+"_CLASS"]).mean()

    #3) calculate biomass for the region selected (default is None)
    # if region is specified, then select the column containing values for that region
    if region is not None:
        biomass_by_region = grouped[region]
    
    # if region is not specified (if region left as None) get average values for Italy
    else:
        biomass_by_region = grouped['Italia']
        
    return biomass_by_region


def get_combustion_factor(path_to_bovio_conversion_table,path_to_fire_damage_table,landcover,scorch_height=None):
    """
    Retrieve combustion factor for each forest type.
    These value are retrieved from Table 4 and Table 5 in Chiriacò et al.(2013), in turn taken from Bovio et al.(2007) 
    where vegetation classes need to be matched to EFFIS forest types.

    Parameters:
    - path_to_bovio_conversion_table (str) : location of lookup table matching vegetation classes in Table 4 (Chiriacò et al.,2013) with EFFIS/CLC18 vegetation classes
    - path_to_fire_damage_table (str) : location of scorch height table, corresponding to Table 5 (Chiriacò et al.,2013)
    - landcover (str) : landcover type to use (CLC18 or EFFIS)
    - scorch_height (int): height of the flame, as specified by the user. Default is None.

    Returns:
    - pd.DataFrame: Processed DataFrame with combustion factor values per vegetation type
    """
    import pandas as pd
    
    #read in table with conversion between EFFIS forest types-BOVIO vegetation classes 
    bovio_df= pd.read_csv(path_to_bovio_conversion_table)
    #read in table with fire damage values
    fire_damage_df= pd.read_csv(path_to_fire_damage_table)

    #Merge the two tables, to match EFFIS forest types to fire_damage table
    merged = pd.merge(bovio_df, fire_damage_df, on= "BOVIO_CLASS")

    #Merge the two tables, to match EFFIS forest types to fire_damage table
    merged = pd.merge(bovio_df, fire_damage_df, on= "BOVIO_CLASS")

    #Fire intensity columns that we need to average for each vegetation class
    cols = ['<1', '1-2.5','2.5-3.5','3.5-4.5','>4.5']
    
    #drop Bovio vegetation classes and eliminate duplicates
    merged = merged[[landcover+'_CLASS','BOVIO_CLASS','<1', '1-2.5','2.5-3.5','3.5-4.5','>4.5']].drop_duplicates().sort_values(by=[landcover+'_CLASS','BOVIO_CLASS'])

    #group by EFFIS forest class and calculate average of fire damage values for each EFFIS forest class
    grouped = merged.groupby([landcover+'_CLASS']).mean(cols).reset_index()

    grouped[landcover+"_CLASS"] = grouped[landcover+"_CLASS"].astype(str)

    #Now from this table, select the fire damage value that correspond to the scorch height 
    
    #if scorch height values are not specified (scorch_height = None) take average of the highest two fire damage classes
    if scorch_height is None:
        grouped['COMBUSTION_FACTOR']  = grouped[['3.5-4.5', '>4.5']].mean(axis=1)
    
    # if they are specified, match scorch height values with its corresponding fire damage column
    else:
        if scorch_height < 1:
            grouped['COMBUSTION_FACTOR']  = grouped['<1']
        elif 1 <= scorch_height < 2.5:
            grouped['COMBUSTION_FACTOR']  = grouped['1-2.5']
        elif 2.5 <= scorch_height < 3.5:
            grouped['COMBUSTION_FACTOR']  = grouped['2.5-3.5']
        elif 3.5 <= scorch_height < 4.5:
            grouped['COMBUSTION_FACTOR']  = grouped['3.5-4.5']
        else:
            grouped['COMBUSTION_FACTOR']  = grouped['>4.5']

    #select only forest classes and combustion factor for the specified scorch height
    grouped = grouped[[landcover+'_CLASS','COMBUSTION_FACTOR']]

    # Set the vegetation class column as the index, then tidy up table so that it contains only fire damage and forest classes
    combustion_factor_df = grouped.set_index(landcover+'_CLASS').T.astype(object)
    combustion_factor_df = combustion_factor_df.reset_index(level=0, drop=True)
    combustion_factor_df = combustion_factor_df.rename_axis(None, axis=1)

    return combustion_factor_df

def get_total_annual_GHG_emissions(A,B,C,path_to_emission_factors_table,forest_classes):
    """

    This function puts together all the previous steps of the model and calculates final GHG emissions. 
    
    Parameters: 
    - A (pd.DataFrame) : burnt area for each forest class, as retrieved by function get_total_burnt_area()
    - B (pd.DataFrame) : pre disturbance biomass for each vegetation type, as retrived by function get_biomass()
    - C (pd.DataFrame) : combustion factor for each forest type, as retrieved by function get_combustion_factor()
    - path_to_emission_factors_table (str) : location of table containing GHG emission values (as from IPCC2003)
    - forest_types (list of str) : forest classes 

    Returns:
    - Float: total_emissions_co2eq (total GHG emissions)
    - pd.Dataframe: emissions_by_forest_type_co2eq (GHG emissions by forest class)
    """
    import pandas as pd
    import numpy as np
    
    #### CALCULATE MODEL PARAMETERS #######
    
    # Multiply A X B X C
    # Create empty dataframe to calculate burnt AGB (Above ground biomass) in burnt area 
    df_burnt_biomass_in_area = pd.DataFrame(columns=forest_classes)

    ABC=[]
    # Calculate burnt AGB of burnt area for each forest type. Multiply x 1e3 to convert AGB values from Mg to Kg
    for forest_class in forest_classes:
        row = A[forest_class+"_AREA_HA"] * B[forest_class] * 1000 * C[forest_class][0]
        ABC.append(row)

    df_burnt_biomass_in_area.loc[len(df_burnt_biomass_in_area)] = ABC
    
    # Get emissions for each GHG compound (D)
    # Read table with emission factors for each GHG compound
    df_emission_factors = pd.read_csv(path_to_emission_factors_table)

    #convert Dataframes to array to perform pair-wise multiplication between GHG values and forest types
    array_df_burnt_biomass_in_area = df_burnt_biomass_in_area.to_numpy()
    array_df_emission_factors = df_emission_factors.to_numpy()

    # Perform element-wise multiplication between forest types and GHG emission factors
    result_array = np.multiply.outer(array_df_emission_factors, array_df_burnt_biomass_in_area)
    
    #convert resulting array into Dataframe, and rename columns using forest types
    df_ghg = pd.DataFrame(result_array.reshape(array_df_emission_factors.shape[1], array_df_burnt_biomass_in_area.shape[1]),       columns=df_burnt_biomass_in_area.columns)

    #rename rows with names of GHG emission factors
    df_ghg.index = df_emission_factors.columns.values.tolist()
    
    #multiply all factors by 10^(-6) to get emission factors into units of kgtons
    df_ghg_kton = df_ghg * 1e-6

    #convert CH4 and N2O in GWP units CO2 equivalent
    df_ghg_kton.loc['CH4'] =  df_ghg_kton.loc['CH4'] * 28
    df_ghg_kton.loc['N2O'] =  df_ghg_kton.loc['N2O'] * 273
    
    #select only relevant GHG compounds (exclude precursors)
    ghg_co2eq_columns = ['CO2', 'CH4', 'N2O']
    
    # sum GHG by forest types
    emissions_by_forest_type_co2eq = df_ghg_kton.loc[ghg_co2eq_columns, :].sum()
    
    # sum over ALL forest types to get final GHG per burnt area
    total_emissions_co2eq = emissions_by_forest_type_co2eq.sum()
    
    return total_emissions_co2eq, emissions_by_forest_type_co2eq