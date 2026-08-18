This readme-file contains information on the data generation for the hydrological processes practicals.
In addition, the folder contains the scripts to generate the data for the practicals of HP.

- Convert_data_oldPRacticals.py
    Python script to convert xlsx-files to csv-files. More information can be found in this script.

- Generation of the Strahler orders.
    Makes use of the DEM_DEMER_ClopGLO30_cropped.tif-file. 
    The pipeline described in https://www.youtube.com/watch?v=2Ub0c7Ss-T4&list=PLeuKJkIxCDj17t2MmPhBZIkLI2FQ8S_ea&index=4 (Last accessed: 2026-08-18)
    is used to fill the gaps in the DEM, and further derive the channels and Strahler orders.

    Next, the Upslope area tool (https://youtu.be/2Ub0c7Ss-T4?list=PLeuKJkIxCDj17t2MmPhBZIkLI2FQ8S_ea&t=2785, last accessed: 2026-08-18) is used to derive the subcatchments
    for Boutersem and Zoutleeuw, starting from the generated Strahler orders.

Data sources:
    - Hydrologische_processen\Data\data_download\catchment_template_bare_ZdenkoShanon.xlsx: 
        - accessed from the KULeuven drives 
        - more info; contact Gabrielle.delannoy@kuleuven.be
    
    - QGIS data:
        - Vlaanderen_prov.shp; Vlaanderen_waterlopen.shp; Demer_stroomgebied.shp: 
            - Accesed from the KULeuven drives
            - more info; contact Gabrielle.delannoy@kuleuven.be
        
        - Vlaanderen_grens.gpkg:
            - Created from Vlaanderen_prov.shp using dissolve.
            - more info; contact lucas.boeykens@kuleuven.be

        - DEM_Demer_CopGLO30_cropped.tif and DEM_Demer_CopGLO90.tif:
            - Accessed using the SRTM Downloader plugin of QGIS.
            - Data clipped to the Demer_stroomgebied.shp-file
            - more info; contact lucas.boeykens@kuleuven.be
        
        - Strahler_Demer_CopGLO30.gpkg, Demer_Boutersem(Zoutleeuw)_strhler.gpkg and Demer_Boutersem(Zoutleeuw).shp
            - See description above
            - more info; contact lucas.boeykens@kuleuven.be           


File created on Tuesday 18/08/26 by Lucas Boeykens - lucas.boeykens@kuleuven.be lucas.boeykens@ugent.be