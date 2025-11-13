# Projections of global sea-level change due to ice sheets

The code in this directory will generate and plot projected changes to global mean sea level due to melting of the world's ice sheets.

## Environment configuration
An `environment.yml` file is included to help configure the working environment for this code. Using your installation of conda, create and activate the virtual environment:

```
conda env create -f environment.yml
conda activate project_icesheet
```

## Run the projection generation code
Run the script `generate_icesheet_projections.py` from the command line with your preferred runtime options. For example:

```
python generate_icesheet_projections.py --nsamps 500 --scenario rcp85 --srocc --seed 1234
```

This code will generate netCDF4 files in the `data` directory for each of the world's ice sheets (GIS = Greenland ice sheet, EAIS = East Antarctic ice sheet, WAIS = West Antarctic ice sheet, AIS = Antarctic ice sheet). Each file will contain 500 samples drawn from fitted distributions of possible sea-level change for an RCP8.5 emissions future. Setting the `--srocc` flag will use estimate ice sheet melt rates calibrated to the IPCC Special Report on the Ocean and Cryosphere in a Changing Climate. Not setting the flag will use estimates based off of the IPCC Fifth Assessment Report. Setting the `seed` value will ensure reproducible results. Use `python generate_icesheet_projections.py --help` for a help menu.

Output from the above run will be stored in the `data` directory (which will be created at runtime if the directory does not currently exist). The file names will indicate which ice sheet and emissions scenario were used in making the projections.

## Visualize the projections
Use the `plot_icesheet_projections.py` script to generate a plot of the projected sea-level change for a given ice sheet and emissions scenario (i.e. one of the generated output files from the projection generation code). For example, using output generated from the example above:

```
python plot_icesheet_projections.py --infile ./data/GIS_rcp85_globalsl.nc
```

This will generate a plot in PDF format in the `plots` directory (which will be created at runtime if the directory does not already exist). 