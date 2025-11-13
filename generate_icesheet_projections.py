import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

import constants
from CalcISDists import CalcISDists as calcDists
from ProjectGSL import ProjectGSL
from SampleISDists import SampleISDists


def preprocess_icesheets(
    rcp_scenario: str,
    use_srocc: bool,
) -> tuple[dict, dict]:
    """Preprocesses data for ice sheet projections

    Generates the rate and correlation data used in projecting
    the sea-level change contributions from global ice sheets.

    Args:
        rcp_scenario (str): Future emissions scenario
        use_srocc (str): Replace the IPCC AR5 calibration with SROCC

    Return:
        Two dictionaries. The first is the ice sheet rate data and
        the second is the correlation information.
    """

    # Fill the rates data matricies
    barates = constants.BARATES
    lastdecadegt = constants.LASTDECADEGT

    # Use rates for selected calibration
    if use_srocc:
        aris2090 = constants.SROCCARIS
    else:
        aris2090 = constants.K14ARIS

    # Fill the correlation data matricies
    bacorris = constants.BACORRIS
    arcorris = constants.ARCORRIS

    # Which RCP scenario does the user want?
    rcp_ind = ["rcp85", "rcp60", "rcp45", "rcp26"].index(rcp_scenario)

    # Collate rates data into a single dictionary
    data_is = {
        "barates": barates,
        "lastdecadegt": lastdecadegt,
        "aris2090": aris2090[rcp_ind, :, :],
    }

    # Collate the correlation data into a single dictionary
    corr_is = {"bacorris": bacorris, "arcorris": arcorris, "scenario": rcp_scenario}

    return (data_is, corr_is)


def fit_icesheets(rates_dict: dict) -> dict:
    """Fit distributions to the ice sheet rates

    Generates moments of a log-normal distribution fitted
    to the provided quantiles of the ice sheet contribution rates

    Args:
        rates_dict (dict): Dictionary of ice sheet rates from preprocessing stage

    Return:
        Dictionary of fitted log-normal moments
    """

    # Fit the distributions
    (batheteais, bathetwais, bathetgis, arthetais, arthetgis, islastdecade) = calcDists(
        rates_dict["barates"], rates_dict["lastdecadegt"], rates_dict["aris2090"]
    )

    # Put the results into a dictionary
    fit_dict = {
        "batheteais": batheteais,
        "bathetwais": bathetwais,
        "bathetgis": bathetgis,
        "arthetais": arthetais,
        "arthetgis": arthetgis,
        "islastdecade": islastdecade,
    }

    return fit_dict


def project_icesheets(
    nsamps: int,
    fit_dict: dict,
    corr_dict: dict,
    seed: int,
) -> dict:
    """Project global contributions to sea level from ice sheets

    Generates projections of global sea-level change from ice sheets
    from the provided fitted distributions and correlation data.

    Args:
        nsamps (int): Number of samples to generate
        fit_dict (dict): Dictionary of fitted data from the 'fit' stage
        corr_dict (dict): Dictionary of correlation data from the
            'preprocess' stage
        seed (int): Seed for the random number generator

    Return:
        A dictionary containing samples drawn from the ice sheet distributions

    """

    # Set the parameters for the projection years
    baseyear = constants.BASEYEAR
    pyear_start = constants.PYEAR_START
    pyear_end = constants.PYEAR_END
    pyear_step = constants.PYEAR_STEP

    # Extract the fitted data
    batheteais = fit_dict["batheteais"]
    bathetwais = fit_dict["bathetwais"]
    bathetgis = fit_dict["bathetgis"]
    arthetais = fit_dict["arthetais"]
    arthetgis = fit_dict["arthetgis"]
    islastdecade = fit_dict["islastdecade"]

    # Extract the correlation data
    bacorris = corr_dict["bacorris"]
    arcorris = corr_dict["arcorris"]

    # Generate samples of ice sheet accelerations
    sigmas = np.array([bathetgis[2], bathetwais[2], batheteais[2]])
    mus = np.array([bathetgis[1], bathetwais[1], batheteais[1]])
    offsets = np.array([bathetgis[0], bathetwais[0], batheteais[0]])
    baissamps = SampleISDists(
        nsamps, sigmas, mus, offsets, islastdecade, bacorris, seed
    )

    sigmas = np.array([arthetgis[2], arthetais[2]])
    mus = np.array([arthetgis[1], arthetais[1]])
    offsets = np.array([arthetgis[0], arthetais[0]])
    arislastdecade = np.array([islastdecade[0], islastdecade[1] + islastdecade[2]])
    arissamps = SampleISDists(
        nsamps, sigmas, mus, offsets, arislastdecade, arcorris, seed + 1234
    )

    # Project global sea-level rise over time
    targyears = np.arange(pyear_start, pyear_end + 1, pyear_step)
    targyears = np.union1d(targyears, baseyear)
    (arsamps, basamps, hysamps) = ProjectGSL(
        baissamps, arissamps, islastdecade, targyears
    )

    # Reference these projections to the base year
    baseyear_idx = np.flatnonzero(targyears == baseyear)
    arsamps = arsamps - arsamps[:, baseyear_idx, :]
    basamps = basamps - basamps[:, baseyear_idx, :]
    hysamps = hysamps - hysamps[:, baseyear_idx, :]

    # Put the results into a dictionary
    proj_dict = {
        "arsamps": arsamps,
        "basamps": basamps,
        "hysamps": hysamps,
        "targyears": targyears,
        "baseyear": baseyear,
    }

    return proj_dict


def writeNetCDF(
    data: float,
    icesheet_name: str,
    targyears: int,
    baseyear: int,
    nsamps: int,
    scenario: str,
) -> None:
    """Write the data to a netcdf file

    Generates a netCDF4 file on disk from the sample data provided.

    Args:
        data (float): Sample data as extracted from the dictionary
            produced by the 'projection' stage
        icesheet_name (str): Name of the ice sheet data being stored
        targyears (int): Array of integer years for which projections are valid
        baseyear (int): Year to which projection data are referenced
        nsamps (int): Number of samples in the provided data
        scenario (str): Future emissions scenario of projected data

    Return:
        None. A netCDF4 file is generated in the script's root directory
    """

    # Create the output directory if needed
    outdir = Path(Path(__file__).parent, "data")
    outdir.mkdir(parents=True, exist_ok=True)

    # Define the output file path
    nc_filename = Path(outdir, f"{icesheet_name}_{scenario}_globalsl.nc")

    # Build the xarray data structure
    outds = xr.Dataset(
        {"sea_level_change": (["samples", "time"], data)},
        coords={"samples": np.arange(nsamps), "time": targyears, "baseyear": baseyear},
        attrs={
            "description": f"Global sea-level change projections from {icesheet_name}",
            "history": f"Created {time.ctime(time.time())}",
            "scenario": scenario,
        },
    )
    outds["sea_level_change"].attrs = {"units": "mm"}

    # Write the data to disk
    compression_encoding = {"zlib": True, "complevel": 4}
    outds.to_netcdf(
        nc_filename,
        encoding={var: compression_encoding for var in outds.data_vars},
    )

    return None


def main(nsamps: int, scenario: str, use_srocc: bool, seed: int) -> None:
    """Main function

    Coordinates the generation of global sea-level change
    projections from the world's ice sheets.

    Args:
        nsamps (int): Number of samples to generate
        scenario (str): Future emissions scenario
        use_srocc (bool): Use the IPCC SROCC calibration instead
            of the IPCC AR5 calibration
        seed (int): Seed value for random number generation

    Return:
        None. A netCDF file for each of the world's ice sheets
        is generated in the script's root directory.
    """

    # Preprocess stage
    data_dict, corr_dict = preprocess_icesheets(
        rcp_scenario=scenario,
        use_srocc=use_srocc,
    )

    # Fit stage
    fit_dict = fit_icesheets(rates_dict=data_dict)

    # Projection stage
    proj_dict = project_icesheets(
        nsamps=nsamps,
        fit_dict=fit_dict,
        corr_dict=corr_dict,
        seed=seed,
    )

    # Write combined distribution data for each ice sheet to disk
    for i, isname in enumerate(["GIS", "WAIS", "EAIS"]):
        writeNetCDF(
            data=proj_dict["hysamps"][:, :, i],
            icesheet_name=isname,
            targyears=proj_dict["targyears"],
            baseyear=proj_dict["baseyear"],
            nsamps=nsamps,
            scenario=scenario,
        )
    writeNetCDF(
        data=proj_dict["hysamps"][:, :, 1] + proj_dict["hysamps"][:, :, 2],
        icesheet_name="AIS",
        targyears=proj_dict["targyears"],
        baseyear=proj_dict["baseyear"],
        nsamps=nsamps,
        scenario=scenario,
    )


if __name__ == "__main__":

    # Initialize the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Run the ice sheet projection workflow"
    )

    # Define the command line arguments to be expected
    parser.add_argument(
        "--nsamps",
        help="Number of samples",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--scenario",
        help="RCP Scenario",
        choices=["rcp85", "rcp60", "rcp45", "rcp26"],
        default="rcp85",
    )
    parser.add_argument(
        "--srocc",
        help="Use the SROCC calibration instead of AR5 calibration",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        help="Seed for random number generation",
        type=int,
        default=1234,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the process
    main(
        nsamps=args.nsamps,
        scenario=args.scenario,
        use_srocc=args.srocc,
        seed=args.seed,
    )

    # Done
    sys.exit()
