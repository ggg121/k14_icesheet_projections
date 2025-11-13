import argparse
import os
import re
import sys
from pathlib import Path

import xarray as xr
from matplotlib import pyplot as plt


def plot_projections(ds: xr.Dataset) -> None:
    """Make a plot of the projection data

    Makes a plot of the projection data produced by the
    "generate_icesheet_projections.py" script.

    Args:
        ds (xr.Dataset): Xarray dataset containing
            projection data

    Return:
        None. A plot is generated in the "plots" directory
    """
    # Make the plot directory if needed
    plotdir = Path(Path(__file__).parent, "plots")
    plotdir.mkdir(parents=True, exist_ok=True)

    # Extract some information for the plot labels
    match = re.search("([G|EA|WA|A]IS)", ds.attrs["description"])
    icesheet_name = match.group(1)
    plot_title = f"{icesheet_name} contribution to global\nsea-level change - {ds.attrs['scenario']}"

    # Plot the outer-most shaded region
    this_label = f"{ds['quantile'][0]*100:.0f}%-ile - {ds['quantile'][4]*100:.0f}%-ile"
    lb = ds["sea_level_change"].isel(quantile=0)
    ub = ds["sea_level_change"].isel(quantile=4)
    plt.fill_between(
        ds.time,
        lb,
        ub,
        alpha=0.5,
        lw=0,
        color="green",
        label=this_label,
    )

    # Plot the inside shaded region
    this_label = f"{ds['quantile'][1]*100:.0f}%-ile - {ds['quantile'][3]*100:.0f}%-ile"
    lb = ds["sea_level_change"].isel(quantile=1)
    ub = ds["sea_level_change"].isel(quantile=3)
    plt.fill_between(
        ds.time,
        lb,
        ub,
        lw=0,
        color="green",
        label=this_label,
    )

    # Plot the middle line
    this_label = f"{ds['quantile'][2]*100:.0f}%-ile"
    plt.plot(
        ds.time,
        ds["sea_level_change"].isel(quantile=2),
        color="black",
        label=this_label,
    )

    # Apply the plot titles and labels
    plt.title(plot_title)
    plt.ylabel(f"sea-level change [{ds.sea_level_change.attrs['units']}]")
    plt.xlabel("Year")
    plt.legend(loc="upper left")

    # Save the plot to disk
    plot_filename = Path(
        plotdir, f"{icesheet_name}_{ds.attrs['scenario']}_quantile_plot.pdf"
    )
    plt.savefig(plot_filename)

    return None


def load_data(filepath: os.PathLike) -> xr.Dataset:
    """Loads projection data into memory

    Loads the netCDF file provided by 'filepath' into
    and xarray dataset and stores it in system memory.

    Args:
        filepath (Pathlike): Full path to projection
            data produced by 'generate_icesheet_projections.py'

    Return:
        Xarray dataset of projection data
    """
    return xr.open_dataset(filepath, chunks=None)


def get_quantiles(ds: xr.Dataset, quantiles: float) -> xr.Dataset:
    """Calculates the five requested quantiles of the projection data

    Args:
        ds (xr.Dataset): Xarray dataset with samples of
            'sea_level_change' over which quantiles are produced
        quantiles (float): List of five quantiles to calculate

    Return:
        Xarray Dataset with the quantiles requested
    """
    if len(quantiles) != 5:
        raise ValueError(
            f"'quantiles' must be of length 5. len('quantiles') = {len(quantiles)}"
        )
    return ds.quantile(quantiles, "samples", keep_attrs=True)


def main(filepath: os.PathLike) -> None:
    """Coordinates plotting of projection data

    Args:
        filepath (Pathlike): Full path to projection
            data produced by 'generate_icesheet_projections.py'

    Return:
        None. A plot is generated in the './plots' directory
    """
    quantiles_to_plot = [0.05, 0.33, 0.5, 0.67, 0.95]
    ds = load_data(filepath)
    ds = get_quantiles(ds, quantiles_to_plot)
    plot_projections(ds)

    return None


if __name__ == "__main__":

    # Initialize the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Generate a plot of ice sheet projection data",
        epilog="Input data are expected to be generated from the 'generate_icesheet_projections.py' script",
    )

    # Define the command line arguments to be expected
    parser.add_argument(
        "--infile",
        help="Full path to input file",
        type=Path,
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the process
    main(
        filepath=args.infile,
    )

    # Done
    sys.exit()
