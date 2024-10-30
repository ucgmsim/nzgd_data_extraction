"""
Functions to help with the validation of CPT data.
"""

from dataclasses import dataclass
import pandas as pd
from scipy import interpolate
import numpy as np



@dataclass
class OrganizedWithDepthRange:
    """
    Keeps track of which DataFrames have the largest and smallest depth ranges.
    """

    largest_depth_range: pd.DataFrame
    shortest_depth_range: pd.DataFrame


def organise_with_depth_range(df1: pd.DataFrame, df2: pd.DataFrame) -> OrganizedWithDepthRange:
    """
    Selects the DataFrame with the largest depth range.

    This function compares the depth ranges of two DataFrames and returns the one with the larger range.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame containing a 'Depth' column.
    df2 : pd.DataFrame
        DataFrame containing a 'Depth' column.

    Returns
    -------
    OrganizedWithDepthRange
        An instance of OrganizedWithDepthRange to indicate which Dataframes have the largest and shortest
        depth ranges.
    """

    d1_range = df1["Depth"].max() - df1["Depth"].min()
    d2_range = df2["Depth"].max() - df2["Depth"].min()

    if d1_range > d2_range:
        return OrganizedWithDepthRange(largest_depth_range=df1, shortest_depth_range=df2)
    else:
        return OrganizedWithDepthRange(largest_depth_range=df2, shortest_depth_range=df1)

def get_interpolated_df(organised_dfs: OrganizedWithDepthRange) -> pd.DataFrame:

    """
    Interpolates the DataFrame with the largest depth range onto the DataFrame with the smallest depth range so that
    every point in the smallest depth range has a corresponding point in the largest depth range.

    Parameters
    ----------
    organised_dfs : OrganizedWithDepthRange
        An instance of OrganizedWithDepthRange containing the DataFrames.

    Returns
    -------
    pd.DataFrame: The DataFrame with the largest depth range with interpolated onto the Depth values of the DataFrame
    with the smallest depth range.
    """

    qc_interp = interpolate.interp1d(organised_dfs.largest_depth_range["Depth"], organised_dfs.largest_depth_range["qc"], kind="linear", bounds_error=False)
    fs_interp = interpolate.interp1d(organised_dfs.largest_depth_range["Depth"], organised_dfs.largest_depth_range["fs"], kind="linear", bounds_error=False)
    u_interp = interpolate.interp1d(organised_dfs.largest_depth_range["Depth"], organised_dfs.largest_depth_range["u"], kind="linear", bounds_error=False)

    interpolated_df = organised_dfs.shortest_depth_range.copy()

    interpolated_df.loc[:,"qc"] = qc_interp(interpolated_df["Depth"])
    interpolated_df.loc[:,"fs"] = fs_interp(interpolated_df["Depth"])
    interpolated_df.loc[:,"u"] = u_interp(interpolated_df["Depth"])

    return interpolated_df



def sigma_clip_indices(data: np.array, n_sigma: int) -> np.array:
    """
    Apply a 3-sigma clip to the residual data.

    Parameters
    ----------
    data : np.array
        The data to be clipped.

    Returns
    -------
    tuple

    """

    data = np.abs(data)

    keep_indices_bool_mask = data < np.median(data) + (n_sigma * np.std(data))


    return keep_indices_bool_mask
