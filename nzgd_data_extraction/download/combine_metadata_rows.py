import pandas as pd
from pathlib import Path

def combine_metadata_rows(metadata_dir: Path, output_file: Path):
    """
    Combine metadata rows from multiple files into a single file.

    Parameters
    ----------
    metadata_dir : Path
        The directory containing the metadata files.
    output_file : Path
        The file to write the combined metadata to.
    """
    # List to store the metadata dataframes
    metadata_dfs = []

    # Iterate over the files in the metadata directory
    for metadata_file in metadata_dir.glob("*.csv"):
        # Read the metadata file into a dataframe
        metadata_df = pd.read_csv(metadata_file)
        # Append the dataframe to the list
        metadata_dfs.append(metadata_df)

    # Concatenate the dataframes in the list
    combined_metadata_df = pd.concat(metadata_dfs)

    # Write the combined metadata to the output file
    combined_metadata_df.to_csv(output_file, index=False)

    return combined_metadata_df

if __name__ == "__main__":
    metadata_dir = Path("/home/arr65/data/nzgd/name_to_metadata_per_record")
    output_file = Path("/home/arr65/data/nzgd/metadata/vsvp/vsvp_metadata.csv")
    metadata_df = combine_metadata_rows(metadata_dir, output_file)
    #
    # print()

    #metadata_df = pd.read_csv(output_file)

    ### Print every unique value and the number of times they each appear in the 'technician' column
    print(metadata_df["technician"].value_counts())