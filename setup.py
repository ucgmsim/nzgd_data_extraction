from setuptools import find_packages, setup

setup(
    name="download_nzgd_data",  # Your package name
    version="1.0",
    packages=find_packages(),  # Automatically finds all packages
    include_package_data=True,  # Ensures additional files (like data) are included
    url="https://github.com/ucgmsim/download_nzgd_data",
    description="For downloading and processing data from the New Zealand Geotechnical Database (NZGD)",
)