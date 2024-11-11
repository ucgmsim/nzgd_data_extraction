from collections import Counter

from pathlib import Path


import pandas as pd

from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from cpt_locations2db import CPTLocation,CPTDepthRecord

from qcore import coordinates, geo

#from getCPTdata import getCPTdata

Base = declarative_base()
# out_dir = Path("/Users/sungbae/CPT/outdir")
# plot_dir = out_dir / "validation_plots"

out_dir = Path("/home/arr65/data/nzgd/processed_data_as_sql")
plot_dir = out_dir / "validation_plots"
plot_dir.mkdir(parents=True, exist_ok=True)

######################################################################################
# cpt_df = pd.read_csv("/Users/sungbae/CPT/cpt_locations_20200909.csv", sep=",")

### cpt_df has columns:
### cpt.CombinedName, 'TTGD Only' (True/False), InvestigationType (SCPT/CPT), NZTM_X, NZTM_Y

processed_new_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")

new_df = pd.read_parquet(processed_new_data_dir, columns=["record_name", "longitude", "latitude"])
new_df = new_df.drop_duplicates(keep='first').reset_index(drop=True)

latlon_array = new_df[["latitude", "longitude"]].to_numpy()

nztm_yx = coordinates.wgs_depth_to_nztm(latlon_array)

#reconvert_to_latlon = coordinates.nztm_to_wgs_depth(nztm_yx)

cpt_df = pd.DataFrame({"CombinedName": new_df["record_name"],
                       "TTGD Only": False,
                       "InvestigationType": "CPT",
                       "NZTM_X": nztm_yx[:,1],
                       "NZTM_Y": nztm_yx[:,0]})

######################################################################################

# cpt_root_path = Path("/Users/sungbae/CPT/CPT_Depth_Profile_CSVs")
# skipped_fp = open(out_dir / "skipped_cpts" / f"error_depth.log", "w")
# results = {}


def log_error(skipped_fp, cpt_name, error):
    skipped_fp.write(f"{cpt_name} - {error}\n")



if __name__ == '__main__':
    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.
    engine = create_engine('sqlite:///nz_cpt.db')

    #Base.metadata.drop_all(engine)

    # Create all tables in the engine. This is equivalent to "Create Table"
    # statements in raw SQL.
    #Base.metadata.create_all(engine)

    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    #cpt_files=cpt_root_path.glob("*/*csv")
    cpt_files = processed_new_data_dir.glob("*.parquet")
    cpt_files = list(cpt_files)
    for i, cpt_file in enumerate(cpt_files):
        cpt_name = cpt_file.stem

        cur_data = session.query(CPTDepthRecord).filter(CPTDepthRecord.cpt_name == cpt_name).all()
        if (len(cur_data)>0):
            skipped = "skipped"
            print(f"{i + 1}/{len(cpt_files)} {cpt_name} @ {cpt_file} - skipping")
            continue

        print(f"{i+1}/{len(cpt_files)} {cpt_name} @ {cpt_file}")

        cur_data = session.query(CPTLocation).filter(CPTLocation.name == cpt_name).all()

        # if len(cur_data) > 1:
        #     log_error(skipped_fp, cpt_name, f"{cpt_name} has duplicates")
        #     continue
        # if len(cur_data) == 0:
        #     log_error(skipped_fp, cpt_name, f"{cpt_name} is unknown location")
        #     continue

        loc_id = cur_data[0].id

        # try:
        #     (z, qc, fs, u2) = getCPTdata(cpt_file)
        # except (ValueError, Exception) as e:
        #     log_error(skipped_fp, cpt_name, f"could not read file: {str(e)}")
        #     continue
        new_cpt_data = pd.read_parquet(cpt_file)
        z = new_cpt_data["Depth"].to_numpy()
        qc = new_cpt_data["qc"].to_numpy()
        fs = new_cpt_data["fs"].to_numpy()
        u2 = new_cpt_data["u"].to_numpy()

        for i in range(len(z)):
            new_record = CPTDepthRecord(
                cpt_name=cpt_name,
                depth=z[i],
                qc=qc[i],
                fs=fs[i],
                u=u2[i],
                loc_id=loc_id,
            )
            session.add(new_record)
        session.commit()



