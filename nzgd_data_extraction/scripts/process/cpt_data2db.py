from collections import Counter

from pathlib import Path


import pandas as pd
from tqdm import tqdm

from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from cpt_locations2db import CPTLocation,CPTDepthRecord

from qcore import coordinates, geo

#from getCPTdata import getCPTdata

Base = declarative_base()




######################################################################################

# cpt_root_path = Path("/Users/sungbae/CPT/CPT_Depth_Profile_CSVs")
# skipped_fp = open(out_dir / "skipped_cpts" / f"error_depth.log", "w")
# results = {}


def log_error(skipped_fp, cpt_name, error):
    skipped_fp.write(f"{cpt_name} - {error}\n")



if __name__ == '__main__':

    processed_new_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")
    out_dir = Path("/home/arr65/data/nzgd/processed_data_as_sql")

    cpt_locations = pd.read_csv(out_dir / "cpt_locations.csv")

    cpt_names = cpt_locations["CombinedName"].to_list()

    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.
    #engine = create_engine('sqlite:///nz_cpt.db')
    engine = create_engine(f'sqlite:///{out_dir}/nz_cpt.db')

    #Base.metadata.drop_all(engine)

    # Create all tables in the engine. This is equivalent to "Create Table"
    # statements in raw SQL.
    #Base.metadata.create_all(engine)

    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    ### cpt_files=cpt_root_path.glob("*/*csv")

    print("making file list")
    cpt_files = [processed_new_data_dir / f"{cpt_name}.parquet" for cpt_name in cpt_names]

    print("Putting data in the SQLite database")
    for i, cpt_file in tqdm(enumerate(cpt_files), total=len(cpt_files)):
        try:
            cpt_name = cpt_file.stem
            cur_data = session.query(CPTDepthRecord).filter(CPTDepthRecord.cpt_name == cpt_name).all()

            # if (len(cur_data)>0):
            #     skipped = "skipped"
            #     print(f"{i + 1}/{len(cpt_files)} {cpt_name} @ {cpt_file} - skipping")
            #     continue
            # print(f"{i+1}/{len(cpt_files)} {cpt_name} @ {cpt_file}")

            cur_data = session.query(CPTLocation).filter(CPTLocation.name == cpt_name).all()

            if len(cur_data) > 1:
                #log_error(skipped_fp, cpt_name, f"{cpt_name} has duplicates")
                print(f"{cpt_name} has duplicates")
                continue
            if len(cur_data) == 0:
                #log_error(skipped_fp, cpt_name, f"{cpt_name} is unknown location")
                print(f"{cpt_name} is unknown location")
                continue
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
        except Exception as e:
            print(f"Trying to process {cpt_name} resulted in this error: {e}")
            continue