from collections import Counter

from pathlib import Path


import pandas as pd

from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from getCPTdata import getCPTdata

Base = declarative_base()
out_dir = Path("/Users/sungbae/CPT/outdir")
plot_dir = out_dir / "validation_plots"

cpt_df = pd.read_csv("/Users/sungbae/CPT/cpt_locations_20200909.csv", sep=",")


skipped_fp = open(out_dir / "skipped_cpts" / f"error_location.log", "w")
results = {}


def log_error(skipped_fp, cpt_name, error):
    skipped_fp.write(f"{cpt_name} - {error}\n")


class CPTLocation(Base):
    __tablename__ = 'cpt_location'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    #    customer_id=Column(Integer, ForeignKey('customers.id'))
    name = Column(String(20), nullable=False)  # 20210427_17
    private = Column(Integer) #true / false
    type = Column(String(5)) # CPT or SCPT
    nztm_x = Column(Float)
    nztm_y = Column(Float)


class CPTDepthRecord(Base):
    __tablename__ = 'cpt_depth_record'
    # Here we define columns for the table person
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    cpt_name = Column(String(20), nullable=False)  # 20210427_17
    depth = Column(Float) #
    qc = Column(Float) #
    fs = Column(Float)
    u = Column(Float)
    loc_id = Column(Integer, ForeignKey('cpt_location.id'))


if __name__ == '__main__':
    # Create an engine that stores data in the local directory's
    # sqlalchemy_example.db file.
    engine = create_engine('sqlite:///nz_cpt.db')
    Base.metadata.drop_all(engine)

    # Create all tables in the engine. This is equivalent to "Create Table"
    # statements in raw SQL.
    Base.metadata.create_all(engine)

    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    for row_n, cpt in cpt_df.iterrows():
        new_record = CPTLocation(
            name=cpt.CombinedName,
            private=1 if cpt['TTGD Only'] else 0,
            type= "sCPT" if cpt.InvestigationType=='SCPT' else "CPT",
            nztm_x = cpt.NZTM_X,
            nztm_y = cpt.NZTM_Y,
        )
        cur_data=session.query(CPTLocation).filter(CPTLocation.name == cpt.CombinedName).all()
        if len(cur_data) == 0:
            session.add(new_record)

            #print("Successfully inserted")
        else:
            print("Duplicate data. Not inserted")

    session.commit()