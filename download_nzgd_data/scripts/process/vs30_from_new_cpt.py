import pandas as pd
from pathlib import Path

import VsViewer.vs_calc.CPT as CPT






print()

cpts = []

cpts.append(
    CPT(
        cpt_loc.name,
        cpt_records[:, 0],
        cpt_records[:, 1],
        cpt_records[:, 2],
        cpt_records[:, 3],
        cpt_loc.nztm_x,
        cpt_loc.nztm_y,
    )
)



#processed_cpt_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")

#cpt_df = pd.read_parquet(processed_cpt_dir / "CPT_1.parquet")




print()

