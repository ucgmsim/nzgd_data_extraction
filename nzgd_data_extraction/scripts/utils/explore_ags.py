from pathlib import Path
from python_ags4 import AGS4

file_path = Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt/CPT_4/CPT_4_AGS01.ags")

tables, headings = AGS4.AGS4_to_dataframe(file_path)

output_dir = Path(f"/home/arr65/data/nzgd/resources/cpt_ags_tables/{file_path.parent.name}")
output_dir.mkdir(parents=True, exist_ok=True)

for heading in headings:
    tables[heading].to_csv(output_dir / f"{heading}.csv",index=False)



print()

