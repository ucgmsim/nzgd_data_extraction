from pathlib import Path
from collections import defaultdict
from python_ags4 import AGS4

borehole_dir = Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/borehole")
borehole_example_output_dir = Path("/home/arr65/data/nzgd/resources/borehole_ags_example")

bh_to_files = defaultdict(list)
ags_files = []


borehole_files = list(borehole_dir.rglob("*"))

for item in borehole_files:

    if item.is_file():

        bh_to_files[item.parent.name].append(item)
        if item.suffix == ".ags":
            ags_files.append(item)

tables, headings = AGS4.AGS4_to_dataframe(ags_files[0])

for heading in headings:

    df = tables[heading]
    df.to_csv(borehole_example_output_dir / f"{heading}.csv", index=False)

print()
