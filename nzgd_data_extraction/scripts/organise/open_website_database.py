import pandas as pd
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

record_name = "CPT_136028"
cpt_df = pd.read_parquet("/home/arr65/src/nzgd_map_from_webplate/instance/extracted_cpt_and_scpt_data.parquet",
                             filters=[("record_name", "==", record_name)]).reset_index()

fig = make_subplots(rows=1, cols=3)

fig.add_trace(go.Scatter(x=cpt_df["qc"], y=cpt_df["Depth"]), row=1, col=1)
fig.add_trace(go.Scatter(x=cpt_df["fs"], y=cpt_df["Depth"]), row=1, col=2)
fig.add_trace(go.Scatter(x=cpt_df["u"], y=cpt_df["Depth"]), row=1, col=3)

fig.update_yaxes(title_text="qc (Mpa)", row=1, col=1)
fig.update_yaxes(title_text="fs (Mpa)", row=1, col=2)
fig.update_yaxes(title_text="u2 (Mpa)", row=1, col=2)