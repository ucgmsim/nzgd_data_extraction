"""
Borehole Report Processor
--------------------------

This script is a command-line interface tool for processing borehole PDF reports
to extract Standard Penetration Test (SPT) values and associated soil classifications.
It consolidates the extracted data into a structured format, which is saved as a
Parquet file for further analysis.

Features
--------
- Extracts depth, SPT values, and soil classifications from borehole PDF reports.
- Supports bulk processing of multiple reports in a directory.
- Outputs consolidated data in a Parquet format for efficient storage and retrieval.

Usage
-----
Run the script from the command line with the required arguments. Example usage:

    python miner.py /path/to/reports /path/to/output.parquet

Positional Arguments
---------------------
report_directory : Path
    Path to the directory containing borehole PDF reports.
output_path : Path
    Path to save the consolidated output as a Parquet file.

Dependencies
------------
- Python >= 3.8
- pdfminer.six
- pandas
- numpy
- typer
- tqdm

Notes
-----
- Ensure that the input PDF reports are formatted in a way that the script can parse.
- The script attempts to extract data robustly but may fail for non-standard or
  corrupted reports.
- Warnings are emitted for reports that cannot be processed, but execution will
  continue for other reports.
"""

import multiprocessing
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Generator, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import tqdm
import typer
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTFigure, LTTextBoxHorizontal, LTTextBoxVertical
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfparser import PDFParser

# Initialize Typer app
app = typer.Typer()

# Configure warnings
warnings.simplefilter("error", np.exceptions.RankWarning)


@dataclass
class TextObject:
    """Represents a text object with positional and textual data.

    Attributes
    ----------
    y0 : float
        The lower y-coordinate of the text object.
    y1 : float
        The upper y-coordinate of the text object.
    x0 : float
        The left x-coordinate of the text object.
    x1 : float
        The right x-coordinate of the text object.
    text : str
        The textual content of the object.
    """

    y0: float
    y1: float
    x0: float
    x1: float
    text: str

    @property
    def yc(self) -> float:
        """float: The vertical centre coordinate."""
        return (self.y1 + self.y0) / 2

    @property
    def xc(self) -> float:
        """float: The horizontal centre coordinate."""
        return (self.x1 + self.x0) / 2


def extract_soil_report(description: str) -> set[str]:
    """Extract soil types mentioned in a description.

    Parameters
    ----------
    description : str
        The input text to search for soil types.

    Returns
    -------
    set[str]
        A set of identified soil types from the input.
    """
    soil_types = {"SAND", "SILT", "CLAY", "GRAVEL", "COBBLES", "BOULDERS"}
    return soil_types & {word.strip(",.;") for word in description.split()}


def extract_spt_value(text: str) -> Optional[int]:
    """Extract the SPT (Standard Penetration Test) value from text.

    Parameters
    ----------
    text : str
        The input text containing the SPT value.

    Returns
    -------
    Optional[int]
        The extracted SPT value, or None if not found.
    """
    if match := re.search(r"\bN\s*(=|\>)\s*(\d+)", text):
        return int(match.group(2))
    return None


def extract_pdf_text_objects(lt_objs: list[Any]) -> Generator[TextObject, None, None]:
    """Recursively extract text objects from PDF layout elements.

    Parameters
    ----------
    lt_objs : list[Any]
        A list of PDF layout objects.

    Yields
    ------
    TextObject
        Extracted text objects with positional and text data.
    """
    for obj in lt_objs:
        if isinstance(obj, (LTTextBoxHorizontal, LTTextBoxVertical)):
            yield TextObject(
                y0=obj.y0, y1=obj.y1, x0=obj.x0, x1=obj.x1, text=obj.get_text()
            )
        elif isinstance(obj, LTFigure):
            yield from extract_pdf_text_objects(obj._objs)


def is_number(text: str) -> bool:
    """Check if a given string represents a number.

    Parameters
    ----------
    text : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a valid number, otherwise False.
    """
    try:
        x = float(text.strip(" \nm"))
        return 0 <= x <= 60
    except ValueError:
        return False


def borehole_id(report: Path) -> int:
    """Extract the borehole ID from a report filename.

    Parameters
    ----------
    report : Path
        The path to the borehole report.

    Returns
    -------
    int
        The extracted borehole ID.

    Raises
    ------
    ValueError
        If the report name does not follow the expected format.
    """
    if match := re.search(r"_(\d+)_", report.stem):
        return int(match.group(1))
    raise ValueError(f"Report name {report.stem} lacks proper structure")


def process_borehole(report: Path) -> pd.DataFrame:
    """Process a borehole report to extract SPT values and soil types.

    Parameters
    ----------
    report : Path
        The path to the borehole report PDF.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing depth, soil types, and SPT values.

    Raises
    ------
    PDFTextExtractionNotAllowed
        If text extraction is not permitted for the PDF.
    ValueError
        If depth column or SPT values are missing.
    """
    text_objects = []
    with report.open("rb") as file:
        parser = PDFParser(file)
        document = PDFDocument(parser)

        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed(
                f"Text extraction not allowed in {report}"
            )

        rsrcmgr = PDFResourceManager()
        laparams = LAParams(detect_vertical=True)
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
            layout = device.get_result()
            page_text_objects = sorted(
                extract_pdf_text_objects(layout._objs), key=lambda obj: (obj.y0, obj.x0)
            )
            text_objects.append(page_text_objects)

    return _analyze_text_objects(text_objects, report)


def interval_intersects(a: float, b: float, c: float, d: float) -> bool:
    """Check if the interval [a, b] intersects the interval [c, d].

    Parameters
    ----------
    a : float
        The left bound of the first interval.
    b : float
        The right bound of the first interval.
    c : float
        The left bound of the second interval.
    d : float
        The right bound of the second interval.


    Returns
    -------
    bool
        True if the intersection of the interval [a, b] with [c, d] is non-trivial.
    """
    return not (b < c or d < a)


def invalid_scale(scale: list[float]) -> bool:
    """Detect if a depth scale is invalid.

    A depth scale is a list of depth values extracted from the PDF.
    The depth scale is considered invalid if it is not strictly
    increasing and non-negative, i.e.

        scale[i] > scale[i - 1] and
        scale[0] >= 0.

    Parameters
    ----------
    scale : list[float]
        The list of depth values to check.


    Returns
    -------
    bool
        True if the depth scale is strictly increasing.
    """
    return all(scale[i] > scale[i - 1] for i in range(1, len(scale))) and scale[0] >= 0


def extract_depth_scale(
    depth_node: TextObject,
    nodes: list[TextObject],
    eps_max: float = 5,
    max_iterations: int = 10,
) -> npt.NDArray[np.float64]:
    """Extracts the depth scaling parameters for a given depth column by iteratively
    refining the search space to identify valid depth values.

    Parameters
    ----------
    depth_node : TextObject
        The text object representing the depth column, with `x0`, `x1`, and `yc`
        attributes specifying its bounding box and centre.
    nodes : list[TextObject]
        A list of text objects containing potential depth values, typically all
        the text nodes on a page.
    eps_max : float, optional
        The maximum allowed horizontal deviation from the centre of the `depth_node`
        during the search for valid depth values. Default is 5.
    max_iterations : int, optional
        The maximum number of iterations to perform when refining the search space.
        Default is 10.

    Returns
    -------
    npt.NDArray[np.float64]
        A 1D array of two values `[m, c]` representing the depth scale equation:
        `depth = m * y + c`, where `y` is the PDF y-coordinate of the text object.

    Raises
    ------
    ValueError
        If the function fails to converge to a valid set of depth values within the
        specified number of iterations.

    Notes
    -----
    This function identifies text objects aligned horizontally with the depth column,
    refines the search bounds using binary search to filter valid depth values, and
    computes a linear fit.

    The process for extraction is illustrated below:


          +-------------------------+         ^
          |    Depth   Col 1  Col 2 |         | y-axis
          |                         |         |
          |       100         -1    |         |
          |                         |         |
          |             150         |         |
          |                         |         |
          |       200               |         |
          +-------------------------+         v
    <-------- eps_max ---------->  (Initial bounds include everything)
      <------ eps_1 ------>         (Bounds narrowed, some invalid values excluded)
             <eps_2>               (Bounds too narrow, missing valid values)
         <--- eps_3 -->      (Final valid bounds after adjustment)

    The diagram illustrates:
    - `eps_max` starts as the widest range.
    - Iterative refinement adjusts the bounds (`eps_1`, `eps_2`, `eps_3`) using binary search.
    - Extra numbers (e.g., `-1` and `150` in `Col 2`) are excluded from the depth column.

    If the search bounds accidentally shrink too far (as in `eps_1` ->
    `eps_2`), the algorithm expands the bounds again to include valid
    depth values. A successful result is achieved when the process
    converges to the smallest possible epsilon that collects a valid
    depth scale.
    """
    eps_max += (depth_node.x1 - depth_node.x0) / 2
    eps_low = 0.0
    eps_high = eps_max
    eps = eps_max / 2
    params: Optional[npt.NDArray[np.float64]] = None
    for _ in range(max_iterations):
        depth_values = [
            obj
            for obj in nodes
            if (
                interval_intersects(
                    depth_node.x0, depth_node.x1, obj.xc - eps, obj.xc + eps
                )
            )
            and is_number(obj.text)
        ]
        if len(depth_values) < 2:
            # Not enough depth values found, widen the search space for depth values.
            eps_low = eps
            eps = (eps_high + eps_low) / 2
            continue
        scale = [float(obj.text.strip(" \nm")) for obj in depth_values]
        if invalid_scale(scale):
            # The depth scale doesn't make sense: this likely means we have
            # included values in our depth that are not part of the depth
            # column. Narrow the search to filter them out.

            eps_high = eps
            eps = (eps_high + eps_low) / 2
            continue

        params = np.polyfit(
            [node.yc for node in depth_values],
            scale,
            1,
        )
    if isinstance(params, np.ndarray):
        return params
    raise ValueError("Failed to converge to a valid depth column")


def _analyze_text_objects(
    text_objects: list[list[TextObject]], report: Path
) -> pd.DataFrame:
    """Analyse extracted text objects to extract borehole data.

    Parameters
    ----------
    text_objects : list[list[TextObject]]
        A list of pages containing lists of TextObjects.
    report : Path
        The path to the borehole report.

    Returns
    -------
    pd.DataFrame
        A DataFrame with borehole data.

    Raises
    ------
    ValueError
        If depth column or SPT values are not found.
    """
    spt_values, extracted_soil_depths, soil_types = [], [], []

    try:
        depth_node = next(
            node
            for page in text_objects
            for node in page
            if re.match(r"(depth|length)\s*(\(m\))?", node.text.lower())
        )
    except StopIteration as exc:
        raise ValueError(f"Depth column not found in {report}") from exc

    for page in text_objects:
        try:
            m, c = extract_depth_scale(depth_node, page)
        except ValueError:
            continue
        for node in sorted(page, key=lambda n: n.yc, reverse=True):
            depth = m * node.yc + c
            if soil_report := extract_soil_report(node.text):
                extracted_soil_depths.append(depth)
                soil_types.append(soil_report)

        soil_depths = np.array(extracted_soil_depths)
        for node in page:
            depth = m * node.yc + c
            soil_type = (
                soil_types[
                    min(np.searchsorted(soil_depths, depth), len(soil_depths) - 1)
                ]
                if soil_types
                else set()
            )

            n = extract_spt_value(node.text)
            if n is not None:
                spt_values.append(
                    {
                        "Depth": round(depth, 2),
                        "Soil Type": [str(tag) for tag in soil_type],
                        "N": n,
                    }
                )

    if not spt_values:
        raise ValueError(f"No SPT values found in {report}")

    df = pd.DataFrame(spt_values)
    df["NZGD_ID"] = borehole_id(report)
    min_depth = df["Depth"].min()
    max_depth = df["Depth"].max()
    if min_depth < 0 or max_depth > 70:
        raise ValueError(
            f"Invalid depth calculation detected (minimum depth = {min_depth}, max depth = {max_depth})."
        )
    return df


def process_borehole_no_exceptions(report: Path) -> Optional[pd.DataFrame]:
    """Process a borehole report while suppressing exceptions.

    Parameters
    ----------
    report : Path
        The path to the borehole report.

    Returns
    -------
    Optional[pd.DataFrame]
        A DataFrame with borehole data, or None if an exception occurs.
    """
    try:
        return process_borehole(report)
    except Exception as e:
        warnings.warn(f"Failed to process {report}: {e}")
        return None


@app.command(help="Extract borehole report data.")
def mine_borehole_log(
    report_directory: Annotated[
        Path,
        typer.Argument(
            help="Path to the directory containing borehole PDF reports.",
            exists=True,
            readable=True,
            file_okay=False,
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(
            help="Path to save the consolidated output as a Parquet file.",
            writable=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Extract and consolidate borehole log data from a directory of reports.

    Parameters
    ----------
    report_directory : Path
        Path to the directory containing borehole PDF reports.
    output_path : Path
        Path to save the consolidated output as a Parquet file.
    """
    pdfs = list(report_directory.glob("*.pdf"))
    with multiprocessing.Pool() as pool:
        combined_df = (
            pd.concat(
                tqdm.tqdm(
                    pool.imap(process_borehole_no_exceptions, pdfs), total=len(pdfs)
                )
            )
            .set_index(["NZGD_ID", "Depth"])
            .sort_index()
        )

    combined_df.to_parquet(output_path)


if __name__ == "__main__":
    app()
