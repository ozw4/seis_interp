"""Read the fixed-width FORGE survey exports, retaining source line numbers."""

from pathlib import Path

import pandas as pd


def read_forge_navigation(path: Path) -> pd.DataFrame:
    """Parse this dataset's NAD83 UTM12N, decimetre-encoded SP1 records.

    Column interpretation is checked against SEG-Y coordinates in the audit;
    this is not a general SPS/SP1 reader. Unparsed records are never skipped.
    """
    lines = path.read_text(encoding="latin1").splitlines()
    preamble = "\n".join(lines[:15])
    if "Datum: NAD83" not in preamble or "System: Zone 12N" not in preamble:
        raise ValueError(f"unexpected FORGE navigation CRS: {path.name}")
    records = []
    for number, line in enumerate(lines[15:], start=16):
        if len(line) != 80:
            raise ValueError(f"invalid navigation width: {path.name}:{number}")
        try:
            records.append(
                {
                    "line": int(line[19:22]),
                    "point": int(line[22:25]),
                    "x_m": int(line[46:53]) / 10,
                    "y_m": int(line[53:61]) / 10,
                    "elevation_m": int(line[61:66]) / 10,
                    "file_line_number": number,
                }
            )
        except ValueError as error:
            raise ValueError(f"invalid navigation record: {path.name}:{number}") from error
    table = pd.DataFrame(records)
    if table.empty or table.duplicated(["line", "point"]).any():
        raise ValueError(f"empty or duplicate navigation keys: {path.name}")
    return table
