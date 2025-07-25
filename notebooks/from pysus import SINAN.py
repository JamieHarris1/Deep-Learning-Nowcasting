from pathlib import Path
import dbcpy

project_dir = Path.cwd().parent

dbcpy.to_csv(project_dir / "data" / "raw" / "counts" / "DENGBR13.dbc", "DENGBR13.csv")
