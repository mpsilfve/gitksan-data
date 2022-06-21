"""
Filter out tables which only contain a ROOT form, where the ROOT form
is not a noun or verb.

Miikka Silfverberg 2022

"""

from pathlib import Path
import click
import csv
import pandas as pd
from io import StringIO

SCRIPT_PATH=Path(__file__).absolute()

# Path to directory with a manually prepared CSV file containing
# lexemes which need to be filtered out
MISC_PATH=SCRIPT_PATH.parents[1]/"misc"

# Empty slots in inflection tables correspond to lines of underscores
EMPTY=set("_")

# Column number 3 in inflection tables gives the orthographic form
ORTHOGRAPHIC=3

def read_tables(data_fn):
    with open(data_fn) as f:
        table = ""
        dfs = []
        header = ""
        for i, line in enumerate(f):
            line = line.strip()
            # Skip header
            if i == 0:
                header = line
                continue
            if line:
                table += line + "\n"
            else:
                sio= StringIO(table)
                df = pd.read_csv(sio, sep="\t", header=None)                
                dfs.append(df)
                table = ""
        return dfs, header

def read_filtered_forms():
    df = pd.read_csv(MISC_PATH/"roots_evaluated.csv")
    exclude_lines = df[df["EXCLUDE"] == 1]
    return set(exclude_lines["ORTHOGRAPHIC"].values.tolist())
    
def has_form(tab,form_name):
    return set(tab.loc[tab[0] == form_name].values[0,1:].tolist()) != EMPTY

def has_root(tab):
    return has_form(tab, "ROOT")

def has_non_root(tab):
    forms = tab[0].values.tolist()
    return any([has_form(tab,form) for form in forms if form != "ROOT"])
    
@click.command()
@click.option("--data_fn", required=True)
def main(data_fn):
    tables, file_header = read_tables(data_fn)
    print(f"Read {len(tables)} tables from {data_fn}")

    filtered_roots = read_filtered_forms()
    print(f"Read {len(filtered_roots)} unique roots which will be filtered from table")

    with open(f"{data_fn}.filtered", "w") as out_f:
        print(file_header, file=out_f)
        excluded = 0
        for tab in tables:
            if has_root(tab) and not has_non_root(tab):
                roots = tab.loc[tab[0] == "ROOT"][ORTHOGRAPHIC].values.tolist()
                if filtered_roots.intersection(roots) != set():
                    excluded += 1
                    continue
            print(tab.to_csv(sep="\t", index=False, header=False), file=out_f)

    print(f"Excluded altogether {excluded} tables")
    
if __name__=="__main__":
    main()


