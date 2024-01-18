import pandas as pd
import traceback
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# dictionary helps nomarlization
dict_norm = {"data_science": "artificial_intelligence",
             "machine_learning": "artificial_intelligence"

}

def normalize_numeric(infile_numeric):
    """
    Normalize numeric values in range 0 to 1 with example from Corona Vaccnine data set.
    :param infile_numeric:
    :return:
    """

    df_in = pd.read_csv(infile_numeric)

    # Step1: calculate state-wise mean number fir weekly corona vaccine distribution
    df = df_in.groupby("jurisdiction")["_1st_dose_allocations"].mean()\
        .to_frame("mean_vaccine_count").reset_index()

    print("State wise mean calculated for # vaccine distributed weekly")
    print(tabulate(df.head(10), tablefmt="psql", headers="keys"))

    # Step2: Calculate normalized mean vaccine count
    df["norm_vaccine_count"] = df["mean_vaccine_count"] / df["mean_vaccine_count"].max()
    print(tabulate(df.head(10), tablefmt="psql", headers="keys"))


if __name__ == "__main__":
    # input file of google scholar
    infile = "/Users/quangtn/Desktop/01_work/" \
             "01_job/02_ml/bentoml/prj/Chapter_2/google_scholar/output.csv"

    # output file for normalized research_interest field in google scholar
    outfile = "./output_normalize.csv"

    # Input file for weekly corona vaccine distribution
    infile_corona = "/Users/quangtn/Desktop/01_work/01_job/02_ml/" \
                    "bentoml/prj/Chapter_2/csv_data/" \
                    "data/cdc-moderna-covid-19-vaccine-distribution-by-state.csv"

    normalize_numeric(infile_corona)