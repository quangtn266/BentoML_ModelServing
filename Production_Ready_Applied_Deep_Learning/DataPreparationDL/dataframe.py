# This script introduces you to table-like data structure named data frame from Pandas libraries
import pandas as pd
from tabulate import tabulate
in_file = "/Users/quangtn/Desktop/01_work/01_job/02_ml/" \
          "bentoml/prj/Chapter_2/csv_data/data/" \
          "cdc-moderna-covid-19-vaccine-distribution-by-state.csv"

# read the csv file and store the returned dataframe to a variable "df_vacc"
df_vacc = pd.read_csv(in_file)

# pretty print the sample 5 rows
print(tabulate(df_vacc.head(5), headers="keys", tablefmt="psql"))