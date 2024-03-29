# One hot encoding

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate

# Output file includes new column "is_artifical_intelligence"
file_out = "./research_interest_onehot.csv"

# output file pointer
f = open(file_out, "w")

# header for output file
header = "author_name, email, affiliation, coauthors_names," \
         " is_artificial_intelligence"
f.write(header +"\n")

# read the google scholar crawled file
df = pd.read_csv("/Users/quangtn/Desktop/01_work/01_job/02_ml/bentoml/"
                 "prj/Chapter_2/google_scholar/output.csv")

print(list(df))

lst_ds = ["data_science", "machine_learning",
          "is_artificial_intelligence", "data_mining"]

# iterate each line of data frame
for index, row in df.iterrows():
    # default value if an author is in artificial intelligence is zero
    # (for new column is_artificial_intelligence)
    is_artificial_intelligence = "no"

    # split based on delimeter to get all the invidual research interest
    lst_research = str(row['research_interest']).replace("##", " ")\
        .replace("_", " ")

    curr_authorname = row['author_name']
    curr_email = row['email']
    curr_affliation = row['affiliation']
    curr_coauthors_names = row['coauthors_names']
    curr_research_interest = str(row['research_interest']).\
        replace("##", " ").replace("_", " ")

    # iterate each of the research interest and add to global list
    for j in lst_research:
        # create new column is_artificial_intelligence: if
        # an author is in artificial_intelligence, mark it as yes
        if j in lst_ds:
            is_artificial_intelligence = "yes"

            # append invidual research interest to global (ordinal
            # encoding purpose)
    curr_line = f"{curr_authorname}, {curr_email}, {curr_affliation}, {curr_coauthors_names}" \
                f", {is_artificial_intelligence}"

    # write to output file (Onehot encoding purpose)
    f.write(curr_line+ "\n")

# close file
f.close()

# Start of one hot encoding

# read the file "research_interest_onehot.csv" that has new column
df_new = pd.read_csv(file_out)

# one hot encoder instance
enc = OneHotEncoder(handle_unknown='ignore')

# pass either "is_artificial_intelligence" or "ri_integer"
enc_df = pd.DataFrame(enc.fit_transform(df_new[['is_artificial_intelligence']]).toarray())

# merge
df_merge = df_new.join(enc_df)

# renamed column related on hot encoding
df_merge.columns = ["author_name", "email", "affiliation", " coauthors_names"
                    "is_artificial_intelligence", "no", "yes"]

print("Only df haveing new column => is_artificial_intelligence"
      " (see last column)")
print(tabulate(df_new.head(10), headers="keys"))
print(tabulate(df_merge.head(10), headers="keys", tablefmt="psql"))
print("Print only is_artificial_intelligence and its related one-hot-encoder")
df_one_hot = df_merge[["is_artificial_intelligence", "no", "yest"]]
print(tabulate(df_one_hot.head(10), headers="keys", tablefmt="psql"))