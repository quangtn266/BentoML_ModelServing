from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tabulate import tabulate
import pandas as pd

## Start of ordinal encoder
# label encoder intance
labelencoder = LabelEncoder()

# Output file that will include new column "is_artificial_intelligence"
file_output = "./research_interest_onehot.csv"

# output file pointer
f = open(file_output, "w")

# header for output file
header = "author_name, email, affiliation, coauthors_names, is_artificial_intelligence"
f.write(header + "\n")

# read the google scholar crawled file
df = pd.read_csv("/Users/quangtn/Desktop/01_work/"
                 "01_job/02_ml/bentoml/prj/Chapter_2/"
                 "google_scholar/output.csv")
print(list(df))

lst_ds = ["data_science", "machine_learning", "artificial_intelligence", "data_mining"]

# hold all unique research interest
lst_global_research = []

# iterate each line of dataframe
for index, row in df.iterrows():
    # default value if an author is in artificial intelligence is zero
    # (for new column is_artificial_intelligence)
    is_artificial_intelligence = "no"

    # split based on delimiter to get all the invidual reserach interest.
    lst_research = str(row['research_interest']).split("##")

    # read all other columns
    curr_authorname = row['author_name']
    curr_email = row['email']
    curr_affiliation = row['affiliation']
    curr_coauthors_names = row['coauthors_names']
    curr_research_interest = str(row['research_interest'])\
        .replace("##", " ").replace("_", " ")

    # iterate each of the research interest and add to global list
    for j in lst_research:
        # create new column is_artificial_intelligence: if an author is
        # in artificial_intelligence, mark it as yes
        if j in lst_ds:
            is_artificial_intelligence = "yes"
        # append invidual research interest to global (ordinal encoding purpose)
        lst_global_research.append(j)
    curr_line = f"{curr_authorname}, {curr_email}, {curr_affiliation}, {curr_coauthors_names}" \
                f",{is_artificial_intelligence}"

    # write to output file (one hot encoding purpose)
    f.write(curr_line + "\n")
# convert global research interest list to dataframe
df_research = pd.DataFrame(lst_global_research, columns=["research_interest"])

# add a new column ri_integer that has the ordinal encoding
df_research['encoded_research_interest'] = labelencoder.fit_transform(df_research['research_interest'])
print(tabulate(df_research.head(10), headers='keys', tablefmt="psql"))