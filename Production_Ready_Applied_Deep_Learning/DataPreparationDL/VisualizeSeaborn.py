import traceback
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tabulate import tabulate
import numpy as np

# image format to save
image_format = 'eps'

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 30

# control default text sizes
plt.rc('font', size=SMALL_SIZE)

# fontsize of the axes title
plt.rc('axes', titlesize=SMALL_SIZE)

# fontsize of the x and y labels
plt.rc('axes', labelsize=SMALL_SIZE)

# fontsize of the tick labels
plt.rc('xtick', labelsize=SMALL_SIZE)

# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)

# legend fontsize
plt.rc('legend', fontsize=SMALL_SIZE)

# fontsize of the figure title
plt.rc('figure', titlesize=SMALL_SIZE)

def seaborn_line_histogram(in_file):
    try:
        # read csv file
        df = pd.read_csv(in_file)

        # pretty print of data
        print(tabulate(df.head(10), headers='keys', tablefmt='psql'))

        # groupby state and output sequence is converted to dataframe
        df_mean = df.groupby(['jurisdiction'])['_1st_dose_allocations'].mean().reset_index()

        # rename column names of dataframe
        df_mean.columns = ['state', 'count_vaccine']

        # sort descending by # vaccine dose 1
        df_mean_sorted = df_mean.sort_values(by=['count_vaccine'], ascending=False)

        # top 10 stats by largest mean
        df_mean_sorted_top10 = df_mean_sorted[0:10]

        # top 10 sorted mean print
        print(tabulate(df_mean_sorted_top10, headers='keys', tablefmt='psql'))

        ### #LINE chart plotting
        # line plot
        sns.lineplot(data=df_mean_sorted_top10, x="state", y="count_vaccine")

        # rotate x-axis labels
        plt.xticks(rotation=45)

        # alternate option withou .gcf
        plt.subplots_adjust(left=0.2, bottom=0.25)

        # write to file
        image_name = 'linediagram.eps' # image name
        plt.savefig(image_name, format=image_format, dpi=1200)

        # show the actual plot
        plt.show()

        ### Histogram chart plotting
        sns.set(font_scale=1.2)

        # plot histogram bars with top10 states mean distribution count of vaccine
        p = sns.displot(df_mean_sorted_top10['count_vaccine'], kde=False)
        p.set_xlabels("count_vaccine", fontsize=20)
        p.set_ylabels("count", fontsize=20)
        plt.subplots_adjust(bottom=0.22)
        plt.xticks(rotation=90)

        # write to a file
        image_name = 'histogram.eps' # image name
        plt.savefig(image_name, format=image_format, dpi=1200)
        plt.show()
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":

    # Modrena vaccination distribution file
    in_file = "/Users/quangtn/Desktop/01_work/01_job/02_ml/" \
              "bentoml/prj/Chapter_2/csv_data/" \
              "data/cdc-moderna-covid-19-vaccine-distribution-by-state.csv"

    # create bag of words for feature research interest with NLTK
    seaborn_line_histogram(in_file)