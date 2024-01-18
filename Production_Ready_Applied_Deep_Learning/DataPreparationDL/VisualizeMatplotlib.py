import traceback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  tabulate import tabulate

# image format to save
image_format ="eps"

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 30

# control default text sizes
plt.rc('font', size=SMALL_SIZE)

# font size of the axes title
plt.rc('axes', titlesize=SMALL_SIZE)

# font size of the x and y labels
plt.rc('axes', labelsize=BIGGER_SIZE)

# font size of the tick labels
plt.rc('xtick', labelsize=SMALL_SIZE)

# font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)

# legend font size
plt.rc('legend', fontsize=BIGGER_SIZE)

# font size of the figure title
plt.rc('figure', titlesize=BIGGER_SIZE)

def matplotlib_pie_bar(in_file):
    try:
        # read vaccination csv file "cdc-moderna-covid-19-vaccine-distribution-by-state.csv"
        df = pd.read_csv(in_file)

        # pretty print of data
        print(tabulate(df.head(10), headers='keys', tablefmt='psql'))

        # group by state and output sequence is converted to dataframe
        df_mean = df.groupby(['jurisdiction'])['_1st_dose_allocations'].mean().reset_index()

        # rename column names of dataframe
        df_mean.columns = ["state", "count_vaccine"]

        # sort decending by accine dose 1
        df_mean_sorted = df_mean.sort_values(by=['count_vaccine'], ascending=False)

        # top 10 stars by largest mean
        df_mean_sorted_top10 = df_mean_sorted[0:10]

        # top 10 sorted mean print
        print(tabulate(df_mean_sorted_top10, headers='keys', tablefmt="psql"))

        #convert top 10 states dataframe to dictionary
        dict_top10 = dict(zip(df_mean_sorted_top10.state, df_mean_sorted_top10.count_vaccine))

        ### Pine chart plotting
        # colors for pie chart
        colors = ["orange", 'green', 'cyan', 'skyblue', 'yellow', 'red', 'lightblue',
                  'grey', '#ffcc99', 'pink']

        # pie chart plot
        plt.pie(list(dict_top10.values()), labels=dict_top10.keys(), colors=colors,
                autopct='%2.1f%%', shadow=True, startangle=90)

        # set axis
        plt.axis('equal')
        image_name = 'piechart.jpg' # image name
        plt.savefig(image_name, format=image_format, dpi=1200)

        # show the actual plot
        plt.show()

        ### Bar chart plotting

        x_states =dict_top10.keys()
        y_vaccine_dist_1 = dict_top10.values()

        fig = plt.figure(figsize=(12, 6)) # figure chart with size
        ax = fig.add_subplot(111)

        # bar values filling with x-axis/ y-axis values
        ax.bar(np.arange(len(x_states)), y_vaccine_dist_1, log=1)

        # x-axis ticks range and labels
        ax.set_xticks(np.arange(len(x_states)))
        ax.set_xticklabels(x_states, rotation=20, zorder=100)

        # alternate option without .gcf
        plt.subplots_adjust(left=0.2, bottom=0.25)

        image_name = 'barchart.jpg' # image name
        plt.savefig(image_name, format=image_format, dpi=1200)
        plt.show()
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    in_file = "/Users/quangtn/Desktop/01_work/01_job/02_ml/" \
              "bentoml/prj/Chapter_2/csv_data/data/" \
              "cdc-moderna-covid-19-vaccine-distribution-by-state.csv"

    # create bag of words for feature research interest with NLTK
    matplotlib_pie_bar(in_file)
