import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tabulate import tabulate
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import traceback
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")
print(stopwords.words("english"))

stop_words = set(stopwords.words('english'))

lst_research_interest = []

ps = PorterStemmer()

BoW = []

def create_bow_with_nltk(in_file, out_file_word_token, out_file):
    """
    Create a bog of words for feature "research_interest" from Google Scholar data set and saved into
    a csv file named "output_bow.csv"
    :param in_file:
    :param out_file_word_token:
    :param out_file:
    :return:
    """

    try:
        # read csv file Google scholar data set output.csv
        df = pd.read_csv(in_file)

        # write csv file for Bag of Words
        f = open(out_file, "w")

        # write csv file for words tokens of research_interest
        f_wt = open(out_file_word_token, "w")

        # header for the output csv with research interest bag of words
        header = "author.name, email, affiliation, coauthors_names, research_bag_of_words\n"
        f_wt.write(header)

        # read each line in dataframe (i.e., each line of input file)
        for index, row in df.iterrows():
            curr_authorname = row['author_name']
            curr_email = row['email']
            curr_affiliation = row['affiliation']
            curr_coauthors_names = row['coauthors_names']
            curr_research_interest = str(row['research_interest']).replace("##", " ").replace("_", " ")
            lst_research_interest.append(curr_research_interest)

            # word tokenizer
            curr_research_int_token = word_tokenize(curr_research_interest)

            # remove stop words from the word tokens
            curr_filtered_research = [w for w in curr_research_int_token\
                                      if not w.lower() in stop_words]

            # word stemming (porter stemmer)
            curr_stem_research = [ps.stem(w) for w in curr_filtered_research]

            # construct line having all features
            curr_line = f"{curr_authorname}, {curr_email}, {curr_affiliation}," \
                        f"{curr_coauthors_names}, {curr_stem_research}"

            # write to csv file
            f_wt.write(curr_line + "\n")

        # 1- gram (i.e, single word token used for Bow creation)
        count_vector = CountVectorizer(ngram_range=(1, 1), stop_words="english")

        # transform the list of research list from all authors
        count_fit = count_vector.fit_transform(lst_research_interest)

        # create dataframe
        df_bow = pd.DataFrame(count_fit.toarray(), columns=count_vector.get_feature_names_out())
        print(tabulate(df_bow.head(10), headers="keys", tablefmt="psql"))

        df_bow.to_csv(out_file, index=None, header=True)
        f.close()
    except:
        traceback.print_exc()

def create_tf_idf_with_scikit():
    try:
        # create an instance for tf-idf vectorized
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)

        # user the td-idf instance to fit list of research_interest (feature)
        tfidf = tfidf_vectorizer.fit_transform(lst_research_interest)

        # tfidf[0].T.todense() provides the tf-idf dense vector calculated for the research_interest
        df = pd.DataFrame(tfidf[0].T.todense(), index=tfidf_vectorizer.get_feature_names_out() \
                          ,columns=["tf-idf"])

        # sort the tf-idf calculated using 'sort-values' of dataframe.
        df = df.sort_values('tf-idf', ascending=False)

        # top 30 words with highes tf-idf
        print(df.head(30))
    except:
        traceback.print_exc()

if __name__ == "__main__":
    in_file = "/Users/quangtn/Desktop/01_work/01_job/02_ml/" \
              "bentoml/prj/Chapter_2/google_scholar/output.csv"

    out_file_word_token = "./out_file_word_token.csv"
    out_file = "./output_bow.csv"

    # create bag of word for feature research interest with NLTK
    create_bow_with_nltk(in_file, out_file_word_token, out_file)

    # create tf-idf for feature research interest with Scikit-learn
    create_tf_idf_with_scikit()