from bs4 import BeautifulSoup
import requests
from requests import get
import random
import sys
import subprocess

# output file
f = open("./out_author.jpg", "w")

def extract_img_url(url):
    # http request
    response = get(url)

    # inside response, we have text field/ parameter
    html_soup = BeautifulSoup(response.text, "html.parser")

    # find all image tag and iterate
    for i in html_soup.select("img"):
        # get "src" attribute from the <img> tag
        link = i["src"]

        # author photo contains string view_photo, so match based on that
        if link.find("view_photo") >= 0:
            # print the image to output file
            print(link)

            # write the image to output file
            f.write(requests.get(link).content)

if __name__ == "__main__":
    csv_lin = ""

    # seed ur to start with
    url = "https://scholar.google.com/citations?user=VlJQwSgAAAAJ&hl=en&oi=ao"
    extract_img_url(url)