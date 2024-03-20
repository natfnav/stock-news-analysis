import nltk
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER Sentiment Lexicon: a built-in model used for sentiment analysis
#nltk.download("vader_lexicon")

finviz_url = "https://finviz.com/quote.ashx?t="

# List of tickers to analyze
tickers = ["SMR"]

# Dictionary containing news tables (value) for each ticker (key)
news_tables = {}

# Traverse tickers list
for ticker in tickers:
    url = finviz_url + ticker

    # Send request to FinViz website and retrieve HTML content
    req = Request(url=url, headers={"user-agent": "my-app"})
    response = urlopen(req)

    # Get HTML source code
    html = BeautifulSoup(response, "html.parser")

    # Stores news table (in html) for each ticker into dictionary
    news_table = html.find(id="news-table")
    news_tables[ticker] = news_table

parsed_data = []

for ticker, news_table in news_tables.items():
    for row in news_table.findAll("tr"):
        try:
            # Get news headline
            headline = row.a.text
            # Get news source
            source = row.span.text

            parsed_data.append([ticker, source, headline])

        except AttributeError:
            continue

# Create Pandas table and initialize sentiment analyzer
df = pd.DataFrame(parsed_data, columns=["ticker", "source", "headline"])
vader = SentimentIntensityAnalyzer()

# Add a "compound" column and calculate the compound for each headline
f = lambda headline: vader.polarity_scores(headline)["compound"]
df["compound"] = df["headline"].apply(f)

plt.figure(figsize=(10,8))

# Group by "ticker" and "source", and calculate the mean compound score
mean_df = df.drop("headline", axis=1).groupby(["ticker", "source"]).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs("compound", axis="columns").transpose()
mean_df.plot(kind="bar")
plt.show()