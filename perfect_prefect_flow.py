import requests
import time 
import duckdb
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prefect import task, flow
import matplotlib.pyplot as plt
import os

my_api_key = os.environ.get("my_api_key")

year_range = list(range(2024,2025)) # two years just to test
month_range=list(range(1,13,6))

# what i'd want to do is iterate through years. 
# i said i wanted to start at the beginning of the 20th century to now. 
base_url = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={my_api_key}"
db_path = "nyt_db.duckdb"

@task(retries=3, retry_delay_seconds=5, log_prints=True)
def load_relevant_data():
    con = duckdb.connect(db_path)
    con.execute(f"""
                CREATE TABLE IF NOT EXISTS "NYT_TEST5"(
                    headline VARCHAR,
                    snippet VARCHAR,
                    pubdate TIMESTAMP,
                    vader_score_title DOUBLE,
                    vader_score_snippet DOUBLE
                )
                """)

    analyzer = SentimentIntensityAnalyzer()

    rows = []
    for year in year_range:
        for month in month_range: 
            print(f"Fetching articles from month {month}, year {year}")

            url = base_url.format(year=year, month=month, my_api_key=my_api_key)

            response = requests.get(url)
            response.raise_for_status()

            payload = response.json()

            docs = payload["response"]["docs"]

            for article in docs:
                headline = article["headline"].get("main", "")
                snippet = article.get("snippet", "")
                pubdate = article.get("pub_date", "")

                vader_score_title = analyzer.polarity_scores(headline)["compound"]
                vader_score_snippet = analyzer.polarity_scores(snippet)["compound"]

                rows.append({
                    "headline": headline,
                    "snippet": snippet,
                    "pubdate": pubdate,
                    "vader_score_title": vader_score_title,
                    "vader_score_snippet": vader_score_snippet
                })
        time.sleep(12)

    test_df = pd.DataFrame(rows) 
    con.execute(f"INSERT INTO NYT_TEST5 SELECT * FROM test_df")
    print("im finished")


@task(retries=3, retry_delay_seconds=5, log_prints=True)
def transform_data(): 
    con = duckdb.connect(db_path)
    con.execute(f"""CREATE TABLE NYT_TEST5_FINAL AS
                SELECT
                DATE_TRUNC('month', pubdate) AS month,
                AVG(vader_score_title) AS mean_vscore_title,
                AVG(vader_score_snippet) AS mean_vscore_snippet
                FROM NYT_TEST3
                GROUP BY DATE_TRUNC('month', pubdate)
                ORDER BY month;
                """)
    print("im finished x2")

@task(retries=3, retry_delay_seconds=5, log_prints=True)
def generate_chart():
    con = duckdb.connect("nyt_db.duckdb")

    # Load transformed data
    df = con.execute("SELECT * FROM NYT_TEST5_FINAL ORDER BY month").df()

    print(df.head())

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["month"], df["mean_vscore_title"], label="Title Sentiment")
    plt.plot(df["month"], df["mean_vscore_snippet"], label="Snippet Sentiment")

    plt.xlabel("Month")
    plt.ylabel("Average Sentiment Score")
    plt.title("NYT Sentiment Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = "nyt_sentiment_plot.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    return output_path



@flow
def main():
    load_relevant_data()
    transform_data()
    generate_chart()

if __name__ == "__main__":
    main()