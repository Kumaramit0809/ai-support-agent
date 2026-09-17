import pandas as pd

df = pd.read_csv("eval/golden_set.csv", dtype=str, on_bad_lines="warn", engine="python")
print("Total rows:", len(df))
print("Columns:", list(df.columns))
print("Rows with missing gold_intent_id:", df["gold_intent_id"].isna().sum())
print("Rows with missing customer_tweet_id:", df["customer_tweet_id"].isna().sum())

bad = df[df["gold_intent_id"].isna() | df["customer_tweet_id"].isna()]
if len(bad) > 0:
    print("\nBad rows found, dropping them:")
    print(bad)

clean = df.dropna(subset=["gold_intent_id", "customer_tweet_id"]).drop_duplicates(subset=["customer_tweet_id"])
clean.to_csv("eval/golden_set_clean.csv", index=False)
print(f"\nWrote {len(clean)} clean rows -> eval/golden_set_clean.csv")