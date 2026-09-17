import csv
import pandas as pd

VALID_INTENTS = [
    "general_feedback", "customer_service_complaint", "international_support_fr",
    "delivery_delay", "return_pickup_failure", "international_support_es",
    "pending_resolution_followup", "media_link_submission", "order_status_and_payment",
    "missing_or_misdelivered_package", "general_inquiry_and_error",
]

agent_df = pd.read_csv("data/processed/agent_outputs.csv")
agent_df = agent_df[agent_df["intent_id"] != "ERROR"].reset_index(drop=True)

out_path = "eval/golden_set_final.csv"
done_ids = set()
try:
    existing = pd.read_csv(out_path, dtype=str)
    done_ids = set(existing["customer_tweet_id"].astype(str))
except FileNotFoundError:
    pass

print(f"{len(agent_df)} total agent messages, {len(done_ids)} already labeled.\n")
print("Valid intent ids:")
for v in VALID_INTENTS:
    print(f"  {v}")
print()

write_header = len(done_ids) == 0
with open(out_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["customer_tweet_id", "customer_text", "gold_intent_id", "gold_escalate", "gold_escalate_reason"])

    for _, row in agent_df.iterrows():
        tid = str(row["customer_tweet_id"])
        if tid in done_ids:
            continue

        print("-" * 70)
        print(f"Customer: {row['customer_text']}")
        print(f"[Agent predicted intent]: {row['intent_id']}  [escalate]: {row['escalate']}")

        while True:
            intent = input("\nGold intent (exact id from list above): ").strip()
            if intent in VALID_INTENTS:
                break
            print(f"  NOT VALID. Must be exactly one of: {', '.join(VALID_INTENTS)}")

        while True:
            esc = input("Escalate? (y/n only): ").strip().lower()
            if esc in ("y", "n"):
                break
            print("  Type only 'y' or 'n'.")

        reason = input("Reason (keep it short, one line): ").strip()

        writer.writerow([tid, row["customer_text"], intent, esc == "y", reason])
        f.flush()
        done_ids.add(tid)
        print(f"Saved. ({len(done_ids)}/{len(agent_df)} total done)\n")

print(f"\nDone for now. {len(done_ids)}/{len(agent_df)} labeled -> {out_path}")