import pandas as pd

df = pd.read_csv("submit_strange_format.csv")


def make_list(row):
    vals = []
    for i in range(1, 6):
        v = row.get(f"web_id_{i}")
        if pd.notna(v):
            vals.append(int(v))
    return "[" + ", ".join(str(v) for v in vals) + "]"


out = pd.DataFrame(
    {
        "q_id": df["q_id"],
        "web_list": df.apply(make_list, axis=1),
    }
)

out.to_csv("submit_NEW.csv", index=False)