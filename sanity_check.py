import argparse
import textwrap

import pandas as pd


def load_data(questions_csv: str, submit_csv: str, websites_csv: str):
    qdf = pd.read_csv(questions_csv)
    sdf = pd.read_csv(submit_csv)
    wdf = pd.read_csv(websites_csv)

    wdf["web_id"] = wdf["web_id"].astype(int).astype(str)
    w_index = wdf.set_index("web_id")

    return qdf, sdf, w_index


def show_samples(qdf, sdf, w_index, num_samples: int = 5, wrap: int = 120):
    merged = qdf.merge(sdf, on="q_id", how="inner")
    if merged.empty:
        print("[WARN] Нет пересечения q_id между questions и submit")
        return

    num_samples = min(num_samples, len(merged))
    subset = merged.head(num_samples)

    for _, row in subset.iterrows():
        q_id = row["q_id"]
        query = str(row["query"])
        print("=" * 80)
        print(f"q_id: {q_id}")
        print("QUERY:")
        print(textwrap.fill(query, width=wrap))
        print("\nTop-5 predicted web_id:")

        for i in range(1, 6):
            col = f"web_id_{i}"
            if col not in row:
                continue
            web_id = row[col]
            if pd.isna(web_id) or web_id == "":
                continue
            web_id_str = str(int(float(web_id)))
            print("-" * 40)
            print(f"#{i}: web_id = {web_id_str}")
            if web_id_str in w_index.index:
                wrow = w_index.loc[web_id_str]
                title = str(wrow.get("title", ""))
                text = str(wrow.get("text", ""))
                snippet = text[:wrap * 3].replace("\n", " ")
                print("TITLE:")
                print(textwrap.fill(title, width=wrap))
                print("TEXT SNIPPET:")
                print(textwrap.fill(snippet, width=wrap))
            else:
                print("[WARN] web_id not found in websites table")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_csv", required=True)
    parser.add_argument("--submit_csv", required=True)
    parser.add_argument("--websites_csv", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    qdf, sdf, w_index = load_data(args.questions_csv, args.submit_csv, args.websites_csv)
    show_samples(qdf, sdf, w_index, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
