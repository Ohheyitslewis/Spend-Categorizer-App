import pandas as pd
from classifier import load_taxonomy, classify_with_ollama

MODEL = "llama3"

def main():
    input_csv = "items.csv"
    output_csv = "items_classified.csv"

    # Load input
    df = pd.read_csv(input_csv, header=0)
    if "description" not in df.columns:
        raise ValueError("The input CSV must have a column named 'description'.")

    # Prep taxonomy
    taxonomy = load_taxonomy("taxonomy.json")

    # Classify each row
    descriptions = df["description"].astype(str).fillna("").tolist()
    results = []
    for i, desc in enumerate(descriptions, 1):
        res = classify_with_ollama(desc, taxonomy, model_name=MODEL)
        results.append(res.model_dump())
        if i % 10 == 0:
            print(f"Processed {i} rows...")

    # Normalize results into a flat DataFrame and align indexes
    results_df = pd.json_normalize(results)
    results_df.columns = [str(c) for c in results_df.columns]
    out = pd.concat(
        [df.reset_index(drop=True), results_df.reset_index(drop=True)],
        axis=1
    )

    # Save
    out.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

if __name__ == "__main__":
    main()

