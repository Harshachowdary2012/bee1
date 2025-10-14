from datasets import load_dataset

ds = load_dataset("kdave/Indian_Financial_News")

# Display dataset structure
print("Dataset structure:")
print(ds)
print("\n" + "="*80 + "\n")

# Display information about the train split
print("Train split info:")
print(f"Number of rows: {len(ds['train'])}")
print(f"Features: {ds['train'].features}")
print("\n" + "="*80 + "\n")

# Display first 5 examples
print("First 5 examples from the dataset:")
for i in range(min(5, len(ds['train']))):
    print(f"\nExample {i+1}:")
    print(ds['train'][i])
    print("-" * 80)

# Display a sample using pandas for better formatting (if data fits in memory)
try:
    import pandas as pd
    df = pd.DataFrame(ds['train'][:10])  # First 10 rows
    print("\n" + "="*80 + "\n")
    print("First 10 rows in table format:")
    print(df.to_string())
    print("\n" + "="*80 + "\n")
    print("Data types:")
    print(df.dtypes)
except Exception as e:
    print(f"\nCouldn't display as DataFrame: {e}")