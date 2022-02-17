import pandas as pd

samples_database_path = './samples_database.csv'
df = pd.read_csv(samples_database_path)

if 'recognized' not in df:
    df = df.reindex(columns=df.columns.tolist() + ['recognized'])

# df.loc[2, "recognized"] = True

print("\n", df)

print("\n\nThis script tags samples in their 'recognizable' column.")
print("Iterate through already tagged samples? (y/*)")

q = input().lower()

tag_again = 0
if q == 'y':
    tag_again = 1

print("\nRecognized: 'y' | Unrecognized: * | Quit: 'q'\n")


if tag_again:
    # iterate through every row
    for index, row in df.iterrows():
        print("\n", row)

        q = input().lower()
        if q == "q":
            break
        elif q == "y":
            df.loc[index, "recognized"] = True
        else:
            df.loc[index, "recognized"] = False

else:
    # iterate through every row
    for index, row in df.iterrows():

        #  Only recog column is empty
        if pd.isnull(row['recognized']):
            print("\n", row)

            q = input().lower()
            if q == "q":
                break
            elif q == "y":
                df.loc[index, "recognized"] = True
            else:
                df.loc[index, "recognized"] = False


print("\nSaving csv...\n")

# Write to output csv file
df.to_csv(samples_database_path, index=False)

print(df)
print()

print("Recognized samples:\n")
print(df[df['recognized'] == True])
