from os import listdir
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv('/Users/anuratibhosekar/Desktop/UMASS/Fall_23/COMPSCI 532/Final project/valorant-stats-master/agents_data/complete_file.csv')
# df=df.fillna(value=0)
# print(df.head())
# print(df.info())
# df.drop("Unnamed: 0.1", axis=1, inplace=True)
# df.drop("Unnamed: 0.2", axis=1, inplace=True)
# df.drop("Unnamed: 0", axis=1, inplace=True)
# df["Damage Per Round"] = label


# df['Headshot'] = df['Headshot'].str.rstrip('%').astype('float') / 100.0
# df['Bodyshot'] = df['Bodyshot'].str.rstrip('%').astype('float') / 100.0
# df['Legshot'] = df['Legshot'].str.rstrip('%').astype('float') / 100.0
# print(df.head())

# le = LabelEncoder()
# label = le.fit_transform(df['Agent Name'])
# df['Agent Name'] = label
# df['Win Rate'] = df['Win Rate'].str.rstrip('%').astype('float') / 100.0
# df['Pick Rate'] = df['Pick Rate'].str.rstrip('%').astype('float') / 100.0
# df['First Blood'] = df['First Blood'].str.rstrip('%').astype('float') / 100.0
# df['Num Matches'] = df['Num Matches'].replace(',', '', regex=True)
# df['Num Matches'] = df['Num Matches'].astype(float)
# df['Label'] = pd.cut(x=df['Win Rate'], bins=[0, 0.3, 0.7, 1], labels=[0, 1, 2])
# df.drop("Win Rate", axis=1, inplace=True)
# print(df.head())
# print(df.info())


# df = pd.read_csv('/Users/anuratibhosekar/Desktop/UMASS/Fall_23/COMPSCI 532/Final project/valorant-stats-master/weapons_data/complete_file.csv')
# df['Label'] = pd.cut(x=df['Damage Per Round'], bins=[0, 100, 120, 1000], labels=[0, 1, 2])
# print(df.head())
# df.drop("Unnamed: 0.1", axis=1, inplace=True)
# df.drop("Unnamed: 0.2", axis=1, inplace=True)
# df.drop("Unnamed: 0", axis=1, inplace=True)
# df["Damage Per Round"] = label


# df['Headshot'] = df['Headshot'].str.rstrip('%').astype('float') / 100.0
# df['Bodyshot'] = df['Bodyshot'].str.rstrip('%').astype('float') / 100.0
# df['Legshot'] = df['Legshot'].str.rstrip('%').astype('float') / 100.0
# print(df.head())

# le = LabelEncoder()
# label = le.fit_transform(df['Weapon Name'])
# # df.drop("Weapon Name", axis=1, inplace=True)
# df["Weapon Name"] = label


# print(df.head(25))
# df.to_csv('/Users/anuratibhosekar/Desktop/UMASS/Fall_23/COMPSCI 532/Final project/valorant-stats-master/agents_data/complete_file.csv', index=False)
# print(df.describe)

# df_list = []
# the path
# directory_path = '/Users/anuratibhosekar/Desktop/UMASS/Fall_23/COMPSCI 532/Final project/valorant-stats-master/weapons_data'
# folderList = listdir(directory_path)
# for folder in folderList:
#     folder_path = os.path.join(directory_path, folder)
#     fileList = listdir(folder_path)
#     csv_files = [f for f in fileList if f.endswith('.csv')]
#     for csv in csv_files:
#         file_path = os.path.join(folder_path, csv)
#         try:
#             # Try reading the file using default UTF-8 encoding
#             df = pd.read_csv(file_path)
#             df_list.append(df)
#         except UnicodeDecodeError:
#             try:
#                 # If UTF-8 fails, try reading the file using UTF-16 encoding with tab separator
#                 df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
#                 df_list.append(df)
#             except Exception as e:
#                 print(f"Could not read file {csv} because of error: {e}")
#         except Exception as e:
#             print(f"Could not read file {csv} because of error: {e}")
#     # print(fileList)
# # Concatenate all data into one DataFrame
# big_df = pd.concat(df_list, ignore_index=True)

# # Save the final result to a new CSV file
# big_df.to_csv(os.path.join(directory_path, 'complete_file.csv'), index=False)

# path = '/Users/anuratibhosekar/Desktop/UMASS/Fall_23/COMPSCI 532/Final project/valorant-stats-master/weapons_data/complete_file.csv'
# df = pd.read_csv(path)
# try:
#     df.drop('Unnamed: 0', axis=1, inplace=True)
# except KeyError:
#     print('Column not found')

# df.to_csv(path, index=False)

# # Dmamage per round
# 0 -> low -> <100
# 1 -> avergae -> 100-120
# 2 -> high -> >120

# # Win rate
# 0 -> low -> <0.3
# 1 -> average -> 0.3-0.7
# 2 -> high -> >0.7