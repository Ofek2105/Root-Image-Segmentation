import pandas as pd

"""
This script takes two reports (csv files) and compares them
by creating a merged csv
"""
def merge_csvs(model_csv_path, hand_annotated_csv_hairs_folder, hand_annotated_csv_root_folder, destination_path):
    model_df = pd.read_csv(model_csv_path)
    hand_annotated_hairs_df = pd.read_csv(hand_annotated_csv_hairs_folder, header=None, names=['label', 'annotated_hair_length'])
    hand_annotated_root_df = pd.read_csv(hand_annotated_csv_root_folder, header=None, names=['label', 'annotated_root_length'])

    # Merge the dataframes on the 'label' column
    merged_df = pd.merge(model_df, hand_annotated_hairs_df, on='label', how='left')
    merged_df = pd.merge(merged_df, hand_annotated_root_df, on='label', how='left')

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(destination_path, index=False)

    # Display the merged DataFrame
    print(merged_df)

if __name__ == '__main__':

    # Define file paths
    model_csv_path = r'images_to_test\results.csv'
    hand_annotated_csv_hairs_folder = r'C:\Users\Ofek\Desktop\Ofek_Root hairs\crops\TRL_hairs.csv'
    hand_annotated_csv_root_folder = r'C:\Users\Ofek\Desktop\Ofek_Root hairs\crops\TRL_main.csv'
    destination_path = r'C:\Users\Ofek\Desktop\Merged_Results.csv'

    merge_csvs(model_csv_path, hand_annotated_csv_hairs_folder, hand_annotated_csv_root_folder, destination_path)
