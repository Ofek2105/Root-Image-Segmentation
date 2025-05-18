import os
from tqdm import tqdm
import pandas as pd
from predict import validation

def multiple_datasets_evaluation(output_csv="comparison_results.csv"):
    paths = [
        r'runs/performance_datasets\dataset_baseline\weights\best.pt',
        r'runs/performance_datasets\dataset_randomwalk_all\weights\best.pt',
        r'runs/performance_datasets\dataset_bezier_all\weights\best.pt',
        r'runs/performance_datasets\dataset_channel_noise\weights\best.pt',
        r'runs/performance_datasets\dataset_gaussian_blurr\weights\best.pt',
        r'runs/performance_datasets\dataset_light_effect\weights\best.pt',
        r'runs/performance_datasets\dataset_perlin_all\weights\best.pt',
        r'runs/performance_datasets\dataset_real_all\weights\best.pt',
        r'runs/performance_datasets\dataset_root_darker_middle\weights\best.pt'
    ]

    all_metrics = {}

    for path in tqdm(paths):
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        metrics = validation(path, print_metrics_=False)
        all_metrics[dataset_name] = metrics

    df = pd.DataFrame(all_metrics).T  # Transpose to get datasets as rows

    baseline = df.loc["dataset_baseline"]
    df_diff = df.subtract(baseline)
    df_pct = df_diff.divide(baseline).multiply(100)

    # Rename columns to show what they represent
    df_diff.columns = [f"{col}_diff" for col in df.columns]
    df_pct.columns = [f"{col}_pct" for col in df.columns]

    # Combine all into one DataFrame
    combined_df = pd.concat([df, df_diff, df_pct], axis=1)

    # Export to CSV
    combined_df.to_csv(output_csv)

    return combined_df

def multiple_datasets_evaluation_html_pretty(output_html="comparison_tablev2.html"):
    base_path = r'runs\performance_datasets_70_epochs_v2'
    folder_names = os.listdir(base_path)
    paths = [fr'{base_path}\{folder_name}\weights\best.pt' for folder_name in folder_names]

    all_metrics = {}

    for path in tqdm(paths):
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        metrics = validation(path, print_metrics_=False)
        all_metrics[dataset_name] = metrics

    df = pd.DataFrame(all_metrics).T.round(4)
    baseline = df.loc["baseline"]
    df_pct = df.divide(baseline).subtract(1).multiply(100).round(2)

    def format_cell(val, pct):
        color = "green" if pct > 0 else "red" if pct < 0 else "gray"
        sign = "+" if pct > 0 else ""
        return f"<div style='font-family:monospace; text-align:center'>{val:.4f} <span style='font-size:10px; color:{color}; opacity:0.75'>({sign}{pct:.2f}%)</span></div>"

    styled_df = df.copy()
    for col in df.columns:
        styled_df[col] = [
            format_cell(df.loc[row, col], df_pct.loc[row, col])
            for row in df.index
        ]

    styled_df = styled_df.rename_axis("Dataset").reset_index()

    table_style = """
    <style>
    table {
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        width: 100%;
    }
    th, td {
        border: 1px solid #ccc;
        padding: 6px 8px;
        text-align: center;
        vertical-align: middle;
    }
    th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    </style>
    """

    html = table_style + styled_df.to_html(escape=False, index=False)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    return html


if __name__ == '__main__':

    multiple_datasets_evaluation_html_pretty(output_html="comparison_table_70_epochs_v2.html")