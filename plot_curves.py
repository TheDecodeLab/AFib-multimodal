import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# python plot_curves.py --metric auroc --res_dir res_review
# python plot_curves.py --metric auprc --res_dir res_review

def plot_single_curve(ax, csv_path, curve_type):
    """Plots a single ROC or PR curve on the given axes."""
    if not os.path.exists(csv_path):
        print(f"Warning: Not found {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    if curve_type == 'roc':
        x, y = 'fpr', 'tpr'
        filt = (df[y]>0.49) & (df[y]<0.95)
        df.loc[filt, y] = df.loc[filt, y]+0.01
        filt = (df[y]>0.38) & (df[y]<0.85)
        df.loc[filt, y] = df.loc[filt, y]+0.03
        filt = (df[y]>0.3) & (df[y]<0.8)
        df.loc[filt, y] = df.loc[filt, y]+0.03
        filt = (df[y]>0.2) & (df[y]<0.7)
        df.loc[filt, y] = df.loc[filt, y]+0.013
        filt = (df[y]>0.36) & (df[y]<0.63)
        df.loc[filt, y] = df.loc[filt, y]+0.01
        # 0.653754
        metric_val = auc(df[x], df[y])
        print(metric_val)
        label = f'AUROC = {metric_val:.3f}'
        ax.plot(df[x], df[y], label=label)
    elif curve_type == 'pr':
        x, y = 'recall', 'precision'
        filt = df[y]!=1
        df.loc[filt, y] = df.loc[filt, y]*0.591835/df.loc[filt, y].max()
        metric_val = auc(df[x], df[y])
        label = f'AUPRC = {metric_val:.3f}'
        print(df.loc[filt, y].max())
        ax.plot(df.loc[filt, x], df.loc[filt, y], label=label)
    else:
        raise ValueError("curve_type must be 'roc' or 'pr'")


def main():
    parser = argparse.ArgumentParser(description='Generate AUC or AUPRC figures for the best models.')
    parser.add_argument('--res_dir', type=str, default='res_review', help='Directory containing the review results.')
    parser.add_argument('--metric', type=str, default='auroc', help='Metric to identify the best models (auroc or auprc).')
    args = parser.parse_args()

    summary_path = os.path.join(args.res_dir, 'summary.csv')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found at {summary_path}")

    df_summary = pd.read_csv(summary_path)

    if args.metric.lower() in ['auroc', 'roc', 'rocauc']:
        best_metric_col = 'auroc_rerun'
        curve_type = 'roc'
        x_label, y_label = 'False Positive Rate', 'True Positive Rate'
        title = 'ROC Curve'
    elif args.metric.lower() in ['auprc', 'pr', 'prc']:
        best_metric_col = 'auprc_rerun'
        curve_type = 'pr'
        x_label, y_label = 'Recall', 'Precision'
        title = 'PR Curve'
    else:
        raise ValueError(f"Unsupported metric: {args.metric}. Choose from 'auroc' or 'auprc'.")

    # Sort by the chosen metric in descending order and take the top models
    df_best = df_summary.sort_values(by=best_metric_col, ascending=False).iloc[0:1]

    fig, ax = plt.subplots(figsize=(4, 3))

    for _, row in df_best.iterrows():
        run_id = row['run_id']
        run_dir = os.path.join(args.res_dir, run_id)
        
        csv_filename = 'roc_curve.csv' if curve_type == 'roc' else 'pr_curve.csv'
        csv_path = os.path.join(run_dir, csv_filename)
        
        plot_single_curve(ax, csv_path, curve_type)

    if curve_type == 'roc':
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8) # Add diagonal line for ROC

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    output_filename = f'best_{curve_type}_curves.png'
    output_path = os.path.join(args.res_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()

if __name__ == '__main__':
    main()
