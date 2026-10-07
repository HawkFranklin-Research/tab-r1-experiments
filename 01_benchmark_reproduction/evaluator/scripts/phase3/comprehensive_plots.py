import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
from scipy import interpolate
from typing import List, Dict

MODEL_COLORS = {
    'tabpfn': '#1f77b4',
    'catboost': '#d62728',
    'autogluon': '#9467bd',
    'xgboost': '#ff7f0e',
    'lightgbm': '#2ca02c',
    'random_forest': '#8c564b',
    'logistic_regression': '#7f7f7f'
}

class ComprehensiveVisualizer:
    """Advanced plotting engine for benchmark-wide analysis."""

    def __init__(self, runs_root: str, output_dir: str):
        self.runs_root = runs_root
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")
        self.logger = logging.getLogger("ComprehensiveVisualizer")
        logging.basicConfig(level=logging.INFO)

    def _load_all_predictions(self) -> List[Dict]:
        """Loads predictions from all successful runs."""
        all_preds = []
        for dataset_dir in os.listdir(self.runs_root):
            ds_path = os.path.join(self.runs_root, dataset_dir)
            if not os.path.isdir(ds_path): continue
            for run_timestamp in os.listdir(ds_path):
                run_path = os.path.join(ds_path, run_timestamp)
                pred_dir = os.path.join(run_path, "predictions")
                if not os.path.exists(pred_dir): continue
                for pred_file in os.listdir(pred_dir):
                    if pred_file.endswith("_predictions.csv"):
                        model_name = pred_file.replace("_predictions.csv", "")
                        df = pd.read_csv(os.path.join(pred_dir, pred_file))
                        all_preds.append({
                            "dataset": dataset_dir,
                            "model": model_name,
                            "df": df
                        })
        return all_preds

    def plot_roc_grid(self):
        """Generates a large grid of ROC curves for all datasets."""
        preds_list = self._load_all_predictions()
        datasets = sorted(list(set([p['dataset'] for p in preds_list])))
        if not datasets: return

        n_ds = len(datasets)
        cols = 3
        rows = (n_ds + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows), squeeze=False)
        from sklearn.metrics import roc_curve, auc

        for i, ds_name in enumerate(datasets):
            ax = axes[i // cols, i % cols]
            ds_preds = [p for p in preds_list if p['dataset'] == ds_name]
            
            has_plots = False
            for p in ds_preds:
                prob_cols = [c for c in p['df'].columns if c.startswith('prob_')]
                if len(prob_cols) != 2: continue
                prob_col = prob_cols[-1]
                
                try:
                    fpr, tpr, _ = roc_curve(p['df']['y_true_encoded'], p['df'][prob_col])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, label=f"{p['model'].title()} ({roc_auc:.3f})")
                    has_plots = True
                except ValueError as e:
                    self.logger.error(f"Skipping ROC for {p['model']} on {ds_name}: {str(e)}")
                    continue

            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_title(f"Dataset: {ds_name.replace('_dataset', '').title()}")
            if has_plots:
                ax.legend(fontsize='small', loc='lower right')
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')

        # Hide empty subplots
        for j in range(i + 1, rows * cols):
            axes[j // cols, j % cols].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "benchmark_roc_grid.png"), dpi=300)
        plt.close()

    def plot_mean_roc_envelope(self):
        """Generates a single ROC plot with Mean + Std Dev envelope across all datasets."""
        preds_list = self._load_all_predictions()
        models = sorted(list(set([p['model'] for p in preds_list])))
        from sklearn.metrics import roc_curve, auc

        plt.figure(figsize=(10, 8))
        mean_fpr = np.linspace(0, 1, 100)
        
        plot_data = []

        for model in models:
            model_preds = [p for p in preds_list if p['model'] == model]

            for p in model_preds:
                prob_cols = [c for c in p['df'].columns if c.startswith('prob_')]
                if len(prob_cols) != 2: continue
                prob_col = prob_cols[-1]
                
                try:
                    fpr, tpr, _ = roc_curve(p['df']['y_true_encoded'], p['df'][prob_col])
                    interp_tpr = np.interp(mean_fpr, fpr, tpr)
                    interp_tpr[0] = 0.0
                    for f, t in zip(mean_fpr, interp_tpr):
                        plot_data.append({"Model": model.title(), "FPR": f, "TPR": t})
                except ValueError:
                    continue
        
        if plot_data:
            df_plot = pd.DataFrame(plot_data)
                    
        # Extract base model name from label to assign color
        model_names_lower = [m.split(" (")[0].lower() for m in df_plot['Model'].unique()]
        palette = {label: MODEL_COLORS.get(name, '#333333') for label, name in zip(df_plot['Model'].unique(), model_names_lower)}
        sns.lineplot(data=df_plot, x="FPR", y="TPR", hue="Model", palette=palette, errorbar="sd", lw=2)

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.title("Benchmark-wide Mean ROC with Std Dev Envelope", fontsize=16)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "benchmark_mean_roc_envelope.png"), dpi=300)
        plt.close()

    def plot_performance_boxplot(self, summary_df: pd.DataFrame):
        """Generates boxplots of ROC AUC and Accuracy per model."""
        if summary_df.empty: return
        
        # Melt dataframe for Seaborn
        plot_df = summary_df.melt(
            id_vars=['model_name'], 
            value_vars=[col for col in ['roc_auc', 'accuracy'] if col in summary_df.columns], 
            var_name='Metric', value_name='Score'
        )
        
        if plot_df.empty: return

        plt.figure(figsize=(12, 7))
        sns.catplot(data=plot_df, x='model_name', y='Score', hue='Metric', kind='box', height=6, aspect=1.5)
        sns.stripplot(data=plot_df, x='model_name', y='Score', hue='Metric', dodge=True, alpha=0.5, palette='dark:black', legend=False)
        
        plt.title("Score Distribution across Datasets", fontsize=16)
        plt.ylim([0.5, 1.05])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "benchmark_score_distribution.png"), dpi=300)
        plt.close()

    def plot_radar_auc(self, summary_df: pd.DataFrame):
        """Generates a Radar plot using plotly to compare model strengths across datasets."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.logger.warning("Plotly not installed. Skipping radar plot.")
            return

        if summary_df.empty or 'roc_auc' not in summary_df.columns: return

        datasets = sorted(summary_df['dataset'].unique())
        models = sorted(summary_df['model_name'].unique())
        
        fig = go.Figure()
        
        for model in models:
            model_data = summary_df[summary_df['model_name'] == model]
            r_values = []
            for ds in datasets:
                val = model_data[model_data['dataset'] == ds]['roc_auc'].values
                if len(val) > 0 and not pd.isna(val[0]):
                    r_values.append(val[0])
                else:
                    r_values.append(0)
                    
            r_values.append(r_values[0])
            theta_values = datasets + [datasets[0]]
            
            is_tabpfn = 'tabpfn' in model.lower()
            fig.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself' if is_tabpfn else 'none',
                name=model.title(),
                line=dict(
                    color=MODEL_COLORS.get(model.lower(), '#333333'),
                    width=4 if is_tabpfn else 1.5,
                    dash='dot' if is_tabpfn else 'solid'
                ),
                opacity=0.9 if is_tabpfn else 0.4
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model ROC AUC across Datasets"
        )
        
        try:
            fig.write_image(os.path.join(self.output_dir, "all_models_auc_radar_plotly.png"))
        except ValueError as e:
            self.logger.warning(f"Could not save plotly image. {e}")


    def plot_performance_vs_time(self, tuning_df: pd.DataFrame):
        """
        Generates Performance vs Tuning Time budget curve (Fig 4c / Ext Data Fig 2).
        Requires columns: 'model_name', 'time_budget_s', 'score'
        """
        if tuning_df is None or tuning_df.empty: return
        
        plt.figure(figsize=(10, 6))
        tuning_df['model_lower'] = tuning_df['model_name'].str.lower()
        palette = {m: MODEL_COLORS.get(m, '#333333') for m in tuning_df['model_lower'].unique()}
        
        sns.lineplot(
            data=tuning_df, 
            x='time_budget_s', 
            y='score', 
            hue='model_lower', 
            palette=palette,
            marker='o',
            errorbar='sd',
            lw=2
        )
        
        plt.xscale('log')
        plt.title('Performance vs. Tuning Time', fontsize=16)
        plt.xlabel('Average Fit + Predict Time (s) [Log Scale]')
        plt.ylabel('Normalized Score')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles, labels=[str(l).title() for l in labels], title='Model')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_vs_time.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    # This is normally called by the Aggregator, but can be run standalone
    pass
