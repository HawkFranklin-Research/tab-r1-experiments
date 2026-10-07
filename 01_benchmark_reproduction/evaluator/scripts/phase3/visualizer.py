import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import logging

MODEL_COLORS = {
    'tabpfn': '#1f77b4',
    'catboost': '#d62728',
    'autogluon': '#9467bd',
    'xgboost': '#ff7f0e',
    'lightgbm': '#2ca02c',
    'random_forest': '#8c564b',
    'logistic_regression': '#7f7f7f'
}

class Visualizer:
    """Publication-grade visualization engine for Evaluate-TABPFN."""
    
    def __init__(self, run_dir: str, palette: str = "muted"):
        self.run_dir = run_dir
        self.output_dir = os.path.join(run_dir, "plots_phase3")
        os.makedirs(self.output_dir, exist_ok=True)
        
        with open(os.path.join(run_dir, "metadata/dataset_metadata.json"), 'r') as f:
            self.dataset_metadata = json.load(f)
        
        with open(os.path.join(run_dir, "metrics/metrics_summary.json"), 'r') as f:
            self.metrics_summary = json.load(f)
            
        self.task_type = self.dataset_metadata['task_type']
        self.dataset_name = self.dataset_metadata['dataset_name']
        
        sns.set_theme(style="whitegrid", palette=palette)
        plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
        
        self.logger = logging.getLogger("Visualizer")
        logging.basicConfig(level=logging.INFO)

    def _get_predictions(self, model_name: str) -> pd.DataFrame:
        pred_path = os.path.join(self.run_dir, f"predictions/{model_name}_predictions.csv")
        if os.path.exists(pred_path):
            return pd.read_csv(pred_path)
        return None

    def plot_classification_suite(self):
        """Generates ROC, PR, and Confusion Matrix plots."""
        if self.task_type not in ["binary", "multiclass"]:
            return

        successful_models = [m['model_name'] for m in self.metrics_summary['rows'] if m['status'] == 'success']
        if not successful_models: return

        # 1. Combined ROC Curve
        plt.figure(figsize=(10, 8))
        has_roc = False
        for model in successful_models:
            preds = self._get_predictions(model)
            if preds is None or 'y_true_encoded' not in preds.columns: continue
            
            prob_cols = [c for c in preds.columns if c.startswith('prob_')]
            if len(prob_cols) == 2:
                prob_col = prob_cols[-1]
                fpr, tpr, _ = roc_curve(preds['y_true_encoded'], preds[prob_col])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, color=MODEL_COLORS.get(model.lower(), '#333333'), label=f'{model.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
                has_roc = True
        
        if has_roc:
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves - {self.dataset_name.replace("_", " ").title()}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "combined_roc_curve.png"), dpi=300)
        plt.close()

        # 2. Combined Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        has_pr = False
        for model in successful_models:
            preds = self._get_predictions(model)
            if preds is None or 'y_true_encoded' not in preds.columns: continue
            
            prob_cols = [c for c in preds.columns if c.startswith('prob_')]
            if len(prob_cols) < 2: continue
            
            y_true = preds['y_true_encoded']
            if len(prob_cols) == 2:
                prob_col = prob_cols[-1]
                precision, recall, _ = precision_recall_curve(y_true, preds[prob_col])
                ap = average_precision_score(y_true, preds[prob_col])
            else:
                classes = list(range(len(prob_cols)))
                y_bin = label_binarize(y_true, classes=classes)
                precision_dict = {}
                recall_dict = {}
                ap_list = []
                for j in range(len(prob_cols)):
                    precision_dict[j], recall_dict[j], _ = precision_recall_curve(y_bin[:, j], preds[prob_cols[j]])
                    ap_list.append(average_precision_score(y_bin[:, j], preds[prob_cols[j]]))
                
                all_recall = np.unique(np.concatenate([recall_dict[j] for j in range(len(prob_cols))]))
                mean_precision = np.zeros_like(all_recall)
                for j in range(len(prob_cols)):
                    sort_idx = np.argsort(recall_dict[j])
                    mean_precision += np.interp(all_recall, recall_dict[j][sort_idx], precision_dict[j][sort_idx])
                mean_precision /= len(prob_cols)
                recall, precision = all_recall, mean_precision
                ap = np.mean(ap_list)
                
            plt.plot(recall, precision, lw=2, color=MODEL_COLORS.get(model.lower(), '#333333'), label=f'{model.replace("_", " ").title()} (AP = {ap:.3f})')
            has_pr = True
        
        if has_pr:
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves - {self.dataset_name.replace("_", " ").title()}')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "combined_pr_curve.png"), dpi=300)
        plt.close()

        # 3. Individual Confusion Matrices
        for model in successful_models:
            preds = self._get_predictions(model)
            if preds is None or 'y_true_encoded' not in preds.columns or 'y_pred_encoded' not in preds.columns: continue
            
            cm = confusion_matrix(preds['y_true_encoded'], preds['y_pred_encoded'])
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
            plt.title(f'Normalized Confusion Matrix: {model.replace("_", " ").title()}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{model}_confusion_matrix_norm.png"), dpi=300)
            plt.close()

    def plot_regression_suite(self):
        """Generates Actual vs Predicted and Residual plots."""
        if self.task_type != "regression":
            return

        successful_models = [m['model_name'] for m in self.metrics_summary['rows'] if m['status'] == 'success']
        if not successful_models: return
        
        plt.figure(figsize=(10, 8))
        all_y_true = []
        for model in successful_models:
            preds = self._get_predictions(model)
            if preds is None: continue
            plt.scatter(preds['y_true'], preds['y_pred'], alpha=0.5, label=model.replace("_", " ").title())
            all_y_true.extend(preds['y_true'].tolist())
        
        if all_y_true:
            min_val, max_val = min(all_y_true), max(all_y_true)
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Identity')
            
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted - {self.dataset_name.replace("_", " ").title()}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "combined_actual_vs_predicted.png"), dpi=300)
        plt.close()

        for model in successful_models:
            preds = self._get_predictions(model)
            if preds is None: continue
            
            residuals = preds['y_true'] - preds['y_pred']
            plt.figure(figsize=(10, 6))
            plt.scatter(preds['y_pred'], residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title(f'Residuals vs Predicted: {model.replace("_", " ").title()}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{model}_residuals.png"), dpi=300)
            plt.close()

    def run_all(self):
        self.logger.info(f"Starting visualization for {self.dataset_name}...")
        if self.task_type in ["binary", "multiclass"]:
            self.plot_classification_suite()
        elif self.task_type == "regression":
            self.plot_regression_suite()
        self.logger.info(f"Plots saved to {self.output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        viz = Visualizer(sys.argv[1])
        viz.run_all()
    else:
        print("Usage: python visualizer.py <run_directory>")
