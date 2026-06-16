import os
import shutil
import json
import argparse
from glob import glob
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def map_metric_name(name: str) -> Tuple[str, bool]:
    key = name.strip().lower()
    mapping = {
        'auroc': ('AUROC', True),
        'rocauc': ('AUROC', True),
        'auc': ('AUROC', True),
        'accuracy': ('Accuracy', True),
        'acc': ('Accuracy', True),
        'f1': ('F1 score', True),
        'f1_score': ('F1 score', True),
        'precision': ('Precision', True),
        'recall': ('Sensitivity', True),
        'sensitivity': ('Sensitivity', True),
        'specificity': ('Specificity', True),
    }
    if key not in mapping:
        raise ValueError(f"Unknown metric '{name}'. Supported: auroc, accuracy, f1, precision, recall/sensitivity, specificity")
    return mapping[key]


def load_ready_data(use_bmi_filter: bool = False):
    ehr_data = pd.read_csv('ready_EHR.csv')
    [ecg_data, ecg_data_denoise, ecg_data_denoise_filtered, fft_data, fft_fake, ecg_fake] = np.load('ready_ECG.npy', allow_pickle=True)
    y_dl = pd.read_csv('ready_Y.csv')

    if use_bmi_filter and 'BMI @ Index Stroke' in ehr_data.columns:
        bmi_filter = ehr_data['BMI @ Index Stroke'] <= 35
        ehr_data = ehr_data[bmi_filter].reset_index(drop=True)
        ecg_data = ecg_data[bmi_filter]
        ecg_data_denoise = ecg_data_denoise[bmi_filter]
        ecg_data_denoise_filtered = ecg_data_denoise_filtered[bmi_filter]
        fft_data = fft_data[bmi_filter]
        fft_fake = fft_fake[bmi_filter]
        ecg_fake = ecg_fake[bmi_filter]
        y_dl = y_dl[bmi_filter].reset_index(drop=True)

    return ehr_data, ecg_data, ecg_data_denoise, ecg_data_denoise_filtered, fft_data, fft_fake, ecg_fake, y_dl


def normalize_inputs(ehr_data: pd.DataFrame,
                     ecg_data_variants: Dict[str, np.ndarray],
                     ecg_app: int) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    cols_to_drop = ['MRN', 'Mechanical Valve based on data from Index Stroke\n\n1:No\n2:Yes']
    ehr_cols = [c for c in ehr_data.columns if c not in cols_to_drop]
    ehr_proc = ehr_data[ehr_cols].copy()
    ehr_min = ehr_proc.min()
    ehr_max = ehr_proc.max()
    ehr_proc = (ehr_proc - ehr_min) / (ehr_max - ehr_min)

    if ecg_app == 0:
        ecg_base = ecg_data_variants['raw']
    elif ecg_app == 1:
        ecg_base = ecg_data_variants['denoise']
    elif ecg_app == 2:
        ecg_base = ecg_data_variants['denoise_filtered']
    elif ecg_app == 3:
        ecg_base = ecg_data_variants['fft']
    elif ecg_app == 4:
        ecg_base = ecg_data_variants['fft_fake']
    elif ecg_app == 5:
        ecg_base = ecg_data_variants['ecg_fake']
    else:
        raise ValueError(f"Unsupported ecg_app: {ecg_app}")

    ecg_min = np.min(ecg_base, axis=(0, 1))
    ecg_max = np.max(ecg_base, axis=(0, 1))
    ecg_proc = (ecg_base - ecg_min) / (ecg_max - ecg_min)

    return ehr_proc, ecg_proc, ehr_cols


# Model definition mirrors experiment_on_best*.py
import keras
from keras import layers, Model


def build_ecg_compressor_factory(com_arch: int):
    if com_arch == 0:
        def ecg_compressor(ecg_input, n_comp, num_heads=4, key_dim=32):
            ecg_conv = layers.Conv1D(64, kernel_size=5, activation='relu')(ecg_input)
            ecg_conv = layers.BatchNormalization()(ecg_conv)
            ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
            ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
            ecg_reshape = layers.Reshape((1, 32))(ecg_pool)
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(ecg_reshape, ecg_reshape)
            attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ecg_reshape)
            ecg_compressed = layers.Dense(n_comp, activation='relu')(attention_output)
            ecg_compressed = layers.GlobalAveragePooling1D()(ecg_compressed)
            return ecg_compressed
        return ecg_compressor
    if com_arch == 1:
        def ecg_compressor(ecg_input, n_comp, num_heads=4, key_dim=32):
            ecg_conv = layers.Conv1D(64, kernel_size=5, activation='relu')(ecg_input)
            ecg_conv = layers.BatchNormalization()(ecg_conv)
            ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
            ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
            ecg_reshape = layers.Reshape((1, 32))(ecg_pool)
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(ecg_reshape, ecg_reshape)
            attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ecg_reshape)
            attention_output2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(attention_output, attention_output)
            attention_output2 = layers.LayerNormalization(epsilon=1e-6)(attention_output2 + attention_output)
            ecg_compressed = layers.Dense(n_comp, activation='relu')(attention_output2)
            ecg_compressed = layers.GlobalAveragePooling1D()(ecg_compressed)
            return ecg_compressed
        return ecg_compressor
    if com_arch == 2:
        def ecg_compressor(ecg_input, n_comp, num_heads=4, key_dim=32):
            ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_input)
            ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
            ecg_reshape = layers.Reshape((1, 32))(ecg_pool)
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(ecg_reshape, ecg_reshape)
            attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ecg_reshape)
            ecg_compressed = layers.Dense(n_comp, activation='relu')(attention_output)
            ecg_compressed = layers.GlobalAveragePooling1D()(ecg_compressed)
            return ecg_compressed
        return ecg_compressor
    if com_arch == 3:
        def ecg_compressor(ecg_input, n_comp):
            ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_input)
            ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
            ecg_conv = layers.Conv1D(8, kernel_size=5, activation='relu')(ecg_conv)
            ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
            ecg_compressed = layers.Dense(n_comp, activation='relu')(ecg_pool)
            return ecg_compressed
        return ecg_compressor
    if com_arch == 4:
        def ecg_compressor(ecg_input, n_comp):
            ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_input)
            ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
            ecg_conv = layers.Conv1D(8, kernel_size=5, activation='relu')(ecg_conv)
            ecg_conv = layers.Conv1D(4, kernel_size=5, activation='relu')(ecg_conv)
            ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
            ecg_compressed = layers.Dense(n_comp, activation='relu')(ecg_pool)
            return ecg_compressed
        return ecg_compressor
    if com_arch == 5:
        def ecg_compressor(ecg_input, n_comp):
            ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_input)
            ecg_conv = layers.Conv1D(32, kernel_size=5, activation='relu')(ecg_conv)
            ecg_conv = layers.Conv1D(8, kernel_size=5, activation='relu')(ecg_conv)
            ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
            ecg_compressed = layers.Dense(2 * n_comp, activation='relu')(ecg_pool)
            ecg_compressed = layers.Dense(n_comp, activation='relu')(ecg_compressed)
            return ecg_compressed
        return ecg_compressor
    if com_arch == 6:
        def ecg_compressor(ecg_input, n_comp):
            ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_input)
            ecg_conv = layers.Conv1D(16, kernel_size=5, activation='relu')(ecg_conv)
            ecg_conv = layers.Conv1D(8, kernel_size=5, activation='relu')(ecg_conv)
            ecg_pool = layers.GlobalAveragePooling1D()(ecg_conv)
            ecg_compressed = layers.Dense(n_comp, activation='relu')(ecg_pool)
            return ecg_compressed
        return ecg_compressor
    raise ValueError(f"Unsupported com_arch: {com_arch}")


def build_model(ecg_shape: Tuple[int, int],
                ehr_shape: Tuple[int],
                n_comp_ecg: int,
                n_comp_ehr: int,
                com_arch: int,
                num_heads: int,
                initial_learning_rate: float):
    ecg_input = layers.Input(shape=ecg_shape, name='ecg_input')
    ehr_input = layers.Input(shape=ehr_shape, name='ehr_input')

    ecg_compressor = build_ecg_compressor_factory(com_arch)
    if n_comp_ecg != 0:
        try:
            ecg_compressed = ecg_compressor(ecg_input, n_comp=n_comp_ecg, num_heads=num_heads, key_dim=32)
        except TypeError:
            ecg_compressed = ecg_compressor(ecg_input, n_comp=n_comp_ecg)
    else:
        ecg_compressed = None

    if n_comp_ehr != 0:
        ehr_processed = layers.Dense(2 * n_comp_ehr, activation='relu')(ehr_input)
        ehr_processed = layers.Dense(n_comp_ehr, activation='relu')(ehr_processed)
    else:
        ehr_processed = None

    if n_comp_ecg * n_comp_ehr != 0:
        fused = layers.Concatenate()([ecg_compressed, ehr_processed])
    elif n_comp_ecg == 0:
        fused = ehr_processed
    elif n_comp_ehr == 0:
        fused = ecg_compressed
    else:
        raise ValueError('Both n_comp_ecg and n_comp_ehr are zero')

    hidden = layers.Dense(max(1, (n_comp_ecg + n_comp_ehr) // 2), activation='relu')(fused)
    hidden = layers.Dense(max(1, (n_comp_ecg + n_comp_ehr) // 4), activation='relu')(hidden)
    output = layers.Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=[ecg_input, ehr_input], outputs=output)
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def augment_ecg(ecg_data: np.ndarray, aug_p: float = 0.2, num_augmented: int = 1) -> np.ndarray:
    # Simplified: Gaussian noise and small shifts
    augmented = []
    for _ in range(num_augmented):
        x = ecg_data.copy()
        if np.random.rand() < aug_p:
            noise = np.random.normal(0.0, 0.01, size=x.shape)
            x = x + noise
        if np.random.rand() < aug_p:
            shift = np.random.randint(-4, 5)
            x = np.roll(x, shift, axis=0)
        augmented.append(x)
    return np.concatenate(augmented, axis=0)


def balance_data(x_ecg: np.ndarray, x_ehr: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    minority_class = int(y.mean() < 0.5)
    filt_min = y == minority_class
    filt_maj = ~filt_min
    x_ecg_min, x_ehr_min, y_min = x_ecg[filt_min], x_ehr[filt_min], y[filt_min]
    x_ecg_maj, x_ehr_maj, y_maj = x_ecg[filt_maj], x_ehr[filt_maj], y[filt_maj]
    if len(y_min) == 0 or len(y_maj) == 0:
        return x_ecg, x_ehr, y
    sample_index = np.random.choice(np.arange(len(y_min)), size=len(y_maj), replace=True)
    x_ecg_bal = x_ecg_min[sample_index]
    x_ehr_bal = x_ehr_min[sample_index]
    y_bal = y_min[sample_index]
    x_ecg_out = np.concatenate([x_ecg_maj, x_ecg_bal], axis=0)
    x_ehr_out = np.concatenate([x_ehr_maj, x_ehr_bal], axis=0)
    y_out = np.concatenate([y_maj, y_bal], axis=0)
    return x_ecg_out, x_ehr_out, y_out


def train_and_predict(config: Dict[str, Any],
                      ehr_proc: pd.DataFrame,
                      ecg_proc: np.ndarray,
                      y_series: pd.Series,
                      n_splits: int = 5,
                      epochs: int = 100) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n_inc = int(config['n_inc'])
    ecg_shape = (ecg_proc.shape[1], ecg_proc.shape[2])
    ehr_shape = (ehr_proc.shape[1],)
    y_values = y_series.values.reshape(-1)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all = []
    y_score_all = []
    last_artifacts = {}

    for fold_idx, (train_index, test_index) in enumerate(kf.split(y_values)):
        ehr_train = ehr_proc.iloc[train_index].values
        ehr_test = ehr_proc.iloc[test_index].values
        ecg_train = ecg_proc[train_index]
        ecg_test = ecg_proc[test_index]
        y_train = y_values[train_index]
        y_test = y_values[test_index]

        # Train-time augmentation
        aug_list = []
        for _ in range(n_inc - 1):
            d_aug = []
            for j in range(ecg_train.shape[0]):
                d_aug.append(augment_ecg(ecg_train[j], aug_p=float(config['aug_p']), num_augmented=1))
            aug_list.append(d_aug)
        ehr_train_rep = np.concatenate([ehr_train] * n_inc, axis=0)
        ecg_train_aug = np.concatenate([ecg_train] + aug_list, axis=0)
        y_train_rep = np.concatenate([y_train] * n_inc, axis=0)
        inds = np.arange(y_train_rep.shape[0])
        np.random.shuffle(inds)
        ehr_train_rep = ehr_train_rep[inds]
        ecg_train_aug = ecg_train_aug[inds]
        y_train_rep = y_train_rep[inds]
        ecg_train_bal, ehr_train_bal, y_train_bal = balance_data(ecg_train_aug, ehr_train_rep, y_train_rep)

        model = build_model(
            ecg_shape=ecg_shape,
            ehr_shape=ehr_shape,
            n_comp_ecg=int(config['n_comp_ecg']),
            n_comp_ehr=int(config['n_comp_ehr']),
            com_arch=int(config['com_arch']),
            num_heads=int(config['num_heads']),
            initial_learning_rate=float(config['initial_learning_rate']),
        )

        model.fit([ecg_train_bal, ehr_train_bal], y_train_bal, epochs=epochs, batch_size=32, validation_split=0.0, verbose=0)

        # Test-time augmentation and probability averaging
        aug_test_list = []
        for _ in range(n_inc - 1):
            d_aug_t = []
            for j in range(ecg_test.shape[0]):
                d_aug_t.append(augment_ecg(ecg_test[j], aug_p=float(config['aug_p']), num_augmented=1))
            aug_test_list.append(d_aug_t)
        ecg_test_aug = np.concatenate([ecg_test] + aug_test_list, axis=0)
        ehr_test_aug = np.concatenate([ehr_test] * n_inc, axis=0)
        y_pred_aug = model.predict([ecg_test_aug, ehr_test_aug], verbose=0).reshape(n_inc, -1)
        y_score = y_pred_aug.mean(axis=0).reshape(-1)

        y_true_all.append(y_test.reshape(-1))
        y_score_all.append(y_score)

        last_artifacts = {
            'model': model,
            'ecg_test': ecg_test,
            'ehr_test': ehr_test,
            'y_test': y_test,
            'y_score': y_score,
            'test_index': test_index,
        }

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_score_all = np.concatenate(y_score_all, axis=0)
    return y_true_all, y_score_all, last_artifacts


def save_curves(y_true: np.ndarray, y_score: np.ndarray, out_dir: str) -> Dict[str, float]:
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_score)
    auroc = auc(fpr, tpr)
    auprc = auc(rec, prec)

    # Robust alignment of thresholds with points; handle version differences
    def align_thresholds(points_len: int, thresholds: np.ndarray):
        aligned = np.full(points_len, np.nan)
        if thresholds is None:
            return aligned, True
        tlen = len(thresholds)
        if tlen == points_len:
            aligned = thresholds.astype(float)
            return aligned, True
        if tlen == points_len - 1:
            aligned[1:] = thresholds.astype(float)
            return aligned, True
        if tlen == points_len + 1:
            aligned = thresholds[:-1].astype(float)
            return aligned, True
        return aligned, False

    roc_thresh_aligned, roc_ok = align_thresholds(len(fpr), roc_thresh)
    pr_thresh_aligned, pr_ok = align_thresholds(len(prec), pr_thresh)

    pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': roc_thresh_aligned}).to_csv(os.path.join(out_dir, 'roc_curve.csv'), index=False)
    pd.DataFrame({'precision': prec, 'recall': rec, 'threshold': pr_thresh_aligned}).to_csv(os.path.join(out_dir, 'pr_curve.csv'), index=False)

    # If alignment didn't fit cleanly, also dump raw thresholds separately
    if not roc_ok and roc_thresh is not None:
        pd.DataFrame({'threshold': roc_thresh}).to_csv(os.path.join(out_dir, 'roc_thresholds_only.csv'), index=False)
    if not pr_ok and pr_thresh is not None:
        pd.DataFrame({'threshold': pr_thresh}).to_csv(os.path.join(out_dir, 'pr_thresholds_only.csv'), index=False)

    # ROC fig
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUROC={auroc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=200)
    plt.close()

    # PR fig
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, label=f'AUPRC={auprc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pr_curve.png'), dpi=200)
    plt.close()

    return {'auroc': float(auroc), 'auprc': float(auprc)}


def try_compute_shap(ehr_features: List[str], artifacts: Dict[str, Any], out_dir: str) -> None:
    try:
        import shap
        model = artifacts['model']
        ecg_test = artifacts['ecg_test']
        ehr_test = artifacts['ehr_test']
        y_score = artifacts['y_score']

        # Subsample for speed
        n_samples = min(100, ehr_test.shape[0])
        idx = np.argsort(-y_score)[:n_samples]
        ecg_bg = ecg_test[idx]
        ehr_bg = ehr_test[idx]

        explainer = shap.Explainer(model, [ecg_bg, ehr_bg])
        shap_values = explainer([ecg_bg, ehr_bg])
        # shap_values is a list-like; second element corresponds to EHR input
        sv_ehr = shap_values[1].values if isinstance(shap_values, list) else shap_values.values[1]
        base_values = shap_values[1].base_values if isinstance(shap_values, list) else shap_values.base_values[1]

        # Mean absolute importance
        mean_abs = np.mean(np.abs(sv_ehr), axis=0)
        imp_df = pd.DataFrame({'feature': ehr_features, 'mean_abs_shap': mean_abs})
        imp_df.sort_values('mean_abs_shap', ascending=False).to_csv(os.path.join(out_dir, 'ehr_shap_importance.csv'), index=False)

        # Waterfall for the top-risk sample
        top_idx = int(np.argmax(y_score))
        # If top_idx not in idx, fallback to first of idx
        local_index = np.where(idx == top_idx)[0]
        if len(local_index) == 0:
            local_index = [0]
        li = int(local_index[0])

        # Use shap's legacy waterfall to avoid version-specific issues
        try:
            shap.plots.waterfall(shap.Explanation(values=sv_ehr[li], base_values=base_values[li], data=ehr_bg[li], feature_names=ehr_features), max_display=20, show=False)
        except Exception:
            from shap.plots._waterfall import waterfall_legacy
            explanation = shap.Explanation(values=sv_ehr[li], base_values=base_values[li], data=ehr_bg[li], feature_names=ehr_features)
            waterfall_legacy(explanation, max_display=20, show=False)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'shap_waterfall_ehr.png'), dpi=200)
        plt.close()

        # Save raw SHAP values for reproducibility
        np.savez_compressed(
            os.path.join(out_dir, 'ehr_shap_values_top_subset.npz'),
            shap_values=sv_ehr,
            base_values=base_values,
            ehr_data=ehr_bg,
            features=np.array(ehr_features),
            idx=idx,
        )
    except Exception as e:
        with open(os.path.join(out_dir, 'shap_error.txt'), 'w') as f:
            f.write(str(e))


def collect_results(res_dir: str, metric_col: str) -> pd.DataFrame:
    rows = []
    for csv_path in glob(os.path.join(res_dir, '*.csv')):
        stem = os.path.splitext(os.path.basename(csv_path))[0]
        npy_path = os.path.join(res_dir, stem + '.npy')
        if not os.path.exists(npy_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            score = float(df[metric_col].mean())
            info = np.load(npy_path, allow_pickle=True)
            [com_arch, n_comp, num_heads, ecg_app, aug_p, n_inc, n_comp_ecg, n_comp_ehr, initial_learning_rate] = info.tolist()
            rows.append({
                'run_id': stem,
                'metric': score,
                'csv': csv_path,
                'npy': npy_path,
                'com_arch': int(com_arch),
                'n_comp': int(n_comp),
                'num_heads': int(num_heads),
                'ecg_app': int(ecg_app),
                'aug_p': float(aug_p),
                'n_inc': int(n_inc),
                'n_comp_ecg': int(n_comp_ecg),
                'n_comp_ehr': int(n_comp_ehr),
                'initial_learning_rate': float(initial_learning_rate),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Review and rerun top models, generate curves and SHAP analysis.')
    parser.add_argument('--metric', type=str, default='auroc', help='Metric to rank by (auroc, accuracy, f1, precision, recall, specificity).')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top models to rerun.')
    parser.add_argument('--res_dir', type=str, default='res', help='Directory containing original run results (.csv/.npy pairs).')
    parser.add_argument('--out_dir', type=str, default='res_review', help='Directory to save review outputs.')
    parser.add_argument('--use_bmi_filter', action='store_true', help='Apply BMI<=35 filter like experiment_on_best2.')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs per fold.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of CV folds.')
    parser.add_argument('--keep_only_best', action='store_true', help='After reruns, keep only best-by-AUROC and best-by-AUPRC run folders; remove the rest.')
    args = parser.parse_args()

    metric_col, higher_is_better = map_metric_name(args.metric)

    ensure_dir(args.out_dir)

    df_all = collect_results(args.res_dir, metric_col)
    if df_all.empty:
        raise RuntimeError(f"No results found in {args.res_dir}")

    print(df_all)
    df_sorted = df_all.sort_values('metric', ascending=not higher_is_better).reset_index(drop=True)
    top_k = min(args.top_k, len(df_sorted))
    df_selected = df_sorted.iloc[:top_k].copy()
    print(df_selected)    
    df_selected.to_csv(os.path.join(args.out_dir, 'selected_models.csv'), index=False)

    ehr_data, ecg_data, ecg_data_denoise, ecg_data_denoise_filtered, fft_data, fft_fake, ecg_fake, y_dl = load_ready_data(args.use_bmi_filter)
    ecg_variants = {
        'raw': ecg_data,
        'denoise': ecg_data_denoise,
        'denoise_filtered': ecg_data_denoise_filtered,
        'fft': fft_data,
        'fft_fake': fft_fake,
        'ecg_fake': ecg_fake,
    }

    summary_rows = []
    created_run_ids = []

    for i, row in df_selected.iterrows():
        run_id = row['run_id']
        run_dir = os.path.join(args.out_dir, run_id)
        ensure_dir(run_dir)
        created_run_ids.append(run_id)

        config = {
            'com_arch': row['com_arch'],
            'n_comp': row['n_comp'],
            'num_heads': row['num_heads'],
            'ecg_app': row['ecg_app'],
            'aug_p': row['aug_p'],
            'n_inc': row['n_inc'],
            'n_comp_ecg': row['n_comp_ecg'],
            'n_comp_ehr': row['n_comp_ehr'],
            'initial_learning_rate': row['initial_learning_rate'],
        }

        ehr_proc, ecg_proc, ehr_features = normalize_inputs(ehr_data, ecg_variants, ecg_app=config['ecg_app'])

        y_true, y_score, artifacts = train_and_predict(config, ehr_proc, ecg_proc, y_dl.iloc[:, 0], n_splits=args.n_splits, epochs=args.epochs)

        metrics = save_curves(y_true, y_score, out_dir=run_dir)
        with open(os.path.join(run_dir, 'meta.json'), 'w') as f:
            json.dump({
                'source_run_id': run_id,
                'rank_metric': metric_col,
                'rank_value': float(row['metric']),
                'config': {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v) for k, v in config.items()},
                'curves': metrics,
            }, f, indent=2)

        summary_rows.append({
            'run_id': run_id,
            'rank_metric': metric_col,
            'rank_value': float(row['metric']),
            'auroc_rerun': metrics['auroc'],
            'auprc_rerun': metrics['auprc'],
        })

        if i < 10:
            try_compute_shap(ehr_features, artifacts, out_dir=run_dir)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)

    # Optionally keep only the best AUROC and AUPRC runs
    if args.keep_only_best and not df_summary.empty:
        keep_ids = set()
        try:
            best_roc_id = df_summary.loc[df_summary['auroc_rerun'].idxmax(), 'run_id']
            keep_ids.add(str(best_roc_id))
        except Exception:
            pass
        try:
            best_pr_id = df_summary.loc[df_summary['auprc_rerun'].idxmax(), 'run_id']
            keep_ids.add(str(best_pr_id))
        except Exception:
            pass

        for rid in created_run_ids:
            if str(rid) not in keep_ids:
                dir_path = os.path.join(args.out_dir, rid)
                if os.path.isdir(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                    except Exception:
                        pass

        kept_df = df_summary[df_summary['run_id'].astype(str).isin(keep_ids)]
        kept_df.to_csv(os.path.join(args.out_dir, 'kept_runs.csv'), index=False)


if __name__ == '__main__':
    main()


