"""
Trojan Detector Training Script

Combines features from cycle-level and full-graph analysis to train
classifiers for hardware trojan detection.

Given the small dataset size (~30 samples), we use:
1. Classical ML models (Random Forest, XGBoost, SVM)
2. Proper cross-validation (Leave-One-Out or stratified k-fold)
3. Feature selection to avoid overfitting
"""

import os
import json
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ML imports
from sklearn.model_selection import (
    StratifiedKFold, LeaveOneOut, cross_val_score, cross_val_predict
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, skipping...")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, skipping...")

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CYCLE_FEATURES_PATH = os.path.join(OUTPUT_DIR, "enhanced_features_dataset.json")
FULL_GRAPH_FEATURES_PATH = os.path.join(OUTPUT_DIR, "full_graph_features.json")
MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "trained_model.pkl")


# Old top features (commented out)
# TOP_CYCLE_FEATURES = [
#     'rare_gate_density_max', 'high_rare_gate_nodes', 'rare_gate_density_std',
#     'dag_longest_path', 'in_out_degree_correlation', 'largest_scc_fraction',
#     'gate_type_entropy', 'xor_xnor_ratio', 'cycles_per_node', 'transitivity',
#     'fan_out_std', 'edges_per_node', 'num_sccs', 'rare_gate_density_mean'
# ]
#
TOP_FULL_GRAPH_FEATURES = [
     'rare_gate_out_degree_mean', 'dff_in_degree_mean', 'max_fan_in', 'in_degree_max',
     'gate_entropy', 'in_out_correlation', 'is_dag', 'rare_gate_out_degree_max',
     'dag_longest_path_norm', 'xor_xnor_ratio', 'rare_gate_ratio', 'transitivity',
     'high_in_degree_fraction', 'xor_ratio', 'reciprocity'
]

# New top discriminative features from analyzer (by correlation with label)
TOP_CYCLE_FEATURES = [
    'rare_gate_density_std',
    'high_rare_gate_nodes',
    'rare_gate_density_mean',
    'rare_gate_density_max',
    'xor_xnor_ratio',
    'rare_logic_ratio',
    'in_degree_max',
    'fan_out_mean',
    'gate_type_entropy',
    'rare_gate_heterogeneity',
    'dag_longest_path',
    'neighborhood_fanin_mean',
    'low_connectivity_nodes',
    'fanin_cone_std',
    'edges_per_node',
    'in_degree_mean',
    'out_degree_mean',
    'neighbor_count_mean',
    'transitivity',
    'cone_ratio_mean',
    'fan_out_std',
    'source_node_fraction',
    'num_wccs',
    'inverter_heterogeneity',
    'bottleneck_ratio'
]



# =============================================================================
# Data Loading
# =============================================================================

def load_cycle_features() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load cycle-level features."""
    with open(CYCLE_FEATURES_PATH, 'r') as f:
        data = json.load(f)
    
    features = []
    labels = []
    design_ids = []
    feature_names = data[0]['feature_names']
    
    for item in data:
        features.append(item['discriminative_features'])
        labels.append(item['label'])
        design_ids.append(item['design_id'])
    
    return np.array(features), np.array(labels), feature_names, design_ids


def load_full_graph_features() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load full graph features."""
    with open(FULL_GRAPH_FEATURES_PATH, 'r') as f:
        data = json.load(f)
    
    features = []
    labels = []
    design_ids = []
    feature_names = data[0]['feature_names']
    
    for item in data:
        features.append(item['discriminative_features'])
        labels.append(item['label'])
        design_ids.append(item['design_id'])
    
    return np.array(features), np.array(labels), feature_names, design_ids


def select_features(X: np.ndarray, feature_names: List[str], 
                   selected_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Select specific features by name."""
    indices = []
    valid_names = []
    for name in selected_names:
        if name in feature_names:
            indices.append(feature_names.index(name))
            valid_names.append(name)
    
    if not indices:
        return X, feature_names
    
    return X[:, indices], valid_names


# =============================================================================
# Model Definitions
# =============================================================================

def get_models() -> Dict[str, object]:
    """Get dictionary of models to evaluate."""
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=2,
            class_weight='balanced', random_state=42
        ),
        'RandomForest_Deep': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=1,
            class_weight='balanced', random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=2, random_state=42
        ),
        'SVM_RBF': SVC(
            kernel='rbf', C=1.0, gamma='scale',
            class_weight='balanced', probability=True, random_state=42
        ),
        'SVM_Linear': SVC(
            kernel='linear', C=1.0,
            class_weight='balanced', probability=True, random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=3, weights='distance'
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50, learning_rate=0.5, random_state=42
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500,
            early_stopping=True, random_state=42
        ),
    }
    
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            scale_pos_weight=1, use_label_encoder=False,
            eval_metric='logloss', random_state=42
        )
    
    if HAS_LGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            class_weight='balanced', random_state=42, verbose=-1
        )
    
    return models


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, X: np.ndarray, y: np.ndarray, 
                  cv_strategy='loo') -> Dict[str, float]:
    """Evaluate model using cross-validation."""
    
    if cv_strategy == 'loo':
        cv = LeaveOneOut()
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get predictions
    y_pred = cross_val_predict(model, X, y, cv=cv)
    
    # Try to get probabilities for AUC
    try:
        y_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        auc = roc_auc_score(y, y_prob)
    except:
        auc = 0.5
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc': auc
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics['true_neg'] = tn
    metrics['false_pos'] = fp
    metrics['false_neg'] = fn
    metrics['true_pos'] = tp
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics, y_pred


def print_results(results: Dict[str, Dict], title: str):
    """Print evaluation results."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}")
    print("-"*80)
    
    # Sort by F1 score
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['f1'])
    
    for name, metrics in sorted_results:
        print(f"{name:<25} {metrics['accuracy']:>8.3f} {metrics['precision']:>8.3f} "
              f"{metrics['recall']:>8.3f} {metrics['specificity']:>8.3f} "
              f"{metrics['f1']:>8.3f} {metrics['auc']:>8.3f}")
    
    return sorted_results


# =============================================================================
# Feature Selection
# =============================================================================

def automatic_feature_selection(X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str],
                                n_features: int = 15) -> Tuple[np.ndarray, List[str]]:
    """Select best features automatically."""
    
    # Method 1: Mutual Information
    mi_selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
    mi_selector.fit(X, y)
    mi_scores = mi_selector.scores_
    
    # Method 2: F-statistic
    f_selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
    f_selector.fit(X, y)
    f_scores = f_selector.scores_
    
    # Combine scores (normalized)
    mi_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
    f_norm = (f_scores - np.nanmin(f_scores)) / (np.nanmax(f_scores) - np.nanmin(f_scores) + 1e-10)
    f_norm = np.nan_to_num(f_norm, 0)
    
    combined_scores = mi_norm + f_norm
    
    # Get top features
    top_indices = np.argsort(combined_scores)[-n_features:][::-1]
    selected_names = [feature_names[i] for i in top_indices]
    
    print(f"\nAutomatically selected {n_features} features:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. {feature_names[idx]:<40} (MI={mi_scores[idx]:.3f}, F={f_scores[idx]:.3f})")
    
    return X[:, top_indices], selected_names


# =============================================================================
# Main Training Pipeline
# =============================================================================

def train_on_dataset(X: np.ndarray, y: np.ndarray, 
                     feature_names: List[str],
                     dataset_name: str,
                     use_feature_selection: bool = True,
                     n_features: int = 15) -> Tuple[object, Dict]:
    """Train and evaluate models on a dataset."""
    
    print(f"\n{'#'*80}")
    print(f"# TRAINING ON: {dataset_name}")
    print(f"# Samples: {len(y)} ({sum(y)} trojaned, {len(y)-sum(y)} clean)")
    print(f"# Features: {X.shape[1]}")
    print(f"{'#'*80}")
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Feature selection
    if use_feature_selection and X.shape[1] > n_features:
        X_selected, selected_names = automatic_feature_selection(X, y, feature_names, n_features)
    else:
        X_selected, selected_names = X, feature_names
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Get models
    models = get_models()
    
    # Evaluate each model
    results = {}
    predictions = {}
    model_filenames = []

    for name, model in models.items():
        try:
            metrics, y_pred = evaluate_model(model, X_scaled, y, cv_strategy='loo')
            results[name] = metrics
            predictions[name] = y_pred

            # Train model on all data
            model.fit(X_scaled, y)

            # Determine prefix: 'cycle' or 'full'
            if 'cycle' in dataset_name.lower():
                prefix = 'cycle'
            elif 'full' in dataset_name.lower():
                prefix = 'full'
            else:
                prefix = 'other'

            model_filename = f"{prefix}_{name}.pkl"
            model_path = os.path.join(OUTPUT_DIR, model_filename)
            save_data = {
                'model': model,
                'scaler': scaler,
                'feature_names': selected_names,
                'metrics': metrics,
                'model_type': name,
                'dataset': dataset_name
            }
            with open(model_path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"Saved model: {model_path}")
            model_filenames.append(model_path)
        except Exception as e:
            print(f"  Error with {name}: {e}")

    # Print results
    sorted_results = print_results(results, f"Results for {dataset_name}")

    # Train best model on all data (already done above, but select for return)
    best_model_name = sorted_results[0][0]
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name}")
    print(f"  F1 Score: {sorted_results[0][1]['f1']:.3f}")
    print(f"  Confusion Matrix: TP={sorted_results[0][1]['true_pos']}, "
          f"TN={sorted_results[0][1]['true_neg']}, "
          f"FP={sorted_results[0][1]['false_pos']}, "
          f"FN={sorted_results[0][1]['false_neg']}")

    # Return model info
    model_info = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': selected_names,
        'metrics': sorted_results[0][1],
        'all_results': results,
        'all_model_paths': model_filenames
    }

    return model_info


def create_ensemble(models_info: List[Dict]) -> Dict:
    """Create an ensemble from multiple trained models."""
    
    # This would require retraining, so we'll just combine predictions
    # at inference time
    return {
        'type': 'ensemble',
        'models': models_info
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*80)
    print("TROJAN DETECTOR TRAINING")
    print("="*80)
    
    # Load data
    print("\nLoading features...")
    
    # Cycle features
    if os.path.exists(CYCLE_FEATURES_PATH):
        X_cycle, y_cycle, cycle_feat_names, cycle_ids = load_cycle_features()
        print(f"Cycle features: {X_cycle.shape}")
    else:
        print("Cycle features not found!")
        X_cycle = None
    
    # Full graph features
    if os.path.exists(FULL_GRAPH_FEATURES_PATH):
        X_full, y_full, full_feat_names, full_ids = load_full_graph_features()
        print(f"Full graph features: {X_full.shape}")
    else:
        print("Full graph features not found!")
        X_full = None
    

    trained_models = {}

    # Train on cycle features
    if X_cycle is not None:
        cycle_model_info = train_on_dataset(
            X_cycle, y_cycle, cycle_feat_names,
            "Cycle-Level Features (Trojaned cycles vs Clean cycles)",
            use_feature_selection=True, n_features=12
        )
        trained_models['cycle'] = cycle_model_info

    # Train on full graph features
    if X_full is not None:
        full_model_info = train_on_dataset(
            X_full, y_full, full_feat_names,
            "Full Graph Features (Trojaned designs vs Clean designs)",
            use_feature_selection=True, n_features=15
        )
        trained_models['full_graph'] = full_model_info

    # Train on new top discriminative features (cycle)
    if X_cycle is not None:
        X_top_cycle, top_cycle_names = select_features(X_cycle, cycle_feat_names, TOP_CYCLE_FEATURES)
        if X_top_cycle.shape[1] > 0:
            top_cycle_model_info = train_on_dataset(
                X_top_cycle, y_cycle, top_cycle_names,
                "Top Cycle Features (analyzer selected)",
                use_feature_selection=False
            )
            trained_models['top_cycle'] = top_cycle_model_info

    # Train on manually selected top features (full graph)
    # (Old list, commented out)
    # if X_full is not None:
    #     X_top_full, top_full_names = select_features(X_full, full_feat_names, TOP_FULL_GRAPH_FEATURES)
    #     if X_top_full.shape[1] > 0:
    #         top_full_model_info = train_on_dataset(
    #             X_top_full, y_full, top_full_names,
    #             "Top Full Graph Features (manually selected)",
    #             use_feature_selection=False
    #         )
    #         trained_models['top_full_graph'] = top_full_model_info

    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    best_overall = None
    best_f1 = 0

    for name, info in trained_models.items():
        f1 = info['metrics']['f1']
        print(f"\n{name}:")
        print(f"  Best Model: {type(info['model']).__name__}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  Accuracy: {info['metrics']['accuracy']:.3f}")
        print(f"  Features: {len(info['feature_names'])}")

        if f1 > best_f1:
            best_f1 = f1
            best_overall = name

    print(f"\n{'='*80}")
    print(f"BEST OVERALL: {best_overall} (F1={best_f1:.3f})")
    print(f"{'='*80}")

    # Save all best models with descriptive names (one file per model type, using new naming)
    for name, info in trained_models.items():
        # Determine prefix: 'cycle' or 'full'
        if 'cycle' in name:
            prefix = 'cycle'
        elif 'full' in name:
            prefix = 'full'
        else:
            prefix = 'other'
        model_filename = f"{prefix}_{info['model'].__class__.__name__}.pkl"
        model_path = os.path.join(OUTPUT_DIR, model_filename)
        save_data = {
            'model': info['model'],
            'scaler': info['scaler'],
            'feature_names': info['feature_names'],
            'metrics': info['metrics'],
            'model_type': name
        }
        with open(model_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to: {model_path}")

    # Also save the best overall model to the default path for convenience
    best_info = trained_models[best_overall]
    save_data = {
        'model': best_info['model'],
        'scaler': best_info['scaler'],
        'feature_names': best_info['feature_names'],
        'metrics': best_info['metrics'],
        'model_type': best_overall
    }
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nBest overall model also saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
