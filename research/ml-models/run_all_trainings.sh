#!/bin/bash

# Set up logging
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to run training with error handling
run_training() {
    local config=$1
    local model_name=$2
    local output_dir="outputs/${model_name}_${TIMESTAMP}"
    
    log_message "Starting training for $model_name using config: $config"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run the training
    python run_trainer.py \
        --config "$config" \
        --output-dir "$output_dir" \
        --log-level INFO 2>&1 | tee -a "$LOG_FILE"
    
    # Check if training was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_message "Successfully completed training for $model_name"
    else
        log_message "ERROR: Training failed for $model_name"
    fi
}

# Main execution
log_message "Starting batch training process"

# Create base config if it doesn't exist
if [ ! -f "configs/base.yaml" ]; then
    log_message "Creating base configuration..."
    cat > configs/base.yaml << EOL
data:
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"
  batch_size: 32
  num_workers: 4

model_registry:
  Stacking:
    base_estimators:
      - LogisticRegression
      - RandomForest
      - XGBoost
    meta_estimator: LogisticRegression
    cv: 5
  
  RandomForest:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  XGBoost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    random_state: 42
  
  LightGBM:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    random_state: 42
  
  LogisticRegression:
    max_iter: 1000
    random_state: 42
  
  SVC_linear:
    kernel: linear
    probability: true
    random_state: 42
  
  SVC_rbf:
    kernel: rbf
    probability: true
    random_state: 42
  
  LinearSVC:
    max_iter: 1000
    random_state: 42
  
  KNN:
    n_neighbors: 5
    weights: uniform
  
  DecisionTree:
    max_depth: 10
    random_state: 42
  
  ExtraTrees:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  
  AdaBoost:
    n_estimators: 100
    random_state: 42
  
  GradientBoosting:
    n_estimators: 100
    learning_rate: 0.1
    random_state: 42
  
  GaussianNB:
    priors: null
  
  LDA:
    solver: svd
  
  QDA:
    reg_param: 0.0
  
  MLP:
    hidden_layer_sizes: [100, 50]
    max_iter: 1000
    random_state: 42
  
  Ridge:
    alpha: 1.0
    random_state: 42
  
  SGD:
    max_iter: 1000
    random_state: 42
  
  CatBoost:
    iterations: 100
    learning_rate: 0.1
    random_state: 42

training:
  tasks:
    - ripeness
    - firmness
  models:
    - Stacking
    - RandomForest
    - XGBoost
    - LightGBM
    - LogisticRegression
    - SVC_linear
    - SVC_rbf
    - LinearSVC
    - KNN
    - DecisionTree
    - ExtraTrees
    - AdaBoost
    - GradientBoosting
    - GaussianNB
    - LDA
    - QDA
    - MLP
    - Ridge
    - SGD
    - CatBoost
EOL
fi

# Run training for each model
models=(
    "Stacking"
    "RandomForest"
    "XGBoost"
    "LightGBM"
    "LogisticRegression"
    "SVC_linear"
    "SVC_rbf"
    "LinearSVC"
    "KNN"
    "DecisionTree"
    "ExtraTrees"
    "AdaBoost"
    "GradientBoosting"
    "GaussianNB"
    "LDA"
    "QDA"
    "MLP"
    "Ridge"
    "SGD"
    "CatBoost"
)

# Create individual configs for each model
for model in "${models[@]}"; do
    config_file="configs/${model,,}.yaml"
    if [ ! -f "$config_file" ]; then
        log_message "Creating config for $model..."
        cat > "$config_file" << EOL
data:
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"
  batch_size: 32
  num_workers: 4

model_registry:
  $model:
    $(grep -A 10 "^  $model:" configs/base.yaml | tail -n +2)

training:
  tasks:
    - ripeness
    - firmness
  models:
    - $model
EOL
    fi
    
    # Run training for this model
    run_training "$config_file" "$model"
    
    # Add a small delay between trainings to prevent resource contention
    sleep 5
done

log_message "Batch training process completed"

# Generate summary report
log_message "Generating training summary..."
python -c "
import json
import glob
import os
from datetime import datetime

def parse_log(log_file):
    results = {'success': False, 'errors': []}
    with open(log_file, 'r') as f:
        for line in f:
            if 'Successfully completed training for' in line:
                results['success'] = True
            elif 'ERROR:' in line:
                results['errors'].append(line.strip())
    return results

# Get all log files from today
today = datetime.now().strftime('%Y%m%d')
log_files = glob.glob(f'logs/training_{today}*.log')

summary = {}
for log_file in log_files:
    model_name = os.path.basename(log_file).split('_')[1]
    results = parse_log(log_file)
    summary[model_name] = results

# Save summary
with open(f'logs/training_summary_{today}.json', 'w') as f:
    json.dump(summary, f, indent=2)
"

log_message "Training summary generated in logs/training_summary_$(date +%Y%m%d).json" 