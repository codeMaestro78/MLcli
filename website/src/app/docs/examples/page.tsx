import { CodeBlock } from '@/components/code';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BookOpen, Brain, Download, Folder, GitBranch, Layers, Settings, Sparkles, Zap } from 'lucide-react';
import Link from 'next/link';

export const metadata = {
  title: 'Examples',
  description: 'Example configurations and usage patterns for mlcli.',
};

const quickExamples = [
  {
    title: 'Train Random Forest',
    command: 'mlcli train -d data.csv -m random_forest --target label',
    description: 'Train a basic Random Forest classifier',
  },
  {
    title: 'Train with Config',
    command: 'mlcli train --config configs/rf_config.json',
    description: 'Train using a JSON configuration file',
  },
  {
    title: 'Hyperparameter Tuning',
    command: 'mlcli tune --config configs/tune_rf.json --method random --n-trials 50',
    description: 'Tune hyperparameters with random search',
  },
  {
    title: 'Model Explanation',
    command: 'mlcli explain --model models/model.pkl --data test.csv --method shap',
    description: 'Generate SHAP explanations for your model',
  },
  {
    title: 'List Experiments',
    command: 'mlcli list-runs',
    description: 'View all experiment runs',
  },
  {
    title: 'Interactive UI',
    command: 'mlcli ui',
    description: 'Launch the interactive terminal UI',
  },
];

const configExamples = {
  classification: {
    randomForest: `{
  "model_type": "random_forest",
  "data": {
    "train_path": "data/train.csv",
    "test_path": "data/test.csv",
    "target_column": "label"
  },
  "preprocessing": {
    "scaler": "standard",
    "handle_missing": "mean"
  },
  "params": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2,
    "random_state": 42
  },
  "output": {
    "model_dir": "models/",
    "run_dir": "runs/"
  }
}`,
    lightgbm: `{
  "model_type": "lightgbm",
  "task_type": "classification",
  "data": {
    "train_path": "data/train.csv",
    "test_path": "data/test.csv",
    "target_column": "target"
  },
  "preprocessing": {
    "scaler": "standard",
    "handle_missing": "mean",
    "encode_categorical": true
  },
  "params": {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "boosting_type": "gbdt"
  },
  "early_stopping_rounds": 50,
  "output": {
    "model_dir": "models/",
    "formats": ["pickle", "lightgbm"]
  }
}`,
    catboost: `{
  "model_type": "catboost",
  "task_type": "classification",
  "data": {
    "train_path": "data/train.csv",
    "test_path": "data/test.csv",
    "target_column": "target",
    "categorical_features": ["category", "type"]
  },
  "preprocessing": {
    "handle_missing": "mean"
  },
  "params": {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "border_count": 128
  },
  "output": {
    "model_dir": "models/",
    "formats": ["pickle", "catboost"]
  }
}`,
  },
  clustering: {
    kmeans: `{
  "model_type": "kmeans",
  "data": {
    "train_path": "data/unlabeled.csv",
    "feature_columns": null
  },
  "preprocessing": {
    "scaler": "standard",
    "handle_missing": "mean"
  },
  "params": {
    "n_clusters": 5,
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "algorithm": "lloyd"
  },
  "output": {
    "model_dir": "models/",
    "formats": ["pickle", "joblib"]
  }
}`,
    dbscan: `{
  "model_type": "dbscan",
  "data": {
    "train_path": "data/unlabeled.csv",
    "feature_columns": null
  },
  "preprocessing": {
    "scaler": "standard",
    "handle_missing": "mean"
  },
  "params": {
    "eps": 0.5,
    "min_samples": 5,
    "metric": "euclidean",
    "algorithm": "auto"
  },
  "output": {
    "model_dir": "models/",
    "formats": ["pickle"]
  }
}`,
  },
  anomaly: {
    isolationForest: `{
  "model_type": "isolation_forest",
  "data": {
    "train_path": "data/normal_data.csv",
    "feature_columns": null
  },
  "preprocessing": {
    "scaler": "standard",
    "handle_missing": "mean"
  },
  "params": {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": false
  },
  "output": {
    "model_dir": "models/",
    "formats": ["pickle", "joblib"]
  }
}`,
    oneClassSvm: `{
  "model_type": "one_class_svm",
  "data": {
    "train_path": "data/normal_data.csv",
    "feature_columns": null
  },
  "preprocessing": {
    "scaler": "standard",
    "handle_missing": "mean"
  },
  "params": {
    "kernel": "rbf",
    "gamma": "scale",
    "nu": 0.1,
    "shrinking": true
  },
  "output": {
    "model_dir": "models/",
    "formats": ["pickle"]
  }
}`,
  },
  tuning: {
    randomSearch: `{
  "model_type": "random_forest",
  "tuning": {
    "method": "random",
    "n_trials": 50,
    "cv_folds": 5,
    "scoring": "accuracy"
  },
  "param_grid": {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [5, 10, 15, 20, null],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
  },
  "data": {
    "train_path": "data/train.csv",
    "target_column": "label"
  },
  "output": {
    "model_dir": "models/",
    "save_best": true
  }
}`,
    optuna: `{
  "model_type": "xgboost",
  "tuning": {
    "method": "optuna",
    "n_trials": 100,
    "cv_folds": 5,
    "scoring": "f1",
    "early_stopping": 20
  },
  "param_distributions": {
    "n_estimators": {"type": "int", "low": 50, "high": 500},
    "max_depth": {"type": "int", "low": 3, "high": 12},
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": true},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0}
  },
  "data": {
    "train_path": "data/train.csv",
    "target_column": "label"
  },
  "output": {
    "model_dir": "models/",
    "save_best": true
  }
}`,
  },
};

export default function ExamplesPage() {
  return (
    <div>
      <h1>Examples</h1>
      <p className="lead">
        Comprehensive examples and configuration templates to help you get started with mlcli.
        Copy and adapt these examples for your own projects.
      </p>

      {/* Quick Start Examples */}
      <h2 id="quick-start">
        <Zap className="inline h-5 w-5 mr-2" />
        Quick Start Commands
      </h2>
      <p>
        Get up and running quickly with these common commands:
      </p>

      <div className="not-prose mt-6 grid gap-3 md:grid-cols-2">
        {quickExamples.map((example) => (
          <Card key={example.title} className="overflow-hidden">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">{example.title}</CardTitle>
              <CardDescription>{example.description}</CardDescription>
            </CardHeader>
            <CardContent className="pt-0">
              <code className="block rounded bg-muted px-3 py-2 text-sm font-mono break-all">
                {example.command}
              </code>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Directory Structure */}
      <h2 id="project-structure">
        <Folder className="inline h-5 w-5 mr-2" />
        Recommended Project Structure
      </h2>
      <CodeBlock
        code={`my-ml-project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── raw/
├── configs/
│   ├── random_forest.json
│   ├── lightgbm.json
│   └── tuning/
│       └── tune_rf.json
├── models/
│   └── (trained models saved here)
├── runs/
│   └── (experiment logs saved here)
└── notebooks/
    └── analysis.ipynb`}
        language="text"
        filename="Project Structure"
      />

      {/* Classification Examples */}
      <h2 id="classification">
        <Layers className="inline h-5 w-5 mr-2" />
        Classification Examples
      </h2>
      <p>
        Configuration examples for classification tasks:
      </p>

      <Tabs defaultValue="random_forest" className="not-prose mt-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="random_forest">Random Forest</TabsTrigger>
          <TabsTrigger value="lightgbm">LightGBM</TabsTrigger>
          <TabsTrigger value="catboost">CatBoost</TabsTrigger>
        </TabsList>
        <TabsContent value="random_forest">
          <CodeBlock
            code={configExamples.classification.randomForest}
            language="json"
            filename="configs/random_forest.json"
          />
          <p className="mt-2 text-sm text-muted-foreground">
            Run with: <code className="bg-muted px-1 rounded">mlcli train --config configs/random_forest.json</code>
          </p>
        </TabsContent>
        <TabsContent value="lightgbm">
          <CodeBlock
            code={configExamples.classification.lightgbm}
            language="json"
            filename="configs/lightgbm.json"
          />
          <p className="mt-2 text-sm text-muted-foreground">
            Run with: <code className="bg-muted px-1 rounded">mlcli train --config configs/lightgbm.json</code>
          </p>
        </TabsContent>
        <TabsContent value="catboost">
          <CodeBlock
            code={configExamples.classification.catboost}
            language="json"
            filename="configs/catboost.json"
          />
          <p className="mt-2 text-sm text-muted-foreground">
            Run with: <code className="bg-muted px-1 rounded">mlcli train --config configs/catboost.json</code>
          </p>
        </TabsContent>
      </Tabs>

      {/* Clustering Examples */}
      <h2 id="clustering">
        <GitBranch className="inline h-5 w-5 mr-2" />
        Clustering Examples
      </h2>
      <p>
        Configuration examples for unsupervised clustering:
      </p>

      <Tabs defaultValue="kmeans" className="not-prose mt-4">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="kmeans">K-Means</TabsTrigger>
          <TabsTrigger value="dbscan">DBSCAN</TabsTrigger>
        </TabsList>
        <TabsContent value="kmeans">
          <CodeBlock
            code={configExamples.clustering.kmeans}
            language="json"
            filename="configs/clustering/kmeans.json"
          />
          <p className="mt-2 text-sm text-muted-foreground">
            Run with: <code className="bg-muted px-1 rounded">mlcli train --config configs/clustering/kmeans.json</code>
          </p>
        </TabsContent>
        <TabsContent value="dbscan">
          <CodeBlock
            code={configExamples.clustering.dbscan}
            language="json"
            filename="configs/clustering/dbscan.json"
          />
          <p className="mt-2 text-sm text-muted-foreground">
            Run with: <code className="bg-muted px-1 rounded">mlcli train --config configs/clustering/dbscan.json</code>
          </p>
        </TabsContent>
      </Tabs>

      {/* Anomaly Detection Examples */}
      <h2 id="anomaly-detection">
        <Sparkles className="inline h-5 w-5 mr-2" />
        Anomaly Detection Examples
      </h2>
      <p>
        Configuration examples for anomaly and outlier detection:
      </p>

      <Tabs defaultValue="isolation_forest" className="not-prose mt-4">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="isolation_forest">Isolation Forest</TabsTrigger>
          <TabsTrigger value="one_class_svm">One-Class SVM</TabsTrigger>
        </TabsList>
        <TabsContent value="isolation_forest">
          <CodeBlock
            code={configExamples.anomaly.isolationForest}
            language="json"
            filename="configs/anomaly/isolation_forest.json"
          />
          <p className="mt-2 text-sm text-muted-foreground">
            Run with: <code className="bg-muted px-1 rounded">mlcli train --config configs/anomaly/isolation_forest.json</code>
          </p>
        </TabsContent>
        <TabsContent value="one_class_svm">
          <CodeBlock
            code={configExamples.anomaly.oneClassSvm}
            language="json"
            filename="configs/anomaly/one_class_svm.json"
          />
          <p className="mt-2 text-sm text-muted-foreground">
            Run with: <code className="bg-muted px-1 rounded">mlcli train --config configs/anomaly/one_class_svm.json</code>
          </p>
        </TabsContent>
      </Tabs>

      {/* Hyperparameter Tuning Examples */}
      <h2 id="tuning">
        <Settings className="inline h-5 w-5 mr-2" />
        Hyperparameter Tuning Examples
      </h2>
      <p>
        Configuration examples for hyperparameter optimization:
      </p>

      <Tabs defaultValue="random_search" className="not-prose mt-4">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="random_search">Random Search</TabsTrigger>
          <TabsTrigger value="optuna">Optuna (Bayesian)</TabsTrigger>
        </TabsList>
        <TabsContent value="random_search">
          <CodeBlock
            code={configExamples.tuning.randomSearch}
            language="json"
            filename="configs/tuning/tune_rf.json"
          />
          <p className="mt-2 text-sm text-muted-foreground">
            Run with: <code className="bg-muted px-1 rounded">mlcli tune --config configs/tuning/tune_rf.json</code>
          </p>
        </TabsContent>
        <TabsContent value="optuna">
          <CodeBlock
            code={configExamples.tuning.optuna}
            language="json"
            filename="configs/tuning/tune_xgb_optuna.json"
          />
          <p className="mt-2 text-sm text-muted-foreground">
            Run with: <code className="bg-muted px-1 rounded">mlcli tune --config configs/tuning/tune_xgb_optuna.json</code>
          </p>
        </TabsContent>
      </Tabs>

      {/* Python API Examples */}
      <h2 id="python-api">
        <Brain className="inline h-5 w-5 mr-2" />
        Python API Examples
      </h2>
      <p>
        Use mlcli programmatically in your Python scripts:
      </p>

      <CodeBlock
        code={`from mlcli.trainers import (
    RandomForestTrainer,
    LightGBMTrainer,
    KMeansTrainer,
    IsolationForestTrainer
)
from mlcli.preprocessor import PreprocessingPipeline
import pandas as pd

# Load data
df = pd.read_csv('data/train.csv')
X = df.drop('target', axis=1)
y = df['target']

# Option 1: Simple training
trainer = RandomForestTrainer(n_estimators=100, max_depth=10)
trainer.train(X, y)
predictions = trainer.predict(X_test)
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Option 2: With preprocessing
pipeline = PreprocessingPipeline([
    ('scaler', 'standard'),
    ('selector', 'select_k_best', {'k': 10})
])
X_processed = pipeline.fit_transform(X)

# Option 3: LightGBM with early stopping
lgb_trainer = LightGBMTrainer(
    n_estimators=500,
    learning_rate=0.1,
    num_leaves=31
)
lgb_trainer.train(X, y, X_val=X_val, y_val=y_val, early_stopping_rounds=50)

# Option 4: Clustering
kmeans = KMeansTrainer(n_clusters=5)
kmeans.train(X)
clusters = kmeans.predict(X)
print(f"Silhouette Score: {kmeans.evaluate(X)['silhouette_score']:.4f}")

# Option 5: Anomaly Detection
iso_forest = IsolationForestTrainer(contamination=0.1)
iso_forest.train(X_normal)
anomalies = iso_forest.get_anomalies(X_test)
print(f"Found {len(anomalies)} anomalies")`}
        language="python"
        filename="example_usage.py"
      />

      {/* End-to-End Workflow */}
      <h2 id="workflow">
        <BookOpen className="inline h-5 w-5 mr-2" />
        End-to-End Workflow
      </h2>
      <p>
        Complete workflow from data to deployed model:
      </p>

      <CodeBlock
        code={`# Step 1: Preprocess data
mlcli preprocess \\
  --data data/raw.csv \\
  --output data/processed.csv \\
  --methods standard_scaler,select_k_best

# Step 2: Train multiple models
mlcli train --config configs/random_forest.json
mlcli train --config configs/lightgbm.json
mlcli train --config configs/xgboost.json

# Step 3: Compare experiments
mlcli list-runs

# Step 4: Tune the best model
mlcli tune \\
  --config configs/tuning/tune_lightgbm.json \\
  --method optuna \\
  --n-trials 100

# Step 5: Explain the model
mlcli explain \\
  --model models/best_model.pkl \\
  --data data/test.csv \\
  --method shap \\
  --output explanations/

# Step 6: Evaluate on test set
mlcli eval \\
  --model models/best_model.pkl \\
  --data data/test.csv`}
        language="bash"
        filename="workflow.sh"
      />

      {/* Download Examples */}
      <h2 id="download">
        <Download className="inline h-5 w-5 mr-2" />
        Download Example Configs
      </h2>
      <p>
        Get all example configurations from the repository:
      </p>

      <CodeBlock
        code={`# Clone repository
git clone https://github.com/codeMaestro78/MLcli.git

# Navigate to examples
cd MLcli/examples

# View available configs
ls configs/`}
        language="bash"
      />

      <div className="not-prose mt-6 rounded-lg border bg-muted/30 p-6">
        <h3 className="text-lg font-semibold mb-2">Need More Examples?</h3>
        <p className="text-muted-foreground mb-4">
          Check out the GitHub repository for more examples, or open an issue to request specific examples.
        </p>
        <div className="flex gap-4">
          <Link
            href="https://github.com/codeMaestro78/MLcli/tree/master/examples"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            View on GitHub →
          </Link>
          <Link
            href="https://github.com/codeMaestro78/MLcli/issues/new"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline"
          >
            Request Example →
          </Link>
        </div>
      </div>
    </div>
  );
}
