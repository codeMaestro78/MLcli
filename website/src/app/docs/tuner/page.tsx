import { CodeBlock } from '@/components/code';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowLeft, Grid, Search, Zap } from 'lucide-react';
import { Metadata } from 'next';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Hyperparameter Tuning',
  description: 'Optimize your models with automated hyperparameter tuning.',
};

const strategies = [
  {
    name: 'Grid Search',
    icon: Grid,
    description:
      'Exhaustively search through a specified parameter grid.',
    pros: ['Thorough', 'Reproducible', 'Simple'],
    cons: ['Slow for large spaces', 'Curse of dimensionality'],
  },
  {
    name: 'Random Search',
    icon: Search,
    description:
      'Sample random combinations from the parameter space.',
    pros: ['Faster than grid', 'Good for large spaces', 'Often finds good solutions'],
    cons: ['May miss optimal', 'Less thorough'],
  },
  {
    name: 'Bayesian Optimization',
    icon: Zap,
    description:
      'Use probabilistic models to guide the search.',
    pros: ['Sample efficient', 'Handles complex spaces', 'Best for expensive evaluations'],
    cons: ['More complex', 'Overhead for simple models'],
  },
];

export default function TunerPage() {
  return (
    <div className="prose prose-gray dark:prose-invert max-w-none">
      <h1>Hyperparameter Tuning</h1>
      <p className="lead">
        mlcli includes powerful hyperparameter tuning capabilities to help you
        find the best model configuration. Choose from grid search, random search,
        or Bayesian optimization.
      </p>

      <h2>Quick Start</h2>
      <p>Run hyperparameter tuning with a single command:</p>
      <CodeBlock
        code={`mlcli tune data/train.csv \\
  --model random_forest \\
  --strategy random \\
  --n-iter 50 \\
  --cv 5`}
        language="bash"
      />

      <h2>Tuning Strategies</h2>
      <div className="not-prose mb-8 grid gap-4 md:grid-cols-3">
        {strategies.map((strategy) => (
          <Card key={strategy.name}>
            <CardHeader>
              <div className="flex items-center gap-2">
                <strategy.icon className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">{strategy.name}</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription className="mb-3">
                {strategy.description}
              </CardDescription>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    Pros:
                  </span>
                  <span className="ml-1 text-muted-foreground">
                    {strategy.pros.join(', ')}
                  </span>
                </div>
                <div>
                  <span className="font-medium text-orange-600 dark:text-orange-400">
                    Cons:
                  </span>
                  <span className="ml-1 text-muted-foreground">
                    {strategy.cons.join(', ')}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <h2>Grid Search</h2>
      <p>
        Grid search exhaustively evaluates all combinations of hyperparameters.
        Best for small parameter spaces.
      </p>
      <CodeBlock
        code={`mlcli tune data/train.csv \\
  --model xgboost \\
  --strategy grid \\
  --param n_estimators=100,200,300 \\
  --param max_depth=3,5,7 \\
  --param learning_rate=0.01,0.1 \\
  --cv 5 \\
  --scoring accuracy`}
        language="bash"
      />

      <h2>Random Search</h2>
      <p>
        Random search samples random combinations from the parameter space.
        More efficient for larger spaces.
      </p>
      <CodeBlock
        code={`mlcli tune data/train.csv \\
  --model lightgbm \\
  --strategy random \\
  --param n_estimators=int:50:500 \\
  --param max_depth=int:3:15 \\
  --param learning_rate=float:0.001:0.3:log \\
  --param subsample=float:0.5:1.0 \\
  --n-iter 100 \\
  --cv 5`}
        language="bash"
      />

      <h2>Bayesian Optimization</h2>
      <p>
        Uses probabilistic models to intelligently explore the parameter space.
        Best for expensive model evaluations.
      </p>
      <CodeBlock
        code={`mlcli tune data/train.csv \\
  --model random_forest \\
  --strategy bayesian \\
  --param n_estimators=int:50:500 \\
  --param max_depth=int:3:20 \\
  --param min_samples_split=int:2:20 \\
  --param min_samples_leaf=int:1:10 \\
  --n-iter 50 \\
  --cv 5`}
        language="bash"
      />

      <h2>Parameter Specification</h2>
      <p>
        Define parameter search spaces using different formats:
      </p>

      <h3>Categorical Values</h3>
      <CodeBlock
        code={`--param criterion=gini,entropy,log_loss`}
        language="bash"
      />

      <h3>Integer Range</h3>
      <CodeBlock
        code={`--param n_estimators=int:100:1000`}
        language="bash"
      />

      <h3>Float Range</h3>
      <CodeBlock
        code={`--param learning_rate=float:0.001:0.1`}
        language="bash"
      />

      <h3>Log Scale</h3>
      <CodeBlock
        code={`--param learning_rate=float:0.0001:1.0:log`}
        language="bash"
      />

      <h2>YAML Configuration</h2>
      <p>Define complex tuning configurations in YAML:</p>
      <CodeBlock
        code={`# tuning_config.yaml
model: xgboost
strategy: bayesian
n_iter: 100
cv: 5
scoring: f1_weighted

parameters:
  n_estimators:
    type: int
    low: 50
    high: 500
  max_depth:
    type: int
    low: 3
    high: 15
  learning_rate:
    type: float
    low: 0.001
    high: 0.3
    log: true
  subsample:
    type: float
    low: 0.5
    high: 1.0
  colsample_bytree:
    type: float
    low: 0.5
    high: 1.0
  reg_alpha:
    type: float
    low: 0.0001
    high: 10.0
    log: true
  reg_lambda:
    type: float
    low: 0.0001
    high: 10.0
    log: true

early_stopping:
  patience: 10
  min_delta: 0.001

output:
  best_params: best_params.json
  trials: trials.csv
  plots: tuning_plots/`}
        language="yaml"
      />

      <p>Run with the config file:</p>
      <CodeBlock
        code={`mlcli tune data/train.csv --config tuning_config.yaml`}
        language="bash"
      />

      <h2>Cross-Validation Options</h2>
      <CodeBlock
        code={`# Standard K-Fold
mlcli tune data.csv --model rf --cv 5

# Stratified K-Fold (for classification)
mlcli tune data.csv --model rf --cv stratified:5

# Time Series Split
mlcli tune data.csv --model rf --cv timeseries:5

# Repeated K-Fold
mlcli tune data.csv --model rf --cv repeated:5:3`}
        language="bash"
      />

      <h2>Scoring Metrics</h2>
      <p>Available scoring metrics for optimization:</p>

      <h3>Classification</h3>
      <ul>
        <li><code>accuracy</code> - Classification accuracy</li>
        <li><code>f1</code> - F1 score (binary)</li>
        <li><code>f1_weighted</code> - Weighted F1 score</li>
        <li><code>roc_auc</code> - ROC AUC score</li>
        <li><code>precision</code> - Precision score</li>
        <li><code>recall</code> - Recall score</li>
      </ul>

      <h3>Regression</h3>
      <ul>
        <li><code>neg_mse</code> - Negative mean squared error</li>
        <li><code>neg_rmse</code> - Negative root mean squared error</li>
        <li><code>neg_mae</code> - Negative mean absolute error</li>
        <li><code>r2</code> - R² score</li>
      </ul>

      <h2>Output</h2>
      <p>
        After tuning, mlcli outputs the best parameters and can save detailed
        results:
      </p>
      <CodeBlock
        code={`mlcli tune data.csv \\
  --model xgboost \\
  --strategy bayesian \\
  --n-iter 50 \\
  --output-params best_params.json \\
  --output-trials trials.csv \\
  --output-plots plots/`}
        language="bash"
      />

      <h2>Using Best Parameters</h2>
      <p>Train a model with the tuned parameters:</p>
      <CodeBlock
        code={`# Use the best parameters from tuning
mlcli train data.csv \\
  --model xgboost \\
  --params best_params.json \\
  --output models/tuned_xgboost.pkl`}
        language="bash"
      />

      {/* Navigation */}
      <div className="not-prose mt-12 flex items-center justify-between border-t pt-6">
        <Button variant="ghost" asChild>
          <Link href="/docs/explainer">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Explainability
          </Link>
        </Button>
        <Button variant="ghost" asChild>
          <Link href="/docs">
            Back to Docs
          </Link>
        </Button>
      </div>
    </div>
  );
}
