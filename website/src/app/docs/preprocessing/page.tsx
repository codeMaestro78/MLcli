import { CodeBlock } from '@/components/code';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowLeft, ArrowRight, Database, Filter, Hash, Scaling } from 'lucide-react';
import { Metadata } from 'next';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Data Preprocessing',
  description: 'Learn how to preprocess your data with mlcli.',
};

const preprocessors = [
  {
    name: 'Scalers',
    icon: Scaling,
    description: 'Normalize feature values to a common scale.',
    options: ['standard', 'minmax', 'robust', 'maxabs'],
  },
  {
    name: 'Encoders',
    icon: Hash,
    description: 'Convert categorical variables to numeric.',
    options: ['label', 'onehot', 'ordinal', 'target'],
  },
  {
    name: 'Imputers',
    icon: Database,
    description: 'Handle missing values in your data.',
    options: ['mean', 'median', 'most_frequent', 'constant'],
  },
  {
    name: 'Feature Selectors',
    icon: Filter,
    description: 'Select the most important features.',
    options: ['variance', 'kbest', 'mutual_info', 'rfe'],
  },
];

export default function PreprocessingPage() {
  return (
    <div className="prose prose-gray dark:prose-invert max-w-none">
      <h1>Data Preprocessing</h1>
      <p className="lead">
        mlcli includes a comprehensive data preprocessing module to prepare your
        data for training. Preprocess your data with a single command or
        configure preprocessing in your YAML files.
      </p>

      <h2>Quick Start</h2>
      <p>Preprocess a CSV file with default settings:</p>
      <CodeBlock
        code={`mlcli preprocess data/raw.csv \\
  --output data/processed.csv \\
  --scaler standard \\
  --encoder onehot \\
  --imputer mean`}
        language="bash"
      />

      <h2>Available Preprocessors</h2>
      <div className="not-prose mb-8 grid gap-4 md:grid-cols-2">
        {preprocessors.map((preprocessor) => (
          <Card key={preprocessor.name}>
            <CardHeader>
              <div className="flex items-center gap-2">
                <preprocessor.icon className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">{preprocessor.name}</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription className="mb-2">
                {preprocessor.description}
              </CardDescription>
              <div className="flex flex-wrap gap-1">
                {preprocessor.options.map((opt) => (
                  <code
                    key={opt}
                    className="rounded bg-muted px-1.5 py-0.5 text-xs"
                  >
                    {opt}
                  </code>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <h2>Scalers</h2>
      <p>
        Scalers normalize feature values to a common scale, which is important
        for algorithms that are sensitive to feature magnitudes.
      </p>

      <h3>Standard Scaler</h3>
      <p>
        Standardizes features by removing the mean and scaling to unit variance.
        Best for normally distributed data.
      </p>
      <CodeBlock
        code={`mlcli preprocess data.csv --scaler standard`}
        language="bash"
      />

      <h3>MinMax Scaler</h3>
      <p>
        Scales features to a given range (default 0 to 1). Good for neural networks
        and algorithms that expect bounded inputs.
      </p>
      <CodeBlock
        code={`mlcli preprocess data.csv --scaler minmax`}
        language="bash"
      />

      <h3>Robust Scaler</h3>
      <p>
        Uses median and IQR for scaling, making it robust to outliers.
      </p>
      <CodeBlock
        code={`mlcli preprocess data.csv --scaler robust`}
        language="bash"
      />

      <h2>Encoders</h2>
      <p>
        Encoders convert categorical variables to numeric values that ML algorithms
        can process.
      </p>

      <h3>Label Encoder</h3>
      <p>
        Converts each unique category to an integer. Simple but introduces ordinal
        relationship.
      </p>
      <CodeBlock
        code={`mlcli preprocess data.csv --encoder label --categorical-cols category1,category2`}
        language="bash"
      />

      <h3>One-Hot Encoder</h3>
      <p>
        Creates binary columns for each category. Best for nominal categorical
        variables.
      </p>
      <CodeBlock
        code={`mlcli preprocess data.csv --encoder onehot --categorical-cols category1,category2`}
        language="bash"
      />

      <h2>Feature Selection</h2>
      <p>
        Select the most relevant features to improve model performance and reduce
        training time.
      </p>

      <h3>Select K Best</h3>
      <p>Select the top K features based on statistical tests.</p>
      <CodeBlock
        code={`mlcli preprocess data.csv --selector kbest --k 10`}
        language="bash"
      />

      <h3>Variance Threshold</h3>
      <p>Remove features with low variance.</p>
      <CodeBlock
        code={`mlcli preprocess data.csv --selector variance --threshold 0.01`}
        language="bash"
      />

      <h2>YAML Configuration</h2>
      <p>Configure preprocessing in your experiment configuration file:</p>
      <CodeBlock
        code={`# config.yaml
preprocessing:
  scaler: standard
  encoder: onehot
  imputer:
    strategy: mean
    columns: [age, income]
  feature_selection:
    method: kbest
    k: 20
  categorical_columns:
    - category
    - status
    - type
  drop_columns:
    - id
    - timestamp`}
        language="yaml"
      />

      <h2>Pipeline Example</h2>
      <p>
        Run preprocessing as part of a training pipeline:
      </p>
      <CodeBlock
        code={`# Full training pipeline with preprocessing
mlcli train data/raw.csv \\
  --model random_forest \\
  --preprocess \\
  --scaler standard \\
  --encoder onehot \\
  --imputer median \\
  --output models/rf_model.pkl`}
        language="bash"
      />

      {/* Navigation */}
      <div className="not-prose mt-12 flex items-center justify-between border-t pt-6">
        <Button variant="ghost" asChild>
          <Link href="/docs/trainers">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Trainers
          </Link>
        </Button>
        <Button variant="ghost" asChild>
          <Link href="/docs/explainer">
            Explainability
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </div>
  );
}
