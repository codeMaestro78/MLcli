import { CodeBlock } from '@/components/code';
import { Button } from '@/components/ui/button';
import { ArrowRight, CheckCircle } from 'lucide-react';
import Link from 'next/link';

export const metadata = {
  title: 'Quickstart',
  description: 'Get started with mlcli in under 5 minutes.',
};

export default function QuickstartPage() {
  return (
    <div>
      <h1>Quickstart</h1>
      <p className="lead">
        Get started with mlcli in under 5 minutes. This guide will walk you through
        installation and your first training run.
      </p>

      <h2 id="installation">Installation</h2>
      <p>Install mlcli using pip:</p>
      <CodeBlock code="pip install mlcli-toolkit" language="bash" />

      <p>Or using pipx for an isolated installation:</p>
      <CodeBlock code="pipx install mlcli-toolkit" language="bash" />

      <p>Verify the installation:</p>
      <CodeBlock
        code={`mlcli --version
# mlcli v0.3.0`}
        language="bash"
      />

      <h2 id="first-training">Your First Training Run</h2>
      <p>
        Train a Random Forest model on your dataset with a single command:
      </p>
      <CodeBlock
        code={`mlcli train \\
  --data data/train.csv \\
  --model random_forest \\
  --target label \\
  --output models/`}
        language="bash"
      />

      <p>This command will:</p>
      <ul>
        <li>
          <CheckCircle className="mr-2 inline h-4 w-4 text-green-500" />
          Load and preprocess your data
        </li>
        <li>
          <CheckCircle className="mr-2 inline h-4 w-4 text-green-500" />
          Train a Random Forest classifier
        </li>
        <li>
          <CheckCircle className="mr-2 inline h-4 w-4 text-green-500" />
          Evaluate on a validation split
        </li>
        <li>
          <CheckCircle className="mr-2 inline h-4 w-4 text-green-500" />
          Save the trained model
        </li>
        <li>
          <CheckCircle className="mr-2 inline h-4 w-4 text-green-500" />
          Log the experiment
        </li>
      </ul>

      <h2 id="using-config">Using Config Files</h2>
      <p>
        For more control over your training, use a JSON configuration file:
      </p>
      <CodeBlock
        code={`{
  "model_type": "random_forest",
  "dataset_path": "data/train.csv",
  "target_column": "label",
  "test_size": 0.2,
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2
  },
  "output_dir": "models/"
}`}
        language="json"
        filename="config.json"
      />

      <p>Then run training with the config:</p>
      <CodeBlock code="mlcli train --config config.json" language="bash" />

      <h2 id="evaluate">Evaluate Your Model</h2>
      <p>Evaluate your trained model on a test set:</p>
      <CodeBlock
        code={`mlcli eval \\
  --model models/random_forest_model.pkl \\
  --data data/test.csv`}
        language="bash"
      />

      <h2 id="view-experiments">View Experiments</h2>
      <p>List all your experiment runs:</p>
      <CodeBlock code="mlcli list-runs" language="bash" />

      <p>View details of a specific run:</p>
      <CodeBlock code="mlcli show-run --run-id run_abc123" language="bash" />

      <h2 id="next-steps">Next Steps</h2>
      <div className="not-prose mt-6 flex gap-4">
        <Button asChild>
          <Link href="/docs/trainers">
            Explore Trainers
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/docs/config">Configuration Guide</Link>
        </Button>
      </div>
    </div>
  );
}
