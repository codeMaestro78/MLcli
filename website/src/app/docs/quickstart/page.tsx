import { CodeBlock } from '@/components/code';
import { Button } from '@/components/ui/button';
import { ArrowRight, CheckCircle, Rocket } from 'lucide-react';
import Link from 'next/link';

export const metadata = {
  title: 'Quickstart',
  description: 'Get started with mlcli in under 5 minutes.',
};

export default function QuickstartPage() {
  return (
    <div>
      {/* Hero Section */}
      <div className="not-prose mb-10">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 text-green-600 dark:text-green-400 text-sm font-medium mb-4">
          <Rocket className="h-3.5 w-3.5" />
          5 minute setup
        </div>
        <h1 className="text-4xl font-bold tracking-tight mb-4">Quickstart</h1>
        <p className="text-xl text-muted-foreground leading-relaxed max-w-2xl">
          Get started with mlcli in under 5 minutes. This guide will walk you through
          installation and your first training run.
        </p>
      </div>

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
      <div className="not-prose my-6 grid gap-3">
        {[
          'Load and preprocess your data',
          'Train a Random Forest classifier',
          'Evaluate on a validation split',
          'Save the trained model',
          'Log the experiment',
        ].map((item, index) => (
          <div
            key={item}
            className="flex items-center gap-3 p-3 rounded-lg bg-green-500/5 border border-green-500/20 animate-fade-in-up"
            style={{ animationDelay: `${index * 100}ms` }}
          >
            <CheckCircle className="h-5 w-5 text-green-500 shrink-0" />
            <span className="text-sm font-medium">{item}</span>
          </div>
        ))}
      </div>

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
      <div className="not-prose mt-8 flex flex-wrap gap-4">
        <Button asChild size="lg" className="group">
          <Link href="/docs/trainers">
            Explore Trainers
            <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
          </Link>
        </Button>
        <Button asChild variant="outline" size="lg">
          <Link href="/docs/config">Configuration Guide</Link>
        </Button>
      </div>
    </div>
  );
}
