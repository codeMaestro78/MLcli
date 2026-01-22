import { CodeBlock } from '@/components/code';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowLeft, ArrowRight, BarChart2, Lightbulb, LineChart, Zap } from 'lucide-react';
import { Metadata } from 'next';
import Link from 'next/link';

export const metadata: Metadata = {
  title: 'Model Explainability',
  description: 'Understand your models with SHAP and LIME explanations.',
};

const explainers = [
  {
    name: 'SHAP',
    icon: BarChart2,
    description:
      'SHapley Additive exPlanations - game-theoretic approach to explain model predictions.',
    features: ['Feature importance', 'Dependency plots', 'Force plots', 'Summary plots'],
  },
  {
    name: 'LIME',
    icon: Lightbulb,
    description:
      'Local Interpretable Model-agnostic Explanations - explains individual predictions.',
    features: ['Local explanations', 'Model-agnostic', 'Interpretable models', 'Feature weights'],
  },
  {
    name: 'Feature Importance',
    icon: Zap,
    description:
      'Built-in feature importance from tree-based models.',
    features: ['Fast computation', 'Native support', 'Gini importance', 'Permutation importance'],
  },
  {
    name: 'Partial Dependence',
    icon: LineChart,
    description:
      'Show the marginal effect of features on predictions.',
    features: ['1D plots', '2D interaction plots', 'ICE plots', 'Feature effects'],
  },
];

export default function ExplainerPage() {
  return (
    <div className="prose prose-gray dark:prose-invert max-w-none">
      <h1>Model Explainability</h1>
      <p className="lead">
        mlcli includes built-in model explainability tools to help you understand
        why your models make specific predictions. Use SHAP, LIME, and other
        techniques to build trust in your models.
      </p>

      <h2>Quick Start</h2>
      <p>Generate SHAP explanations for a trained model:</p>
      <CodeBlock
        code={`# Generate SHAP summary plot
mlcli explain models/rf_model.pkl \\
  --data data/test.csv \\
  --method shap \\
  --output explanations/`}
        language="bash"
      />

      <h2>Available Explainers</h2>
      <div className="not-prose mb-8 grid gap-4 md:grid-cols-2">
        {explainers.map((explainer) => (
          <Card key={explainer.name}>
            <CardHeader>
              <div className="flex items-center gap-2">
                <explainer.icon className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">{explainer.name}</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <CardDescription className="mb-3">
                {explainer.description}
              </CardDescription>
              <ul className="space-y-1 text-sm text-muted-foreground">
                {explainer.features.map((feature) => (
                  <li key={feature} className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    {feature}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        ))}
      </div>

      <h2>SHAP Explanations</h2>
      <p>
        SHAP (SHapley Additive exPlanations) uses game theory to explain the
        output of any machine learning model. It connects optimal credit
        allocation with local explanations.
      </p>

      <h3>Summary Plot</h3>
      <p>
        Shows the importance and impact of all features across all predictions:
      </p>
      <CodeBlock
        code={`mlcli explain model.pkl --method shap --plot summary`}
        language="bash"
      />

      <h3>Force Plot</h3>
      <p>Explains a single prediction by showing feature contributions:</p>
      <CodeBlock
        code={`mlcli explain model.pkl --method shap --plot force --sample 0`}
        language="bash"
      />

      <h3>Dependence Plot</h3>
      <p>Shows the effect of a single feature across all predictions:</p>
      <CodeBlock
        code={`mlcli explain model.pkl --method shap --plot dependence --feature age`}
        language="bash"
      />

      <h2>LIME Explanations</h2>
      <p>
        LIME (Local Interpretable Model-agnostic Explanations) explains individual
        predictions by approximating the model locally with an interpretable model.
      </p>

      <h3>Single Prediction</h3>
      <CodeBlock
        code={`mlcli explain model.pkl \\
  --method lime \\
  --sample 42 \\
  --num-features 10`}
        language="bash"
      />

      <h3>Batch Explanations</h3>
      <CodeBlock
        code={`mlcli explain model.pkl \\
  --method lime \\
  --samples 0,1,2,3,4 \\
  --output explanations/lime/`}
        language="bash"
      />

      <h2>Feature Importance</h2>
      <p>
        For tree-based models, get built-in feature importance scores:
      </p>
      <CodeBlock
        code={`# Feature importance from model
mlcli explain model.pkl --method importance

# Permutation importance (model-agnostic)
mlcli explain model.pkl \\
  --method permutation \\
  --data test.csv \\
  --n-repeats 10`}
        language="bash"
      />

      <h2>Partial Dependence Plots</h2>
      <p>
        Visualize the marginal effect of one or two features on predictions:
      </p>
      <CodeBlock
        code={`# 1D PDP
mlcli explain model.pkl --method pdp --features age,income

# 2D interaction PDP
mlcli explain model.pkl --method pdp --features age,income --interaction`}
        language="bash"
      />

      <h2>YAML Configuration</h2>
      <p>Configure explainability in your experiment file:</p>
      <CodeBlock
        code={`# config.yaml
explainability:
  methods:
    - shap
    - lime
  shap:
    plot_types:
      - summary
      - force
      - dependence
    max_samples: 100
  lime:
    num_features: 10
    num_samples: 5000
  output_dir: ./explanations`}
        language="yaml"
      />

      <h2>Programmatic Access</h2>
      <p>Use the Python API for more control:</p>
      <CodeBlock
        code={`from mlcli.explainer import SHAPExplainer, LIMEExplainer

# Load model and data
model = load_model("models/rf_model.pkl")
X_test = pd.read_csv("data/test.csv")

# SHAP explanations
shap_explainer = SHAPExplainer(model)
shap_values = shap_explainer.explain(X_test)
shap_explainer.plot_summary(shap_values, X_test)

# LIME explanations
lime_explainer = LIMEExplainer(model, X_test)
explanation = lime_explainer.explain_instance(X_test.iloc[0])
explanation.show_in_notebook()`}
        language="python"
      />

      <h2>Best Practices</h2>
      <ul>
        <li>
          <strong>Sample size:</strong> For SHAP, use a representative sample
          (100-1000 instances) for summary plots
        </li>
        <li>
          <strong>Background data:</strong> SHAP requires background data - use
          training data or a summary
        </li>
        <li>
          <strong>Interpretation:</strong> SHAP values are additive - they sum to
          the difference from the expected value
        </li>
        <li>
          <strong>Model type:</strong> Use TreeExplainer for tree models
          (faster), KernelExplainer for others
        </li>
      </ul>

      {/* Navigation */}
      <div className="not-prose mt-12 flex items-center justify-between border-t pt-6">
        <Button variant="ghost" asChild>
          <Link href="/docs/preprocessing">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Preprocessing
          </Link>
        </Button>
        <Button variant="ghost" asChild>
          <Link href="/docs/tuner">
            Hyperparameter Tuning
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
    </div>
  );
}
