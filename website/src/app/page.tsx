import { CodeBlock } from '@/components/code';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { fetchLatestRelease, fetchRepoStats } from '@/lib/api';
import {
    ArrowRight,
    BarChart3,
    Brain,
    CheckCircle,
    Download,
    FlaskConical,
    GitFork,
    Github,
    Layers,
    Settings,
    Star,
    Terminal,
    Zap,
} from 'lucide-react';
import Link from 'next/link';

const features = [
  {
    icon: Terminal,
    title: 'CLI & TUI Interface',
    description:
      'Train models from the command line or use the interactive TUI for a rich experience.',
  },
  {
    icon: Layers,
    title: 'Multiple Algorithms',
    description:
      'Support for Random Forest, XGBoost, SVM, Logistic Regression, and deep learning models.',
  },
  {
    icon: FlaskConical,
    title: 'Experiment Tracking',
    description:
      'Track experiments, compare runs, and visualize metrics with built-in tracking.',
  },
  {
    icon: Settings,
    title: 'Hyperparameter Tuning',
    description:
      'Grid search, random search, and Bayesian optimization for optimal parameters.',
  },
  {
    icon: Brain,
    title: 'Model Explainability',
    description:
      'Understand your models with SHAP and LIME explanations and feature importance.',
  },
  {
    icon: BarChart3,
    title: 'Data Preprocessing',
    description:
      'Built-in preprocessing pipelines with scalers, encoders, and feature selection.',
  },
];

const quickstartCode = `# Install mlcli
pip install mlcli

# Train a Random Forest model
mlcli train -d data.csv -m random_forest --target label

# Evaluate the model
mlcli evaluate -m models/rf_model.pkl -d test.csv

# Track experiments
mlcli list-experiments

# Tune hyperparameters
mlcli tune -d data.csv -m random_forest --method bayesian`;

export default async function HomePage() {
  const [latestRelease, repoStats] = await Promise.all([
    fetchLatestRelease(),
    fetchRepoStats(),
  ]);

  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="hero-gradient relative overflow-hidden">
        <div className="container py-24 md:py-32">
          <div className="mx-auto max-w-4xl text-center">
            {/* Badges */}
            <div className="mb-6 flex flex-wrap items-center justify-center gap-2">
              {latestRelease && (
                <Badge variant="secondary" className="gap-1">
                  <Download className="h-3 w-3" />
                  {latestRelease.tag_name}
                </Badge>
              )}
              <Badge variant="outline" className="gap-1">
                <Star className="h-3 w-3" />
                {repoStats.stars} stars
              </Badge>
              <Badge variant="outline" className="gap-1">
                <GitFork className="h-3 w-3" />
                {repoStats.forks} forks
              </Badge>
            </div>

            {/* Title */}
            <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">
              Train ML Models
              <br />
              <span className="text-mlcli-500">From the Command Line</span>
            </h1>

            {/* Description */}
            <p className="mx-auto mb-8 max-w-2xl text-lg text-muted-foreground md:text-xl">
              A production-ready CLI tool for training, evaluating, and managing machine learning
              models with experiment tracking, hyperparameter tuning, and model explainability.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button asChild size="lg">
                <Link href="/docs/quickstart">
                  Get Started
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link
                  href="https://github.com/codeMaestro78/MLcli"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Github className="mr-2 h-4 w-4" />
                  View on GitHub
                </Link>
              </Button>
              <Button asChild variant="ghost" size="lg">
                <Link href="/ui-demo">
                  <Terminal className="mr-2 h-4 w-4" />
                  Try Demo
                </Link>
              </Button>
            </div>
          </div>
        </div>

        {/* Decorative gradient */}
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
      </section>

      {/* Install Section */}
      <section className="border-b py-12">
        <div className="container">
          <div className="mx-auto max-w-2xl">
            <CodeBlock
              code="pip install mlcli"
              language="bash"
              filename="Installation"
            />
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24">
        <div className="container">
          <div className="mx-auto mb-16 max-w-2xl text-center">
            <h2 className="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">
              Everything You Need for ML Workflows
            </h2>
            <p className="text-lg text-muted-foreground">
              From data preprocessing to model deployment, mlcli provides a complete toolkit
              for machine learning practitioners.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {features.map((feature) => (
              <Card key={feature.title} className="transition-shadow hover:shadow-lg">
                <CardHeader>
                  <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                    <feature.icon className="h-6 w-6 text-primary" />
                  </div>
                  <CardTitle>{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-base">{feature.description}</CardDescription>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Code Example Section */}
      <section className="border-y bg-muted/30 py-24">
        <div className="container">
          <div className="grid items-center gap-12 lg:grid-cols-2">
            <div>
              <h2 className="mb-4 text-3xl font-bold tracking-tight">
                Simple Yet Powerful CLI
              </h2>
              <p className="mb-6 text-lg text-muted-foreground">
                Get started in seconds with intuitive commands. Train models, track experiments,
                and tune hyperparameters with just a few keystrokes.
              </p>
              <ul className="space-y-3">
                {[
                  'Train multiple model types with one command',
                  'Automatic experiment tracking and logging',
                  'Built-in hyperparameter optimization',
                  'SHAP and LIME model explanations',
                  'Preprocessing pipelines included',
                ].map((item) => (
                  <li key={item} className="flex items-center gap-2">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <CodeBlock code={quickstartCode} language="bash" filename="Terminal" />
            </div>
          </div>
        </div>
      </section>

      {/* Quick Links Section */}
      <section className="py-24">
        <div className="container">
          <div className="mx-auto mb-12 max-w-2xl text-center">
            <h2 className="mb-4 text-3xl font-bold tracking-tight">Quick Links</h2>
            <p className="text-lg text-muted-foreground">
              Jump into the documentation or explore the experiment dashboard.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            <Link href="/docs/quickstart" className="group">
              <Card className="h-full transition-all hover:border-primary hover:shadow-lg">
                <CardHeader>
                  <Zap className="mb-2 h-8 w-8 text-primary" />
                  <CardTitle className="group-hover:text-primary">Quickstart</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>
                    Get up and running with mlcli in under 5 minutes.
                  </CardDescription>
                </CardContent>
              </Card>
            </Link>

            <Link href="/docs" className="group">
              <Card className="h-full transition-all hover:border-primary hover:shadow-lg">
                <CardHeader>
                  <Layers className="mb-2 h-8 w-8 text-primary" />
                  <CardTitle className="group-hover:text-primary">Documentation</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>
                    Comprehensive guides for all features and APIs.
                  </CardDescription>
                </CardContent>
              </Card>
            </Link>

            <Link href="/runs" className="group">
              <Card className="h-full transition-all hover:border-primary hover:shadow-lg">
                <CardHeader>
                  <FlaskConical className="mb-2 h-8 w-8 text-primary" />
                  <CardTitle className="group-hover:text-primary">Experiments</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>
                    View and analyze your experiment runs and metrics.
                  </CardDescription>
                </CardContent>
              </Card>
            </Link>

            <Link href="/releases" className="group">
              <Card className="h-full transition-all hover:border-primary hover:shadow-lg">
                <CardHeader>
                  <Download className="mb-2 h-8 w-8 text-primary" />
                  <CardTitle className="group-hover:text-primary">Releases</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>
                    Download the latest version and view changelog.
                  </CardDescription>
                </CardContent>
              </Card>
            </Link>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="border-t bg-muted/30 py-24">
        <div className="container">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="mb-4 text-3xl font-bold tracking-tight">Ready to Get Started?</h2>
            <p className="mb-8 text-lg text-muted-foreground">
              Join the community of ML practitioners using mlcli to streamline their workflows.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button asChild size="lg">
                <Link href="/docs/quickstart">
                  Read the Docs
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link href="/contribute">Contribute</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
