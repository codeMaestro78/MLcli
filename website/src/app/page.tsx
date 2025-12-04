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
    Code2,
    Download,
    FlaskConical,
    GitFork,
    Github,
    Layers,
    Rocket,
    Settings,
    Sparkles,
    Star,
    Terminal,
    TrendingUp,
    Zap,
} from 'lucide-react';
import Link from 'next/link';

const features = [
  {
    icon: Terminal,
    title: 'CLI & TUI Interface',
    description:
      'Train models from the command line or use the interactive TUI for a rich experience.',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    icon: Layers,
    title: 'Multiple Algorithms',
    description:
      'Support for Random Forest, XGBoost, SVM, Logistic Regression, and deep learning models.',
    color: 'from-purple-500 to-pink-500',
  },
  {
    icon: FlaskConical,
    title: 'Experiment Tracking',
    description:
      'Track experiments, compare runs, and visualize metrics with built-in tracking.',
    color: 'from-orange-500 to-red-500',
  },
  {
    icon: Settings,
    title: 'Hyperparameter Tuning',
    description:
      'Grid search, random search, and Bayesian optimization for optimal parameters.',
    color: 'from-green-500 to-emerald-500',
  },
  {
    icon: Brain,
    title: 'Model Explainability',
    description:
      'Understand your models with SHAP and LIME explanations and feature importance.',
    color: 'from-indigo-500 to-purple-500',
  },
  {
    icon: BarChart3,
    title: 'Data Preprocessing',
    description:
      'Built-in preprocessing pipelines with scalers, encoders, and feature selection.',
    color: 'from-teal-500 to-cyan-500',
  },
];

const quickstartCode = `# Install mlcli-toolkit
pip install mlcli-toolkit

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
        <div className="container py-24 md:py-32 lg:py-40">
          <div className="mx-auto max-w-4xl text-center">
            {/* Badges */}
            <div className="mb-6 flex flex-wrap items-center justify-center gap-2">
              {latestRelease && (
                <Badge variant="secondary" className="gap-1 animate-in">
                  <Sparkles className="h-3 w-3" />
                  {latestRelease.tag_name}
                </Badge>
              )}
              <Badge variant="outline" className="gap-1 animate-in">
                <Star className="h-3 w-3 text-yellow-500" />
                {repoStats.stars} stars
              </Badge>
              <Badge variant="outline" className="gap-1 animate-in">
                <GitFork className="h-3 w-3" />
                {repoStats.forks} forks
              </Badge>
            </div>

            {/* Title */}
            <h1 className="mb-6 text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
              Train ML Models
              <br />
              <span className="gradient-text">From the Command Line</span>
            </h1>

            {/* Description */}
            <p className="mx-auto mb-8 max-w-2xl text-lg text-muted-foreground md:text-xl">
              A production-ready CLI tool for training, evaluating, and managing machine learning
              models with experiment tracking, hyperparameter tuning, and model explainability.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button asChild size="lg" className="shine-effect">
                <Link href="/docs/quickstart">
                  <Rocket className="mr-2 h-4 w-4" />
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
              code="pip install mlcli-toolkit"
              language="bash"
              filename="Installation"
            />
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 border-b">
        <div className="container">
          <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Code2 className="h-6 w-6 text-primary mr-2" />
              </div>
              <p className="text-3xl font-bold stat-number">7+</p>
              <p className="text-sm text-muted-foreground">ML Models</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Settings className="h-6 w-6 text-primary mr-2" />
              </div>
              <p className="text-3xl font-bold stat-number">3</p>
              <p className="text-sm text-muted-foreground">Tuning Methods</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <TrendingUp className="h-6 w-6 text-primary mr-2" />
              </div>
              <p className="text-3xl font-bold stat-number">10+</p>
              <p className="text-sm text-muted-foreground">Preprocessors</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Brain className="h-6 w-6 text-primary mr-2" />
              </div>
              <p className="text-3xl font-bold stat-number">2</p>
              <p className="text-sm text-muted-foreground">Explainability Tools</p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24">
        <div className="container">
          <div className="mx-auto mb-16 max-w-2xl text-center">
            <Badge variant="outline" className="mb-4">
              <Sparkles className="h-3 w-3 mr-1" />
              Features
            </Badge>
            <h2 className="mb-4 text-3xl font-bold tracking-tight sm:text-4xl">
              Everything You Need for ML Workflows
            </h2>
            <p className="text-lg text-muted-foreground">
              From data preprocessing to model deployment, mlcli provides a complete toolkit
              for machine learning practitioners.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 stagger-in">
            {features.map((feature) => (
              <Card key={feature.title} className="feature-card group">
                <CardHeader>
                  <div className={`mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br ${feature.color} float-animation`}>
                    <feature.icon className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="group-hover:text-primary transition-colors">{feature.title}</CardTitle>
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
            <Badge variant="outline" className="mb-4">
              <Zap className="h-3 w-3 mr-1" />
              Quick Access
            </Badge>
            <h2 className="mb-4 text-3xl font-bold tracking-tight">Quick Links</h2>
            <p className="text-lg text-muted-foreground">
              Jump into the documentation or explore the experiment dashboard.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4 stagger-in">
            <Link href="/docs/quickstart" className="group">
              <Card className="h-full feature-card">
                <CardHeader>
                  <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-yellow-500 to-orange-500">
                    <Zap className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="group-hover:text-primary transition-colors">Quickstart</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>
                    Get up and running with mlcli in under 5 minutes.
                  </CardDescription>
                </CardContent>
              </Card>
            </Link>

            <Link href="/docs" className="group">
              <Card className="h-full feature-card">
                <CardHeader>
                  <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-indigo-500">
                    <Layers className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="group-hover:text-primary transition-colors">Documentation</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>
                    Comprehensive guides for all features and APIs.
                  </CardDescription>
                </CardContent>
              </Card>
            </Link>

            <Link href="/runs" className="group">
              <Card className="h-full feature-card">
                <CardHeader>
                  <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-pink-500">
                    <FlaskConical className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="group-hover:text-primary transition-colors">Experiments</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription>
                    View and analyze your experiment runs and metrics.
                  </CardDescription>
                </CardContent>
              </Card>
            </Link>

            <Link href="/releases" className="group">
              <Card className="h-full feature-card">
                <CardHeader>
                  <div className="mb-2 flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-green-500 to-emerald-500">
                    <Download className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="group-hover:text-primary transition-colors">Releases</CardTitle>
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
      <section className="border-t bg-muted/30 py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-primary/5" />
        <div className="container relative">
          <div className="mx-auto max-w-2xl text-center">
            <div className="mb-4 inline-flex items-center justify-center rounded-full bg-primary/10 p-3">
              <Rocket className="h-6 w-6 text-primary" />
            </div>
            <h2 className="mb-4 text-3xl font-bold tracking-tight">Ready to Get Started?</h2>
            <p className="mb-8 text-lg text-muted-foreground">
              Join the community of ML practitioners using mlcli to streamline their workflows.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button asChild size="lg" className="shine-effect">
                <Link href="/docs/quickstart">
                  Read the Docs
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link href="https://github.com/codeMaestro78/MLcli" target="_blank" rel="noopener noreferrer">
                  <Github className="mr-2 h-4 w-4" />
                  Star on GitHub
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
