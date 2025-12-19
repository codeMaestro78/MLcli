import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowRight, Box, Brain, FlaskConical, Layers, Settings, Sparkles, Terminal } from 'lucide-react';
import Link from 'next/link';

export const metadata = {
  title: 'Documentation',
  description: 'Learn how to use mlcli to train, evaluate, and manage machine learning models.',
};

const sections = [
  {
    icon: Terminal,
    title: 'Getting Started',
    description: 'Install mlcli and run your first training job in minutes.',
    href: '/docs/quickstart',
    gradient: 'from-blue-500/20 to-cyan-500/20',
    iconColor: 'text-blue-500',
  },
  {
    icon: Settings,
    title: 'Configuration',
    description: 'Learn about config files, environment variables, and customization.',
    href: '/docs/config',
    gradient: 'from-purple-500/20 to-pink-500/20',
    iconColor: 'text-purple-500',
  },
  {
    icon: Layers,
    title: 'Trainers',
    description: 'Explore 15+ model trainers from scikit-learn, XGBoost, and TensorFlow.',
    href: '/docs/trainers',
    gradient: 'from-orange-500/20 to-red-500/20',
    iconColor: 'text-orange-500',
  },
  {
    icon: Box,
    title: 'Preprocessing',
    description: 'Data preprocessing pipelines, scalers, encoders, and feature selection.',
    href: '/docs/preprocessing',
    gradient: 'from-green-500/20 to-emerald-500/20',
    iconColor: 'text-green-500',
  },
  {
    icon: FlaskConical,
    title: 'Experiment Tracking',
    description: 'Track experiments, compare runs, log metrics, and visualize results.',
    href: '/docs/experiments',
    gradient: 'from-yellow-500/20 to-orange-500/20',
    iconColor: 'text-yellow-500',
  },
  {
    icon: Brain,
    title: 'Model Explainability',
    description: 'Understand your models with SHAP, LIME, and feature importance.',
    href: '/docs/explainer',
    gradient: 'from-pink-500/20 to-rose-500/20',
    iconColor: 'text-pink-500',
  },
];

export default function DocsPage() {
  return (
    <div>
      {/* Hero Section */}
      <div className="relative mb-12">
        <div className="absolute -top-4 -left-4 w-24 h-24 bg-primary/10 rounded-full blur-3xl" />
        <div className="relative">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4">
            <Sparkles className="h-3.5 w-3.5" />
            v0.3.0 Now Available
          </div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 bg-gradient-to-r from-foreground via-foreground to-foreground/50 bg-clip-text">
            Documentation
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl leading-relaxed">
            Learn how to use mlcli to train, evaluate, and manage machine learning models
            from the command line with ease.
          </p>
        </div>
      </div>

      {/* Quick Links Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 stagger-children">
        {sections.map((section) => (
          <Link key={section.href} href={section.href} className="group">
            <Card className="h-full docs-card border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className={`rounded-xl bg-gradient-to-br ${section.gradient} p-3 ring-1 ring-inset ring-white/10`}>
                    <section.icon className={`h-5 w-5 ${section.iconColor}`} />
                  </div>
                  <ArrowRight className="h-4 w-4 text-muted-foreground opacity-0 -translate-x-2 transition-all duration-200 group-hover:opacity-100 group-hover:translate-x-0" />
                </div>
                <CardTitle className="text-lg mt-4 group-hover:text-primary transition-colors duration-200">
                  {section.title}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-sm leading-relaxed">
                  {section.description}
                </CardDescription>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {/* Quick Start Section */}
      <div className="mt-16 relative">
        <div className="absolute -right-4 top-1/2 w-32 h-32 bg-primary/5 rounded-full blur-3xl" />
        <div className="relative rounded-2xl border border-border/50 bg-gradient-to-br from-muted/30 to-muted/10 p-8 backdrop-blur-sm">
          <h2 className="text-2xl font-semibold mb-3">Quick Install</h2>
          <p className="text-muted-foreground mb-6 max-w-xl">
            Get started with mlcli in seconds. Install via pip and you're ready to train your first model.
          </p>
          <div className="flex flex-col sm:flex-row gap-4">
            <code className="flex-1 block rounded-lg bg-zinc-950 dark:bg-zinc-900 px-4 py-3 font-mono text-sm text-zinc-100">
              <span className="text-zinc-500">$</span> pip install mlcli-toolkit
            </code>
            <Link
              href="/docs/quickstart"
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Get Started
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* Help Section */}
      <div className="mt-12 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 rounded-xl border border-border/50 bg-muted/20 p-6 backdrop-blur-sm">
        <div>
          <h3 className="font-semibold mb-1">Need Help?</h3>
          <p className="text-sm text-muted-foreground">
            Can't find what you're looking for? We're here to help.
          </p>
        </div>
        <div className="flex gap-3">
          <Link
            href="https://github.com/codeMaestro78/MLcli/discussions"
            className="inline-flex items-center gap-2 rounded-lg border border-border/50 bg-background px-4 py-2 text-sm font-medium hover:bg-muted transition-colors"
            target="_blank"
          >
            Discussions
          </Link>
          <Link
            href="https://github.com/codeMaestro78/MLcli/issues"
            className="inline-flex items-center gap-2 rounded-lg border border-border/50 bg-background px-4 py-2 text-sm font-medium hover:bg-muted transition-colors"
            target="_blank"
          >
            Report Issue
          </Link>
        </div>
      </div>
    </div>
  );
}
