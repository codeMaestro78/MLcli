import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowRight, Box, Brain, FlaskConical, Layers, Settings, Terminal } from 'lucide-react';
import Link from 'next/link';

export const metadata = {
  title: 'Documentation',
  description: 'Learn how to use mlcli to train, evaluate, and manage machine learning models.',
};

const sections = [
  {
    icon: Terminal,
    title: 'Getting Started',
    description: 'Install mlcli and run your first training job.',
    href: '/docs/quickstart',
  },
  {
    icon: Settings,
    title: 'Configuration',
    description: 'Learn about config files and environment variables.',
    href: '/docs/config',
  },
  {
    icon: Layers,
    title: 'Trainers',
    description: 'Explore available model trainers and how to use them.',
    href: '/docs/trainers',
  },
  {
    icon: Box,
    title: 'Preprocessing',
    description: 'Data preprocessing pipelines, scalers, and encoders.',
    href: '/docs/preprocessing',
  },
  {
    icon: FlaskConical,
    title: 'Experiment Tracking',
    description: 'Track experiments, compare runs, and log metrics.',
    href: '/docs/experiments',
  },
  {
    icon: Brain,
    title: 'Model Explainability',
    description: 'Understand your models with SHAP and LIME.',
    href: '/docs/explainability',
  },
];

export default function DocsPage() {
  return (
    <div>
      <div className="mb-8">
        <h1 className="mb-4 text-4xl font-bold tracking-tight">Documentation</h1>
        <p className="text-xl text-muted-foreground">
          Learn how to use mlcli to train, evaluate, and manage machine learning models
          from the command line.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {sections.map((section) => (
          <Link key={section.href} href={section.href} className="group">
            <Card className="h-full transition-all hover:border-primary hover:shadow-md">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className="rounded-lg bg-primary/10 p-2">
                    <section.icon className="h-5 w-5 text-primary" />
                  </div>
                  <CardTitle className="group-hover:text-primary">{section.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription className="flex items-center justify-between">
                  {section.description}
                  <ArrowRight className="h-4 w-4 opacity-0 transition-opacity group-hover:opacity-100" />
                </CardDescription>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      <div className="mt-12 rounded-lg border bg-muted/30 p-6">
        <h2 className="mb-2 text-lg font-semibold">Need Help?</h2>
        <p className="text-muted-foreground">
          If you cant find what youre looking for, check out our{' '}
          <Link href="https://github.com/codeMaestro78/MLcli/discussions" className="text-primary hover:underline">
            GitHub Discussions
          </Link>{' '}
          or{' '}
          <Link href="https://github.com/codeMaestro78/MLcli/issues" className="text-primary hover:underline">
            open an issue
          </Link>
          .
        </p>
      </div>
    </div>
  );
}
