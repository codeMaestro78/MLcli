import { NextRequest, NextResponse } from 'next/server';

interface SearchResult {
  title: string;
  href: string;
  section: string;
  preview: string;
}

// Static searchable content - in production, this could be generated from MDX files
const searchableContent: SearchResult[] = [
  {
    title: 'Introduction',
    href: '/docs',
    section: 'Getting Started',
    preview: 'mlcli is a production-ready CLI tool for training, evaluating, and managing machine learning models.',
  },
  {
    title: 'Quickstart',
    href: '/docs/quickstart',
    section: 'Getting Started',
    preview: 'Install mlcli and run your first training command in minutes.',
  },
  {
    title: 'Configuration',
    href: '/docs/config',
    section: 'Reference',
    preview: 'Configure mlcli with YAML files and command-line options.',
  },
  {
    title: 'Trainers',
    href: '/docs/trainers',
    section: 'Reference',
    preview: 'Available trainers: Random Forest, XGBoost, LightGBM, SVM, and Deep Learning.',
  },
  {
    title: 'Data Preprocessing',
    href: '/docs/preprocessing',
    section: 'Features',
    preview: 'Preprocess your data with scalers, encoders, imputers, and feature selectors.',
  },
  {
    title: 'Model Explainability',
    href: '/docs/explainer',
    section: 'Features',
    preview: 'Understand your models with SHAP and LIME explanations.',
  },
  {
    title: 'Hyperparameter Tuning',
    href: '/docs/tuner',
    section: 'Features',
    preview: 'Optimize your models with grid search, random search, or Bayesian optimization.',
  },
  {
    title: 'Experiment Runs',
    href: '/runs',
    section: 'Dashboard',
    preview: 'View and compare your experiment runs with metrics and visualizations.',
  },
  {
    title: 'Releases',
    href: '/releases',
    section: 'Downloads',
    preview: 'View all mlcli releases and changelogs from GitHub.',
  },
  {
    title: 'Download',
    href: '/download',
    section: 'Downloads',
    preview: 'Install mlcli via pip or from source.',
  },
  {
    title: 'About',
    href: '/about',
    section: 'Info',
    preview: 'Learn about the mlcli project, mission, and team.',
  },
  {
    title: 'Contribute',
    href: '/contribute',
    section: 'Info',
    preview: 'Learn how to contribute to mlcli - bug reports, code, and documentation.',
  },
  {
    title: 'Random Forest',
    href: '/docs/trainers',
    section: 'Trainers',
    preview: 'Train Random Forest models with scikit-learn for classification and regression.',
  },
  {
    title: 'XGBoost',
    href: '/docs/trainers',
    section: 'Trainers',
    preview: 'Train XGBoost models for high-performance gradient boosting.',
  },
  {
    title: 'LightGBM',
    href: '/docs/trainers',
    section: 'Trainers',
    preview: 'Train LightGBM models for fast and efficient gradient boosting.',
  },
  {
    title: 'Deep Learning',
    href: '/docs/trainers',
    section: 'Trainers',
    preview: 'Train deep neural networks with TensorFlow and Keras.',
  },
  {
    title: 'SHAP',
    href: '/docs/explainer',
    section: 'Explainability',
    preview: 'SHapley Additive exPlanations for understanding model predictions.',
  },
  {
    title: 'LIME',
    href: '/docs/explainer',
    section: 'Explainability',
    preview: 'Local Interpretable Model-agnostic Explanations for individual predictions.',
  },
];

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get('q')?.toLowerCase() || '';
  const limit = parseInt(searchParams.get('limit') || '10', 10);

  if (!query) {
    return NextResponse.json({
      results: [],
      success: true,
      meta: { query: '', total: 0 },
    });
  }

  const results = searchableContent
    .filter(
      (item) =>
        item.title.toLowerCase().includes(query) ||
        item.preview.toLowerCase().includes(query) ||
        item.section.toLowerCase().includes(query)
    )
    .slice(0, limit);

  return NextResponse.json({
    results,
    success: true,
    meta: {
      query,
      total: results.length,
    },
  });
}
