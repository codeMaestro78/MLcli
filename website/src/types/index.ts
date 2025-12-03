// Type definitions for mlcli website

// Experiment run types
export interface ExperimentRun {
  run_id: string;
  timestamp: string;
  model_type: string;
  framework: string;
  dataset: string;
  target_column: string;
  hyperparameters: Record<string, unknown>;
  metrics: RunMetrics;
  artifacts?: RunArtifacts;
  mlflow_run_id?: string;
  mlflow_tracking_uri?: string;
  epoch_history?: EpochMetric[];
  notes?: string;
  tags?: string[];
}

export interface RunMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  auc?: number;
  loss?: number;
  mse?: number;
  mae?: number;
  r2?: number;
  [key: string]: number | undefined;
}

export interface RunArtifacts {
  model_path?: string;
  config_path?: string;
  logs_path?: string;
  plots?: string[];
}

export interface EpochMetric {
  epoch: number;
  loss: number;
  accuracy?: number;
  val_loss?: number;
  val_accuracy?: number;
}

// GitHub Release types
export interface GitHubRelease {
  id: number;
  tag_name: string;
  name: string;
  body: string;
  draft: boolean;
  prerelease: boolean;
  created_at: string;
  published_at: string;
  html_url: string;
  assets: GitHubAsset[];
  author: {
    login: string;
    avatar_url: string;
    html_url: string;
  };
}

export interface GitHubAsset {
  id: number;
  name: string;
  size: number;
  download_count: number;
  browser_download_url: string;
  content_type: string;
}

// Documentation types
export interface DocPage {
  slug: string;
  title: string;
  description?: string;
  content: string;
  order?: number;
  section?: string;
  version?: string;
}

export interface DocSection {
  title: string;
  pages: DocPage[];
}

export interface DocSidebarItem {
  title: string;
  href: string;
  items?: DocSidebarItem[];
}

// Search types
export interface SearchResult {
  title: string;
  description: string;
  href: string;
  section: string;
  content?: string;
}

// API response types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  error?: string;
  meta?: {
    total?: number;
    page?: number;
    limit?: number;
  };
}

export interface RunsFilter {
  model_type?: string;
  framework?: string;
  date_from?: string;
  date_to?: string;
  min_accuracy?: number;
  max_accuracy?: number;
  tags?: string[];
}

// Component prop types
export interface BadgeProps {
  variant?: 'default' | 'secondary' | 'success' | 'warning' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  children: React.ReactNode;
}

export interface CardProps {
  title?: string;
  description?: string;
  className?: string;
  children: React.ReactNode;
}

// Chart types
export interface ChartDataPoint {
  name: string;
  value: number;
  [key: string]: string | number;
}

export interface MetricHistogram {
  range: string;
  count: number;
}
