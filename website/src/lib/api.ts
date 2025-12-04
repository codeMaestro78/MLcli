import type { ExperimentRun, GitHubRelease } from '@/types';
import { promises as fs } from 'fs';
import path from 'path';

const GITHUB_REPO = process.env.GITHUB_REPO || 'codeMaestro78/MLcli';
const GITHUB_API_BASE = 'https://api.github.com';

/**
 * Fetch experiments from JSON file or URL
 */
export async function fetchExperiments(): Promise<ExperimentRun[]> {
  try {
    // Try to read from local file first (works in both dev and build)
    const filePath = path.join(process.cwd(), 'public', 'runs', 'experiments.json');
    const fileContents = await fs.readFile(filePath, 'utf-8');
    const data = JSON.parse(fileContents);
    return data.runs || data;
  } catch {
    // Fallback: try to fetch from configured URL
    const experimentsUrl = process.env.EXPERIMENTS_JSON_URL;

    if (experimentsUrl) {
      try {
        const response = await fetch(experimentsUrl, {
          next: { revalidate: 60 },
        });

        if (response.ok) {
          const data = await response.json();
          return data.runs || data;
        }
      } catch (fetchError) {
        console.error('Error fetching experiments from URL:', fetchError);
      }
    }

    // Return empty array if all else fails
    return [];
  }
}

/**
 * Fetch a single experiment run by ID
 */
export async function fetchExperimentById(runId: string): Promise<ExperimentRun | null> {
  const runs = await fetchExperiments();
  return runs.find((run) => run.run_id === runId) || null;
}

/**
 * Fetch GitHub releases
 */
export async function fetchGitHubReleases(): Promise<GitHubRelease[]> {
  try {
    const headers: HeadersInit = {
      Accept: 'application/vnd.github.v3+json',
    };

    // Add token if available
    if (process.env.GITHUB_TOKEN) {
      headers.Authorization = `token ${process.env.GITHUB_TOKEN}`;
    }

    const response = await fetch(
      `${GITHUB_API_BASE}/repos/${GITHUB_REPO}/releases`,
      {
        headers,
        next: { revalidate: 300 }, // Cache for 5 minutes
      }
    );

    if (!response.ok) {
      // If rate limited or error, return fallback data
      if (response.status === 403 || response.status === 404) {
        return getFallbackReleases();
      }
      throw new Error(`GitHub API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching GitHub releases:', error);
    return getFallbackReleases();
  }
}

/**
 * Fetch latest release only
 */
export async function fetchLatestRelease(): Promise<GitHubRelease | null> {
  try {
    const headers: HeadersInit = {
      Accept: 'application/vnd.github.v3+json',
    };

    if (process.env.GITHUB_TOKEN) {
      headers.Authorization = `token ${process.env.GITHUB_TOKEN}`;
    }

    const response = await fetch(
      `${GITHUB_API_BASE}/repos/${GITHUB_REPO}/releases/latest`,
      {
        headers,
        next: { revalidate: 300 },
      }
    );

    if (!response.ok) {
      const fallback = getFallbackReleases();
      return fallback[0] || null;
    }

    return await response.json();
  } catch (error) {
    console.error('Error fetching latest release:', error);
    const fallback = getFallbackReleases();
    return fallback[0] || null;
  }
}

/**
 * Get repository stats (stars, forks, watchers)
 */
export async function fetchRepoStats(): Promise<{
  stars: number;
  forks: number;
  watchers: number;
  open_issues: number;
}> {
  try {
    const headers: HeadersInit = {
      Accept: 'application/vnd.github.v3+json',
    };

    if (process.env.GITHUB_TOKEN) {
      headers.Authorization = `token ${process.env.GITHUB_TOKEN}`;
    }

    const response = await fetch(
      `${GITHUB_API_BASE}/repos/${GITHUB_REPO}`,
      {
        headers,
        next: { revalidate: 600 }, // Cache for 10 minutes
      }
    );

    if (!response.ok) {
      return { stars: 0, forks: 0, watchers: 0, open_issues: 0 };
    }

    const data = await response.json();
    return {
      stars: data.stargazers_count || 0,
      forks: data.forks_count || 0,
      watchers: data.watchers_count || 0,
      open_issues: data.open_issues_count || 0,
    };
  } catch (error) {
    console.error('Error fetching repo stats:', error);
    return { stars: 0, forks: 0, watchers: 0, open_issues: 0 };
  }
}

/**
 * Fallback releases data when GitHub API is unavailable
 */
function getFallbackReleases(): GitHubRelease[] {
  return [
    {
      id: 1,
      tag_name: 'v0.1.0',
      name: 'mlcli v0.1.0 - Initial Release',
      body: `## 🚀 Initial Release

### Features
- **CLI Training**: Train ML models from the command line
- **Multiple Algorithms**: Support for Random Forest, XGBoost, SVM, Logistic Regression
- **Deep Learning**: TensorFlow DNN, CNN, RNN trainers
- **Experiment Tracking**: Track and compare experiment runs
- **Hyperparameter Tuning**: Grid, Random, and Bayesian optimization
- **Model Explainability**: SHAP and LIME explanations
- **Data Preprocessing**: StandardScaler, Normalization, Encoding, Feature Selection

### Installation
\`\`\`bash
pip install mlcli-toolkit
\`\`\`

### Quick Start
\`\`\`bash
mlcli train -d data.csv -m random_forest --target label
\`\`\`
`,
      draft: false,
      prerelease: false,
      created_at: '2025-12-01T10:00:00Z',
      published_at: '2025-12-01T10:00:00Z',
      html_url: `https://github.com/${GITHUB_REPO}/releases/tag/v0.1.0`,
      assets: [
        {
          id: 1,
          name: 'mlcli-0.1.0-py3-none-any.whl',
          size: 156000,
          download_count: 42,
          browser_download_url: `https://github.com/${GITHUB_REPO}/releases/download/v0.1.0/mlcli-0.1.0-py3-none-any.whl`,
          content_type: 'application/zip',
        },
        {
          id: 2,
          name: 'mlcli-0.1.0.tar.gz',
          size: 145000,
          download_count: 28,
          browser_download_url: `https://github.com/${GITHUB_REPO}/releases/download/v0.1.0/mlcli-0.1.0.tar.gz`,
          content_type: 'application/gzip',
        },
      ],
      author: {
        login: 'codeMaestro78',
        avatar_url: 'https://avatars.githubusercontent.com/u/0?v=4',
        html_url: 'https://github.com/codeMaestro78',
      },
    },
  ];
}
