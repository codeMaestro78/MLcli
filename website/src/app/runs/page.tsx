'use client';

import { AccuracyHistogram, ModelComparisonChart, RunFilters, RunTable } from '@/components/runs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import type { ExperimentRun, RunsFilter } from '@/types';
import { Activity, BarChart3, FlaskConical, TrendingUp } from 'lucide-react';
import * as React from 'react';
import useSWR from 'swr';

const fetcher = (url: string) => fetch(url).then((res) => res.json());

export default function RunsPage() {
  const [filters, setFilters] = React.useState<RunsFilter>({});

  const { data, error, isLoading } = useSWR<{ runs: ExperimentRun[] }>(
    '/api/runs',
    fetcher
  );

  const runs = data?.runs || [];

  // Get unique model types and frameworks for filters
  const modelTypes = [...new Set(runs.map((r) => r.model_type))];
  const frameworks = [...new Set(runs.map((r) => r.framework))];

  // Apply filters
  const filteredRuns = runs.filter((run) => {
    if (filters.model_type && run.model_type !== filters.model_type) return false;
    if (filters.framework && run.framework !== filters.framework) return false;
    if (filters.min_accuracy && (run.metrics.accuracy || 0) < filters.min_accuracy) return false;
    return true;
  });

  // Calculate stats
  const stats = React.useMemo(() => {
    const runsWithAccuracy = runs.filter((r) => r.metrics.accuracy !== undefined);
    const avgAccuracy =
      runsWithAccuracy.length > 0
        ? runsWithAccuracy.reduce((sum, r) => sum + (r.metrics.accuracy || 0), 0) /
          runsWithAccuracy.length
        : 0;
    const bestAccuracy = Math.max(...runsWithAccuracy.map((r) => r.metrics.accuracy || 0), 0);

    return {
      totalRuns: runs.length,
      avgAccuracy,
      bestAccuracy,
      modelTypes: modelTypes.length,
    };
  }, [runs, modelTypes.length]);

  if (error) {
    return (
      <div className="container py-12">
        <div className="text-center text-destructive">
          Failed to load experiments. Please try again later.
        </div>
      </div>
    );
  }

  return (
    <div className="container py-12">
      {/* Header */}
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold tracking-tight">Experiment Runs</h1>
        <p className="text-muted-foreground">
          View and analyze your machine learning experiment runs.
        </p>
      </div>

      {/* Stats Cards */}
      <div className="mb-8 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Runs</CardTitle>
            <FlaskConical className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalRuns}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Accuracy</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(stats.avgAccuracy * 100).toFixed(1)}%
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Best Accuracy</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">
              {(stats.bestAccuracy * 100).toFixed(1)}%
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Types</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.modelTypes}</div>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      {runs.length > 0 && (
        <div className="mb-8 grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Accuracy Distribution</CardTitle>
              <CardDescription>Distribution of accuracy scores across all runs</CardDescription>
            </CardHeader>
            <CardContent>
              <AccuracyHistogram runs={runs} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Model Comparison</CardTitle>
              <CardDescription>Average accuracy by model type</CardDescription>
            </CardHeader>
            <CardContent>
              <ModelComparisonChart runs={runs} />
            </CardContent>
          </Card>
        </div>
      )}

      {/* Filters */}
      <div className="mb-6">
        <RunFilters
          filters={filters}
          onFiltersChange={setFilters}
          modelTypes={modelTypes}
          frameworks={frameworks}
        />
      </div>

      {/* Runs Table */}
      <RunTable runs={filteredRuns} isLoading={isLoading} />
    </div>
  );
}
