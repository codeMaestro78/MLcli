'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { formatDate, formatPercent } from '@/lib/utils';
import type { ExperimentRun } from '@/types';
import { ArrowLeft, Calendar, Cpu, Database, Download, ExternalLink, Settings } from 'lucide-react';
import Link from 'next/link';
import { MetricsChart } from './metrics-chart';

interface RunDetailProps {
  run: ExperimentRun;
}

export function RunDetail({ run }: RunDetailProps) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm">
              <Link href="/runs">
                <ArrowLeft className="mr-1 h-4 w-4" />
                Back to runs
              </Link>
            </Button>
          </div>
          <h1 className="text-3xl font-bold tracking-tight">{run.run_id}</h1>
          <div className="flex items-center gap-4 text-muted-foreground">
            <span className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              {formatDate(run.timestamp)}
            </span>
            <Badge variant="secondary">{run.framework}</Badge>
            <Badge>{run.model_type}</Badge>
          </div>
        </div>
        <div className="flex gap-2">
          {run.mlflow_tracking_uri && (
            <Button variant="outline" size="sm" asChild>
              <Link
                href={`${run.mlflow_tracking_uri}/#/experiments/0/runs/${run.mlflow_run_id}`}
                target="_blank"
                rel="noopener noreferrer"
              >
                <ExternalLink className="mr-1 h-4 w-4" />
                View in MLflow
              </Link>
            </Button>
          )}
          {run.artifacts?.model_path && (
            <Button size="sm">
              <Download className="mr-1 h-4 w-4" />
              Download Model
            </Button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="hyperparameters">Hyperparameters</TabsTrigger>
          <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {/* Accuracy Card */}
            {run.metrics.accuracy !== undefined && (
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Accuracy</CardTitle>
                  <Cpu className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-500">
                    {formatPercent(run.metrics.accuracy)}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* F1 Score Card */}
            {run.metrics.f1 !== undefined && (
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">F1 Score</CardTitle>
                  <Settings className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{formatPercent(run.metrics.f1)}</div>
                </CardContent>
              </Card>
            )}

            {/* Precision Card */}
            {run.metrics.precision !== undefined && (
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Precision</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{formatPercent(run.metrics.precision)}</div>
                </CardContent>
              </Card>
            )}

            {/* Recall Card */}
            {run.metrics.recall !== undefined && (
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Recall</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{formatPercent(run.metrics.recall)}</div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Dataset Info */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Dataset
              </CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid gap-4 sm:grid-cols-2">
                <div>
                  <dt className="text-sm font-medium text-muted-foreground">Dataset Path</dt>
                  <dd className="mt-1 font-mono text-sm">{run.dataset}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-muted-foreground">Target Column</dt>
                  <dd className="mt-1 font-mono text-sm">{run.target_column}</dd>
                </div>
              </dl>
            </CardContent>
          </Card>

          {/* Notes */}
          {run.notes && (
            <Card>
              <CardHeader>
                <CardTitle>Notes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{run.notes}</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Metrics Tab */}
        <TabsContent value="metrics" className="space-y-4">
          {run.epoch_history && run.epoch_history.length > 0 ? (
            <Card>
              <CardHeader>
                <CardTitle>Training History</CardTitle>
                <CardDescription>Loss and accuracy over epochs</CardDescription>
              </CardHeader>
              <CardContent>
                <MetricsChart epochHistory={run.epoch_history} />
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>Final Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {Object.entries(run.metrics).map(([key, value]) => (
                    <div key={key}>
                      <dt className="text-sm font-medium capitalize text-muted-foreground">
                        {key.replace(/_/g, ' ')}
                      </dt>
                      <dd className="mt-1 text-2xl font-bold">
                        {typeof value === 'number' && value <= 1
                          ? formatPercent(value)
                          : value?.toFixed(4) ?? '—'}
                      </dd>
                    </div>
                  ))}
                </dl>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Hyperparameters Tab */}
        <TabsContent value="hyperparameters">
          <Card>
            <CardHeader>
              <CardTitle>Hyperparameters</CardTitle>
              <CardDescription>Model configuration used for this run</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="overflow-auto rounded-lg bg-muted p-4 text-sm">
                {JSON.stringify(run.hyperparameters, null, 2)}
              </pre>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Artifacts Tab */}
        <TabsContent value="artifacts">
          <Card>
            <CardHeader>
              <CardTitle>Artifacts</CardTitle>
              <CardDescription>Model files and outputs from this run</CardDescription>
            </CardHeader>
            <CardContent>
              {run.artifacts ? (
                <dl className="space-y-4">
                  {run.artifacts.model_path && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground">Model Path</dt>
                      <dd className="mt-1 font-mono text-sm">{run.artifacts.model_path}</dd>
                    </div>
                  )}
                  {run.artifacts.config_path && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground">Config Path</dt>
                      <dd className="mt-1 font-mono text-sm">{run.artifacts.config_path}</dd>
                    </div>
                  )}
                  {run.artifacts.logs_path && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground">Logs Path</dt>
                      <dd className="mt-1 font-mono text-sm">{run.artifacts.logs_path}</dd>
                    </div>
                  )}
                  {run.artifacts.plots && run.artifacts.plots.length > 0 && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground">Plots</dt>
                      <dd className="mt-2 flex flex-wrap gap-2">
                        {run.artifacts.plots.map((plot, index) => (
                          <Badge key={index} variant="outline">
                            {plot}
                          </Badge>
                        ))}
                      </dd>
                    </div>
                  )}
                </dl>
              ) : (
                <p className="text-sm text-muted-foreground">No artifacts available for this run.</p>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
