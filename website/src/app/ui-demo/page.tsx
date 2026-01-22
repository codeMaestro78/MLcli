'use client';

import { CodeBlock } from '@/components/code';
import { MetricsChart } from '@/components/runs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { formatPercent, generateRunId } from '@/lib/utils';
import type { EpochMetric } from '@/types';
import { CheckCircle, Loader2, Play, RotateCcw, Settings } from 'lucide-react';
import * as React from 'react';

interface DemoConfig {
  modelType: string;
  epochs: number;
  learningRate: number;
  hiddenLayers: number;
}

interface DemoResult {
  runId: string;
  accuracy: number;
  loss: number;
  epochHistory: EpochMetric[];
  duration: number;
}

export default function UIDemoPage() {
  const [config, setConfig] = React.useState<DemoConfig>({
    modelType: 'random_forest',
    epochs: 20,
    learningRate: 0.01,
    hiddenLayers: 3,
  });
  const [isRunning, setIsRunning] = React.useState(false);
  const [progress, setProgress] = React.useState(0);
  const [result, setResult] = React.useState<DemoResult | null>(null);

  const runDemo = async () => {
    setIsRunning(true);
    setProgress(0);
    setResult(null);

    const startTime = Date.now();
    const epochHistory: EpochMetric[] = [];

    // Simulate training epochs
    for (let epoch = 1; epoch <= config.epochs; epoch++) {
      await new Promise((resolve) => setTimeout(resolve, 100));

      // Generate realistic-looking metrics
      const baseLoss = 0.7 - (epoch / config.epochs) * 0.4 + Math.random() * 0.05;
      const baseAccuracy = 0.5 + (epoch / config.epochs) * 0.35 + Math.random() * 0.03;

      epochHistory.push({
        epoch,
        loss: Math.max(0.1, baseLoss),
        accuracy: Math.min(0.95, baseAccuracy),
        val_loss: Math.max(0.15, baseLoss + 0.02 + Math.random() * 0.03),
        val_accuracy: Math.min(0.93, baseAccuracy - 0.01 + Math.random() * 0.02),
      });

      setProgress(Math.round((epoch / config.epochs) * 100));
    }

    const duration = Date.now() - startTime;
    const finalEpoch = epochHistory[epochHistory.length - 1];

    setResult({
      runId: generateRunId(),
      accuracy: finalEpoch?.val_accuracy || 0.85,
      loss: finalEpoch?.val_loss || 0.3,
      epochHistory,
      duration,
    });

    setIsRunning(false);
    setProgress(100);
  };

  const resetDemo = () => {
    setResult(null);
    setProgress(0);
  };

  const cliCommand = `mlcli train \\
  --data data/sample.csv \\
  --model ${config.modelType} \\
  --target label \\
  ${config.modelType.startsWith('tf_') ? `--epochs ${config.epochs} \\
  --learning-rate ${config.learningRate} \\
  --hidden-layers ${config.hiddenLayers}` : '--n-estimators 100'}`;

  return (
    <div className="container py-12">
      {/* Header */}
      <div className="mb-8 text-center">
        <h1 className="mb-4 text-4xl font-bold tracking-tight">Interactive Demo</h1>
        <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
          Experience mlcli without installing anything. Configure parameters below
          and run a simulated training session.
        </p>
      </div>

      <div className="mx-auto max-w-4xl">
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Configuration Panel */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Configuration
              </CardTitle>
              <CardDescription>
                Adjust hyperparameters for your training run
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Model Type</label>
                <Select
                  value={config.modelType}
                  onValueChange={(value) => setConfig({ ...config, modelType: value })}
                  disabled={isRunning}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="random_forest">Random Forest</SelectItem>
                    <SelectItem value="xgboost">XGBoost</SelectItem>
                    <SelectItem value="tf_dnn">Deep Neural Network</SelectItem>
                    <SelectItem value="logistic">Logistic Regression</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {config.modelType.startsWith('tf_') && (
                <>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Epochs</label>
                    <Input
                      type="number"
                      value={config.epochs}
                      onChange={(e) =>
                        setConfig({ ...config, epochs: parseInt(e.target.value) || 10 })
                      }
                      min={5}
                      max={100}
                      disabled={isRunning}
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Learning Rate</label>
                    <Input
                      type="number"
                      value={config.learningRate}
                      onChange={(e) =>
                        setConfig({ ...config, learningRate: parseFloat(e.target.value) || 0.01 })
                      }
                      step={0.001}
                      min={0.0001}
                      max={1}
                      disabled={isRunning}
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Hidden Layers</label>
                    <Input
                      type="number"
                      value={config.hiddenLayers}
                      onChange={(e) =>
                        setConfig({ ...config, hiddenLayers: parseInt(e.target.value) || 2 })
                      }
                      min={1}
                      max={10}
                      disabled={isRunning}
                    />
                  </div>
                </>
              )}

              <div className="flex gap-2 pt-4">
                <Button onClick={runDemo} disabled={isRunning} className="flex-1">
                  {isRunning ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Training... {progress}%
                    </>
                  ) : (
                    <>
                      <Play className="mr-2 h-4 w-4" />
                      Run Demo
                    </>
                  )}
                </Button>
                <Button variant="outline" onClick={resetDemo} disabled={isRunning}>
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </div>

              {/* Progress bar */}
              {isRunning && (
                <div className="space-y-1">
                  <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="text-center text-sm text-muted-foreground">
                    Epoch {Math.ceil((progress / 100) * config.epochs)} of {config.epochs}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* CLI Command */}
          <Card>
            <CardHeader>
              <CardTitle>Equivalent CLI Command</CardTitle>
              <CardDescription>
                This is the command you would run in the terminal
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CodeBlock code={cliCommand} language="bash" />
            </CardContent>
          </Card>
        </div>

        {/* Results */}
        {result && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                Training Complete
              </CardTitle>
              <CardDescription>
                Run ID: <code className="rounded bg-muted px-1">{result.runId}</code>
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Metrics */}
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="rounded-lg border p-4 text-center">
                  <p className="text-sm text-muted-foreground">Final Accuracy</p>
                  <p className="text-2xl font-bold text-green-500">
                    {formatPercent(result.accuracy)}
                  </p>
                </div>
                <div className="rounded-lg border p-4 text-center">
                  <p className="text-sm text-muted-foreground">Final Loss</p>
                  <p className="text-2xl font-bold">{result.loss.toFixed(4)}</p>
                </div>
                <div className="rounded-lg border p-4 text-center">
                  <p className="text-sm text-muted-foreground">Duration</p>
                  <p className="text-2xl font-bold">{(result.duration / 1000).toFixed(1)}s</p>
                </div>
              </div>

              {/* Training Chart */}
              {result.epochHistory.length > 0 && (
                <div>
                  <h4 className="mb-4 font-medium">Training History</h4>
                  <MetricsChart epochHistory={result.epochHistory} />
                </div>
              )}

              {/* Tags */}
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Tags:</span>
                <Badge variant="secondary">demo</Badge>
                <Badge variant="secondary">{config.modelType}</Badge>
                <Badge variant="outline">simulated</Badge>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
