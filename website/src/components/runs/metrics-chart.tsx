'use client';

import type { EpochMetric, ExperimentRun } from '@/types';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts';

interface MetricsChartProps {
  epochHistory: EpochMetric[];
}

export function MetricsChart({ epochHistory }: MetricsChartProps) {
  return (
    <div className="h-80 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={epochHistory}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="epoch"
            tick={{ fill: 'hsl(var(--muted-foreground))' }}
            tickLine={{ stroke: 'hsl(var(--muted-foreground))' }}
          />
          <YAxis
            tick={{ fill: 'hsl(var(--muted-foreground))' }}
            tickLine={{ stroke: 'hsl(var(--muted-foreground))' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--popover))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
            labelStyle={{ color: 'hsl(var(--foreground))' }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="#ef4444"
            strokeWidth={2}
            dot={{ fill: '#ef4444', strokeWidth: 2 }}
            name="Training Loss"
          />
          {epochHistory[0]?.val_loss !== undefined && (
            <Line
              type="monotone"
              dataKey="val_loss"
              stroke="#f97316"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: '#f97316', strokeWidth: 2 }}
              name="Validation Loss"
            />
          )}
          {epochHistory[0]?.accuracy !== undefined && (
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#22c55e"
              strokeWidth={2}
              dot={{ fill: '#22c55e', strokeWidth: 2 }}
              name="Training Accuracy"
            />
          )}
          {epochHistory[0]?.val_accuracy !== undefined && (
            <Line
              type="monotone"
              dataKey="val_accuracy"
              stroke="#10b981"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: '#10b981', strokeWidth: 2 }}
              name="Validation Accuracy"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

interface AccuracyHistogramProps {
  runs: ExperimentRun[];
}

export function AccuracyHistogram({ runs }: AccuracyHistogramProps) {
  // Create histogram buckets
  const buckets = [
    { range: '0-50%', min: 0, max: 0.5, count: 0 },
    { range: '50-60%', min: 0.5, max: 0.6, count: 0 },
    { range: '60-70%', min: 0.6, max: 0.7, count: 0 },
    { range: '70-80%', min: 0.7, max: 0.8, count: 0 },
    { range: '80-90%', min: 0.8, max: 0.9, count: 0 },
    { range: '90-100%', min: 0.9, max: 1.0, count: 0 },
  ];

  runs.forEach((run) => {
    if (run.metrics.accuracy !== undefined) {
      const bucket = buckets.find(
        (b) => run.metrics.accuracy! >= b.min && run.metrics.accuracy! < b.max
      );
      if (bucket) {
        bucket.count++;
      } else if (run.metrics.accuracy === 1.0) {
        // Handle edge case for 100% accuracy
        buckets[buckets.length - 1]!.count++;
      }
    }
  });

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={buckets} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="range"
            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
            tickLine={{ stroke: 'hsl(var(--muted-foreground))' }}
          />
          <YAxis
            tick={{ fill: 'hsl(var(--muted-foreground))' }}
            tickLine={{ stroke: 'hsl(var(--muted-foreground))' }}
            allowDecimals={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--popover))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
            labelStyle={{ color: 'hsl(var(--foreground))' }}
            formatter={(value: number) => [value, 'Runs']}
          />
          <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} name="Runs" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

interface ModelComparisonChartProps {
  runs: ExperimentRun[];
}

export function ModelComparisonChart({ runs }: ModelComparisonChartProps) {
  // Group runs by model type and calculate average accuracy
  const modelStats = runs.reduce(
    (acc, run) => {
      if (run.metrics.accuracy !== undefined) {
        if (!acc[run.model_type]) {
          acc[run.model_type] = { total: 0, count: 0 };
        }
        acc[run.model_type]!.total += run.metrics.accuracy;
        acc[run.model_type]!.count++;
      }
      return acc;
    },
    {} as Record<string, { total: number; count: number }>
  );

  const data = Object.entries(modelStats).map(([model, stats]) => ({
    model,
    accuracy: (stats.total / stats.count) * 100,
    runs: stats.count,
  }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 80, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            type="number"
            domain={[0, 100]}
            tick={{ fill: 'hsl(var(--muted-foreground))' }}
            tickLine={{ stroke: 'hsl(var(--muted-foreground))' }}
            tickFormatter={(value) => `${value}%`}
          />
          <YAxis
            type="category"
            dataKey="model"
            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
            tickLine={{ stroke: 'hsl(var(--muted-foreground))' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'hsl(var(--popover))',
              border: '1px solid hsl(var(--border))',
              borderRadius: '6px',
            }}
            labelStyle={{ color: 'hsl(var(--foreground))' }}
            formatter={(value: number, name: string) => [
              `${value.toFixed(1)}%`,
              name === 'accuracy' ? 'Avg Accuracy' : name,
            ]}
          />
          <Bar dataKey="accuracy" fill="#22c55e" radius={[0, 4, 4, 0]} name="Avg Accuracy" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
