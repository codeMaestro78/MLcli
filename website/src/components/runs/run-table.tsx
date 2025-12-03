'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table';
import { formatPercent, formatRelativeTime, stringToColor } from '@/lib/utils';
import type { ExperimentRun } from '@/types';
import { Clock, Database, ExternalLink, TrendingUp } from 'lucide-react';
import Link from 'next/link';

interface RunTableProps {
  runs: ExperimentRun[];
  isLoading?: boolean;
}

export function RunTable({ runs, isLoading }: RunTableProps) {
  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="flex items-center gap-2 text-muted-foreground">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
          Loading experiments...
        </div>
      </div>
    );
  }

  if (runs.length === 0) {
    return (
      <div className="flex h-64 flex-col items-center justify-center gap-4 rounded-lg border border-dashed">
        <Database className="h-12 w-12 text-muted-foreground/50" />
        <div className="text-center">
          <h3 className="font-semibold">No experiments found</h3>
          <p className="text-sm text-muted-foreground">
            Run your first experiment to see it here.
          </p>
        </div>
        <Button asChild variant="outline" size="sm">
          <Link href="/docs/quickstart">Get Started</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[200px]">Run ID</TableHead>
            <TableHead>Model</TableHead>
            <TableHead>Framework</TableHead>
            <TableHead>Accuracy</TableHead>
            <TableHead>F1 Score</TableHead>
            <TableHead>Time</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <RunRow key={run.run_id} run={run} />
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

interface RunRowProps {
  run: ExperimentRun;
}

function RunRow({ run }: RunRowProps) {
  const modelColor = stringToColor(run.model_type);

  return (
    <TableRow className="group">
      <TableCell className="font-mono text-sm">
        <Link
          href={`/runs/${run.run_id}`}
          className="text-primary hover:underline"
        >
          {run.run_id.slice(0, 16)}...
        </Link>
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-2">
          <span
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: modelColor }}
            aria-hidden="true"
          />
          <span className="font-medium">{run.model_type}</span>
        </div>
      </TableCell>
      <TableCell>
        <Badge variant="secondary">{run.framework}</Badge>
      </TableCell>
      <TableCell>
        {run.metrics.accuracy !== undefined ? (
          <div className="flex items-center gap-1">
            <TrendingUp className="h-4 w-4 text-green-500" />
            <span className="font-medium">{formatPercent(run.metrics.accuracy)}</span>
          </div>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </TableCell>
      <TableCell>
        {run.metrics.f1 !== undefined ? (
          <span>{formatPercent(run.metrics.f1)}</span>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </TableCell>
      <TableCell>
        <div className="flex items-center gap-1 text-muted-foreground">
          <Clock className="h-4 w-4" />
          <span className="text-sm">{formatRelativeTime(run.timestamp)}</span>
        </div>
      </TableCell>
      <TableCell className="text-right">
        <div className="flex items-center justify-end gap-2 opacity-0 transition-opacity group-hover:opacity-100">
          <Button asChild variant="ghost" size="sm">
            <Link href={`/runs/${run.run_id}`}>
              View
              <ExternalLink className="ml-1 h-3 w-3" />
            </Link>
          </Button>
        </div>
      </TableCell>
    </TableRow>
  );
}
