import { RunDetail } from '@/components/runs';
import { fetchExperimentById } from '@/lib/api';
import { notFound } from 'next/navigation';

interface RunDetailPageProps {
  params: {
    runId: string;
  };
}

export default async function RunDetailPage({ params }: RunDetailPageProps) {
  const run = await fetchExperimentById(params.runId);

  if (!run) {
    notFound();
  }

  return (
    <div className="container py-12">
      <RunDetail run={run} />
    </div>
  );
}

export async function generateMetadata({ params }: RunDetailPageProps) {
  const run = await fetchExperimentById(params.runId);

  if (!run) {
    return {
      title: 'Run Not Found',
    };
  }

  return {
    title: `${run.model_type} Run - ${run.run_id.slice(0, 8)}`,
    description: `Experiment run details for ${run.model_type} model trained on ${run.dataset}`,
  };
}
