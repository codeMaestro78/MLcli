import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    name: 'mlcli-website',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    endpoints: {
      runs: '/api/runs',
      runById: '/api/runs/[runId]',
      releases: '/api/releases',
      search: '/api/search?q=query',
      health: '/api/health',
    },
  });
}
