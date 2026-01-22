import { fetchExperimentById } from '@/lib/api';
import { NextResponse } from 'next/server';

export async function GET(
  _request: Request,
  { params }: { params: { runId: string } }
) {
  try {
    const run = await fetchExperimentById(params.runId);

    if (!run) {
      return NextResponse.json(
        {
          run: null,
          success: false,
          error: 'Run not found',
        },
        { status: 404 }
      );
    }

    return NextResponse.json({
      run,
      success: true,
    });
  } catch (error) {
    console.error('Error fetching run:', error);
    return NextResponse.json(
      {
        run: null,
        success: false,
        error: 'Failed to fetch run',
      },
      { status: 500 }
    );
  }
}
