import { fetchExperiments } from '@/lib/api';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const runs = await fetchExperiments();

    return NextResponse.json({
      runs,
      success: true,
      meta: {
        total: runs.length,
      },
    });
  } catch (error) {
    console.error('Error fetching runs:', error);
    return NextResponse.json(
      {
        runs: [],
        success: false,
        error: 'Failed to fetch experiments',
      },
      { status: 500 }
    );
  }
}
