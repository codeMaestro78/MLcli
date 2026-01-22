import { fetchGitHubReleases } from '@/lib/api';
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const releases = await fetchGitHubReleases();

    return NextResponse.json({
      releases,
      success: true,
      meta: {
        total: releases.length,
      },
    });
  } catch (error) {
    console.error('Error fetching releases:', error);
    return NextResponse.json(
      {
        releases: [],
        success: false,
        error: 'Failed to fetch releases',
      },
      { status: 500 }
    );
  }
}
