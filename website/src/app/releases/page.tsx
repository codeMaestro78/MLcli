import { ReleaseList } from '@/components/releases';
import { fetchGitHubReleases } from '@/lib/api';

export const metadata = {
  title: 'Releases',
  description: 'Download the latest version of mlcli and view the changelog.',
};

export default async function ReleasesPage() {
  const releases = await fetchGitHubReleases();

  return (
    <div className="container py-12">
      {/* Header */}
      <div className="mb-8">
        <h1 className="mb-2 text-3xl font-bold tracking-tight">Releases</h1>
        <p className="text-muted-foreground">
          Download the latest version of mlcli and view the changelog.
        </p>
      </div>

      {/* Releases List */}
      <ReleaseList releases={releases} />
    </div>
  );
}
