import { CodeBlock } from '@/components/code';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { fetchLatestRelease } from '@/lib/api';
import { Download, ExternalLink, FileDown, Package } from 'lucide-react';
import Link from 'next/link';

export const metadata = {
  title: 'Download',
  description: 'Download mlcli and get started with machine learning from the command line.',
};

export default async function DownloadPage() {
  const latestRelease = await fetchLatestRelease();

  return (
    <div className="container py-12">
      {/* Header */}
      <div className="mb-12 text-center">
        <h1 className="mb-4 text-4xl font-bold tracking-tight">Download mlcli</h1>
        <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
          Get started with mlcli by installing it via pip, pipx, or downloading directly.
        </p>
        {latestRelease && (
          <Badge variant="secondary" className="mt-4">
            Latest: {latestRelease.tag_name}
          </Badge>
        )}
      </div>

      {/* Installation Methods */}
      <div className="mx-auto mb-12 max-w-3xl space-y-8">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Package className="h-5 w-5 text-primary" />
              <CardTitle>Install via pip (Recommended)</CardTitle>
            </div>
            <CardDescription>
              The simplest way to install mlcli is using pip.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CodeBlock code="pip install mlcli-toolkit" language="bash" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Package className="h-5 w-5 text-primary" />
              <CardTitle>Install via pipx</CardTitle>
            </div>
            <CardDescription>
              Use pipx for isolated installation without affecting your system Python.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CodeBlock code="pipx install mlcli-toolkit" language="bash" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Package className="h-5 w-5 text-primary" />
              <CardTitle>Install from source</CardTitle>
            </div>
            <CardDescription>
              Clone the repository and install in development mode.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CodeBlock
              code={`git clone https://github.com/codeMaestro78/MLcli.git
cd MLcli
pip install -e .`}
              language="bash"
            />
          </CardContent>
        </Card>
      </div>

      {/* Direct Downloads */}
      {latestRelease && latestRelease.assets.length > 0 && (
        <div className="mx-auto max-w-3xl">
          <h2 className="mb-6 text-2xl font-bold">Direct Downloads</h2>
          <Card>
            <CardHeader>
              <CardTitle>{latestRelease.name}</CardTitle>
              <CardDescription>
                Released on {new Date(latestRelease.published_at).toLocaleDateString()}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {latestRelease.assets.map((asset) => (
                <Link
                  key={asset.id}
                  href={asset.browser_download_url}
                  className="flex items-center justify-between rounded-lg border p-4 transition-colors hover:bg-muted"
                >
                  <div className="flex items-center gap-3">
                    <FileDown className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="font-medium">{asset.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {(asset.size / 1024).toFixed(0)} KB • {asset.download_count} downloads
                      </p>
                    </div>
                  </div>
                  <Download className="h-4 w-4 text-muted-foreground" />
                </Link>
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Verify Installation */}
      <div className="mx-auto mt-12 max-w-3xl">
        <h2 className="mb-6 text-2xl font-bold">Verify Installation</h2>
        <Card>
          <CardContent className="pt-6">
            <CodeBlock
              code={`# Check version
mlcli --version

# View available commands
mlcli --help

# Run a quick test
mlcli list-trainers`}
              language="bash"
            />
          </CardContent>
        </Card>
      </div>

      {/* Next Steps */}
      <div className="mx-auto mt-12 max-w-3xl text-center">
        <h2 className="mb-4 text-2xl font-bold">Next Steps</h2>
        <p className="mb-6 text-muted-foreground">
          Now that you have mlcli installed, check out the documentation to get started.
        </p>
        <div className="flex justify-center gap-4">
          <Button asChild>
            <Link href="/docs/quickstart">Quickstart Guide</Link>
          </Button>
          <Button asChild variant="outline">
            <Link
              href="https://github.com/codeMaestro78/MLcli"
              target="_blank"
              rel="noopener noreferrer"
            >
              <ExternalLink className="mr-2 h-4 w-4" />
              View on GitHub
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
