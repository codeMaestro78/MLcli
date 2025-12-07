'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { formatDate, formatFileSize } from '@/lib/utils';
import type { GitHubAsset, GitHubRelease } from '@/types';
import { Calendar, Download, ExternalLink, FileDown, Tag, User } from 'lucide-react';
import Link from 'next/link';
import * as React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface ReleaseListProps {
  releases: GitHubRelease[];
}

export function ReleaseList({ releases }: ReleaseListProps) {
  if (releases.length === 0) {
    return (
      <div className="flex h-64 flex-col items-center justify-center gap-4 rounded-lg border border-dashed">
        <Tag className="h-12 w-12 text-muted-foreground/50" />
        <div className="text-center">
          <h3 className="font-semibold">No releases yet</h3>
          <p className="text-sm text-muted-foreground">
            Check back soon for new releases.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {releases.map((release, index) => (
        <ReleaseItem key={release.id} release={release} isLatest={index === 0} />
      ))}
    </div>
  );
}

interface ReleaseItemProps {
  release: GitHubRelease;
  isLatest?: boolean;
}

export function ReleaseItem({ release, isLatest }: ReleaseItemProps) {
  const [expanded, setExpanded] = React.useState(isLatest);

  return (
    <Card className={isLatest ? 'border-primary/50' : undefined}>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-xl">
                <Link
                  href={release.html_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:underline"
                >
                  {release.name}
                </Link>
              </CardTitle>
              {isLatest && <Badge variant="success">Latest</Badge>}
              {release.prerelease && <Badge variant="warning">Pre-release</Badge>}
            </div>
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <Tag className="h-4 w-4" />
                {release.tag_name}
              </span>
              <span className="flex items-center gap-1">
                <Calendar className="h-4 w-4" />
                {formatDate(release.published_at)}
              </span>
              <span className="flex items-center gap-1">
                <User className="h-4 w-4" />
                {release.author.login}
              </span>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Collapse' : 'Expand'}
          </Button>
        </div>
      </CardHeader>

      {expanded && (
        <CardContent className="space-y-4">
          {/* Release notes */}
          <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:font-bold prose-h2:text-xl prose-h2:mt-6 prose-h2:mb-3 prose-h3:text-lg prose-h3:mt-4 prose-h3:mb-2 prose-ul:my-2 prose-li:my-1 prose-p:my-2 prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-pre:bg-transparent prose-pre:p-0">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={oneDark}
                      language={match[1]}
                      PreTag="div"
                      className="rounded-lg !mt-2 !mb-4"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
              }}
            >
              {release.body}
            </ReactMarkdown>
          </div>

          {/* Assets */}
          {release.assets.length > 0 && (
            <div className="space-y-2">
              <h4 className="font-semibold">Downloads</h4>
              <div className="grid gap-2 sm:grid-cols-2">
                {release.assets.map((asset) => (
                  <AssetDownload key={asset.id} asset={asset} />
                ))}
              </div>
            </div>
          )}

          {/* GitHub link */}
          <div className="flex justify-end">
            <Button variant="ghost" size="sm" asChild>
              <Link
                href={release.html_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                View on GitHub
                <ExternalLink className="ml-1 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </CardContent>
      )}
    </Card>
  );
}

interface AssetDownloadProps {
  asset: GitHubAsset;
}

function AssetDownload({ asset }: AssetDownloadProps) {
  return (
    <Link
      href={asset.browser_download_url}
      className="flex items-center justify-between rounded-lg border p-3 transition-colors hover:bg-muted"
    >
      <div className="flex items-center gap-2">
        <FileDown className="h-5 w-5 text-muted-foreground" />
        <div>
          <p className="text-sm font-medium">{asset.name}</p>
          <p className="text-xs text-muted-foreground">
            {formatFileSize(asset.size)} • {asset.download_count} downloads
          </p>
        </div>
      </div>
      <Download className="h-4 w-4 text-muted-foreground" />
    </Link>
  );
}

interface DownloadCardProps {
  title: string;
  description: string;
  downloads: {
    name: string;
    url: string;
    size?: string;
  }[];
}

export function DownloadCard({ title, description, downloads }: DownloadCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-2">
        {downloads.map((download, index) => (
          <Link
            key={index}
            href={download.url}
            className="flex items-center justify-between rounded-lg border p-3 transition-colors hover:bg-muted"
          >
            <div className="flex items-center gap-2">
              <FileDown className="h-5 w-5 text-muted-foreground" />
              <span className="text-sm font-medium">{download.name}</span>
            </div>
            {download.size && (
              <span className="text-xs text-muted-foreground">{download.size}</span>
            )}
          </Link>
        ))}
      </CardContent>
    </Card>
  );
}
