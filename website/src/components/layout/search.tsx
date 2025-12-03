'use client';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Command, FileText, Search } from 'lucide-react';
import { useRouter } from 'next/navigation';
import * as React from 'react';

interface SearchResult {
  title: string;
  href: string;
  section: string;
  preview: string;
}

const searchableContent: SearchResult[] = [
  {
    title: 'Getting Started',
    href: '/docs/quickstart',
    section: 'Docs',
    preview: 'Install mlcli and run your first training command.',
  },
  {
    title: 'Configuration',
    href: '/docs/config',
    section: 'Docs',
    preview: 'Configure mlcli with YAML files and command-line options.',
  },
  {
    title: 'Trainers',
    href: '/docs/trainers',
    section: 'Docs',
    preview: 'Available trainers: Random Forest, XGBoost, LightGBM, and more.',
  },
  {
    title: 'Experiment Runs',
    href: '/runs',
    section: 'Dashboard',
    preview: 'View and compare your experiment runs.',
  },
  {
    title: 'Releases',
    href: '/releases',
    section: 'Downloads',
    preview: 'View all mlcli releases and changelogs.',
  },
  {
    title: 'Download',
    href: '/download',
    section: 'Downloads',
    preview: 'Install mlcli via pip or from source.',
  },
  {
    title: 'About',
    href: '/about',
    section: 'Info',
    preview: 'Learn about the mlcli project and team.',
  },
  {
    title: 'Contribute',
    href: '/contribute',
    section: 'Info',
    preview: 'Learn how to contribute to mlcli.',
  },
];

export function SearchDialog() {
  const router = useRouter();
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState('');
  const inputRef = React.useRef<HTMLInputElement>(null);

  const results = React.useMemo(() => {
    if (!query.trim()) return [];
    const lowerQuery = query.toLowerCase();
    return searchableContent.filter(
      (item) =>
        item.title.toLowerCase().includes(lowerQuery) ||
        item.preview.toLowerCase().includes(lowerQuery) ||
        item.section.toLowerCase().includes(lowerQuery)
    );
  }, [query]);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  React.useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
    }
  }, [open]);

  const handleSelect = (href: string) => {
    setOpen(false);
    setQuery('');
    router.push(href);
  };

  if (!open) {
    return (
      <Button
        variant="outline"
        className="relative h-9 w-full justify-start text-sm text-muted-foreground sm:w-64"
        onClick={() => setOpen(true)}
      >
        <Search className="mr-2 h-4 w-4" />
        <span className="hidden sm:inline-flex">Search documentation...</span>
        <span className="inline-flex sm:hidden">Search...</span>
        <kbd className="pointer-events-none absolute right-2 hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-xs font-medium sm:flex">
          <Command className="h-3 w-3" />K
        </kbd>
      </Button>
    );
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm"
        onClick={() => setOpen(false)}
      />

      {/* Dialog */}
      <div className="fixed left-1/2 top-[20%] z-50 w-full max-w-lg -translate-x-1/2 rounded-lg border bg-background shadow-lg">
        <div className="flex items-center border-b px-3">
          <Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
          <Input
            ref={inputRef}
            placeholder="Search documentation..."
            className="h-12 border-0 focus-visible:ring-0"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>

        <div className="max-h-[300px] overflow-y-auto p-2">
          {query && results.length === 0 && (
            <div className="py-6 text-center text-sm text-muted-foreground">
              No results found for &quot;{query}&quot;
            </div>
          )}

          {results.length > 0 && (
            <ul className="space-y-1">
              {results.map((result) => (
                <li key={result.href}>
                  <button
                    className="flex w-full items-start gap-3 rounded-md px-3 py-2 text-left hover:bg-muted"
                    onClick={() => handleSelect(result.href)}
                  >
                    <FileText className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                    <div className="flex-1 overflow-hidden">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{result.title}</span>
                        <span className="rounded bg-muted px-1.5 py-0.5 text-xs text-muted-foreground">
                          {result.section}
                        </span>
                      </div>
                      <p className="truncate text-sm text-muted-foreground">
                        {result.preview}
                      </p>
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          )}

          {!query && (
            <div className="space-y-2 p-2">
              <p className="text-xs font-medium text-muted-foreground">
                QUICK LINKS
              </p>
              {searchableContent.slice(0, 5).map((item) => (
                <button
                  key={item.href}
                  className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-muted"
                  onClick={() => handleSelect(item.href)}
                >
                  <FileText className="h-4 w-4 text-muted-foreground" />
                  {item.title}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="flex items-center justify-end border-t px-3 py-2 text-xs text-muted-foreground">
          Press{' '}
          <kbd className="mx-1 rounded border bg-muted px-1.5 py-0.5 font-mono">
            Esc
          </kbd>{' '}
          to close
        </div>
      </div>
    </>
  );
}
