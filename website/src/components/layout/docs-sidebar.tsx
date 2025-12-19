'use client';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { BookOpen, Box, ChevronDown, Code, FlaskConical, Lightbulb, Settings, Sliders, Zap } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import * as React from 'react';

interface SidebarItem {
  title: string;
  href: string;
  icon?: React.ComponentType<{ className?: string }>;
  items?: SidebarItem[];
}

const docsNavigation: SidebarItem[] = [
  {
    title: 'Getting Started',
    href: '/docs',
    icon: Zap,
    items: [
      { title: 'Introduction', href: '/docs' },
      { title: 'Quickstart', href: '/docs/quickstart' },
      { title: 'Examples', href: '/docs/examples' },
    ],
  },
  {
    title: 'Configuration',
    href: '/docs/config',
    icon: Settings,
    items: [
      { title: 'Config Reference', href: '/docs/config' },
    ],
  },
  {
    title: 'Trainers',
    href: '/docs/trainers',
    icon: Code,
    items: [
      { title: 'Overview', href: '/docs/trainers' },
      { title: 'Gradient Boosting', href: '/docs/trainers#gradient-boosting' },
      { title: 'Clustering', href: '/docs/trainers#clustering' },
      { title: 'Anomaly Detection', href: '/docs/trainers#anomaly-detection' },
    ],
  },
  {
    title: 'Preprocessing',
    href: '/docs/preprocessing',
    icon: Box,
    items: [
      { title: 'Overview', href: '/docs/preprocessing' },
    ],
  },
  {
    title: 'Explainability',
    href: '/docs/explainer',
    icon: Lightbulb,
    items: [
      { title: 'Overview', href: '/docs/explainer' },
    ],
  },
  {
    title: 'Hyperparameter Tuning',
    href: '/docs/tuner',
    icon: Sliders,
    items: [
      { title: 'Overview', href: '/docs/tuner' },
    ],
  },
  {
    title: 'Experiment Tracking',
    href: '/docs/experiments',
    icon: FlaskConical,
    items: [
      { title: 'Tracker Overview', href: '/docs/experiments' },
    ],
  },
  {
    title: 'API Reference',
    href: '/docs/api',
    icon: BookOpen,
    items: [
      { title: 'CLI Commands', href: '/docs/api' },
    ],
  },
];

const versions = ['v0.3.0', 'v0.2.0', 'v0.1.1', 'latest'];

interface DocsSidebarProps {
  className?: string;
}

export function DocsSidebar({ className }: DocsSidebarProps) {
  const pathname = usePathname();
  const [openSections, setOpenSections] = React.useState<string[]>(['Getting Started']);
  const [selectedVersion, setSelectedVersion] = React.useState('latest');

  // Auto-expand section based on current path
  React.useEffect(() => {
    const currentSection = docsNavigation.find(
      (section) =>
        section.href === pathname || section.items?.some((item) => item.href === pathname)
    );
    if (currentSection && !openSections.includes(currentSection.title)) {
      setOpenSections((prev) => [...prev, currentSection.title]);
    }
  }, [pathname, openSections]);

  const toggleSection = (title: string) => {
    setOpenSections((prev) =>
      prev.includes(title) ? prev.filter((t) => t !== title) : [...prev, title]
    );
  };

  return (
    <aside className={cn('w-64 shrink-0', className)}>
      <div className="sticky top-20 space-y-6">
        {/* Version selector */}
        <div className="px-2">
          <Select value={selectedVersion} onValueChange={setSelectedVersion}>
            <SelectTrigger className="w-full bg-background/50 backdrop-blur-sm border-border/50 hover:border-primary/50 transition-colors">
              <SelectValue placeholder="Select version" />
            </SelectTrigger>
            <SelectContent>
              {versions.map((version) => (
                <SelectItem key={version} value={version}>
                  {version}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Navigation */}
        <nav className="space-y-1 px-1" aria-label="Docs navigation">
          {docsNavigation.map((section, sectionIndex) => {
            const isOpen = openSections.includes(section.title);
            const Icon = section.icon;
            const isActiveSection = section.href === pathname ||
              section.items?.some((item) => item.href === pathname);

            return (
              <div
                key={section.title}
                className="animate-fade-in-up"
                style={{ animationDelay: `${sectionIndex * 50}ms` }}
              >
                <button
                  onClick={() => toggleSection(section.title)}
                  className={cn(
                    'group flex w-full items-center justify-between rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200',
                    isActiveSection
                      ? 'bg-primary/10 text-primary'
                      : 'text-foreground hover:bg-muted/80'
                  )}
                  aria-expanded={isOpen}
                >
                  <span className="flex items-center gap-3">
                    {Icon && (
                      <Icon className={cn(
                        'h-4 w-4 transition-colors duration-200',
                        isActiveSection ? 'text-primary' : 'text-muted-foreground group-hover:text-foreground'
                      )} />
                    )}
                    {section.title}
                  </span>
                  {section.items && (
                    <span className={cn(
                      'transition-transform duration-200',
                      isOpen ? 'rotate-0' : '-rotate-90'
                    )}>
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </span>
                  )}
                </button>

                {/* Animated submenu */}
                <div
                  className={cn(
                    'overflow-hidden transition-all duration-300 ease-in-out',
                    isOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
                  )}
                >
                  {section.items && (
                    <ul className="ml-4 mt-1 space-y-0.5 border-l-2 border-border/50 pl-4">
                      {section.items.map((item, itemIndex) => {
                        const isActive = pathname === item.href;
                        return (
                          <li
                            key={item.href}
                            style={{ animationDelay: `${(sectionIndex * 50) + (itemIndex * 30)}ms` }}
                          >
                            <Link
                              href={item.href}
                              className={cn(
                                'group relative block rounded-md px-3 py-2 text-sm transition-all duration-200',
                                isActive
                                  ? 'bg-primary/10 font-medium text-primary'
                                  : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'
                              )}
                            >
                              {/* Active indicator */}
                              {isActive && (
                                <span className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-[1.125rem] h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
                              )}
                              {item.title}
                            </Link>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              </div>
            );
          })}
        </nav>

        {/* Help card */}
        <div className="mx-2 rounded-lg border border-border/50 bg-muted/30 p-4 backdrop-blur-sm">
          <p className="text-xs font-medium text-foreground mb-1">Need help?</p>
          <p className="text-xs text-muted-foreground mb-3">
            Check our GitHub for issues and discussions.
          </p>
          <Link
            href="https://github.com/codeMaestro78/MLcli"
            target="_blank"
            className="inline-flex items-center text-xs font-medium text-primary hover:underline"
          >
            View on GitHub →
          </Link>
        </div>
      </div>
    </aside>
  );
}
