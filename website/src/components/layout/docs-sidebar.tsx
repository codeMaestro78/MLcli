'use client';

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { BookOpen, Box, ChevronDown, ChevronRight, Code, FlaskConical, Lightbulb, Settings, Sliders, Zap } from 'lucide-react';
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

const versions = ['v0.1.1', 'latest'];

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
      <div className="sticky top-20 space-y-4">
        {/* Version selector */}
        <div className="px-4">
          <Select value={selectedVersion} onValueChange={setSelectedVersion}>
            <SelectTrigger className="w-full">
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
        <nav className="space-y-1 px-2" aria-label="Docs navigation">
          {docsNavigation.map((section) => {
            const isOpen = openSections.includes(section.title);
            const Icon = section.icon;

            return (
              <div key={section.title}>
                <button
                  onClick={() => toggleSection(section.title)}
                  className="flex w-full items-center justify-between rounded-md px-2 py-2 text-sm font-medium text-foreground hover:bg-muted"
                  aria-expanded={isOpen}
                >
                  <span className="flex items-center gap-2">
                    {Icon && <Icon className="h-4 w-4 text-muted-foreground" />}
                    {section.title}
                  </span>
                  {section.items && (
                    <span className="text-muted-foreground">
                      {isOpen ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </span>
                  )}
                </button>

                {isOpen && section.items && (
                  <ul className="ml-4 mt-1 space-y-1 border-l pl-4">
                    {section.items.map((item) => {
                      const isActive = pathname === item.href;
                      return (
                        <li key={item.href}>
                          <Link
                            href={item.href}
                            className={cn(
                              'block rounded-md px-2 py-1.5 text-sm transition-colors',
                              isActive
                                ? 'bg-primary/10 font-medium text-primary'
                                : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                            )}
                          >
                            {item.title}
                          </Link>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            );
          })}
        </nav>
      </div>
    </aside>
  );
}
