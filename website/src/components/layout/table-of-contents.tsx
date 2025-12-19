'use client';

import { cn } from '@/lib/utils';
import { List } from 'lucide-react';
import * as React from 'react';

interface TOCItem {
  id: string;
  title: string;
  level: number;
}

interface TableOfContentsProps {
  className?: string;
}

export function TableOfContents({ className }: TableOfContentsProps) {
  const [headings, setHeadings] = React.useState<TOCItem[]>([]);
  const [activeId, setActiveId] = React.useState<string>('');

  React.useEffect(() => {
    // Get all headings from the article
    const article = document.querySelector('article');
    if (!article) return;

    const elements = article.querySelectorAll('h2, h3');
    const items: TOCItem[] = [];

    elements.forEach((element) => {
      if (element.id) {
        items.push({
          id: element.id,
          title: element.textContent || '',
          level: element.tagName === 'H2' ? 2 : 3,
        });
      }
    });

    setHeadings(items);

    // Set up intersection observer
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        });
      },
      { rootMargin: '-80px 0px -80% 0px' }
    );

    elements.forEach((element) => {
      if (element.id) {
        observer.observe(element);
      }
    });

    return () => observer.disconnect();
  }, []);

  if (headings.length === 0) return null;

  return (
    <nav className={cn('w-56 shrink-0', className)}>
      <div className="sticky top-24 max-h-[calc(100vh-8rem)] overflow-auto">
        <div className="flex items-center gap-2 mb-4 text-sm font-medium text-foreground">
          <List className="h-4 w-4" />
          On this page
        </div>
        <ul className="space-y-2.5 text-sm">
          {headings.map((heading, index) => (
            <li
              key={heading.id}
              style={{
                paddingLeft: heading.level === 3 ? '1rem' : '0',
                animationDelay: `${index * 50}ms`,
              }}
              className="animate-fade-in-up"
            >
              <a
                href={`#${heading.id}`}
                onClick={(e) => {
                  e.preventDefault();
                  const element = document.getElementById(heading.id);
                  if (element) {
                    element.scrollIntoView({ behavior: 'smooth' });
                    setActiveId(heading.id);
                  }
                }}
                className={cn(
                  'block py-1 transition-all duration-200 hover:text-primary border-l-2 pl-3 -ml-px',
                  activeId === heading.id
                    ? 'text-primary border-primary font-medium'
                    : 'text-muted-foreground border-transparent hover:border-muted-foreground/50'
                )}
              >
                {heading.title}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
