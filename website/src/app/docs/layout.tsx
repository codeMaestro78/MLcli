import { DocsSidebar, TableOfContents } from '@/components/layout';

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="container flex gap-8 py-8 lg:py-12">
      {/* Left Sidebar */}
      <DocsSidebar className="hidden lg:block" />

      {/* Main Content */}
      <div className="min-w-0 flex-1">
        <article className="docs-content prose prose-slate dark:prose-invert max-w-none animate-fade-in-up">
          {children}
        </article>
      </div>

      {/* Right Table of Contents */}
      <TableOfContents className="hidden xl:block" />
    </div>
  );
}
