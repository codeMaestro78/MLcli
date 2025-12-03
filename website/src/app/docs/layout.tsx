import { DocsSidebar } from '@/components/layout';

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="container flex gap-8 py-8">
      <DocsSidebar className="hidden lg:block" />
      <div className="min-w-0 flex-1">
        <article className="prose prose-slate dark:prose-invert max-w-none">
          {children}
        </article>
      </div>
    </div>
  );
}
