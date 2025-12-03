import { CodeBlock } from '@/components/code';
import type { MDXComponents } from 'mdx/types';
import Link from 'next/link';

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    // Headings
    h1: ({ children }) => (
      <h1 className="mt-8 scroll-m-20 text-4xl font-bold tracking-tight">
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className="mt-8 scroll-m-20 border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0">
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="mt-6 scroll-m-20 text-2xl font-semibold tracking-tight">
        {children}
      </h3>
    ),
    h4: ({ children }) => (
      <h4 className="mt-4 scroll-m-20 text-xl font-semibold tracking-tight">
        {children}
      </h4>
    ),

    // Paragraphs and text
    p: ({ children }) => (
      <p className="leading-7 [&:not(:first-child)]:mt-4">{children}</p>
    ),
    strong: ({ children }) => (
      <strong className="font-semibold">{children}</strong>
    ),
    em: ({ children }) => <em className="italic">{children}</em>,

    // Lists
    ul: ({ children }) => (
      <ul className="my-4 ml-6 list-disc [&>li]:mt-2">{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className="my-4 ml-6 list-decimal [&>li]:mt-2">{children}</ol>
    ),
    li: ({ children }) => <li>{children}</li>,

    // Links
    a: ({ href, children }) => {
      const isExternal = href?.startsWith('http');
      if (isExternal) {
        return (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-primary underline underline-offset-4"
          >
            {children}
          </a>
        );
      }
      return (
        <Link
          href={href || '#'}
          className="font-medium text-primary underline underline-offset-4"
        >
          {children}
        </Link>
      );
    },

    // Code
    code: ({ children, className }) => {
      // Check if this is a code block (has language class) or inline code
      const isCodeBlock = className?.includes('language-');
      if (isCodeBlock) {
        const language = className?.replace('language-', '') || 'text';
        return <CodeBlock code={String(children)} language={language} />;
      }
      return (
        <code className="relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold">
          {children}
        </code>
      );
    },
    pre: ({ children }) => {
      // The pre tag wraps code blocks, but we handle them in the code component
      return <div className="my-4">{children}</div>;
    },

    // Blockquotes
    blockquote: ({ children }) => (
      <blockquote className="mt-4 border-l-4 border-primary pl-4 italic">
        {children}
      </blockquote>
    ),

    // Tables
    table: ({ children }) => (
      <div className="my-6 w-full overflow-y-auto">
        <table className="w-full">{children}</table>
      </div>
    ),
    thead: ({ children }) => <thead>{children}</thead>,
    tbody: ({ children }) => <tbody>{children}</tbody>,
    tr: ({ children }) => (
      <tr className="m-0 border-t p-0 even:bg-muted">{children}</tr>
    ),
    th: ({ children }) => (
      <th className="border px-4 py-2 text-left font-bold [&[align=center]]:text-center [&[align=right]]:text-right">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="border px-4 py-2 text-left [&[align=center]]:text-center [&[align=right]]:text-right">
        {children}
      </td>
    ),

    // Horizontal rule
    hr: () => <hr className="my-8 border-muted" />,

    // Images
    img: ({ src, alt }) => (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={src}
        alt={alt || ''}
        className="my-4 rounded-lg border"
      />
    ),

    // Custom components can be added here
    ...components,
  };
}
