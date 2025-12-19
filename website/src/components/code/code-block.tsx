'use client';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { Check, Copy, FileCode, Terminal } from 'lucide-react';
import * as React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  showLineNumbers?: boolean;
  className?: string;
}

const languageIcons: Record<string, React.ReactNode> = {
  bash: <Terminal className="h-3.5 w-3.5" />,
  sh: <Terminal className="h-3.5 w-3.5" />,
  shell: <Terminal className="h-3.5 w-3.5" />,
  default: <FileCode className="h-3.5 w-3.5" />,
};

const languageLabels: Record<string, string> = {
  bash: 'Terminal',
  sh: 'Shell',
  shell: 'Shell',
  python: 'Python',
  javascript: 'JavaScript',
  typescript: 'TypeScript',
  json: 'JSON',
  yaml: 'YAML',
  yml: 'YAML',
};

export function CodeBlock({
  code,
  language = 'bash',
  filename,
  showLineNumbers = false,
  className,
}: CodeBlockProps) {
  const [copied, setCopied] = React.useState(false);

  const copyToClipboard = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const icon = languageIcons[language] || languageIcons.default;
  const label = filename || languageLabels[language] || language.toUpperCase();

  return (
    <div className={cn(
      'group relative my-6 overflow-hidden rounded-xl border border-border/50',
      'bg-zinc-950 dark:bg-zinc-900/80 backdrop-blur-sm',
      'transition-all duration-300 hover:border-border/80 hover:shadow-lg hover:shadow-primary/5',
      className
    )}>
      {/* Top gradient line */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800/50 px-4 py-2.5">
        <div className="flex items-center gap-2">
          {/* Mac-style dots */}
          <div className="flex items-center gap-1.5 mr-2">
            <span className="h-3 w-3 rounded-full bg-red-500/80" />
            <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
            <span className="h-3 w-3 rounded-full bg-green-500/80" />
          </div>
          <span className="flex items-center gap-2 text-xs font-medium text-zinc-400">
            {icon}
            {label}
          </span>
        </div>

        {/* Copy button */}
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            'h-7 px-2 text-xs font-medium text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800',
            'opacity-0 transition-all duration-200 group-hover:opacity-100',
            copied && 'opacity-100'
          )}
          onClick={copyToClipboard}
          aria-label="Copy code"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 mr-1 text-green-400" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5 mr-1" />
              Copy
            </>
          )}
        </Button>
      </div>

      {/* Code */}
      <div className="overflow-x-auto">
        <SyntaxHighlighter
          language={language}
          style={oneDark}
          showLineNumbers={showLineNumbers}
          customStyle={{
            margin: 0,
            padding: '1rem 1.25rem',
            background: 'transparent',
            fontSize: '0.875rem',
            lineHeight: '1.7',
          }}
          codeTagProps={{
            style: {
              fontFamily: '"JetBrains Mono", "Fira Code", monospace',
            },
          }}
          lineNumberStyle={{
            minWidth: '2.5em',
            paddingRight: '1em',
            color: '#52525b',
            userSelect: 'none',
          }}
        >
          {code.trim()}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}

interface ConfigViewerProps {
  config: Record<string, unknown>;
  format?: 'json' | 'yaml';
  title?: string;
}

export function ConfigViewer({ config, format = 'json', title }: ConfigViewerProps) {
  const code = format === 'json'
    ? JSON.stringify(config, null, 2)
    : jsonToYaml(config);

  return (
    <div className="space-y-2">
      {title && <h4 className="font-medium">{title}</h4>}
      <CodeBlock code={code} language={format} />
    </div>
  );
}

// Simple JSON to YAML converter
function jsonToYaml(obj: unknown, indent = 0): string {
  const spaces = '  '.repeat(indent);

  if (obj === null || obj === undefined) {
    return 'null';
  }

  if (typeof obj === 'string') {
    return obj.includes('\n') ? `|\n${obj.split('\n').map(line => spaces + '  ' + line).join('\n')}` : obj;
  }

  if (typeof obj === 'number' || typeof obj === 'boolean') {
    return String(obj);
  }

  if (Array.isArray(obj)) {
    if (obj.length === 0) return '[]';
    return obj.map(item => `${spaces}- ${jsonToYaml(item, indent + 1).trim()}`).join('\n');
  }

  if (typeof obj === 'object') {
    const entries = Object.entries(obj);
    if (entries.length === 0) return '{}';
    return entries
      .map(([key, value]) => {
        const valueStr = jsonToYaml(value, indent + 1);
        if (typeof value === 'object' && value !== null) {
          return `${spaces}${key}:\n${valueStr}`;
        }
        return `${spaces}${key}: ${valueStr}`;
      })
      .join('\n');
  }

  return String(obj);
}
