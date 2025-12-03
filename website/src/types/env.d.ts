/* eslint-disable @typescript-eslint/triple-slash-reference */
/// <reference types="react" />
/// <reference types="react-dom" />

declare namespace NodeJS {
  interface ProcessEnv {
    readonly NODE_ENV: 'development' | 'production' | 'test';
    readonly GITHUB_REPO: string;
    readonly GITHUB_TOKEN?: string;
    readonly EXPERIMENTS_JSON_URL?: string;
    readonly SITE_URL?: string;
    readonly NEXT_PUBLIC_GA_ID?: string;
    readonly NEXT_PUBLIC_ALGOLIA_APP_ID?: string;
    readonly NEXT_PUBLIC_ALGOLIA_SEARCH_KEY?: string;
    readonly NEXT_PUBLIC_ALGOLIA_INDEX_NAME?: string;
  }
}

declare module '*.mdx' {
  import type { ComponentType } from 'react';

  interface MDXProps {
    components?: Record<string, ComponentType>;
  }

  const MDXComponent: ComponentType<MDXProps>;
  export default MDXComponent;

  export const frontmatter: {
    title?: string;
    description?: string;
    sidebar?: string;
    order?: number;
  };
}

declare module '*.md' {
  const content: string;
  export default content;
}
