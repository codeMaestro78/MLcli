import { Footer, Navbar } from '@/components/layout';
import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://mlcli.dev'),
  title: {
    default: 'mlcli - Production-Ready ML CLI Tool',
    template: '%s | mlcli',
  },
  description:
    'A production-ready CLI tool for training, evaluating, and managing machine learning models with experiment tracking, hyperparameter tuning, and model explainability.',
  keywords: [
    'machine learning',
    'cli',
    'ml',
    'deep learning',
    'experiment tracking',
    'model training',
    'python',
    'scikit-learn',
    'tensorflow',
    'xgboost',
  ],
  authors: [{ name: 'codeMaestro78' }],
  creator: 'codeMaestro78',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://mlcli.dev',
    title: 'mlcli - Production-Ready ML CLI Tool',
    description:
      'Train, evaluate, and manage ML models from the command line with experiment tracking and hyperparameter tuning.',
    siteName: 'mlcli',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'mlcli - ML CLI Tool',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'mlcli - Production-Ready ML CLI Tool',
    description:
      'Train, evaluate, and manage ML models from the command line with experiment tracking and hyperparameter tuning.',
    images: ['/og-image.png'],
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  icons: {
    icon: '/favicon.svg',
    shortcut: '/favicon.svg',
    apple: '/icons/icon-192.svg',
  },
  manifest: '/site.webmanifest',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="font-sans antialiased">
        <div className="relative flex min-h-screen flex-col">
          <Navbar />
          <main className="flex-1">{children}</main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
