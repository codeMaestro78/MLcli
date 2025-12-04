import { Github, Heart, Mail, Terminal, Twitter } from 'lucide-react';
import Link from 'next/link';

const footerLinks = {
  product: [
    { name: 'Features', href: '/docs' },
    { name: 'Quickstart', href: '/docs/quickstart' },
    { name: 'Releases', href: '/releases' },
    { name: 'Download', href: '/download' },
  ],
  resources: [
    { name: 'Documentation', href: '/docs' },
    { name: 'API Reference', href: '/docs/api' },
    { name: 'Examples', href: '/docs/examples' },
    { name: 'UI Demo', href: '/ui-demo' },
  ],
  community: [
    { name: 'GitHub', href: 'https://github.com/codeMaestro78/MLcli' },
    { name: 'Issues', href: 'https://github.com/codeMaestro78/MLcli/issues' },
    { name: 'Discussions', href: 'https://github.com/codeMaestro78/MLcli/discussions' },
    { name: 'Contribute', href: '/contribute' },
  ],
  legal: [
    { name: 'License', href: '/license' },
    { name: 'About', href: '/about' },
  ],
};

const socialLinks = [
  {
    name: 'GitHub',
    href: 'https://github.com/codeMaestro78/MLcli',
    icon: Github,
  },
  {
    name: 'Twitter',
    href: 'https://twitter.com/mlcli',
    icon: Twitter,
  },
  {
    name: 'Email',
    href: 'mailto:contact@mlcli.dev',
    icon: Mail,
  },
];

export function Footer() {
  return (
    <footer className="border-t bg-muted/30" aria-labelledby="footer-heading">
      <h2 id="footer-heading" className="sr-only">
        Footer
      </h2>
      <div className="container py-12 md:py-16">
        <div className="xl:grid xl:grid-cols-3 xl:gap-8">
          {/* Brand section */}
          <div className="space-y-4">
            <Link href="/" className="flex items-center gap-2 group">
              <Terminal className="h-8 w-8 text-mlcli-500 transition-transform group-hover:scale-110" />
              <span className="text-xl font-bold">mlcli</span>
            </Link>
            <p className="max-w-xs text-sm text-muted-foreground">
              A production-ready CLI tool for training, evaluating, and managing machine learning
              models with experiment tracking and hyperparameter tuning.
            </p>
            <div className="flex space-x-4">
              {socialLinks.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground transition-all hover:text-foreground hover:scale-110"
                  aria-label={item.name}
                >
                  <item.icon className="h-5 w-5" aria-hidden="true" />
                </Link>
              ))}
            </div>
          </div>

          {/* Links section */}
          <div className="mt-12 grid grid-cols-2 gap-8 md:grid-cols-4 xl:col-span-2 xl:mt-0">
            <div>
              <h3 className="text-sm font-semibold text-foreground">Product</h3>
              <ul role="list" className="mt-4 space-y-2">
                {footerLinks.product.map((item) => (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className="text-sm text-muted-foreground transition-colors hover:text-foreground"
                    >
                      {item.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-foreground">Resources</h3>
              <ul role="list" className="mt-4 space-y-2">
                {footerLinks.resources.map((item) => (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className="text-sm text-muted-foreground transition-colors hover:text-foreground"
                    >
                      {item.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-foreground">Community</h3>
              <ul role="list" className="mt-4 space-y-2">
                {footerLinks.community.map((item) => (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      target={item.href.startsWith('http') ? '_blank' : undefined}
                      rel={item.href.startsWith('http') ? 'noopener noreferrer' : undefined}
                      className="text-sm text-muted-foreground transition-colors hover:text-foreground"
                    >
                      {item.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-foreground">Legal</h3>
              <ul role="list" className="mt-4 space-y-2">
                {footerLinks.legal.map((item) => (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className="text-sm text-muted-foreground transition-colors hover:text-foreground"
                    >
                      {item.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>

        {/* Bottom section */}
        <div className="mt-12 border-t pt-8">
          <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
            <p className="text-sm text-muted-foreground">
              &copy; {new Date().getFullYear()} mlcli. Open source under MIT License.
            </p>
            <p className="flex items-center gap-1 text-sm text-muted-foreground">
              Made with <Heart className="h-4 w-4 text-red-500 animate-pulse" aria-label="love" /> for the ML
              community
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
