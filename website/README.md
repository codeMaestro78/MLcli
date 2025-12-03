# mlcli Website

The official documentation and product website for [mlcli](https://github.com/codeMaestro78/MLcli) - a command-line interface for machine learning.

## Tech Stack

- **Framework**: [Next.js 14](https://nextjs.org/) with App Router
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **UI Components**: [shadcn/ui](https://ui.shadcn.com/) + [Radix UI](https://www.radix-ui.com/)
- **Charts**: [Recharts](https://recharts.org/)
- **Data Fetching**: [SWR](https://swr.vercel.app/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Deployment**: [Vercel](https://vercel.com/)

## Getting Started

### Prerequisites

- Node.js 18.17 or later
- npm, yarn, or pnpm

### Installation

```bash
# Navigate to the website directory
cd website

# Install dependencies
npm install

# Start the development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the website.

### Environment Variables

Create a `.env.local` file in the website directory:

```env
# GitHub repository for fetching releases
NEXT_PUBLIC_GITHUB_REPO=codeMaestro78/MLcli

# URL for experiments JSON data
NEXT_PUBLIC_EXPERIMENTS_JSON_URL=/runs/experiments.json
```

## Project Structure

```
website/
├── public/
│   └── runs/
│       └── experiments.json    # Sample experiment data
├── src/
│   ├── app/                    # Next.js App Router pages
│   │   ├── about/              # About page
│   │   ├── api/                # API routes
│   │   ├── contribute/         # Contribute page
│   │   ├── docs/               # Documentation pages
│   │   ├── download/           # Download page
│   │   ├── releases/           # Releases page
│   │   ├── runs/               # Runs dashboard
│   │   ├── ui-demo/            # Interactive demo
│   │   ├── globals.css         # Global styles
│   │   ├── layout.tsx          # Root layout
│   │   └── page.tsx            # Home page
│   ├── components/
│   │   ├── code/               # Code display components
│   │   ├── layout/             # Layout components (Navbar, Footer)
│   │   ├── releases/           # Release-related components
│   │   ├── runs/               # Experiment run components
│   │   └── ui/                 # shadcn/ui components
│   ├── lib/
│   │   ├── api.ts              # API fetching functions
│   │   └── utils.ts            # Utility functions
│   └── types/
│       └── index.ts            # TypeScript interfaces
├── next.config.mjs             # Next.js configuration
├── tailwind.config.ts          # Tailwind CSS configuration
├── tsconfig.json               # TypeScript configuration
└── package.json                # Dependencies and scripts
```

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server on port 3000 |
| `npm run build` | Build for production |
| `npm run start` | Start production server |
| `npm run lint` | Run ESLint |
| `npm run type-check` | Run TypeScript type checking |
| `npm run format` | Format code with Prettier |

## Pages

| Route | Description |
|-------|-------------|
| `/` | Home page with hero, features, and CTA |
| `/docs` | Documentation landing page |
| `/docs/quickstart` | Getting started guide |
| `/docs/config` | Configuration reference |
| `/docs/trainers` | Available trainers documentation |
| `/runs` | Experiment runs dashboard |
| `/runs/[runId]` | Individual run details |
| `/releases` | GitHub releases list |
| `/download` | Download instructions |
| `/ui-demo` | Interactive training demo |
| `/about` | About the project |
| `/contribute` | Contribution guidelines |

## API Routes

| Endpoint | Description |
|----------|-------------|
| `GET /api/runs` | Fetch all experiment runs |
| `GET /api/runs/[runId]` | Fetch a specific run |
| `GET /api/releases` | Fetch GitHub releases |

## Adding New Components

This project uses [shadcn/ui](https://ui.shadcn.com/) components. Components are copied into `src/components/ui/` and can be customized.

To add a new shadcn/ui component:

```bash
npx shadcn-ui@latest add [component-name]
```

## Customization

### Theme

The color theme is defined in `src/app/globals.css` using CSS variables. Modify the `:root` and `.dark` selectors to change colors.

### Fonts

The website uses the Inter font. To change fonts, update `src/app/layout.tsx`.

## Deployment

### Vercel (Recommended)

1. Push your code to GitHub
2. Import the project in [Vercel](https://vercel.com/)
3. Set the root directory to `website`
4. Configure environment variables
5. Deploy!

### Docker

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV production
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run linting: `npm run lint`
5. Commit your changes: `git commit -m 'Add my feature'`
6. Push to the branch: `git push origin feature/my-feature`
7. Open a Pull Request

## License

This project is part of mlcli, which is open source.

---

Built with ❤️ for the mlcli community
