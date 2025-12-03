import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Code, Github, Linkedin, Mail, Target, Users, Zap } from 'lucide-react';
import Link from 'next/link';

export const metadata = {
  title: 'About',
  description: 'Learn about the mlcli project and team.',
};

const values = [
  {
    icon: Zap,
    title: 'Simple & Fast',
    description:
      'Train models with a single command. No boilerplate, no complexity.',
  },
  {
    icon: Target,
    title: 'Accurate Results',
    description:
      'Built on proven ML libraries like scikit-learn, XGBoost, LightGBM, and TensorFlow.',
  },
  {
    icon: Users,
    title: 'Community Driven',
    description:
      'Open source and built by the community, for the community.',
  },
  {
    icon: Code,
    title: 'Developer First',
    description:
      'Designed for developers who want to integrate ML into their workflows.',
  },
];

const timeline = [
  {
    date: 'March 2025',
    title: 'Project Started',
    description: 'Initial commit with basic training functionality.',
  },
  {
    date: 'March 2025',
    title: 'Core Features',
    description: 'Added evaluation, experiment tracking, and hyperparameter tuning.',
  },
  {
    date: 'March 2025',
    title: 'Data Preprocessing',
    description: 'Comprehensive preprocessing module with scalers, encoders, and feature selection.',
  },
  {
    date: 'Coming Soon',
    title: 'Model Deployment',
    description: 'Export models as REST APIs and deploy to cloud platforms.',
  },
];

export default function AboutPage() {
  return (
    <div className="container py-12">
      {/* Hero */}
      <div className="mb-16 text-center">
        <h1 className="mb-4 text-4xl font-bold tracking-tight md:text-5xl">
          About mlcli
        </h1>
        <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
          A command-line interface for machine learning that makes training,
          evaluating, and deploying models simple and reproducible.
        </p>
      </div>

      {/* Mission */}
      <section className="mb-16">
        <Card className="mx-auto max-w-3xl">
          <CardHeader>
            <CardTitle className="text-2xl">Our Mission</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-muted-foreground">
              Machine learning should be accessible to every developer, not just
              specialists. mlcli bridges the gap between data and insights by
              providing a simple, consistent interface for training and
              evaluating models. Whether you are a data scientist building
              production pipelines or a developer adding ML to your application,
              mlcli helps you move faster without sacrificing quality.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* Values */}
      <section className="mb-16">
        <h2 className="mb-8 text-center text-3xl font-bold">Our Values</h2>
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          {values.map((value) => (
            <Card key={value.title} className="text-center">
              <CardHeader>
                <value.icon className="mx-auto h-10 w-10 text-primary" />
                <CardTitle className="text-lg">{value.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>{value.description}</CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Timeline */}
      <section className="mb-16">
        <h2 className="mb-8 text-center text-3xl font-bold">Project Timeline</h2>
        <div className="mx-auto max-w-2xl">
          <div className="relative border-l-2 border-muted pl-6">
            {timeline.map((item, index) => (
              <div key={index} className="relative mb-8 last:mb-0">
                <div className="absolute -left-[31px] h-4 w-4 rounded-full border-2 border-primary bg-background" />
                <div className="text-sm text-muted-foreground">{item.date}</div>
                <h3 className="text-lg font-semibold">{item.title}</h3>
                <p className="text-muted-foreground">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="mb-16">
        <h2 className="mb-8 text-center text-3xl font-bold">The Team</h2>
        <div className="mx-auto max-w-md">
          <Card className="text-center">
            <CardHeader>
              <div className="mx-auto mb-4 flex h-24 w-24 items-center justify-center rounded-full bg-gradient-to-br from-blue-500 to-purple-600 text-3xl font-bold text-white">
                DH
              </div>
              <CardTitle>Devarshi Harsora</CardTitle>
              <CardDescription>Creator & Lead Developer</CardDescription>
            </CardHeader>
            <CardContent>
              <p className="mb-4 text-sm text-muted-foreground">
                Software developer passionate about machine learning and developer tools.
                Building mlcli to make ML accessible to everyone.
              </p>
              <div className="flex justify-center gap-2">
                <Button variant="ghost" size="icon" asChild>
                  <Link
                    href="https://github.com/codeMaestro78"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Github className="h-5 w-5" />
                    <span className="sr-only">GitHub</span>
                  </Link>
                </Button>
                <Button variant="ghost" size="icon" asChild>
                  <Link
                    href="https://linkedin.com/in/"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Linkedin className="h-5 w-5" />
                    <span className="sr-only">LinkedIn</span>
                  </Link>
                </Button>
                <Button variant="ghost" size="icon" asChild>
                  <Link href="mailto:contact@example.com">
                    <Mail className="h-5 w-5" />
                    <span className="sr-only">Email</span>
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="mb-16">
        <h2 className="mb-8 text-center text-3xl font-bold">Built With</h2>
        <div className="flex flex-wrap justify-center gap-3">
          {[
            'Python',
            'Click',
            'scikit-learn',
            'XGBoost',
            'LightGBM',
            'TensorFlow',
            'pandas',
            'numpy',
            'SHAP',
            'LIME',
            'Rich',
            'Typer',
          ].map((tech) => (
            <span
              key={tech}
              className="rounded-full bg-muted px-4 py-2 text-sm font-medium"
            >
              {tech}
            </span>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="text-center">
        <Card className="mx-auto inline-block">
          <CardContent className="flex flex-col items-center gap-4 p-8 sm:flex-row">
            <div className="text-left">
              <h3 className="text-lg font-semibold">Want to get involved?</h3>
              <p className="text-muted-foreground">
                We would love your contributions!
              </p>
            </div>
            <Button asChild>
              <Link href="/contribute">Start Contributing</Link>
            </Button>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
