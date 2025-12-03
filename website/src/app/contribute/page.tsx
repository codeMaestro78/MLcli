import { CodeBlock } from '@/components/code';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Bug, Code, FileText, Github, GitPullRequest, Heart, MessageSquare, Users } from 'lucide-react';
import Link from 'next/link';

export const metadata = {
  title: 'Contribute',
  description: 'Learn how to contribute to mlcli.',
};

export default function ContributePage() {
  return (
    <div className="container py-12">
      {/* Header */}
      <div className="mb-12 text-center">
        <h1 className="mb-4 text-4xl font-bold tracking-tight">Contribute to mlcli</h1>
        <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
          mlcli is open source and we welcome contributions! Whether it is fixing bugs, adding features, or improving documentation.
        </p>
      </div>

      {/* Quick Links */}
      <div className="mb-12 grid gap-4 md:grid-cols-4">
        <Link
          href="https://github.com/codeMaestro78/MLcli"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Card className="h-full transition-all hover:border-primary hover:shadow-md">
            <CardHeader className="text-center">
              <Github className="mx-auto h-8 w-8" />
              <CardTitle className="text-lg">GitHub Repo</CardTitle>
            </CardHeader>
          </Card>
        </Link>
        <Link
          href="https://github.com/codeMaestro78/MLcli/issues"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Card className="h-full transition-all hover:border-primary hover:shadow-md">
            <CardHeader className="text-center">
              <Bug className="mx-auto h-8 w-8" />
              <CardTitle className="text-lg">Report Bug</CardTitle>
            </CardHeader>
          </Card>
        </Link>
        <Link
          href="https://github.com/codeMaestro78/MLcli/pulls"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Card className="h-full transition-all hover:border-primary hover:shadow-md">
            <CardHeader className="text-center">
              <GitPullRequest className="mx-auto h-8 w-8" />
              <CardTitle className="text-lg">Pull Requests</CardTitle>
            </CardHeader>
          </Card>
        </Link>
        <Link
          href="https://github.com/codeMaestro78/MLcli/discussions"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Card className="h-full transition-all hover:border-primary hover:shadow-md">
            <CardHeader className="text-center">
              <MessageSquare className="mx-auto h-8 w-8" />
              <CardTitle className="text-lg">Discussions</CardTitle>
            </CardHeader>
          </Card>
        </Link>
      </div>

      {/* Getting Started */}
      <div className="mx-auto max-w-3xl space-y-12">
        <section>
          <h2 className="mb-4 text-2xl font-bold">Getting Started</h2>
          <Card>
            <CardContent className="pt-6">
              <CodeBlock
                code={`# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/MLcli.git
cd MLcli

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest tests/`}
                language="bash"
              />
            </CardContent>
          </Card>
        </section>

        {/* Ways to Contribute */}
        <section>
          <h2 className="mb-4 text-2xl font-bold">Ways to Contribute</h2>
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Bug className="h-5 w-5 text-red-500" />
                  <CardTitle className="text-lg">Bug Reports</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Found a bug? Open an issue on GitHub with steps to reproduce, expected behavior, and your environment details.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Code className="h-5 w-5 text-blue-500" />
                  <CardTitle className="text-lg">Code Contributions</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Add new trainers, fix bugs, improve performance, or add new features. Check open issues for ideas.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-green-500" />
                  <CardTitle className="text-lg">Documentation</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Improve docs, add examples, fix typos, or translate documentation to other languages.
                </CardDescription>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Users className="h-5 w-5 text-purple-500" />
                  <CardTitle className="text-lg">Community</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Help others in discussions, answer questions, share your use cases, and spread the word!
                </CardDescription>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Adding a Trainer */}
        <section>
          <h2 className="mb-4 text-2xl font-bold">Adding a New Trainer</h2>
          <Card>
            <CardContent className="pt-6">
              <CodeBlock
                code={`# 1. Create a new trainer file
# mlcli/trainers/my_trainer.py

from mlcli.trainers.base_trainer import BaseTrainer

class MyTrainer(BaseTrainer):
    """My custom trainer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Training logic
        pass

    def predict(self, X):
        # Prediction logic
        pass

    def save(self, path):
        # Save model
        pass

    def load(self, path):
        # Load model
        pass

# 2. Register in registry.py
from mlcli.utils.registry import ModelRegistry
ModelRegistry.register("my_trainer", MyTrainer)

# 3. Add tests in tests/test_my_trainer.py
# 4. Update documentation`}
                language="python"
              />
            </CardContent>
          </Card>
        </section>

        {/* Code Style */}
        <section>
          <h2 className="mb-4 text-2xl font-bold">Code Style</h2>
          <Card>
            <CardContent className="prose prose-sm dark:prose-invert pt-6">
              <ul>
                <li>Follow PEP 8 style guidelines</li>
                <li>Use type hints for function signatures</li>
                <li>Write docstrings for all public functions and classes</li>
                <li>Run <code>black</code> and <code>isort</code> before committing</li>
                <li>Ensure all tests pass with <code>pytest</code></li>
                <li>Add tests for new features</li>
              </ul>
              <CodeBlock
                code={`# Format code
black mlcli/
isort mlcli/

# Run linting
flake8 mlcli/

# Run tests
pytest tests/ -v`}
                language="bash"
              />
            </CardContent>
          </Card>
        </section>

        {/* Pull Request Process */}
        <section>
          <h2 className="mb-4 text-2xl font-bold">Pull Request Process</h2>
          <ol className="list-decimal space-y-2 pl-6">
            <li>Fork the repository and create a new branch</li>
            <li>Make your changes and commit with clear messages</li>
            <li>Ensure all tests pass and add new tests if needed</li>
            <li>Update documentation if your changes affect it</li>
            <li>Open a PR with a clear description of your changes</li>
            <li>Address any review feedback</li>
          </ol>
        </section>

        {/* Thanks */}
        <section className="text-center">
          <div className="inline-flex items-center gap-2 rounded-lg bg-muted p-4">
            <Heart className="h-5 w-5 text-red-500" />
            <span>Thank you for contributing to mlcli!</span>
          </div>
        </section>
      </div>
    </div>
  );
}
