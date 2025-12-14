import { CodeBlock } from '@/components/code';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import Link from 'next/link';

export const metadata = {
  title: 'Trainers',
  description: 'Available model trainers in mlcli.',
};

const trainers = [
  {
    name: 'Random Forest',
    command: 'random_forest',
    framework: 'scikit-learn',
    type: 'Classification / Regression',
    description: 'Ensemble of decision trees with bootstrap sampling.',
    category: 'Traditional ML',
  },
  {
    name: 'XGBoost',
    command: 'xgboost',
    framework: 'XGBoost',
    type: 'Classification / Regression',
    description: 'Gradient boosting with regularization for high performance.',
    category: 'Gradient Boosting',
  },
  {
    name: 'LightGBM',
    command: 'lightgbm',
    framework: 'LightGBM',
    type: 'Classification / Regression',
    description: 'Fast gradient boosting with leaf-wise tree growth and native categorical support.',
    category: 'Gradient Boosting',
  },
  {
    name: 'CatBoost',
    command: 'catboost',
    framework: 'CatBoost',
    type: 'Classification / Regression',
    description: 'Gradient boosting with excellent handling of categorical features.',
    category: 'Gradient Boosting',
  },
  {
    name: 'SVM',
    command: 'svm',
    framework: 'scikit-learn',
    type: 'Classification / Regression',
    description: 'Support Vector Machine with various kernels.',
    category: 'Traditional ML',
  },
  {
    name: 'Logistic Regression',
    command: 'logistic',
    framework: 'scikit-learn',
    type: 'Classification',
    description: 'Linear model for binary and multiclass classification.',
    category: 'Traditional ML',
  },
  {
    name: 'K-Means',
    command: 'kmeans',
    framework: 'scikit-learn',
    type: 'Clustering',
    description: 'Partition-based clustering with automatic optimal K detection via elbow method.',
    category: 'Clustering',
  },
  {
    name: 'DBSCAN',
    command: 'dbscan',
    framework: 'scikit-learn',
    type: 'Clustering',
    description: 'Density-based clustering with automatic noise detection and optimal eps finder.',
    category: 'Clustering',
  },
  {
    name: 'Isolation Forest',
    command: 'isolation_forest',
    framework: 'scikit-learn',
    type: 'Anomaly Detection',
    description: 'Tree-based anomaly detection using isolation principle.',
    category: 'Anomaly Detection',
  },
  {
    name: 'One-Class SVM',
    command: 'one_class_svm',
    framework: 'scikit-learn',
    type: 'Anomaly Detection',
    description: 'Novelty detection using support vector methods.',
    category: 'Anomaly Detection',
  },
  {
    name: 'DNN',
    command: 'tf_dnn',
    framework: 'TensorFlow',
    type: 'Classification / Regression',
    description: 'Deep Neural Network with customizable layers.',
    category: 'Deep Learning',
  },
  {
    name: 'CNN',
    command: 'tf_cnn',
    framework: 'TensorFlow',
    type: 'Classification',
    description: 'Convolutional Neural Network for image-like data.',
    category: 'Deep Learning',
  },
  {
    name: 'RNN',
    command: 'tf_rnn',
    framework: 'TensorFlow',
    type: 'Classification / Regression',
    description: 'Recurrent Neural Network for sequential data.',
    category: 'Deep Learning',
  },
];

export default function TrainersPage() {
  const categories = ['Traditional ML', 'Gradient Boosting', 'Clustering', 'Anomaly Detection', 'Deep Learning'];
  
  return (
    <div>
      <h1>Trainers</h1>
      <p className="lead">
        mlcli supports 15+ machine learning algorithms out of the box including classification,
        regression, clustering, and anomaly detection. This page lists all available trainers.
      </p>

      <h2 id="available-trainers">Available Trainers</h2>
      <p>
        List all available trainers with:
      </p>
      <CodeBlock code="mlcli list-trainers" language="bash" />

      {categories.map((category) => {
        const categoryTrainers = trainers.filter((t) => t.category === category);
        const categoryId = category.toLowerCase().replace(/\s+/g, '-');
        return (
          <div key={category} className="mt-8">
            <h3 id={categoryId} className="text-xl font-semibold mb-4">{category}</h3>
            <div className="not-prose grid gap-4">
              {categoryTrainers.map((trainer) => (
                <Card key={trainer.command}>
                  <CardHeader>
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <CardTitle className="text-lg">{trainer.name}</CardTitle>
                      <div className="flex gap-2 flex-wrap">
                        <Badge variant="outline">{trainer.framework}</Badge>
                        <Badge variant="secondary">{trainer.type}</Badge>
                      </div>
                    </div>
                    <CardDescription className="font-mono text-sm">
                      mlcli train -m {trainer.command}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">{trainer.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        );
      })}

      <h2 id="using-trainers">Using Trainers</h2>
      <p>
        Train a model using the <code>--model</code> or <code>-m</code> flag:
      </p>
      <CodeBlock
        code={`# Train Random Forest (Classification/Regression)
mlcli train -d data.csv -m random_forest --target label

# Train LightGBM (Gradient Boosting)
mlcli train -d data.csv -m lightgbm --target label

# Train K-Means (Clustering - no target needed)
mlcli train -d data.csv -m kmeans

# Train Isolation Forest (Anomaly Detection)
mlcli train -d data.csv -m isolation_forest

# Train Deep Neural Network
mlcli train -d data.csv -m tf_dnn --target label`}
        language="bash"
      />

      <h2 id="custom-trainers">Creating Custom Trainers</h2>
      <p>
        You can create custom trainers by extending the <code>BaseTrainer</code> class:
      </p>
      <CodeBlock
        code={`from mlcli.trainers.base_trainer import BaseTrainer

class MyCustomTrainer(BaseTrainer):
    """Custom trainer implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model here

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model."""
        # Implement training logic
        pass

    def predict(self, X):
        """Make predictions."""
        # Implement prediction logic
        pass

    def save(self, path):
        """Save the model."""
        # Implement model saving
        pass

    def load(self, path):
        """Load a saved model."""
        # Implement model loading
        pass`}
        language="python"
        filename="my_trainer.py"
      />

      <p>
        Register your custom trainer with the model registry:
      </p>
      <CodeBlock
        code={`from mlcli.utils.registry import ModelRegistry

# Register the custom trainer
ModelRegistry.register("my_custom", MyCustomTrainer)

# Now use it with mlcli
# mlcli train -d data.csv -m my_custom --target label`}
        language="python"
      />

      <h2 id="next-steps">Next Steps</h2>
      <p>
        Learn more about specific trainers:
      </p>
      <ul>
        <li><Link href="/docs/trainers/random-forest">Random Forest Guide</Link></li>
        <li><Link href="/docs/trainers/xgboost">XGBoost Guide</Link></li>
        <li><Link href="/docs/trainers/deep-learning">Deep Learning Guide</Link></li>
      </ul>
    </div>
  );
}
