import { CodeBlock, ConfigViewer } from '@/components/code';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export const metadata = {
  title: 'Configuration',
  description: 'Learn about mlcli configuration files and options.',
};

const exampleConfig = {
  model_type: 'random_forest',
  dataset_path: 'data/train.csv',
  target_column: 'label',
  test_size: 0.2,
  random_state: 42,
  hyperparameters: {
    n_estimators: 100,
    max_depth: 10,
    min_samples_split: 2,
    min_samples_leaf: 1,
    class_weight: 'balanced',
  },
  output_dir: 'models/',
  experiment_name: 'my_experiment',
  tags: ['production', 'v1'],
};

export default function ConfigPage() {
  return (
    <div>
      <h1>Configuration</h1>
      <p className="lead">
        mlcli supports flexible configuration through JSON files and command-line arguments.
        This guide covers all available options.
      </p>

      <h2 id="config-file">Configuration File</h2>
      <p>
        Create a JSON configuration file to define your training parameters:
      </p>

      <Tabs defaultValue="json" className="not-prose">
        <TabsList>
          <TabsTrigger value="json">JSON</TabsTrigger>
          <TabsTrigger value="yaml">YAML</TabsTrigger>
        </TabsList>
        <TabsContent value="json">
          <ConfigViewer config={exampleConfig} format="json" />
        </TabsContent>
        <TabsContent value="yaml">
          <ConfigViewer config={exampleConfig} format="yaml" />
        </TabsContent>
      </Tabs>

      <h2 id="options">Configuration Options</h2>

      <h3 id="required">Required Options</h3>
      <table>
        <thead>
          <tr>
            <th>Option</th>
            <th>Type</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>model_type</code></td>
            <td>string</td>
            <td>Type of model to train (e.g., random_forest, xgboost, svm)</td>
          </tr>
          <tr>
            <td><code>dataset_path</code></td>
            <td>string</td>
            <td>Path to the training dataset (CSV format)</td>
          </tr>
          <tr>
            <td><code>target_column</code></td>
            <td>string</td>
            <td>Name of the target column in the dataset</td>
          </tr>
        </tbody>
      </table>

      <h3 id="optional">Optional Options</h3>
      <table>
        <thead>
          <tr>
            <th>Option</th>
            <th>Type</th>
            <th>Default</th>
            <th>Description</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>test_size</code></td>
            <td>float</td>
            <td>0.2</td>
            <td>Proportion of data for validation</td>
          </tr>
          <tr>
            <td><code>random_state</code></td>
            <td>int</td>
            <td>42</td>
            <td>Random seed for reproducibility</td>
          </tr>
          <tr>
            <td><code>output_dir</code></td>
            <td>string</td>
            <td>models/</td>
            <td>Directory to save trained models</td>
          </tr>
          <tr>
            <td><code>experiment_name</code></td>
            <td>string</td>
            <td>null</td>
            <td>Name for the experiment run</td>
          </tr>
          <tr>
            <td><code>hyperparameters</code></td>
            <td>object</td>
            <td>{'{}'}</td>
            <td>Model-specific hyperparameters</td>
          </tr>
        </tbody>
      </table>

      <h2 id="cli-args">Command-Line Arguments</h2>
      <p>
        You can also pass options directly via the command line:
      </p>
      <CodeBlock
        code={`mlcli train \\
  --data data/train.csv \\
  --model random_forest \\
  --target label \\
  --test-size 0.2 \\
  --n-estimators 100 \\
  --max-depth 10 \\
  --output models/`}
        language="bash"
      />

      <p>
        CLI arguments take precedence over config file values, allowing you to
        override specific options.
      </p>

      <h2 id="env-vars">Environment Variables</h2>
      <p>
        Some options can be set via environment variables:
      </p>
      <CodeBlock
        code={`# Set default output directory
export MLCLI_OUTPUT_DIR=models/

# Enable MLflow tracking
export MLCLI_MLFLOW_TRACKING_URI=http://localhost:5000

# Set random seed
export MLCLI_RANDOM_STATE=42`}
        language="bash"
      />

      <h2 id="hyperparameters">Hyperparameters by Model</h2>

      <h3>Random Forest</h3>
      <CodeBlock
        code={JSON.stringify({
          n_estimators: 100,
          max_depth: 10,
          min_samples_split: 2,
          min_samples_leaf: 1,
          max_features: 'sqrt',
          class_weight: 'balanced',
        }, null, 2)}
        language="json"
      />

      <h3>XGBoost</h3>
      <CodeBlock
        code={JSON.stringify({
          n_estimators: 100,
          max_depth: 6,
          learning_rate: 0.1,
          subsample: 0.8,
          colsample_bytree: 0.8,
          objective: 'binary:logistic',
        }, null, 2)}
        language="json"
      />

      <h3>Deep Neural Network</h3>
      <CodeBlock
        code={JSON.stringify({
          hidden_layers: [128, 64, 32],
          dropout_rate: 0.3,
          learning_rate: 0.001,
          epochs: 100,
          batch_size: 32,
          optimizer: 'adam',
        }, null, 2)}
        language="json"
      />
    </div>
  );
}
