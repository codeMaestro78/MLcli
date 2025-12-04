# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of MLCLI seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please Do

- **Email us** at [INSERT SECURITY EMAIL] with details of the vulnerability
- **Provide** sufficient information to reproduce the issue
- **Allow** reasonable time for us to address the issue before public disclosure
- **Make** a good faith effort to avoid privacy violations, data destruction, or service disruption

### Please Don't

- **Don't** open a public GitHub issue for security vulnerabilities
- **Don't** access or modify data that doesn't belong to you
- **Don't** perform actions that could harm the service or other users

## What to Include

When reporting a vulnerability, please include:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** of the vulnerability
4. **Suggested fix** (if any)

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days (depending on complexity)

## Security Best Practices for Users

### Model Security

```python
# ❌ Don't load untrusted pickle files
model = pickle.load(open("untrusted_model.pkl", "rb"))  # DANGEROUS

# ✅ Use ONNX format for safer model sharing
mlcli evaluate --model model.onnx --data test.csv
```

### Configuration Security

```python
# ❌ Don't include sensitive data in configs
{
    "database": {
        "password": "secret123"  # NEVER DO THIS
    }
}

# ✅ Use environment variables
{
    "database": {
        "password": "${DB_PASSWORD}"
    }
}
```

### Data Privacy

- Don't commit training data to public repositories
- Use `.gitignore` to exclude data files
- Consider data anonymization for shared datasets

## Dependencies

We regularly update dependencies to patch known vulnerabilities. Users should:

```bash
# Update to latest version
pip install --upgrade mlcli-toolkit

# Check for vulnerabilities
pip-audit
```

## Acknowledgments

We appreciate the security research community and will acknowledge reporters (with permission) in our release notes.

Thank you for helping keep MLCLI and our users safe!
