# Manual Setup Required

This document outlines the manual setup steps required to complete the quantum-devops-ci implementation. Some configurations require repository admin privileges or external service setup that cannot be automated through GitHub Actions.

## Required Manual Actions

### 1. GitHub Repository Settings

#### Branch Protection Rules
Navigate to **Settings → Branches** and configure:

- **Branch name pattern**: `main`
- **Protect matching branches**:
  - ✅ Require a pull request before merging
  - ✅ Require approvals (minimum 1)
  - ✅ Dismiss stale PR approvals when new commits are pushed
  - ✅ Require review from code owners
  - ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - ✅ Require conversation resolution before merging
  - ✅ Restrict pushes that create files larger than 100 MB

#### Required Status Checks
Add these required status checks:
- `ci-status` (from CI workflow)
- `security-scan` (from security workflow)
- `build` (from CI workflow)
- `quantum-tests` (from CI workflow)

#### Repository Topics
Add these topics in **Settings → General**:
- `quantum-computing`
- `devops`
- `ci-cd`
- `qiskit`
- `cirq`
- `github-actions`
- `automation`
- `testing`

### 2. GitHub Actions Secrets

Navigate to **Settings → Secrets and Variables → Actions** and add:

#### Required Secrets
```
# Quantum Platform Credentials
IBMQ_TOKEN=your_ibm_quantum_token
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
GOOGLE_CLOUD_CREDENTIALS=your_gcp_service_account_json

# Monitoring and Alerting
GRAFANA_API_KEY=your_grafana_api_key
SLACK_WEBHOOK_URL=your_slack_webhook_url
DATADOG_API_KEY=your_datadog_api_key (optional)

# Container Registry
DOCKER_HUB_USERNAME=your_dockerhub_username
DOCKER_HUB_TOKEN=your_dockerhub_token

# Code Quality
CODECOV_TOKEN=your_codecov_token
SONAR_TOKEN=your_sonarcloud_token (optional)

# Deployment
PRODUCTION_DEPLOY_KEY=your_production_ssh_key
STAGING_DEPLOY_KEY=your_staging_ssh_key
```

#### Environment Variables
```
# Project Configuration
QUANTUM_PROJECT_ID=your_project_identifier
DEFAULT_QUANTUM_BACKEND=qasm_simulator
MONTHLY_BUDGET_USD=1000

# Monitoring
METRICS_ENDPOINT=https://your-monitoring-endpoint.com/metrics
ALERT_EMAIL=alerts@your-company.com
```

### 3. GitHub Apps and Integrations

#### Install Required GitHub Apps
1. **Dependabot** (for automated dependency updates)
   - Navigate to **Settings → Code security and analysis**
   - Enable **Dependabot alerts**
   - Enable **Dependabot security updates**
   - Enable **Dependabot version updates**

2. **CodeQL** (for security analysis)
   - Navigate to **Settings → Code security and analysis**
   - Enable **Code scanning**
   - Set up **CodeQL analysis**

3. **Codecov** (for test coverage reporting)
   - Install from [GitHub Marketplace](https://github.com/marketplace/codecov)
   - Configure with `CODECOV_TOKEN`

#### Optional GitHub Apps
- **Snyk** (additional security scanning)
- **DeepCode** (AI-powered code review)
- **GitKraken Glo Boards** (project management)

### 4. External Service Configuration

#### Quantum Platform Setup

**IBM Quantum**
1. Create account at [IBM Quantum Experience](https://quantum-computing.ibm.com/)
2. Generate API token in Account settings
3. Add token to GitHub secrets as `IBMQ_TOKEN`

**AWS Braket**
1. Enable AWS Braket in your AWS account
2. Create IAM user with Braket permissions
3. Add credentials to GitHub secrets

**Google Quantum AI**
1. Enable Quantum AI API in Google Cloud Console
2. Create service account with appropriate permissions
3. Download service account JSON and add to secrets

#### Monitoring Stack Setup

**Grafana Cloud** (Recommended)
1. Sign up for [Grafana Cloud](https://grafana.com/products/cloud/)
2. Create API key for CI/CD integration
3. Import dashboards from `monitoring/grafana/dashboards/`

**Self-Hosted Monitoring**
```bash
# Deploy monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Import Grafana dashboards
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboards/quantum-dashboard.json
```

### 5. Team and Access Management

#### Create GitHub Teams
1. **@quantum-devops-team** (Admin access)
2. **@development-team** (Write access)
3. **@quantum-team** (Write access to quantum components)
4. **@devops-team** (Admin access to infrastructure)
5. **@security-team** (Admin access to security components)
6. **@documentation-team** (Write access to docs)

#### Assign Team Members
Update `CODEOWNERS` file with actual GitHub usernames:
```bash
# Replace placeholder usernames with real ones
sed -i 's/@alice-dev/@actual-username/g' CODEOWNERS
sed -i 's/@quantum-alice/@actual-quantum-dev/g' CODEOWNERS
# ... continue for all placeholder usernames
```

### 6. Environment Setup

#### Staging Environment
1. Deploy staging infrastructure
2. Configure DNS records
3. Set up SSL certificates
4. Configure monitoring endpoints

#### Production Environment
1. Deploy production infrastructure
2. Configure load balancers
3. Set up backup systems
4. Configure alerting rules

### 7. Documentation and Communication

#### Update Repository Description
Set repository description to:
```
GitHub Actions templates to bring CI/CD discipline to Qiskit & Cirq workflows
```

#### Set Homepage URL
```
https://quantum-devops-ci.readthedocs.io
```

#### Configure Discussions
1. Enable **Discussions** in repository settings
2. Create discussion categories:
   - General
   - Quantum Computing
   - DevOps Best Practices
   - Feature Requests
   - Q&A

#### Set up Status Page
1. Create status page (e.g., using StatusPage.io)
2. Configure incident management
3. Set up automated updates from monitoring

### 8. Security Configuration

#### Enable Security Features
1. **Private vulnerability reporting**
2. **Dependency graph**
3. **Dependabot alerts**
4. **Code scanning alerts**
5. **Secret scanning alerts**

#### Configure Security Policy
1. Review and update `SECURITY.md`
2. Set up security contact email
3. Configure vulnerability disclosure process

### 9. Compliance and Legal

#### License Configuration
1. Verify MIT license compatibility
2. Add license headers to source files
3. Update copyright notices

#### Terms of Service
1. Create terms of service (if applicable)
2. Add privacy policy (if collecting data)
3. Configure GDPR compliance (if applicable)

### 10. Performance Optimization

#### CDN Configuration
1. Set up CDN for documentation
2. Configure caching policies
3. Optimize asset delivery

#### Database Optimization
1. Configure database connection pooling
2. Set up read replicas (if needed)
3. Configure backup retention policies

## Verification Steps

After completing the manual setup, verify the configuration:

### 1. Test CI/CD Pipeline
```bash
# Create a test branch and PR
git checkout -b test-setup
echo "# Test" >> TEST.md
git add TEST.md
git commit -m "test: verify CI/CD setup"
git push origin test-setup

# Create PR and verify all checks pass
gh pr create --title "Test: Verify Setup" --body "Testing CI/CD configuration"
```

### 2. Test Security Scanning
```bash
# Trigger security scan
gh workflow run security-scan.yml

# Check for vulnerabilities
npm audit
python -m safety check
```

### 3. Test Quantum Workflows
```bash
# Run quantum tests
npm run test:quantum

# Test quantum linting
npm run quantum-lint
```

### 4. Test Monitoring
```bash
# Send test metrics
python scripts/automation/metrics-collection.py

# Verify dashboards are populated
open http://localhost:3000  # or your Grafana URL
```

### 5. Test Deployment
```bash
# Deploy to staging
gh workflow run cd.yml --ref main

# Verify deployment health
curl -f https://staging.your-domain.com/health
```

## Troubleshooting

### Common Issues

1. **GitHub Actions failing**
   - Check secrets are properly set
   - Verify workflow permissions
   - Review action logs for specific errors

2. **Quantum tests failing**
   - Verify quantum platform credentials
   - Check backend availability
   - Review quota limits

3. **Security scans failing**
   - Update vulnerable dependencies
   - Fix reported security issues
   - Verify scan tool configurations

4. **Monitoring not working**
   - Check endpoint configurations
   - Verify API keys and tokens
   - Review network connectivity

### Getting Help

1. **Documentation**: Check the `docs/` directory for detailed guides
2. **Issues**: Create GitHub issues for bugs and feature requests
3. **Discussions**: Use GitHub Discussions for questions and support
4. **Community**: Join the quantum computing DevOps community

## Next Steps

After completing the manual setup:

1. **Team Onboarding**: Train team members on the new workflow
2. **Process Documentation**: Document team-specific processes
3. **Continuous Improvement**: Regularly review and update configurations
4. **Community Engagement**: Share learnings with the quantum computing community

---

**Note**: This document should be updated as the project evolves and new requirements emerge. Consider it a living document that grows with your team's needs.