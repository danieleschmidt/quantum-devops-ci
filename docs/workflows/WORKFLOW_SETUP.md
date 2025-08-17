# Quantum DevOps CI - Workflow Templates

> 📋 **GitHub Actions and CI/CD Workflow Templates for Quantum Computing Projects**

## 🚀 Quick Setup

Due to GitHub App permissions, workflow files need to be manually copied to your `.github/workflows/` directory. 

### Manual Installation Steps

1. **Copy the workflow files**:
   ```bash
   # Create workflows directory
   mkdir -p .github/workflows
   
   # Copy templates
   cp docs/workflows/examples/quantum-ci.yml .github/workflows/
   cp docs/workflows/examples/quantum-deployment.yml .github/workflows/
   ```

2. **Customize for your project**:
   ```bash
   # Edit workflow files to match your project structure
   nano .github/workflows/quantum-ci.yml
   nano .github/workflows/quantum-deployment.yml
   ```

3. **Set up secrets** (in GitHub repository settings):
   ```
   IBMQ_TOKEN - Your IBM Quantum token
   AWS_CREDENTIALS - AWS credentials for Braket
   NPM_TOKEN - NPM publishing token (if needed)
   PYPI_TOKEN - PyPI publishing token (if needed)
   ```

## 📁 Available Workflows

### 1. `quantum-ci.yml` - Main CI/CD Pipeline
- **Purpose**: Continuous integration for quantum projects
- **Features**: 
  - Multi-framework testing (Qiskit, Cirq, PennyLane)
  - Security scanning
  - Performance benchmarks
  - Coverage reporting
- **Triggers**: Push, Pull Request
- **Matrix builds**: Python 3.8-3.11, multiple frameworks

### 2. `quantum-deployment.yml` - Production Deployment
- **Purpose**: Automated deployment to NPM and PyPI
- **Features**:
  - Package publishing
  - Hardware validation tests
  - Documentation deployment
  - Release automation
- **Triggers**: Release creation, manual dispatch
- **Environments**: Staging, Production

---

**⚠️ Important**: Due to GitHub App permissions, these workflow files must be manually copied to your `.github/workflows/` directory. This is a security feature to prevent unauthorized workflow modifications.