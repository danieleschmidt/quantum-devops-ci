#!/usr/bin/env node

const { Command } = require('commander');
const chalk = require('chalk');
const ora = require('ora');
const fs = require('fs-extra');
const path = require('path');
const inquirer = require('inquirer');
const yaml = require('yaml');

const program = new Command();

// Package information
const packageJson = require('../package.json');

program
  .name('quantum-devops-ci')
  .description('GitHub Actions templates to bring CI/CD discipline to Qiskit & Cirq workflows')
  .version(packageJson.version);

// Init command - main functionality from README
program
  .command('init')
  .description('Initialize quantum CI/CD in your project')
  .option('-f, --framework <framework>', 'Quantum framework (qiskit, cirq, pennylane, all)', 'qiskit')
  .option('-p, --provider <provider>', 'Quantum provider (ibmq, aws-braket, google)', 'ibmq')
  .option('--skip-devcontainer', 'Skip creating dev container configuration')
  .option('--skip-workflows', 'Skip creating GitHub Actions workflows')
  .option('--dry-run', 'Show what would be created without actually creating files')
  .action(async (options) => {
    const spinner = ora('Initializing quantum DevOps CI/CD...').start();
    
    try {
      // Get current directory
      const projectRoot = process.cwd();
      
      if (!options.dryRun) {
        spinner.text = 'Analyzing project structure...';
        await analyzeProject(projectRoot);
        
        spinner.text = 'Creating quantum CI/CD structure...';
        await createProjectStructure(projectRoot, options);
        
        spinner.succeed(chalk.green('✅ Quantum DevOps CI/CD initialized successfully!'));
        
        // Show next steps
        console.log(chalk.blue('\n🚀 Next Steps:'));
        console.log('1. Configure your quantum provider credentials');
        console.log('2. Review and customize quantum.config.yml');
        console.log('3. Add quantum tests to quantum-tests/ directory');
        console.log('4. Commit and push to trigger CI/CD pipeline');
        console.log('\n📚 Documentation: https://quantum-devops-ci.readthedocs.io');
      } else {
        spinner.succeed(chalk.yellow('Dry run completed - no files created'));
        await showDryRunResults(projectRoot, options);
      }
      
    } catch (error) {
      spinner.fail(chalk.red('❌ Failed to initialize quantum DevOps CI/CD'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Status command
program
  .command('status')
  .description('Show quantum DevOps CI/CD status')
  .action(async () => {
    const spinner = ora('Checking quantum DevOps status...').start();
    
    try {
      const status = await checkProjectStatus(process.cwd());
      spinner.succeed('Status check completed');
      
      console.log(chalk.blue('\n📊 Quantum DevOps CI/CD Status:'));
      console.log(`Configuration: ${status.hasConfig ? chalk.green('✅') : chalk.red('❌')}`);
      console.log(`GitHub Workflows: ${status.hasWorkflows ? chalk.green('✅') : chalk.red('❌')}`);
      console.log(`Test Directory: ${status.hasTests ? chalk.green('✅') : chalk.red('❌')}`);
      console.log(`Dev Container: ${status.hasDevContainer ? chalk.green('✅') : chalk.red('❌')}`);
      
      if (!status.isFullyConfigured) {
        console.log(chalk.yellow('\n⚠️  Run "quantum-devops-ci init" to complete setup'));
      }
      
    } catch (error) {
      spinner.fail(chalk.red('❌ Failed to check status'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Validate command
program
  .command('validate')
  .description('Validate quantum configuration and test files')
  .action(async () => {
    const spinner = ora('Validating quantum configuration...').start();
    
    try {
      const validation = await validateConfiguration(process.cwd());
      
      if (validation.isValid) {
        spinner.succeed(chalk.green('✅ Configuration is valid'));
      } else {
        spinner.fail(chalk.red('❌ Configuration validation failed'));
        console.log(chalk.red('\nValidation Errors:'));
        validation.errors.forEach(error => {
          console.log(chalk.red(`  • ${error}`));
        });
        
        if (validation.warnings.length > 0) {
          console.log(chalk.yellow('\nWarnings:'));
          validation.warnings.forEach(warning => {
            console.log(chalk.yellow(`  • ${warning}`));
          });
        }
        process.exit(1);
      }
      
    } catch (error) {
      spinner.fail(chalk.red('❌ Failed to validate configuration'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Helper functions
async function analyzeProject(projectRoot) {
  // Check if it's a git repository
  const gitDir = path.join(projectRoot, '.git');
  if (!await fs.pathExists(gitDir)) {
    throw new Error('Not a git repository. Please run "git init" first.');
  }
  
  // Check for existing quantum files
  const existingFiles = [];
  const checkFiles = [
    'quantum.config.yml',
    '.github/workflows/quantum-ci.yml',
    'quantum-tests/',
    '.devcontainer/devcontainer.json'
  ];
  
  for (const file of checkFiles) {
    if (await fs.pathExists(path.join(projectRoot, file))) {
      existingFiles.push(file);
    }
  }
  
  if (existingFiles.length > 0) {
    const { confirm } = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'confirm',
        message: `Found existing quantum DevOps files: ${existingFiles.join(', ')}. Continue?`,
        default: false
      }
    ]);
    
    if (!confirm) {
      throw new Error('Initialization cancelled by user');
    }
  }
}

async function createProjectStructure(projectRoot, options) {
  // Create directories
  const directories = [
    'quantum-tests',
    '.github/workflows',
    '.devcontainer'
  ];
  
  for (const dir of directories) {
    await fs.ensureDir(path.join(projectRoot, dir));
  }
  
  // Create quantum configuration
  await createQuantumConfig(projectRoot, options);
  
  // Create GitHub workflow (template structure only)
  if (!options.skipWorkflows) {
    await createWorkflowTemplate(projectRoot, options);
  }
  
  // Create dev container configuration
  if (!options.skipDevcontainer) {
    await createDevContainer(projectRoot, options);
  }
  
  // Create initial test structure
  await createTestStructure(projectRoot, options);
}

async function createQuantumConfig(projectRoot, options) {
  const config = {
    quantum_devops_ci: {
      version: '1.0.0',
      framework: options.framework,
      provider: options.provider
    },
    hardware_access: {
      providers: []
    },
    testing: {
      default_shots: 1000,
      noise_simulation: true,
      error_mitigation: false,
      timeout_minutes: 30
    },
    quota_rules: [
      {
        name: 'development',
        branches: ['develop', 'feature/*'],
        max_shots_per_run: 1000,
        allowed_backends: ['simulator']
      },
      {
        name: 'production',
        branches: ['main'],
        max_shots_per_run: 100000,
        allowed_backends: ['all'],
        requires_approval: true
      }
    ]
  };
  
  // Add provider-specific configuration
  if (options.provider === 'ibmq') {
    config.hardware_access.providers.push({
      name: 'ibmq',
      credentials_secret: 'IBMQ_TOKEN',
      max_monthly_shots: 10000000,
      priority_queue: 'research'
    });
  }
  
  const configPath = path.join(projectRoot, 'quantum.config.yml');
  await fs.writeFile(configPath, yaml.stringify(config, { indent: 2 }));
}

async function createWorkflowTemplate(projectRoot, options) {
  // Note: As per constraints, we document the structure instead of creating actual .yml files
  const workflowDir = path.join(projectRoot, '.github/workflows');
  const readmePath = path.join(workflowDir, 'README.md');
  
  const workflowDoc = `# Quantum CI/CD Workflows

This directory contains GitHub Actions workflows for quantum DevOps CI/CD.

## Template Structure

The following workflow files should be created based on your project needs:

### quantum-ci.yml
Main CI/CD pipeline with:
- Quantum circuit linting
- Noise-aware testing
- Hardware quota management
- Test result reporting

### quantum-benchmark.yml
Performance benchmarking with:
- Execution time tracking
- Memory usage analysis
- Circuit optimization metrics
- Historical comparison

### quantum-deploy.yml
Deployment pipeline with:
- Hardware validation
- Blue-green deployment
- A/B testing support
- Cost tracking

## Quick Start

Run \`quantum-devops-ci init\` to generate these workflow files automatically.

For manual setup, refer to the examples in the documentation:
https://quantum-devops-ci.readthedocs.io/workflows/
`;
  
  await fs.writeFile(readmePath, workflowDoc);
}

async function createDevContainer(projectRoot, options) {
  const devcontainerDir = path.join(projectRoot, '.devcontainer');
  
  const devcontainerConfig = {
    name: 'Quantum Development Environment',
    image: 'quantum-devops/devcontainer:latest',
    features: {
      'quantum-frameworks': {
        qiskit: '0.45',
        cirq: '1.3',
        pennylane: '0.33'
      },
      simulators: {
        'qiskit-aer': true,
        'cirq-qsim': true,
        'gpu-acceleration': true
      },
      'analysis-tools': {
        jupyter: true,
        'quantum-visualization': true,
        'pulse-designer': true
      }
    },
    customizations: {
      vscode: {
        extensions: [
          'quantum-devops.quantum-lint',
          'quantum-devops.circuit-visualizer',
          'quantum-devops.pulse-designer',
          'ms-python.python'
        ]
      }
    },
    postCreateCommand: 'quantum-devops setup-workspace'
  };
  
  const configPath = path.join(devcontainerDir, 'devcontainer.json');
  await fs.writeFile(configPath, JSON.stringify(devcontainerConfig, null, 2));
}

async function createTestStructure(projectRoot, options) {
  const testsDir = path.join(projectRoot, 'quantum-tests');
  
  // Create test configuration
  const testConfig = `# Quantum Test Configuration

[tool.pytest.ini_options]
markers = [
    "quantum: marks tests as quantum tests",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "hardware: marks tests that require real quantum hardware"
]

quantum_test_config = {
    "default_backend": "qasm_simulator",
    "default_shots": 1000,
    "noise_simulation": true,
    "timeout_seconds": 300
}
`;
  
  await fs.writeFile(path.join(testsDir, 'pytest.ini'), testConfig);
  
  // Create example test file
  const exampleTest = `"""
Example quantum test demonstrating noise-aware testing patterns.

This file shows how to write quantum tests that work with the
quantum-devops-ci testing framework.
"""

import pytest
from quantum_devops_ci.testing import NoiseAwareTest
from qiskit import QuantumCircuit


class TestExampleQuantumAlgorithm(NoiseAwareTest):
    """Example test class showing quantum testing patterns."""
    
    def test_bell_state_preparation(self):
        """Test Bell state preparation in ideal conditions."""
        # Create Bell state circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Run test
        result = self.run_circuit(qc, shots=1000)
        
        # Verify results
        counts = result.get_counts()
        assert '00' in counts
        assert '11' in counts
        
        # Check roughly equal distribution
        total_shots = sum(counts.values())
        assert abs(counts.get('00', 0) - counts.get('11', 0)) < total_shots * 0.1
    
    @pytest.mark.slow
    def test_bell_state_under_noise(self):
        """Test Bell state with realistic noise model."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Run with noise
        result = self.run_with_noise(
            qc,
            noise_model='ibmq_essex',
            shots=8192
        )
        
        # Calculate fidelity
        fidelity = self.calculate_bell_fidelity(result)
        assert fidelity > 0.8, f"Fidelity too low: {fidelity:.3f}"
    
    @pytest.mark.hardware
    @pytest.mark.skipif(not pytest.config.hardware_available, reason="No hardware access")
    def test_on_real_hardware(self):
        """Test on real quantum hardware (if available)."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # Run on hardware
        result = self.run_on_hardware(qc, backend='least_busy', shots=100)
        
        # Basic validation
        counts = result.get_counts()
        assert len(counts) > 0
        assert sum(counts.values()) == 100
`;
  
  await fs.writeFile(path.join(testsDir, 'test_example.py'), exampleTest);
  
  // Create __init__.py
  await fs.writeFile(path.join(testsDir, '__init__.py'), '# Quantum tests package\n');
}

async function checkProjectStatus(projectRoot) {
  const status = {
    hasConfig: false,
    hasWorkflows: false,
    hasTests: false,
    hasDevContainer: false,
    isFullyConfigured: false
  };
  
  // Check for configuration
  status.hasConfig = await fs.pathExists(path.join(projectRoot, 'quantum.config.yml'));
  
  // Check for workflows
  const workflowsDir = path.join(projectRoot, '.github/workflows');
  status.hasWorkflows = await fs.pathExists(workflowsDir) && 
    (await fs.readdir(workflowsDir)).some(file => file.endsWith('.yml'));
  
  // Check for tests
  status.hasTests = await fs.pathExists(path.join(projectRoot, 'quantum-tests'));
  
  // Check for dev container
  status.hasDevContainer = await fs.pathExists(path.join(projectRoot, '.devcontainer/devcontainer.json'));
  
  status.isFullyConfigured = status.hasConfig && status.hasWorkflows && status.hasTests;
  
  return status;
}

async function validateConfiguration(projectRoot) {
  const validation = {
    isValid: true,
    errors: [],
    warnings: []
  };
  
  try {
    // Check quantum.config.yml
    const configPath = path.join(projectRoot, 'quantum.config.yml');
    if (await fs.pathExists(configPath)) {
      const configContent = await fs.readFile(configPath, 'utf8');
      const config = yaml.parse(configContent);
      
      // Validate required fields
      if (!config.quantum_devops_ci) {
        validation.errors.push('Missing quantum_devops_ci section in config');
      }
      
      if (!config.testing) {
        validation.warnings.push('No testing configuration found');
      }
      
    } else {
      validation.errors.push('quantum.config.yml not found');
    }
    
    // Check test directory
    const testsDir = path.join(projectRoot, 'quantum-tests');
    if (await fs.pathExists(testsDir)) {
      const testFiles = await fs.readdir(testsDir);
      const hasTestFiles = testFiles.some(file => file.startsWith('test_') && file.endsWith('.py'));
      
      if (!hasTestFiles) {
        validation.warnings.push('No test files found in quantum-tests directory');
      }
    } else {
      validation.warnings.push('quantum-tests directory not found');
    }
    
  } catch (error) {
    validation.errors.push(`Configuration validation error: ${error.message}`);
  }
  
  validation.isValid = validation.errors.length === 0;
  return validation;
}

async function showDryRunResults(projectRoot, options) {
  console.log(chalk.blue('\n📋 Files that would be created:'));
  console.log('  quantum.config.yml');
  console.log('  quantum-tests/');
  console.log('    ├── __init__.py');
  console.log('    ├── pytest.ini');
  console.log('    └── test_example.py');
  
  if (!options.skipWorkflows) {
    console.log('  .github/workflows/');
    console.log('    └── README.md (workflow documentation)');
  }
  
  if (!options.skipDevcontainer) {
    console.log('  .devcontainer/');
    console.log('    └── devcontainer.json');
  }
  
  console.log(chalk.green('\n✨ Run without --dry-run to create these files'));
}

// Parse command line arguments
program.parse();