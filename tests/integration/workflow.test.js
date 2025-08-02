/**
 * Integration tests for complete workflow scenarios
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from '@jest/globals';
import fs from 'fs-extra';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('End-to-End Workflow Integration', () => {
  let testWorkspace;
  let originalCwd;

  beforeAll(async () => {
    // Save original working directory
    originalCwd = process.cwd();
    
    // Create test workspace
    testWorkspace = await global.testUtils.createTempDir();
    process.chdir(testWorkspace);
  });

  afterAll(async () => {
    // Restore original working directory
    process.chdir(originalCwd);
    
    // Clean up test workspace
    if (testWorkspace) {
      await global.testUtils.cleanupTempDir(testWorkspace);
    }
  });

  beforeEach(async () => {
    // Ensure clean workspace for each test
    const files = await fs.readdir(testWorkspace);
    for (const file of files) {
      await fs.remove(path.join(testWorkspace, file));
    }
  });

  describe('Project Initialization Workflow', () => {
    it('should initialize a complete quantum project', async () => {
      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Run initialization command
      const initResult = await runCliCommand([
        'node', cliPath, 'init',
        '--framework', 'qiskit',
        '--provider', 'ibmq',
        '--templates', 'testing,github-actions',
        '--non-interactive'
      ]);

      expect(initResult.exitCode).toBe(0);
      expect(initResult.stdout).toContain('Project initialized successfully');

      // Verify created files
      expect(await fs.pathExists('quantum.config.yml')).toBe(true);
      expect(await fs.pathExists('.github/workflows/quantum-ci.yml')).toBe(true);
      expect(await fs.pathExists('quantum-tests/conftest.py')).toBe(true);
      expect(await fs.pathExists('quantum-tests/examples/test_basic.py')).toBe(true);

      // Verify configuration content
      const config = await fs.readJSON('quantum.config.yml');
      expect(config.framework).toBe('qiskit');
      expect(config.provider).toBe('ibmq');
    });

    it('should handle initialization with custom configuration', async () => {
      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Create custom config
      const customConfig = {
        framework: 'cirq',
        provider: 'google-quantum',
        testing: {
          default_shots: 2000,
          noise_simulation: false
        }
      };
      
      await fs.writeJSON('custom-config.yml', customConfig);

      // Run initialization with custom config
      const initResult = await runCliCommand([
        'node', cliPath, 'init',
        '--config', 'custom-config.yml',
        '--non-interactive'
      ]);

      expect(initResult.exitCode).toBe(0);

      // Verify configuration was applied
      const finalConfig = await fs.readJSON('quantum.config.yml');
      expect(finalConfig.framework).toBe('cirq');
      expect(finalConfig.provider).toBe('google-quantum');
      expect(finalConfig.testing.default_shots).toBe(2000);
    });
  });

  describe('Testing Workflow', () => {
    beforeEach(async () => {
      // Set up a basic quantum project
      await setupBasicProject();
    });

    it('should run quantum tests successfully', async () => {
      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Run quantum tests
      const testResult = await runCliCommand([
        'node', cliPath, 'test',
        '--framework', 'qiskit',
        '--backend', 'qasm_simulator',
        '--shots', '100'
      ]);

      expect(testResult.exitCode).toBe(0);
      expect(testResult.stdout).toContain('Tests passed');
      expect(testResult.stdout).toContain('quantum-tests/examples/test_basic.py');
    });

    it('should generate test reports', async () => {
      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Run tests with reporting
      const testResult = await runCliCommand([
        'node', cliPath, 'test',
        '--framework', 'qiskit',
        '--coverage',
        '--junit-xml', 'test-results.xml',
        '--html-report', 'test-report.html'
      ]);

      expect(testResult.exitCode).toBe(0);
      expect(await fs.pathExists('test-results.xml')).toBe(true);
      expect(await fs.pathExists('test-report.html')).toBe(true);

      // Verify report content
      const xmlContent = await fs.readFile('test-results.xml', 'utf8');
      expect(xmlContent).toContain('testsuite');
      expect(xmlContent).toContain('testcase');
    });

    it('should handle test failures gracefully', async () => {
      // Create a failing test
      const failingTest = `
import pytest
from quantum_devops_ci.testing import NoiseAwareTest

class TestFailingQuantum(NoiseAwareTest):
    def test_failing_case(self):
        assert False, "This test should fail"
      `;
      
      await fs.ensureDir('quantum-tests/failing');
      await fs.writeFile('quantum-tests/failing/test_fail.py', failingTest);

      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Run tests (should fail)
      const testResult = await runCliCommand([
        'node', cliPath, 'test',
        '--framework', 'qiskit'
      ]);

      expect(testResult.exitCode).toBe(1);
      expect(testResult.stdout).toContain('Tests failed');
      expect(testResult.stdout).toContain('This test should fail');
    });
  });

  describe('Linting Workflow', () => {
    beforeEach(async () => {
      await setupBasicProject();
    });

    it('should lint quantum circuits successfully', async () => {
      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Run linting
      const lintResult = await runCliCommand([
        'node', cliPath, 'lint',
        '--check-circuits',
        '--check-gates',
        '--max-depth', '50'
      ]);

      expect(lintResult.exitCode).toBe(0);
      expect(lintResult.stdout).toContain('Linting completed');
      expect(lintResult.stdout).toContain('No issues found');
    });

    it('should detect circuit issues', async () => {
      // Create a circuit with issues
      const problematicCircuit = `
from qiskit import QuantumCircuit

# Create a circuit with excessive depth
qc = QuantumCircuit(2)
for i in range(200):  # Excessive depth
    qc.h(0)
    qc.cx(0, 1)
      `;
      
      await fs.ensureDir('src/circuits');
      await fs.writeFile('src/circuits/problematic.py', problematicCircuit);

      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Run linting (should find issues)
      const lintResult = await runCliCommand([
        'node', cliPath, 'lint',
        '--check-circuits',
        '--max-depth', '50'
      ]);

      expect(lintResult.exitCode).toBe(1);
      expect(lintResult.stdout).toContain('Circuit depth exceeds maximum');
    });
  });

  describe('Deployment Workflow', () => {
    beforeEach(async () => {
      await setupBasicProject();
    });

    it('should prepare deployment package', async () => {
      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Run deployment preparation
      const deployResult = await runCliCommand([
        'node', cliPath, 'deploy',
        '--target', 'staging',
        '--backend', 'ibmq_qasm_simulator',
        '--dry-run'
      ]);

      expect(deployResult.exitCode).toBe(0);
      expect(deployResult.stdout).toContain('Deployment package prepared');
      expect(await fs.pathExists('deployment-package.zip')).toBe(true);
    });

    it('should validate deployment configuration', async () => {
      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Run deployment validation
      const validateResult = await runCliCommand([
        'node', cliPath, 'deploy',
        '--validate-only',
        '--target', 'production'
      ]);

      expect(validateResult.exitCode).toBe(0);
      expect(validateResult.stdout).toContain('Deployment configuration valid');
    });
  });

  describe('Configuration Workflow', () => {
    it('should manage provider configurations', async () => {
      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Add provider configuration
      const addResult = await runCliCommand([
        'node', cliPath, 'config',
        'add-provider',
        'ibmq',
        '--token', 'test-token-123',
        '--hub', 'ibm-q',
        '--group', 'open'
      ]);

      expect(addResult.exitCode).toBe(0);
      expect(addResult.stdout).toContain('Provider configuration added');

      // List providers
      const listResult = await runCliCommand([
        'node', cliPath, 'config',
        'list-providers'
      ]);

      expect(listResult.exitCode).toBe(0);
      expect(listResult.stdout).toContain('ibmq');
    });

    it('should validate configurations', async () => {
      const cliPath = path.join(__dirname, '../../src/cli.js');
      
      // Create invalid configuration
      await fs.writeJSON('quantum.config.yml', {
        framework: 'invalid-framework',
        testing: {
          default_shots: 'not-a-number'
        }
      });

      // Validate configuration
      const validateResult = await runCliCommand([
        'node', cliPath, 'config',
        'validate'
      ]);

      expect(validateResult.exitCode).toBe(1);
      expect(validateResult.stdout).toContain('Configuration validation failed');
    });
  });

  // Helper functions
  async function setupBasicProject() {
    // Create basic project structure
    await fs.ensureDir('quantum-tests/examples');
    await fs.ensureDir('src/quantum_devops_ci');
    
    // Create basic config
    const config = {
      framework: 'qiskit',
      provider: 'ibmq',
      testing: {
        default_shots: 1000,
        noise_simulation: true
      }
    };
    await fs.writeJSON('quantum.config.yml', config);

    // Create basic test
    const basicTest = `
import pytest
from quantum_devops_ci.testing import NoiseAwareTest
from qiskit import QuantumCircuit

class TestBasicQuantum(NoiseAwareTest):
    def test_simple_circuit(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        result = self.run_circuit(qc, shots=100, backend='qasm_simulator')
        assert 'counts' in result
        assert sum(result['counts'].values()) == 100
    `;
    
    await fs.writeFile('quantum-tests/examples/test_basic.py', basicTest);
  }

  async function runCliCommand(args, options = {}) {
    return new Promise((resolve, reject) => {
      const child = spawn(args[0], args.slice(1), {
        cwd: testWorkspace,
        env: { ...process.env, NODE_ENV: 'test' },
        ...options
      });

      let stdout = '';
      let stderr = '';

      child.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('close', (exitCode) => {
        resolve({
          exitCode,
          stdout,
          stderr
        });
      });

      child.on('error', (error) => {
        reject(error);
      });

      // Set timeout for long-running commands
      setTimeout(() => {
        child.kill('SIGTERM');
        reject(new Error('Command timeout'));
      }, 30000);
    });
  }
});