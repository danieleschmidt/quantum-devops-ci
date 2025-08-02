/**
 * Unit tests for CLI functionality
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import fs from 'fs-extra';
import path from 'path';

// Mock modules before importing CLI
jest.mock('fs-extra');
jest.mock('inquirer');
jest.mock('child_process');

describe('CLI Core Functionality', () => {
  let mockTempDir;

  beforeEach(async () => {
    // Create a mock temporary directory
    mockTempDir = await global.testUtils.createTempDir();
    
    // Reset all mocks
    jest.clearAllMocks();
    
    // Mock fs-extra methods
    fs.pathExists.mockResolvedValue(true);
    fs.readJSON.mockResolvedValue({});
    fs.writeJSON.mockResolvedValue();
    fs.copy.mockResolvedValue();
    fs.ensureDir.mockResolvedValue();
  });

  afterEach(async () => {
    // Clean up temp directory
    if (mockTempDir) {
      await global.testUtils.cleanupTempDir(mockTempDir);
    }
  });

  describe('Command Parsing', () => {
    it('should parse init command correctly', async () => {
      const { parseCommand } = await import('../../src/cli.js');
      
      const result = parseCommand(['node', 'cli.js', 'init', '--framework', 'qiskit']);
      
      expect(result).toEqual({
        command: 'init',
        options: {
          framework: 'qiskit'
        }
      });
    });

    it('should handle invalid commands gracefully', async () => {
      const { parseCommand } = await import('../../src/cli.js');
      
      expect(() => {
        parseCommand(['node', 'cli.js', 'invalid-command']);
      }).toThrow('Unknown command: invalid-command');
    });
  });

  describe('Project Initialization', () => {
    it('should create quantum config file', async () => {
      const { initProject } = await import('../../src/cli.js');
      
      await initProject({
        framework: 'qiskit',
        provider: 'ibmq',
        outputDir: mockTempDir
      });

      expect(fs.writeJSON).toHaveBeenCalledWith(
        path.join(mockTempDir, 'quantum.config.yml'),
        expect.objectContaining({
          framework: 'qiskit',
          provider: 'ibmq'
        }),
        { spaces: 2 }
      );
    });

    it('should copy template files', async () => {
      const { initProject } = await import('../../src/cli.js');
      
      await initProject({
        framework: 'qiskit',
        templates: ['github-actions', 'testing'],
        outputDir: mockTempDir
      });

      expect(fs.copy).toHaveBeenCalledWith(
        expect.stringContaining('templates/github-actions'),
        expect.stringContaining(mockTempDir)
      );
      expect(fs.copy).toHaveBeenCalledWith(
        expect.stringContaining('templates/testing'),
        expect.stringContaining(mockTempDir)
      );
    });

    it('should handle initialization errors gracefully', async () => {
      const { initProject } = await import('../../src/cli.js');
      
      // Mock fs error
      fs.writeJSON.mockRejectedValue(new Error('Permission denied'));
      
      await expect(initProject({
        framework: 'qiskit',
        outputDir: '/invalid/path'
      })).rejects.toThrow('Permission denied');
    });
  });

  describe('Configuration Management', () => {
    it('should validate quantum configuration', async () => {
      const { validateConfig } = await import('../../src/cli.js');
      
      const validConfig = {
        framework: 'qiskit',
        provider: 'ibmq',
        testing: {
          default_shots: 1000,
          noise_simulation: true
        }
      };

      const result = validateConfig(validConfig);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should detect invalid configuration', async () => {
      const { validateConfig } = await import('../../src/cli.js');
      
      const invalidConfig = {
        framework: 'invalid-framework',
        testing: {
          default_shots: 'not-a-number'
        }
      };

      const result = validateConfig(invalidConfig);
      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });
  });

  describe('Template Processing', () => {
    it('should process template variables correctly', async () => {
      const { processTemplate } = await import('../../src/cli.js');
      
      const template = 'framework: {{framework}}\nshots: {{shots}}';
      const variables = {
        framework: 'qiskit',
        shots: 1000
      };

      const result = processTemplate(template, variables);
      expect(result).toBe('framework: qiskit\nshots: 1000');
    });

    it('should handle missing template variables', async () => {
      const { processTemplate } = await import('../../src/cli.js');
      
      const template = 'framework: {{framework}}\nshots: {{missing_var}}';
      const variables = {
        framework: 'qiskit'
      };

      expect(() => {
        processTemplate(template, variables);
      }).toThrow('Missing template variable: missing_var');
    });
  });

  describe('Interactive Prompts', () => {
    it('should handle user input for framework selection', async () => {
      const inquirer = await import('inquirer');
      inquirer.prompt.mockResolvedValue({
        framework: 'qiskit',
        provider: 'ibmq',
        features: ['testing', 'linting']
      });

      const { interactiveInit } = await import('../../src/cli.js');
      
      const result = await interactiveInit();
      
      expect(result).toEqual({
        framework: 'qiskit',
        provider: 'ibmq',
        features: ['testing', 'linting']
      });
    });

    it('should validate user input', async () => {
      const inquirer = await import('inquirer');
      inquirer.prompt.mockResolvedValue({
        framework: 'invalid-framework'
      });

      const { interactiveInit } = await import('../../src/cli.js');
      
      await expect(interactiveInit()).rejects.toThrow('Invalid framework selection');
    });
  });

  describe('Error Handling', () => {
    it('should provide helpful error messages', async () => {
      const { CLIError } = await import('../../src/cli.js');
      
      const error = new CLIError('Configuration file not found', 'CONFIG_NOT_FOUND');
      
      expect(error.message).toBe('Configuration file not found');
      expect(error.code).toBe('CONFIG_NOT_FOUND');
      expect(error.name).toBe('CLIError');
    });

    it('should handle system errors gracefully', async () => {
      const { handleSystemError } = await import('../../src/cli.js');
      
      const systemError = new Error('ENOENT: no such file or directory');
      systemError.code = 'ENOENT';
      
      const result = handleSystemError(systemError);
      
      expect(result).toEqual({
        message: 'File or directory not found',
        code: 'ENOENT',
        suggestion: 'Please check the file path and try again'
      });
    });
  });
});