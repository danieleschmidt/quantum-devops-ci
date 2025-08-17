/**
 * Jest setup file for quantum-devops-ci JavaScript tests
 * This file is automatically loaded before each test suite
 */

// Set up global test environment
// Note: jest is globally available in Jest test environment

// Configure test timeouts
jest.setTimeout(30000) // 30 seconds default timeout

// Set up test environment variables
process.env.NODE_ENV = 'test'
process.env.QUANTUM_DEVOPS_ENV = 'test'
process.env.DEBUG = 'quantum-devops-ci:test'

// Mock console methods for cleaner test output
const originalConsole = { ...console }

beforeEach(() => {
  // Reset console mocks before each test
  console.log = jest.fn()
  console.info = jest.fn()
  console.warn = jest.fn()
  console.error = originalConsole.error // Keep error for debugging
})

afterEach(() => {
  // Restore console after each test
  console.log = originalConsole.log
  console.info = originalConsole.info
  console.warn = originalConsole.warn
  console.error = originalConsole.error
})

// Global test utilities
global.testUtils = {
  // Create a temporary directory for test files
  createTempDir: async() => {
    const fs = require('fs-extra')
    const path = require('path')
    const os = require('os')

    return fs.mkdtemp(path.join(os.tmpdir(), 'quantum-devops-test-'))
  },

  // Clean up temporary directory
  cleanupTempDir: async(dir) => {
    const fs = require('fs-extra')
    if (await fs.pathExists(dir)) {
      await fs.remove(dir)
    }
  },

  // Mock quantum provider responses
  createMockQuantumResponse: (counts = { '00': 500, 11: 500 }) => ({
    job_id: 'test-job-123',
    status: 'completed',
    results: {
      counts,
      success: true,
      shots: Object.values(counts).reduce((sum, count) => sum + count, 0)
    },
    metadata: {
      backend: 'qasm_simulator',
      execution_time: 0.1,
      cost: 0.0
    }
  }),

  // Mock CLI command execution
  mockCliExecution: (command, exitCode = 0, stdout = '', stderr = '') => {
    const mockSpawn = jest.fn().mockReturnValue({
      stdout: { on: jest.fn((event, cb) => event === 'data' && cb(stdout)) },
      stderr: { on: jest.fn((event, cb) => event === 'data' && cb(stderr)) },
      on: jest.fn((event, cb) => event === 'close' && cb(exitCode))
    })

    jest.doMock('child_process', () => ({
      spawn: mockSpawn
    }))

    return mockSpawn
  },

  // Wait for a condition to be true
  waitFor: async(condition, timeout = 5000, interval = 100) => {
    const start = Date.now()
    while (Date.now() - start < timeout) {
      if (await condition()) {
        return true
      }
      await new Promise(resolve => setTimeout(resolve, interval))
    }
    throw new Error(`Condition not met within ${timeout}ms`)
  }
}

// Global test matchers
expect.extend({
  // Custom matcher for quantum results
  toHaveValidQuantumResult(received) {
    const pass = (
      received &&
      typeof received === 'object' &&
      'counts' in received &&
      'shots' in received &&
      typeof received.counts === 'object'
    )

    if (pass) {
      return {
        message: () => `expected ${JSON.stringify(received)} not to be a valid quantum result`,
        pass: true
      }
    } else {
      return {
        message: () => `expected ${JSON.stringify(received)} to be a valid quantum result with counts and shots`,
        pass: false
      }
    }
  },

  // Custom matcher for command line output
  toHaveValidCliOutput(received) {
    const pass = (
      received &&
      typeof received === 'object' &&
      'exitCode' in received &&
      'stdout' in received &&
      'stderr' in received
    )

    if (pass) {
      return {
        message: () => `expected ${JSON.stringify(received)} not to be valid CLI output`,
        pass: true
      }
    } else {
      return {
        message: () => `expected ${JSON.stringify(received)} to have exitCode, stdout, and stderr properties`,
        pass: false
      }
    }
  }
})

// Handle unhandled promise rejections in tests
process.on('unhandledRejection', (reason, _promise) => {
  console.error('Unhandled promise rejection in test:', reason)
  throw reason
})

// Clean up after all tests
afterAll(async() => {
  // Clean up any global resources
  jest.clearAllMocks()
  jest.restoreAllMocks()
})
