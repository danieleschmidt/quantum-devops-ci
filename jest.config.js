module.exports = {
  // Test environment
  testEnvironment: 'node',
  
  // Coverage configuration
  collectCoverage: true,
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    '!src/**/*.d.ts',
    '!src/**/*.test.{js,ts}',
    '!src/**/__tests__/**',
    '!src/cli.js' // CLI entry point
  ],
  coverageDirectory: 'coverage',
  coverageReporters: [
    'text',
    'text-summary',
    'lcov',
    'html',
    'json',
    'cobertura'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  
  // Test file patterns
  testMatch: [
    '**/tests/**/*.test.{js,ts}',
    '**/tests/**/*.spec.{js,ts}',
    '**/__tests__/**/*.{js,ts}'
  ],
  testPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/coverage/',
    '/quantum-tests/' // Python tests handled separately
  ],
  
  // Module resolution
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1',
    '^inquirer$': '<rootDir>/tests/__mocks__/inquirer.js'
  },
  
  // Setup files
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup.js'
  ],
  
  // Transform configuration
  transform: {
    '^.+\\.(js|ts)$': 'babel-jest'
  },
  transformIgnorePatterns: [
    '/node_modules/(?!(chalk|ora)/)'
  ],
  
  // Test timeout
  testTimeout: 30000,
  
  // Verbose output
  verbose: true,
  
  // Error handling
  bail: 0,
  errorOnDeprecated: true,
  
  // Performance
  maxWorkers: '50%',
  
  // Reporter configuration
  reporters: [
    'default'
  ],
  
  // Global configuration
  globals: {
    'ts-jest': {
      useESM: true
    }
  },
  
  // Clear mocks between tests
  clearMocks: true,
  restoreMocks: true,
  
  // Watch mode configuration
  watchPathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    '/coverage/',
    '/docs/',
    '\\.git'
  ],
  
  // Notification configuration (for local development)
  notify: false,
  notifyMode: 'failure-change'
};