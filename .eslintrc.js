module.exports = {
  extends: ['standard'],
  env: {
    node: true,
    es2022: true,
    jest: true
  },
  rules: {
    'no-console': 'off', // Allow console statements in CLI tool
    'space-before-function-paren': ['error', 'never']
  },
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module'
  }
}