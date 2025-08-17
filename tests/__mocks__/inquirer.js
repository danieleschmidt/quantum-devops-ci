// Mock for inquirer module
module.exports = {
  prompt: jest.fn().mockResolvedValue({
    framework: 'qiskit',
    provider: 'ibmq',
    useGitHubActions: true
  })
}
