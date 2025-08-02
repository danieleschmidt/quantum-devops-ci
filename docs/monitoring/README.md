# Monitoring & Observability

This directory contains monitoring and observability configuration and documentation for the quantum-devops-ci project.

## Overview

The monitoring stack provides comprehensive observability for:

- **Application Performance**: Track quantum circuit execution times, throughput, and resource usage
- **Infrastructure Health**: Monitor CI/CD pipelines, container health, and system resources
- **Quantum Hardware**: Track quantum backend availability, queue times, and execution costs
- **Security Events**: Monitor for security threats, failed authentications, and policy violations
- **Business Metrics**: Track usage patterns, cost optimization, and user engagement

## Components

### Health Checks
- [`health-checks.md`](health-checks.md) - Application health check endpoints
- [`monitoring-endpoints.md`](monitoring-endpoints.md) - Available monitoring endpoints

### Metrics & Dashboards
- [`metrics.md`](metrics.md) - Available metrics and their meanings
- [`dashboards.md`](dashboards.md) - Grafana dashboard configurations
- [`alerts.md`](alerts.md) - Alert rules and notification policies

### Logging
- [`logging.md`](logging.md) - Structured logging configuration
- [`log-aggregation.md`](log-aggregation.md) - Log collection and analysis

### Quantum-Specific Monitoring
- [`quantum-metrics.md`](quantum-metrics.md) - Quantum computing specific metrics
- [`hardware-monitoring.md`](hardware-monitoring.md) - Quantum hardware monitoring
- [`cost-tracking.md`](cost-tracking.md) - Quantum resource cost tracking

## Quick Start

### Local Development

```bash
# Start monitoring stack with Docker Compose
docker-compose -f docker-compose.monitoring.yml up -d

# Access monitoring services
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
open http://localhost:5601  # Kibana (if using ELK stack)
```

### Production Setup

Refer to the deployment guides:
- [Production Monitoring Setup](../deployment/monitoring-setup.md)
- [Cloud Provider Integrations](../deployment/cloud-monitoring.md)

## Configuration Files

| File | Purpose |
|------|----------|
| `prometheus.yml` | Prometheus scraping configuration |
| `grafana-datasources.yml` | Grafana data source configuration |
| `alertmanager.yml` | Alert routing and notification rules |
| `otel-collector.yml` | OpenTelemetry collector configuration |
| `fluent-bit.conf` | Log forwarding configuration |

## Metrics Overview

### Application Metrics
- **quantum_circuit_executions_total**: Total number of quantum circuits executed
- **quantum_circuit_execution_duration_seconds**: Histogram of circuit execution times
- **quantum_circuit_gate_count**: Number of gates in executed circuits
- **quantum_hardware_queue_time_seconds**: Time spent waiting in quantum hardware queues
- **quantum_cost_usd_total**: Total cost of quantum hardware usage

### Infrastructure Metrics
- **ci_pipeline_duration_seconds**: CI/CD pipeline execution times
- **ci_pipeline_success_rate**: Pipeline success rate
- **container_cpu_usage**: Container CPU utilization
- **container_memory_usage**: Container memory utilization
- **api_request_duration_seconds**: API request latency

### Security Metrics
- **security_scan_findings_total**: Number of security vulnerabilities found
- **failed_authentication_attempts_total**: Failed login attempts
- **secrets_scan_violations_total**: Detected secrets in code

## Alert Categories

### Critical Alerts (Immediate Response)
- Quantum hardware failures
- Security breaches
- Service outages
- Budget threshold exceeded

### Warning Alerts (Next Business Day)
- Performance degradation
- Increased error rates
- Resource utilization high
- Cost trending up

### Info Alerts (Weekly Review)
- Successful deployments
- Weekly cost reports
- Performance summaries

## Dashboards

### Executive Dashboard
- High-level metrics and KPIs
- Cost trends and budget tracking
- Service availability overview
- Weekly/monthly summaries

### Operations Dashboard
- Real-time system health
- CI/CD pipeline status
- Infrastructure metrics
- Alert status

### Developer Dashboard
- Code quality metrics
- Test coverage trends
- Performance benchmarks
- Error rates and debugging info

### Quantum Dashboard
- Quantum hardware status
- Circuit execution metrics
- Queue times and utilization
- Cost per circuit/experiment

## Best Practices

### Metric Naming
- Use snake_case for metric names
- Include units in metric names (e.g., `_seconds`, `_bytes`)
- Use consistent prefixes (`quantum_`, `ci_`, `security_`)
- Follow Prometheus naming conventions

### Alert Design
- Set appropriate thresholds based on historical data
- Avoid alert fatigue with proper routing
- Include runbook links in alert descriptions
- Test alert delivery regularly

### Dashboard Design
- Group related metrics together
- Use consistent time ranges
- Include context and documentation
- Optimize for different screen sizes

### Log Management
- Use structured logging (JSON format)
- Include correlation IDs for tracing
- Set appropriate log retention policies
- Sanitize sensitive information

## Troubleshooting

### Common Issues
- [Metrics not appearing](troubleshooting/metrics-missing.md)
- [High cardinality metrics](troubleshooting/high-cardinality.md)
- [Alert delivery problems](troubleshooting/alert-delivery.md)
- [Dashboard loading issues](troubleshooting/dashboard-issues.md)

### Performance Optimization
- [Query optimization](troubleshooting/query-optimization.md)
- [Storage management](troubleshooting/storage-management.md)
- [Resource scaling](troubleshooting/resource-scaling.md)

## Integration Guides

### Cloud Providers
- [AWS CloudWatch](integrations/aws-cloudwatch.md)
- [Google Cloud Monitoring](integrations/gcp-monitoring.md)
- [Azure Monitor](integrations/azure-monitor.md)

### Third-Party Services
- [Datadog](integrations/datadog.md)
- [New Relic](integrations/newrelic.md)
- [Splunk](integrations/splunk.md)

### Quantum Platforms
- [IBM Quantum](integrations/ibm-quantum.md)
- [AWS Braket](integrations/aws-braket.md)
- [Google Quantum AI](integrations/google-quantum.md)

## Security Considerations

- Monitor access to quantum resources
- Track API key usage and rotation
- Alert on unusual access patterns
- Ensure monitoring data privacy
- Secure dashboard access with authentication

## Cost Management

- Track monitoring infrastructure costs
- Optimize metric retention policies
- Use sampling for high-volume metrics
- Regular cleanup of unused dashboards
- Monitor quantum hardware spend

## Support

For monitoring-related issues:
1. Check the [troubleshooting guides](troubleshooting/)
2. Review the [runbooks](../runbooks/)
3. Contact the DevOps team
4. Create an issue with monitoring logs