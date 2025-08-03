"""
API request/response schemas for quantum DevOps operations.

Defines validation schemas for all API endpoints.
"""

from typing import Dict, Any


# Job Management Schemas
JobSchema = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'name': {'type': 'string'},
        'circuit': {'type': 'string'},
        'shots': {'type': 'integer', 'minimum': 1, 'maximum': 1000000},
        'priority': {'type': 'string', 'enum': ['low', 'medium', 'high', 'critical']},
        'backend_preferences': {
            'type': 'array',
            'items': {'type': 'string'}
        },
        'max_cost': {'type': 'number', 'minimum': 0},
        'deadline': {'type': 'string', 'format': 'datetime'},
        'parameters': {'type': 'object'},
        'metadata': {'type': 'object'}
    },
    'required': ['circuit', 'shots']
}

BatchJobSchema = {
    'type': 'object',
    'properties': {
        'jobs': {
            'type': 'array',
            'items': JobSchema,
            'minItems': 1,
            'maxItems': 100
        },
        'scheduling_constraints': {
            'type': 'object',
            'properties': {
                'max_cost': {'type': 'number', 'minimum': 0},
                'max_time_minutes': {'type': 'integer', 'minimum': 1},
                'preferred_providers': {
                    'type': 'array',
                    'items': {'type': 'string'}
                }
            }
        }
    },
    'required': ['jobs']
}

ScheduleOptimizationSchema = {
    'type': 'object',
    'properties': {
        'jobs': {
            'type': 'array',
            'items': JobSchema,
            'minItems': 1
        },
        'constraints': {
            'type': 'object',
            'properties': {
                'max_cost': {'type': 'number', 'minimum': 0},
                'max_time_minutes': {'type': 'integer', 'minimum': 1},
                'deadline': {'type': 'string', 'format': 'datetime'},
                'preferred_providers': {
                    'type': 'array',
                    'items': {'type': 'string'}
                }
            }
        },
        'optimization_goal': {
            'type': 'string',
            'enum': ['minimize_cost', 'minimize_time', 'maximize_fidelity']
        }
    },
    'required': ['jobs']
}


# Cost Optimization Schemas
ExperimentSchema = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'circuit': {'type': 'string'},
        'shots': {'type': 'integer', 'minimum': 1, 'maximum': 1000000},
        'priority': {'type': 'string', 'enum': ['low', 'medium', 'high']},
        'backend_preferences': {
            'type': 'array',
            'items': {'type': 'string'}
        },
        'max_cost': {'type': 'number', 'minimum': 0}
    },
    'required': ['id', 'circuit', 'shots']
}

CostOptimizationSchema = {
    'type': 'object',
    'properties': {
        'experiments': {
            'type': 'array',
            'items': ExperimentSchema,
            'minItems': 1,
            'maxItems': 50
        },
        'constraints': {
            'type': 'object',
            'properties': {
                'max_cost_per_experiment': {'type': 'number', 'minimum': 0},
                'deadline': {'type': 'string', 'format': 'datetime'},
                'preferred_providers': {
                    'type': 'array',
                    'items': {'type': 'string'}
                }
            }
        }
    },
    'required': ['experiments']
}

UsageUpdateSchema = {
    'type': 'object',
    'properties': {
        'provider': {'type': 'string', 'enum': ['ibmq', 'aws_braket', 'google_quantum']},
        'shots': {'type': 'integer', 'minimum': 1},
        'cost': {'type': 'number', 'minimum': 0},
        'backend': {'type': 'string'},
        'timestamp': {'type': 'string', 'format': 'datetime'}
    },
    'required': ['provider', 'shots', 'cost']
}


# Testing Schemas
TestRunSchema = {
    'type': 'object',
    'properties': {
        'circuit': {'type': 'string'},
        'shots': {'type': 'integer', 'minimum': 1, 'maximum': 100000},
        'backend': {'type': 'string'},
        'noise_model': {'type': 'string'},
        'test_type': {
            'type': 'string',
            'enum': ['standard', 'noise_aware', 'fidelity', 'hardware_compatibility']
        },
        'parameters': {'type': 'object'}
    },
    'required': ['circuit', 'shots']
}

NoiseTestSchema = {
    'type': 'object',
    'properties': {
        'circuit': {'type': 'string'},
        'shots': {'type': 'integer', 'minimum': 1, 'maximum': 100000},
        'noise_levels': {
            'type': 'array',
            'items': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'minItems': 1
        },
        'noise_model': {'type': 'string'},
        'mitigation_method': {'type': 'string'}
    },
    'required': ['circuit', 'shots', 'noise_levels']
}

FidelityCalculationSchema = {
    'type': 'object',
    'properties': {
        'counts': {
            'type': 'object',
            'patternProperties': {
                '^[01]+$': {'type': 'integer', 'minimum': 0}
            }
        },
        'shots': {'type': 'integer', 'minimum': 1},
        'target_state': {
            'type': 'string',
            'enum': ['bell', 'ghz', 'uniform', 'zero', 'plus']
        },
        'custom_target': {
            'type': 'object',
            'patternProperties': {
                '^[01]+$': {'type': 'number', 'minimum': 0, 'maximum': 1}
            }
        }
    },
    'required': ['counts', 'shots']
}

TestResultSchema = {
    'type': 'object',
    'properties': {
        'test_name': {'type': 'string'},
        'framework': {'type': 'string'},
        'backend': {'type': 'string'},
        'shots': {'type': 'integer', 'minimum': 1},
        'fidelity': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'error_rate': {'type': 'number', 'minimum': 0},
        'status': {'type': 'string', 'enum': ['passed', 'failed', 'skipped']},
        'execution_time': {'type': 'number', 'minimum': 0},
        'metadata': {'type': 'object'}
    },
    'required': ['test_name', 'framework', 'backend', 'shots', 'status']
}


# Monitoring Schemas
BuildRecordSchema = {
    'type': 'object',
    'properties': {
        'commit': {'type': 'string'},
        'branch': {'type': 'string'},
        'circuit_count': {'type': 'integer', 'minimum': 0},
        'total_gates': {'type': 'integer', 'minimum': 0},
        'max_depth': {'type': 'integer', 'minimum': 0},
        'estimated_fidelity': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'noise_tests_passed': {'type': 'integer', 'minimum': 0},
        'noise_tests_total': {'type': 'integer', 'minimum': 0},
        'execution_time_seconds': {'type': 'number', 'minimum': 0},
        'metadata': {'type': 'object'}
    },
    'required': ['commit', 'branch']
}

HardwareUsageSchema = {
    'type': 'object',
    'properties': {
        'backend': {'type': 'string'},
        'provider': {'type': 'string'},
        'shots': {'type': 'integer', 'minimum': 1},
        'queue_time_minutes': {'type': 'number', 'minimum': 0},
        'execution_time_minutes': {'type': 'number', 'minimum': 0},
        'cost_usd': {'type': 'number', 'minimum': 0},
        'circuit_depth': {'type': 'integer', 'minimum': 0},
        'num_qubits': {'type': 'integer', 'minimum': 1},
        'success': {'type': 'boolean'}
    },
    'required': ['backend', 'provider', 'shots']
}

AlertSchema = {
    'type': 'object',
    'properties': {
        'type': {'type': 'string'},
        'severity': {
            'type': 'string',
            'enum': ['info', 'warning', 'critical']
        },
        'message': {'type': 'string'},
        'details': {'type': 'object'}
    },
    'required': ['type', 'severity', 'message']
}


# Deployment Schemas
AlgorithmSchema = {
    'type': 'object',  
    'properties': {
        'name': {'type': 'string'},
        'version': {'type': 'string'},
        'circuit': {'type': 'string'},
        'parameters': {'type': 'object'},
        'description': {'type': 'string'},
        'requirements': {
            'type': 'object',
            'properties': {
                'min_qubits': {'type': 'integer', 'minimum': 1},
                'max_depth': {'type': 'integer', 'minimum': 1},
                'supported_backends': {
                    'type': 'array',
                    'items': {'type': 'string'}
                }
            }
        }
    },
    'required': ['name', 'version', 'circuit']
}

DeploymentSchema = {
    'type': 'object',
    'properties': {
        'algorithm': AlgorithmSchema,
        'strategy': {
            'type': 'string',
            'enum': ['blue_green', 'canary', 'rolling']
        },
        'config': {
            'type': 'object',
            'properties': {
                'rollback_threshold': {'type': 'number', 'minimum': 0, 'maximum': 1},
                'health_check_timeout': {'type': 'integer', 'minimum': 1},
                'traffic_split': {'type': 'object'},
                'batch_size': {'type': 'integer', 'minimum': 1},
                'traffic_percentage': {'type': 'integer', 'minimum': 1, 'maximum': 100}
            }
        }
    },
    'required': ['algorithm', 'strategy']
}

ABTestSchema = {
    'type': 'object',
    'properties': {
        'test_name': {'type': 'string'},
        'description': {'type': 'string'},
        'algorithms': {
            'type': 'object',
            'patternProperties': {
                '^[A-Za-z0-9_-]+$': AlgorithmSchema
            },
            'minProperties': 2
        },
        'test_config': {
            'type': 'object',
            'properties': {
                'shots': {'type': 'integer', 'minimum': 100, 'maximum': 10000},
                'iterations': {'type': 'integer', 'minimum': 3, 'maximum': 50},
                'backend': {'type': 'string'},
                'confidence_level': {'type': 'number', 'minimum': 0.8, 'maximum': 0.99},
                'minimum_difference': {'type': 'number', 'minimum': 0.01, 'maximum': 0.5}
            }
        }
    },
    'required': ['test_name', 'algorithms']
}

RollbackSchema = {
    'type': 'object',
    'properties': {
        'reason': {'type': 'string'},
        'target_version': {'type': 'string'},
        'force': {'type': 'boolean'}
    },
    'required': ['reason']
}


# Response Schemas (for documentation)
JobResponseSchema = {
    'type': 'object',
    'properties': {
        'job_id': {'type': 'string'},
        'status': {'type': 'string'},
        'estimated_cost': {'type': 'number'},
        'estimated_time_minutes': {'type': 'number'},
        'backend': {'type': 'string'},
        'created_at': {'type': 'string', 'format': 'datetime'}
    }
}

ScheduleResponseSchema = {
    'type': 'object',
    'properties': {
        'schedule_id': {'type': 'string'},
        'assignments': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'job_id': {'type': 'string'},
                    'backend': {'type': 'string'},
                    'estimated_start': {'type': 'string', 'format': 'datetime'},
                    'estimated_cost': {'type': 'number'}
                }
            }
        },
        'total_estimated_cost': {'type': 'number'},
        'total_estimated_time_minutes': {'type': 'number'},
        'optimization_score': {'type': 'number'}
    }
}

ErrorResponseSchema = {
    'type': 'object',
    'properties': {
        'error': {'type': 'string'},
        'code': {'type': 'string'},
        'details': {'type': 'string'}
    },
    'required': ['error', 'code']
}


# Schema registry for easy access
SCHEMA_REGISTRY = {
    # Job schemas
    'job': JobSchema,
    'batch_job': BatchJobSchema,
    'schedule_optimization': ScheduleOptimizationSchema,
    
    # Cost schemas  
    'experiment': ExperimentSchema,
    'cost_optimization': CostOptimizationSchema,
    'usage_update': UsageUpdateSchema,
    
    # Testing schemas
    'test_run': TestRunSchema,
    'noise_test': NoiseTestSchema,
    'fidelity_calculation': FidelityCalculationSchema,
    'test_result': TestResultSchema,
    
    # Monitoring schemas
    'build_record': BuildRecordSchema,
    'hardware_usage': HardwareUsageSchema,
    'alert': AlertSchema,
    
    # Deployment schemas
    'algorithm': AlgorithmSchema,
    'deployment': DeploymentSchema,
    'ab_test': ABTestSchema,
    'rollback': RollbackSchema,
    
    # Response schemas
    'job_response': JobResponseSchema,
    'schedule_response': ScheduleResponseSchema,
    'error_response': ErrorResponseSchema
}


def get_schema(name: str) -> Dict[str, Any]:
    """Get schema by name."""
    return SCHEMA_REGISTRY.get(name, {})


def validate_schema_exists(name: str) -> bool:
    """Check if schema exists."""
    return name in SCHEMA_REGISTRY
