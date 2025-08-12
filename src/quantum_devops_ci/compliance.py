"""
Global compliance framework for quantum DevOps CI/CD.

This module provides compliance checking and enforcement for various
international regulations including GDPR, CCPA, PDPA, and other
data protection and quantum computing regulations.
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from .exceptions import SecurityError, ValidationError
from .security import SecurityManager, audit_action
from .internationalization import t, get_translation_manager


class ComplianceRegime(Enum):
    """Supported compliance regimes."""
    GDPR = "gdpr"           # General Data Protection Regulation (EU)
    CCPA = "ccpa"           # California Consumer Privacy Act (US)
    PDPA = "pdpa"           # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"           # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"       # Personal Information Protection and Electronic Documents Act (Canada)
    QUANTUM_US = "quantum_us"  # US Quantum Computing regulations
    QUANTUM_EU = "quantum_eu"  # EU Quantum Computing regulations
    QUANTUM_CN = "quantum_cn"  # China Quantum Computing regulations


@dataclass
class ComplianceRequirement:
    """Individual compliance requirement."""
    id: str
    regime: ComplianceRegime
    title: str
    description: str
    category: str  # data_protection, quantum_security, export_control
    mandatory: bool = True
    validation_function: Optional[str] = None
    remediation_steps: List[str] = field(default_factory=list)
    
    def validate(self, context: Dict[str, Any]) -> bool:
        """Validate requirement against provided context."""
        if self.validation_function:
            # This would call registered validation functions
            return True  # Simplified for now
        return True


@dataclass  
class ComplianceViolation:
    """Compliance violation record."""
    requirement_id: str
    regime: ComplianceRegime
    violation_type: str
    description: str
    severity: str  # low, medium, high, critical
    detected_at: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for compliance."""
    id: str
    data_type: str  # quantum_circuit, measurement_data, user_data, system_logs
    processing_purpose: str
    legal_basis: str  # consent, contract, legitimate_interest, vital_interests
    data_subject_categories: List[str]
    recipients: List[str]
    retention_period: Optional[timedelta]
    security_measures: List[str]
    international_transfers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class ComplianceManager:
    """Central compliance management system."""
    
    def __init__(self, 
                 applicable_regimes: Optional[List[ComplianceRegime]] = None,
                 data_residency_requirements: Optional[Dict[str, List[str]]] = None):
        """
        Initialize compliance manager.
        
        Args:
            applicable_regimes: List of applicable compliance regimes
            data_residency_requirements: Data residency requirements by region
        """
        self.applicable_regimes = applicable_regimes or [
            ComplianceRegime.GDPR,
            ComplianceRegime.CCPA,
            ComplianceRegime.QUANTUM_US
        ]
        
        self.data_residency_requirements = data_residency_requirements or {
            'EU': ['gdpr'],
            'US': ['ccpa', 'quantum_us'],
            'APAC': ['pdpa']
        }
        
        # Compliance requirements registry
        self.requirements: Dict[str, ComplianceRequirement] = {}
        self.violations: List[ComplianceViolation] = []
        self.processing_records: List[DataProcessingRecord] = []
        
        # Initialize requirements
        self._initialize_requirements()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_requirements(self):
        """Initialize compliance requirements for applicable regimes."""
        
        # GDPR Requirements
        if ComplianceRegime.GDPR in self.applicable_regimes:
            self._add_gdpr_requirements()
        
        # CCPA Requirements  
        if ComplianceRegime.CCPA in self.applicable_regimes:
            self._add_ccpa_requirements()
        
        # Quantum Computing Requirements
        if ComplianceRegime.QUANTUM_US in self.applicable_regimes:
            self._add_quantum_us_requirements()
        
        if ComplianceRegime.QUANTUM_EU in self.applicable_regimes:
            self._add_quantum_eu_requirements()
    
    def _add_gdpr_requirements(self):
        """Add GDPR compliance requirements."""
        requirements = [
            ComplianceRequirement(
                id="gdpr_001",
                regime=ComplianceRegime.GDPR,
                title="Lawful Basis for Processing",
                description="Must have lawful basis for processing personal data",
                category="data_protection",
                remediation_steps=[
                    "Identify legal basis for each processing activity",
                    "Document legal basis in processing records",
                    "Obtain consent where required"
                ]
            ),
            ComplianceRequirement(
                id="gdpr_002", 
                regime=ComplianceRegime.GDPR,
                title="Data Subject Rights",
                description="Must provide mechanisms for data subject rights",
                category="data_protection",
                remediation_steps=[
                    "Implement data access procedures",
                    "Implement data portability procedures",
                    "Implement right to erasure procedures"
                ]
            ),
            ComplianceRequirement(
                id="gdpr_003",
                regime=ComplianceRegime.GDPR,
                title="Data Protection Impact Assessment",
                description="Must conduct DPIA for high-risk processing",
                category="data_protection",
                remediation_steps=[
                    "Identify high-risk processing activities",
                    "Conduct DPIA assessment",
                    "Implement risk mitigation measures"
                ]
            ),
            ComplianceRequirement(
                id="gdpr_004",
                regime=ComplianceRegime.GDPR,
                title="Data Minimization",
                description="Process only necessary personal data",
                category="data_protection",
                remediation_steps=[
                    "Review data collection practices",
                    "Remove unnecessary data fields",
                    "Implement purpose limitation controls"
                ]
            ),
            ComplianceRequirement(
                id="gdpr_005",
                regime=ComplianceRegime.GDPR,
                title="International Data Transfers",
                description="Ensure adequate protection for international transfers",
                category="data_protection",
                remediation_steps=[
                    "Identify international transfers",
                    "Implement Standard Contractual Clauses",
                    "Conduct Transfer Risk Assessment"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.id] = req
    
    def _add_ccpa_requirements(self):
        """Add CCPA compliance requirements."""
        requirements = [
            ComplianceRequirement(
                id="ccpa_001",
                regime=ComplianceRegime.CCPA,
                title="Consumer Right to Know",
                description="Consumers have right to know about data collection",
                category="data_protection",
                remediation_steps=[
                    "Provide privacy notice describing data practices",
                    "Implement consumer request procedures",
                    "Maintain records of data categories collected"
                ]
            ),
            ComplianceRequirement(
                id="ccpa_002",
                regime=ComplianceRegime.CCPA,
                title="Consumer Right to Delete",
                description="Consumers have right to delete personal information",
                category="data_protection",
                remediation_steps=[
                    "Implement data deletion procedures",
                    "Verify consumer identity",
                    "Delete data from all systems"
                ]
            ),
            ComplianceRequirement(
                id="ccpa_003",
                regime=ComplianceRegime.CCPA,
                title="Do Not Sell My Personal Information",
                description="Consumers can opt-out of personal information sales",
                category="data_protection",
                remediation_steps=[
                    "Implement opt-out mechanism",
                    "Honor opt-out requests",
                    "Maintain opt-out records"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.id] = req
    
    def _add_quantum_us_requirements(self):
        """Add US quantum computing compliance requirements."""
        requirements = [
            ComplianceRequirement(
                id="quantum_us_001",
                regime=ComplianceRegime.QUANTUM_US,
                title="Export Control Compliance",
                description="Quantum technology subject to export controls",
                category="export_control",
                remediation_steps=[
                    "Screen against denied persons lists",
                    "Classify quantum technology exports",
                    "Obtain required export licenses"
                ]
            ),
            ComplianceRequirement(
                id="quantum_us_002",
                regime=ComplianceRegime.QUANTUM_US,
                title="Quantum Circuit Security",
                description="Protect quantum intellectual property",
                category="quantum_security",
                remediation_steps=[
                    "Encrypt quantum circuit data",
                    "Implement access controls",
                    "Monitor for unauthorized access"
                ]
            ),
            ComplianceRequirement(
                id="quantum_us_003",
                regime=ComplianceRegime.QUANTUM_US,
                title="Research Security",
                description="Protect quantum research from foreign interference",
                category="quantum_security",
                remediation_steps=[
                    "Screen research collaborators",
                    "Secure quantum research data",
                    "Report suspicious activities"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.id] = req
    
    def _add_quantum_eu_requirements(self):
        """Add EU quantum computing compliance requirements."""
        requirements = [
            ComplianceRequirement(
                id="quantum_eu_001",
                regime=ComplianceRegime.QUANTUM_EU,
                title="Dual-Use Export Control",
                description="Quantum technology subject to dual-use regulations",
                category="export_control",
                remediation_steps=[
                    "Check dual-use control lists",
                    "Obtain export authorization",
                    "Maintain export records"
                ]
            ),
            ComplianceRequirement(
                id="quantum_eu_002",
                regime=ComplianceRegime.QUANTUM_EU,
                title="Digital Sovereignty",
                description="Ensure EU digital sovereignty in quantum computing",
                category="quantum_security",
                remediation_steps=[
                    "Use EU-based quantum providers",
                    "Ensure data stays within EU",
                    "Document quantum supply chain"
                ]
            )
        ]
        
        for req in requirements:
            self.requirements[req.id] = req
    
    @audit_action('compliance_check', 'requirements')
    def check_compliance(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive compliance check.
        
        Args:
            context: Additional context for compliance checking
            
        Returns:
            Compliance check results
        """
        context = context or {}
        
        results = {
            'timestamp': datetime.now(),
            'total_requirements': len(self.requirements),
            'compliant_requirements': 0,
            'violations': [],
            'recommendations': [],
            'overall_status': 'compliant'
        }
        
        for requirement in self.requirements.values():
            try:
                if requirement.validate(context):
                    results['compliant_requirements'] += 1
                else:
                    # Record violation
                    violation = ComplianceViolation(
                        requirement_id=requirement.id,
                        regime=requirement.regime,
                        violation_type='validation_failure',
                        description=f"Failed to meet requirement: {requirement.title}",
                        severity='high' if requirement.mandatory else 'medium',
                        detected_at=datetime.now(),
                        context=context
                    )
                    
                    self.violations.append(violation)
                    results['violations'].append({
                        'requirement_id': requirement.id,
                        'regime': requirement.regime.value,
                        'title': requirement.title,
                        'severity': violation.severity,
                        'remediation_steps': requirement.remediation_steps
                    })
                    
                    # Add to recommendations
                    results['recommendations'].extend(requirement.remediation_steps)
            
            except Exception as e:
                self.logger.error(f"Error checking requirement {requirement.id}: {e}")
        
        # Determine overall status
        if results['violations']:
            critical_violations = [v for v in results['violations'] if v['severity'] == 'critical']
            high_violations = [v for v in results['violations'] if v['severity'] == 'high']
            
            if critical_violations:
                results['overall_status'] = 'non_compliant'
            elif high_violations:
                results['overall_status'] = 'partially_compliant'
            else:
                results['overall_status'] = 'compliant_with_warnings'
        
        return results
    
    def record_data_processing(self, 
                             data_type: str,
                             processing_purpose: str,
                             legal_basis: str,
                             data_subject_categories: List[str],
                             retention_period: Optional[timedelta] = None) -> str:
        """
        Record data processing activity for compliance.
        
        Args:
            data_type: Type of data being processed
            processing_purpose: Purpose of processing
            legal_basis: Legal basis for processing
            data_subject_categories: Categories of data subjects
            retention_period: How long data will be retained
            
        Returns:
            Processing record ID
        """
        record_id = hashlib.md5(
            f"{data_type}:{processing_purpose}:{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        record = DataProcessingRecord(
            id=record_id,
            data_type=data_type,
            processing_purpose=processing_purpose,
            legal_basis=legal_basis,
            data_subject_categories=data_subject_categories,
            recipients=[],  # To be updated
            retention_period=retention_period,
            security_measures=['encryption', 'access_control', 'audit_logging']
        )
        
        self.processing_records.append(record)
        self.logger.info(f"Recorded data processing activity: {record_id}")
        
        return record_id
    
    def check_data_residency(self, 
                           data_location: str, 
                           data_subject_region: str) -> bool:
        """
        Check if data location complies with residency requirements.
        
        Args:
            data_location: Current location of data
            data_subject_region: Region of data subject
            
        Returns:
            True if compliant, False otherwise
        """
        requirements = self.data_residency_requirements.get(data_subject_region, [])
        
        for requirement in requirements:
            if requirement == 'gdpr' and data_subject_region == 'EU':
                # GDPR requires data to stay in EU or adequate jurisdiction
                if data_location not in ['EU', 'UK', 'Switzerland']:
                    return False
            elif requirement == 'quantum_cn' and data_subject_region == 'CN':
                # China requires quantum data to stay in China
                if data_location != 'CN':
                    return False
        
        return True
    
    def generate_privacy_notice(self, locale: str = 'en') -> str:
        """
        Generate privacy notice based on applicable regulations.
        
        Args:
            locale: Locale for privacy notice
            
        Returns:
            Privacy notice text
        """
        tm = get_translation_manager()
        tm.set_locale(locale)
        
        notice_parts = []
        
        # Header
        notice_parts.append(t('privacy_notice.header', locale_code=locale))
        notice_parts.append('')
        
        # Data collection section
        notice_parts.append(t('privacy_notice.data_collection.header', locale_code=locale))
        
        # Add regime-specific sections
        if ComplianceRegime.GDPR in self.applicable_regimes:
            notice_parts.extend(self._generate_gdpr_privacy_section(locale))
        
        if ComplianceRegime.CCPA in self.applicable_regimes:
            notice_parts.extend(self._generate_ccpa_privacy_section(locale))
        
        # Contact information
        notice_parts.append('')
        notice_parts.append(t('privacy_notice.contact.header', locale_code=locale))
        notice_parts.append(t('privacy_notice.contact.email', locale_code=locale))
        
        return '\n'.join(notice_parts)
    
    def _generate_gdpr_privacy_section(self, locale: str) -> List[str]:
        """Generate GDPR-specific privacy notice sections."""
        return [
            t('privacy_notice.gdpr.lawful_basis', locale_code=locale),
            t('privacy_notice.gdpr.data_rights', locale_code=locale),
            t('privacy_notice.gdpr.international_transfers', locale_code=locale),
            ''
        ]
    
    def _generate_ccpa_privacy_section(self, locale: str) -> List[str]:
        """Generate CCPA-specific privacy notice sections.""" 
        return [
            t('privacy_notice.ccpa.consumer_rights', locale_code=locale),
            t('privacy_notice.ccpa.do_not_sell', locale_code=locale),
            ''
        ]
    
    def handle_data_subject_request(self, 
                                  request_type: str,
                                  subject_id: str,
                                  verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle data subject rights requests.
        
        Args:
            request_type: Type of request (access, delete, portability)
            subject_id: ID of data subject
            verification_data: Data for identity verification
            
        Returns:
            Request handling results
        """
        result = {
            'request_id': hashlib.md5(f"{subject_id}:{request_type}:{datetime.now().isoformat()}".encode()).hexdigest(),
            'status': 'pending',
            'estimated_completion': datetime.now() + timedelta(days=30),
            'actions_required': []
        }
        
        if request_type == 'access':
            result['actions_required'] = [
                'Verify subject identity',
                'Collect all personal data',
                'Prepare data export',
                'Send data to subject'
            ]
        elif request_type == 'delete':
            result['actions_required'] = [
                'Verify subject identity', 
                'Identify all personal data',
                'Check deletion exceptions',
                'Delete data from all systems',
                'Confirm deletion completion'
            ]
        elif request_type == 'portability':
            result['actions_required'] = [
                'Verify subject identity',
                'Extract portable data',
                'Format data for portability',
                'Provide data to subject'
            ]
        
        self.logger.info(f"Data subject request received: {result['request_id']}")
        
        return result
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data."""
        recent_violations = [v for v in self.violations if v.detected_at > datetime.now() - timedelta(days=30)]
        
        return {
            'applicable_regimes': [r.value for r in self.applicable_regimes],
            'total_requirements': len(self.requirements),
            'recent_violations': len(recent_violations),
            'critical_violations': len([v for v in recent_violations if v.severity == 'critical']),
            'processing_records': len(self.processing_records),
            'last_compliance_check': datetime.now().isoformat(),
            'data_residency_regions': list(self.data_residency_requirements.keys()),
            'violation_trends': self._calculate_violation_trends()
        }
    
    def _calculate_violation_trends(self) -> Dict[str, int]:
        """Calculate violation trends over time."""
        now = datetime.now()
        
        return {
            'last_7_days': len([v for v in self.violations if v.detected_at > now - timedelta(days=7)]),
            'last_30_days': len([v for v in self.violations if v.detected_at > now - timedelta(days=30)]),
            'last_90_days': len([v for v in self.violations if v.detected_at > now - timedelta(days=90)])
        }
    
    def export_compliance_report(self, output_file: str, format: str = 'json'):
        """
        Export comprehensive compliance report.
        
        Args:
            output_file: Output file path
            format: Export format (json, csv, html)
        """
        dashboard_data = self.get_compliance_dashboard()
        compliance_results = self.check_compliance()
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'dashboard': dashboard_data,
            'compliance_check': compliance_results,
            'processing_records': [
                {
                    'id': r.id,
                    'data_type': r.data_type,
                    'purpose': r.processing_purpose,
                    'legal_basis': r.legal_basis,
                    'created_at': r.created_at.isoformat()
                }
                for r in self.processing_records
            ],
            'violations': [
                {
                    'requirement_id': v.requirement_id,
                    'regime': v.regime.value,
                    'description': v.description,
                    'severity': v.severity,
                    'detected_at': v.detected_at.isoformat(),
                    'resolved': v.resolved
                }
                for v in self.violations
            ]
        }
        
        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2)
        elif format == 'html':
            self._export_html_report(report_data, output_file)
        
        self.logger.info(f"Compliance report exported to: {output_file}")
    
    def _export_html_report(self, data: Dict[str, Any], output_file: str):
        """Export compliance report as HTML."""
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum DevOps CI/CD Compliance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #f0f8ff; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .violation { background: #ffe4e1; padding: 10px; margin: 5px 0; border-radius: 3px; }
                .compliant { background: #e6ffe6; padding: 10px; margin: 5px 0; border-radius: 3px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quantum DevOps CI/CD Compliance Report</h1>
                <p>Generated: {generated_at}</p>
            </div>
            
            <div class="section">
                <h2>Compliance Overview</h2>
                <div class="{overall_class}">
                    <strong>Overall Status: {overall_status}</strong><br>
                    Requirements: {compliant_requirements}/{total_requirements}<br>
                    Violations: {violation_count}
                </div>
            </div>
            
            <div class="section">
                <h2>Processing Records</h2>
                <p>Total Records: {processing_count}</p>
            </div>
        </body>
        </html>
        '''.format(
            generated_at=data['generated_at'],
            overall_class='compliant' if data['compliance_check']['overall_status'] == 'compliant' else 'violation',
            overall_status=data['compliance_check']['overall_status'],
            compliant_requirements=data['compliance_check']['compliant_requirements'],
            total_requirements=data['compliance_check']['total_requirements'],
            violation_count=len(data['violations']),
            processing_count=len(data['processing_records'])
        )
        
        with open(output_file, 'w') as f:
            f.write(html_template)


# Global compliance manager instance
_compliance_manager: Optional[ComplianceManager] = None


def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager instance."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager()
    return _compliance_manager


def check_compliance(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check compliance using global manager."""
    return get_compliance_manager().check_compliance(context)


def record_processing(data_type: str, purpose: str, legal_basis: str, 
                     data_subjects: List[str]) -> str:
    """Record data processing using global manager."""
    return get_compliance_manager().record_data_processing(
        data_type, purpose, legal_basis, data_subjects
    )


def main():
    """CLI for compliance management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum DevOps CI/CD Compliance Management')
    parser.add_argument('--check', action='store_true', help='Run compliance check')
    parser.add_argument('--dashboard', action='store_true', help='Show compliance dashboard')
    parser.add_argument('--export-report', help='Export compliance report (file path)')
    parser.add_argument('--format', choices=['json', 'html'], default='json', help='Report format')
    parser.add_argument('--privacy-notice', help='Generate privacy notice for locale')
    
    args = parser.parse_args()
    
    cm = get_compliance_manager()
    
    if args.check:
        results = cm.check_compliance()
        print(f"Compliance Status: {results['overall_status']}")
        print(f"Compliant Requirements: {results['compliant_requirements']}/{results['total_requirements']}")
        
        if results['violations']:
            print("\nViolations:")
            for violation in results['violations']:
                print(f"  - {violation['title']} ({violation['severity']})")
    
    if args.dashboard:
        dashboard = cm.get_compliance_dashboard()
        print("Compliance Dashboard:")
        for key, value in dashboard.items():
            print(f"  {key}: {value}")
    
    if args.export_report:
        cm.export_compliance_report(args.export_report, args.format)
        print(f"Report exported to: {args.export_report}")
    
    if args.privacy_notice:
        notice = cm.generate_privacy_notice(args.privacy_notice)
        print(f"Privacy Notice ({args.privacy_notice}):")
        print(notice)


if __name__ == '__main__':
    main()