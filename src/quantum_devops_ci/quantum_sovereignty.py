"""
Quantum Sovereignty Framework for Generation 4 Intelligence.

This module implements advanced quantum sovereignty controls for global compliance,
export restrictions, dual-use technology management, and strategic autonomy.
"""

import warnings
import json
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import ipaddress

from .exceptions import SovereigntyViolationError, ExportControlError, ComplianceError
from .validation import validate_inputs
from .security import requires_auth, audit_action, SecurityContext
from .caching import CacheManager


class SovereigntyLevel(Enum):
    """Levels of quantum sovereignty control."""
    OPEN = "open"
    RESTRICTED = "restricted"
    CONTROLLED = "controlled"
    CLASSIFIED = "classified"


class TechnologyClassification(Enum):
    """Quantum technology classifications."""
    FUNDAMENTAL_RESEARCH = "fundamental_research"
    APPLIED_RESEARCH = "applied_research"
    DUAL_USE = "dual_use"
    COMMERCIAL = "commercial"
    STRATEGIC = "strategic"
    DEFENSE_CRITICAL = "defense_critical"


class ExportControlRegime(Enum):
    """Export control regimes."""
    WASSENAAR = "wassenaar"
    EAR = "ear"  # US Export Administration Regulations
    ITAR = "itar"  # International Traffic in Arms Regulations
    EU_DUAL_USE = "eu_dual_use"
    AUSTRALIA_GROUP = "australia_group"
    MTCR = "mtcr"  # Missile Technology Control Regime


@dataclass
class SovereigntyPolicy:
    """Quantum sovereignty policy configuration."""
    country_code: str
    sovereignty_level: SovereigntyLevel
    allowed_destinations: List[str] = field(default_factory=list)
    restricted_destinations: List[str] = field(default_factory=list)
    prohibited_destinations: List[str] = field(default_factory=list)
    technology_restrictions: Dict[TechnologyClassification, List[str]] = field(default_factory=dict)
    export_control_regimes: List[ExportControlRegime] = field(default_factory=list)
    data_residency_requirements: Dict[str, str] = field(default_factory=dict)
    algorithmic_restrictions: List[str] = field(default_factory=list)
    collaboration_limits: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default restrictions if not provided."""
        if not self.technology_restrictions:
            self._set_default_restrictions()
            
    def _set_default_restrictions(self):
        """Set default technology restrictions by sovereignty level."""
        if self.sovereignty_level == SovereigntyLevel.CLASSIFIED:
            self.technology_restrictions = {
                TechnologyClassification.DEFENSE_CRITICAL: ["NO_EXPORT"],
                TechnologyClassification.STRATEGIC: ["ALLIES_ONLY"],
                TechnologyClassification.DUAL_USE: ["LICENSE_REQUIRED"],
                TechnologyClassification.APPLIED_RESEARCH: ["REVIEW_REQUIRED"],
                TechnologyClassification.FUNDAMENTAL_RESEARCH: ["PUBLICATION_REVIEW"]
            }
        elif self.sovereignty_level == SovereigntyLevel.CONTROLLED:
            self.technology_restrictions = {
                TechnologyClassification.STRATEGIC: ["ALLIES_ONLY"],
                TechnologyClassification.DUAL_USE: ["LICENSE_REQUIRED"],
                TechnologyClassification.APPLIED_RESEARCH: ["REVIEW_REQUIRED"]
            }
        elif self.sovereignty_level == SovereigntyLevel.RESTRICTED:
            self.technology_restrictions = {
                TechnologyClassification.DUAL_USE: ["NOTIFICATION_REQUIRED"],
                TechnologyClassification.APPLIED_RESEARCH: ["REVIEW_OPTIONAL"]
            }


@dataclass
class AccessRequest:
    """Request for quantum technology access."""
    request_id: str
    requester_info: Dict[str, Any]
    requested_technology: TechnologyClassification
    intended_use: str
    destination_country: str
    duration: timedelta
    justification: str
    security_clearance: Optional[str] = None
    institutional_affiliation: Optional[str] = None
    previous_violations: List[str] = field(default_factory=list)
    
    def risk_score(self) -> float:
        """Calculate risk score for the access request."""
        base_risk = {
            TechnologyClassification.FUNDAMENTAL_RESEARCH: 0.1,
            TechnologyClassification.APPLIED_RESEARCH: 0.3,
            TechnologyClassification.COMMERCIAL: 0.2,
            TechnologyClassification.DUAL_USE: 0.7,
            TechnologyClassification.STRATEGIC: 0.8,
            TechnologyClassification.DEFENSE_CRITICAL: 0.95
        }.get(self.requested_technology, 0.5)
        
        # Adjust for violations
        violation_penalty = len(self.previous_violations) * 0.1
        
        # Adjust for clearance
        clearance_bonus = 0.2 if self.security_clearance else 0.0
        
        return min(1.0, max(0.0, base_risk + violation_penalty - clearance_bonus))


@dataclass
class ComplianceReport:
    """Compliance monitoring report."""
    report_id: str
    reporting_period: Tuple[datetime, datetime]
    total_requests: int
    approved_requests: int
    denied_requests: int
    violations_detected: int
    export_activities: List[Dict[str, Any]] = field(default_factory=list)
    risk_incidents: List[Dict[str, Any]] = field(default_factory=list)
    compliance_score: float = 1.0
    recommendations: List[str] = field(default_factory=list)


class QuantumSovereigntyManager:
    """Manager for quantum sovereignty and export controls."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.sovereignty_policies = {}
        self.access_requests = {}
        self.compliance_records = {}
        self.blocked_entities = set()
        self.monitored_technologies = {}
        
        # Load default policies
        self._initialize_default_policies()
        
    def _initialize_default_policies(self):
        """Initialize default sovereignty policies for major regions."""
        
        # United States Policy
        us_policy = SovereigntyPolicy(
            country_code="US",
            sovereignty_level=SovereigntyLevel.CONTROLLED,
            allowed_destinations=["CA", "GB", "AU", "JP", "KR", "IL"],  # Key allies
            restricted_destinations=["CN", "RU", "IR", "KP"],  # Strategic competitors
            prohibited_destinations=["KP", "IR"],  # Sanctioned countries
            export_control_regimes=[ExportControlRegime.EAR, ExportControlRegime.ITAR],
            data_residency_requirements={"defense": "US", "intelligence": "US"},
            algorithmic_restrictions=["quantum_cryptography", "optimization_algorithms"]
        )
        
        # European Union Policy
        eu_policy = SovereigntyPolicy(
            country_code="EU",
            sovereignty_level=SovereigntyLevel.RESTRICTED,
            allowed_destinations=["US", "CA", "GB", "AU", "JP", "CH", "NO"],
            restricted_destinations=["CN", "RU"],
            export_control_regimes=[ExportControlRegime.EU_DUAL_USE, ExportControlRegime.WASSENAAR],
            data_residency_requirements={"personal_data": "EU"},
            algorithmic_restrictions=["quantum_ai_hybrid"]
        )
        
        # China Policy
        cn_policy = SovereigntyPolicy(
            country_code="CN", 
            sovereignty_level=SovereigntyLevel.CONTROLLED,
            allowed_destinations=["RU", "KZ", "BY"],  # Strategic partners
            restricted_destinations=["US", "GB", "AU", "JP"],
            data_residency_requirements={"all_data": "CN"},
            algorithmic_restrictions=["foreign_encryption", "western_optimization"]
        )
        
        # Australia Policy
        au_policy = SovereigntyPolicy(
            country_code="AU",
            sovereignty_level=SovereigntyLevel.RESTRICTED,
            allowed_destinations=["US", "GB", "CA", "NZ", "JP"],  # AUKUS + close allies
            restricted_destinations=["CN", "RU"],
            export_control_regimes=[ExportControlRegime.AUSTRALIA_GROUP, ExportControlRegime.WASSENAAR],
            data_residency_requirements={"defense": "AU"},
            algorithmic_restrictions=["strategic_quantum_computing"]
        )
        
        self.sovereignty_policies.update({
            "US": us_policy,
            "EU": eu_policy, 
            "CN": cn_policy,
            "AU": au_policy
        })
        
    @requires_auth
    @audit_action("sovereignty_policy_update")
    def set_sovereignty_policy(self, country_code: str, policy: SovereigntyPolicy):
        """Set or update sovereignty policy for a country."""
        self.sovereignty_policies[country_code] = policy
        logging.info(f"Updated sovereignty policy for {country_code}")
        
    @validate_inputs
    def evaluate_access_request(self, request: AccessRequest) -> Dict[str, Any]:
        """Evaluate quantum technology access request."""
        
        # Check if requester is from known policy region
        source_country = request.requester_info.get('country', 'UNKNOWN')
        destination_country = request.destination_country
        
        if source_country not in self.sovereignty_policies:
            logging.warning(f"No sovereignty policy found for {source_country}")
            return self._default_evaluation(request)
            
        policy = self.sovereignty_policies[source_country]
        
        evaluation = {
            'request_id': request.request_id,
            'timestamp': datetime.now(),
            'approved': False,
            'risk_score': request.risk_score(),
            'violations': [],
            'conditions': [],
            'reasoning': []
        }
        
        # Check prohibited destinations
        if destination_country in policy.prohibited_destinations:
            evaluation['violations'].append(f"Destination {destination_country} is prohibited")
            evaluation['approved'] = False
            evaluation['reasoning'].append("Export to prohibited destination")
            return evaluation
            
        # Check blocked entities
        requester_org = request.requester_info.get('organization', '')
        if requester_org in self.blocked_entities:
            evaluation['violations'].append(f"Requester organization {requester_org} is blocked")
            evaluation['approved'] = False
            evaluation['reasoning'].append("Blocked entity")
            return evaluation
            
        # Technology-specific restrictions
        tech_restrictions = policy.technology_restrictions.get(request.requested_technology, [])
        
        for restriction in tech_restrictions:
            if restriction == "NO_EXPORT":
                evaluation['violations'].append("Technology classified as no-export")
                evaluation['approved'] = False
                evaluation['reasoning'].append("No-export technology")
                return evaluation
            elif restriction == "ALLIES_ONLY" and destination_country not in policy.allowed_destinations:
                evaluation['violations'].append(f"Technology restricted to allies only, {destination_country} not in allowed list")
                evaluation['approved'] = False
                evaluation['reasoning'].append("Allies-only restriction violated")
                return evaluation
            elif restriction == "LICENSE_REQUIRED":
                evaluation['conditions'].append("Export license required")
            elif restriction == "REVIEW_REQUIRED":
                evaluation['conditions'].append("Technical review required")
            elif restriction == "NOTIFICATION_REQUIRED":
                evaluation['conditions'].append("Government notification required")
                
        # Risk-based evaluation
        risk_score = request.risk_score()
        
        if risk_score > 0.8:
            evaluation['conditions'].append("High-risk: Additional security measures required")
        elif risk_score > 0.6:
            evaluation['conditions'].append("Medium-risk: Monitoring required")
            
        # Check security clearance requirements
        if (request.requested_technology in [TechnologyClassification.STRATEGIC, TechnologyClassification.DEFENSE_CRITICAL] 
            and not request.security_clearance):
            evaluation['conditions'].append("Security clearance verification required")
            
        # Approve if no violations and conditions can be met
        if not evaluation['violations']:
            evaluation['approved'] = True
            evaluation['reasoning'].append("Request meets sovereignty requirements")
            
        return evaluation
        
    def _default_evaluation(self, request: AccessRequest) -> Dict[str, Any]:
        """Default evaluation when no specific policy exists."""
        return {
            'request_id': request.request_id,
            'timestamp': datetime.now(),
            'approved': False,
            'risk_score': request.risk_score(),
            'violations': ['No sovereignty policy defined for source country'],
            'conditions': ['Manual review required'],
            'reasoning': ['Default deny due to lack of policy']
        }
        
    @requires_auth
    @audit_action("technology_classification")
    def classify_quantum_technology(self, 
                                  algorithm_description: str,
                                  intended_application: str,
                                  technical_parameters: Dict[str, Any]) -> TechnologyClassification:
        """Classify quantum technology based on description and parameters."""
        
        # Keyword-based classification
        description_lower = algorithm_description.lower()
        application_lower = intended_application.lower()
        
        # Defense-critical keywords
        defense_keywords = [
            'cryptanalysis', 'code breaking', 'military', 'defense', 'nuclear',
            'weapons', 'submarine', 'radar', 'sonar', 'ballistic', 'missile'
        ]
        
        # Strategic keywords
        strategic_keywords = [
            'quantum supremacy', 'quantum advantage', 'error correction',
            'fault tolerant', 'scalable quantum', 'quantum internet',
            'quantum communication', 'quantum sensing', 'precision timing'
        ]
        
        # Dual-use keywords
        dual_use_keywords = [
            'optimization', 'machine learning', 'artificial intelligence',
            'drug discovery', 'financial modeling', 'supply chain',
            'logistics', 'scheduling', 'cryptography', 'encryption'
        ]
        
        # Classification logic
        if any(keyword in description_lower or keyword in application_lower 
               for keyword in defense_keywords):
            return TechnologyClassification.DEFENSE_CRITICAL
            
        if any(keyword in description_lower or keyword in application_lower 
               for keyword in strategic_keywords):
            return TechnologyClassification.STRATEGIC
            
        if any(keyword in description_lower or keyword in application_lower 
               for keyword in dual_use_keywords):
            return TechnologyClassification.DUAL_USE
            
        # Check technical parameters
        if technical_parameters:
            qubit_count = technical_parameters.get('qubits', 0)
            gate_fidelity = technical_parameters.get('gate_fidelity', 0.0)
            coherence_time = technical_parameters.get('coherence_time_us', 0.0)
            
            # High-performance quantum systems are strategic
            if (qubit_count > 100 or 
                gate_fidelity > 0.999 or 
                coherence_time > 1000):
                return TechnologyClassification.STRATEGIC
                
            # Medium performance is dual-use
            if qubit_count > 10 or gate_fidelity > 0.99:
                return TechnologyClassification.DUAL_USE
                
        # Default to applied research
        if 'research' in application_lower:
            return TechnologyClassification.APPLIED_RESEARCH
        
        return TechnologyClassification.COMMERCIAL
        
    @validate_inputs 
    def monitor_cross_border_collaboration(self,
                                         participants: List[Dict[str, str]],
                                         project_description: str,
                                         data_sharing_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor cross-border quantum collaboration for compliance."""
        
        monitoring_report = {
            'collaboration_id': f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now(),
            'participants': participants,
            'compliance_status': 'COMPLIANT',
            'warnings': [],
            'violations': [],
            'required_approvals': [],
            'data_residency_issues': []
        }
        
        # Extract countries involved
        participant_countries = set()
        for participant in participants:
            country = participant.get('country', 'UNKNOWN')
            participant_countries.add(country)
            
        # Check for restricted collaborations
        for country in participant_countries:
            if country in self.sovereignty_policies:
                policy = self.sovereignty_policies[country]
                
                # Check if any participant is from restricted destination
                other_countries = participant_countries - {country}
                restricted_involvement = other_countries.intersection(policy.restricted_destinations)
                
                if restricted_involvement:
                    monitoring_report['warnings'].append(
                        f"Collaboration involves participants from restricted countries: {restricted_involvement}"
                    )
                    monitoring_report['required_approvals'].append(f"Government approval required for {country}")
                    
                prohibited_involvement = other_countries.intersection(policy.prohibited_destinations)
                
                if prohibited_involvement:
                    monitoring_report['violations'].append(
                        f"Collaboration involves participants from prohibited countries: {prohibited_involvement}"
                    )
                    monitoring_report['compliance_status'] = 'VIOLATION'
                    
        # Classify the collaboration technology
        tech_classification = self.classify_quantum_technology(
            project_description,
            "collaborative research", 
            {}
        )
        
        # Additional restrictions for sensitive technologies
        if tech_classification in [TechnologyClassification.STRATEGIC, TechnologyClassification.DEFENSE_CRITICAL]:
            monitoring_report['required_approvals'].append("Export license required")
            monitoring_report['required_approvals'].append("Technology transfer review")
            
        # Data residency compliance
        for participant in participants:
            country = participant.get('country')
            if country in self.sovereignty_policies:
                policy = self.sovereignty_policies[country]
                
                for data_type, required_residency in policy.data_residency_requirements.items():
                    planned_residency = data_sharing_plan.get('data_location', 'UNSPECIFIED')
                    
                    if required_residency != planned_residency and required_residency != 'FLEXIBLE':
                        monitoring_report['data_residency_issues'].append(
                            f"{country} requires {data_type} data to remain in {required_residency}, "
                            f"but plan specifies {planned_residency}"
                        )
                        
        # Set compliance status based on findings
        if monitoring_report['violations']:
            monitoring_report['compliance_status'] = 'VIOLATION'
        elif monitoring_report['warnings'] or monitoring_report['required_approvals']:
            monitoring_report['compliance_status'] = 'REVIEW_REQUIRED'
            
        return monitoring_report
        
    @requires_auth
    @audit_action("sovereignty_compliance_report")
    def generate_compliance_report(self,
                                 start_date: datetime,
                                 end_date: datetime,
                                 country_code: Optional[str] = None) -> ComplianceReport:
        """Generate quantum sovereignty compliance report."""
        
        # Filter requests by date range and country
        relevant_requests = []
        for request_id, request_data in self.access_requests.items():
            request_time = request_data.get('timestamp', datetime.min)
            
            if start_date <= request_time <= end_date:
                if country_code is None or request_data.get('country') == country_code:
                    relevant_requests.append(request_data)
                    
        # Calculate statistics
        total_requests = len(relevant_requests)
        approved_requests = sum(1 for req in relevant_requests if req.get('approved', False))
        denied_requests = total_requests - approved_requests
        
        # Count violations
        violations_detected = sum(
            1 for req in relevant_requests 
            if req.get('violations', [])
        )
        
        # Compile export activities
        export_activities = []
        for req in relevant_requests:
            if req.get('approved', False):
                export_activities.append({
                    'technology': req.get('technology', 'unknown'),
                    'destination': req.get('destination', 'unknown'),
                    'value': req.get('estimated_value', 0),
                    'risk_score': req.get('risk_score', 0)
                })
                
        # Calculate compliance score
        if total_requests > 0:
            violation_rate = violations_detected / total_requests
            compliance_score = max(0.0, 1.0 - violation_rate * 2)  # Heavily penalize violations
        else:
            compliance_score = 1.0
            
        # Generate recommendations
        recommendations = []
        
        if violation_rate > 0.1:
            recommendations.append("High violation rate detected - review approval processes")
            
        if denied_requests / max(total_requests, 1) > 0.5:
            recommendations.append("High denial rate - consider policy review")
            
        high_risk_exports = [act for act in export_activities if act['risk_score'] > 0.7]
        if len(high_risk_exports) > total_requests * 0.2:
            recommendations.append("High proportion of risky exports - enhance monitoring")
            
        report_id = f"compliance_{country_code or 'global'}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        return ComplianceReport(
            report_id=report_id,
            reporting_period=(start_date, end_date),
            total_requests=total_requests,
            approved_requests=approved_requests,
            denied_requests=denied_requests,
            violations_detected=violations_detected,
            export_activities=export_activities,
            compliance_score=compliance_score,
            recommendations=recommendations
        )
        
    def add_blocked_entity(self, entity_name: str, reason: str):
        """Add entity to blocked list."""
        self.blocked_entities.add(entity_name)
        logging.info(f"Added {entity_name} to blocked entities: {reason}")
        
    def remove_blocked_entity(self, entity_name: str):
        """Remove entity from blocked list."""
        self.blocked_entities.discard(entity_name)
        logging.info(f"Removed {entity_name} from blocked entities")
        
    def get_sovereignty_status(self, country_code: str) -> Dict[str, Any]:
        """Get sovereignty status and policies for a country."""
        if country_code not in self.sovereignty_policies:
            return {
                'country_code': country_code,
                'has_policy': False,
                'status': 'No quantum sovereignty policy defined'
            }
            
        policy = self.sovereignty_policies[country_code]
        
        return {
            'country_code': country_code,
            'has_policy': True,
            'sovereignty_level': policy.sovereignty_level.value,
            'allowed_destinations': policy.allowed_destinations,
            'restricted_destinations': policy.restricted_destinations,
            'prohibited_destinations': policy.prohibited_destinations,
            'export_control_regimes': [regime.value for regime in policy.export_control_regimes],
            'data_residency_requirements': policy.data_residency_requirements,
            'algorithmic_restrictions': policy.algorithmic_restrictions
        }


class QuantumDataSovereignty:
    """Manager for quantum data sovereignty and cross-border data flows."""
    
    def __init__(self, sovereignty_manager: QuantumSovereigntyManager):
        self.sovereignty_manager = sovereignty_manager
        self.data_flows = {}
        self.encryption_requirements = {}
        
    @requires_auth
    @audit_action("data_sovereignty_assessment")  
    def assess_data_transfer(self,
                           data_type: str,
                           source_country: str,
                           destination_country: str,
                           data_classification: str,
                           transfer_volume_gb: float) -> Dict[str, Any]:
        """Assess quantum data transfer for sovereignty compliance."""
        
        assessment = {
            'transfer_id': f"transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now(),
            'source_country': source_country,
            'destination_country': destination_country,
            'data_type': data_type,
            'classification': data_classification,
            'volume_gb': transfer_volume_gb,
            'approved': False,
            'requirements': [],
            'restrictions': [],
            'encryption_required': False
        }
        
        # Check source country policy
        source_policy = self.sovereignty_manager.sovereignty_policies.get(source_country)
        
        if source_policy:
            # Check data residency requirements
            residency_req = source_policy.data_residency_requirements.get(data_type)
            
            if residency_req and residency_req != source_country and residency_req != 'FLEXIBLE':
                if destination_country != residency_req:
                    assessment['restrictions'].append(
                        f"Data must remain in {residency_req} according to source country policy"
                    )
                    return assessment
                    
            # Check if destination is restricted
            if destination_country in source_policy.restricted_destinations:
                assessment['requirements'].append("Special authorization required for restricted destination")
                assessment['encryption_required'] = True
                
            if destination_country in source_policy.prohibited_destinations:
                assessment['restrictions'].append("Transfer to prohibited destination")
                return assessment
                
        # Check destination country policy  
        dest_policy = self.sovereignty_manager.sovereignty_policies.get(destination_country)
        
        if dest_policy:
            # Check if source is acceptable
            if (source_country in dest_policy.restricted_destinations and 
                source_country not in dest_policy.allowed_destinations):
                assessment['requirements'].append("Import restrictions may apply")
                
        # Classification-based requirements
        if data_classification in ['CLASSIFIED', 'SECRET', 'TOP_SECRET']:
            assessment['requirements'].append("Highest level encryption required")
            assessment['requirements'].append("Government approval required")
            assessment['encryption_required'] = True
        elif data_classification in ['SENSITIVE', 'RESTRICTED']:
            assessment['requirements'].append("Enhanced encryption required")
            assessment['encryption_required'] = True
            
        # Volume-based requirements
        if transfer_volume_gb > 100:  # Large transfers
            assessment['requirements'].append("Bulk transfer notification required")
            
        if transfer_volume_gb > 1000:  # Very large transfers
            assessment['requirements'].append("Strategic data transfer review required")
            
        # Approve if no hard restrictions
        if not assessment['restrictions']:
            assessment['approved'] = True
            
        return assessment
        
    def get_encryption_requirements(self,
                                  source_country: str,
                                  destination_country: str,
                                  data_classification: str) -> Dict[str, str]:
        """Get encryption requirements for quantum data transfer."""
        
        requirements = {
            'minimum_algorithm': 'AES-256',
            'key_management': 'STANDARD',
            'quantum_safe': False
        }
        
        # Classification-based requirements
        if data_classification in ['CLASSIFIED', 'SECRET', 'TOP_SECRET']:
            requirements.update({
                'minimum_algorithm': 'AES-256-GCM',
                'key_management': 'HSM_REQUIRED',
                'quantum_safe': True,
                'additional_controls': ['END_TO_END_ENCRYPTION', 'PERFECT_FORWARD_SECRECY']
            })
        elif data_classification in ['SENSITIVE', 'RESTRICTED']:
            requirements.update({
                'minimum_algorithm': 'AES-256',
                'quantum_safe': True,
                'additional_controls': ['END_TO_END_ENCRYPTION']
            })
            
        # Country-specific enhancements
        source_policy = self.sovereignty_manager.sovereignty_policies.get(source_country)
        dest_policy = self.sovereignty_manager.sovereignty_policies.get(destination_country)
        
        if (source_policy and 
            destination_country in source_policy.restricted_destinations):
            requirements['quantum_safe'] = True
            requirements['key_management'] = 'HSM_REQUIRED'
            
        if (dest_policy and 
            source_country in dest_policy.restricted_destinations):
            requirements['quantum_safe'] = True
            
        return requirements


# Export main classes
__all__ = [
    'SovereigntyLevel',
    'TechnologyClassification',
    'ExportControlRegime',
    'SovereigntyPolicy',
    'AccessRequest',
    'ComplianceReport',
    'QuantumSovereigntyManager',
    'QuantumDataSovereignty'
]