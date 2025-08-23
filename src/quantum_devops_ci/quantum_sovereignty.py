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
from .monitoring import PerformanceMetrics
import geoip2.database
import geoip2.errors


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
    QUANTUM_EXPORT_CONTROL = "quantum_export_control"
    

@dataclass
class QuantumSovereigntyPolicy:
    """Quantum sovereignty policy configuration."""
    country_code: str
    sovereignty_level: SovereigntyLevel
    allowed_technologies: Set[TechnologyClassification]
    export_control_regimes: List[ExportControlRegime]
    quantum_key_length_limit: int = 256
    quantum_algorithm_restrictions: List[str] = field(default_factory=list)
    data_localization_required: bool = True
    audit_level: str = "comprehensive"
    encryption_requirements: Dict[str, Any] = field(default_factory=dict)
    

@dataclass  
class QuantumTechnicalAssessment:
    """Technical assessment of quantum algorithms and circuits."""
    algorithm_classification: TechnologyClassification
    quantum_advantage_factor: float
    cryptographic_impact_score: float
    dual_use_risk_score: float
    export_control_classification: str
    security_clearance_required: Optional[str] = None
    restricted_countries: List[str] = field(default_factory=list)
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    

class QuantumSovereigntyManager:
    """Advanced quantum sovereignty and export control management."""
    
    def __init__(self, config_path: Optional[str] = None, enable_geolocation: bool = True):
        self.policies: Dict[str, QuantumSovereigntyPolicy] = {}
        self.technical_assessments: Dict[str, QuantumTechnicalAssessment] = {}
        self.audit_logger = logging.getLogger(f"{__name__}.audit")
        self.cache_manager = CacheManager()
        self.enable_geolocation = enable_geolocation
        self.geoip_reader = None
        
        if enable_geolocation:
            try:
                # Note: In production, use actual GeoIP2 database
                # self.geoip_reader = geoip2.database.Reader('path/to/GeoLite2-Country.mmdb')
                pass
            except Exception as e:
                logger.warning(f"GeoIP database not available: {e}")
                self.enable_geolocation = False
        
        self._load_default_policies()
        if config_path:
            self._load_policies_from_config(config_path)
    
    def _load_default_policies(self):
        """Load default sovereignty policies for major quantum powers."""
        # United States - ITAR/EAR restrictions
        self.policies['US'] = QuantumSovereigntyPolicy(
            country_code='US',
            sovereignty_level=SovereigntyLevel.CONTROLLED,
            allowed_technologies={
                TechnologyClassification.FUNDAMENTAL_RESEARCH,
                TechnologyClassification.APPLIED_RESEARCH,
                TechnologyClassification.COMMERCIAL
            },
            export_control_regimes=[
                ExportControlRegime.EAR,
                ExportControlRegime.ITAR,
                ExportControlRegime.WASSENAAR
            ],
            quantum_key_length_limit=256,
            quantum_algorithm_restrictions=[
                'shor_algorithm',
                'quantum_cryptanalysis',
                'post_quantum_key_recovery'
            ],
            encryption_requirements={
                'minimum_key_length': 256,
                'approved_algorithms': ['AES', 'RSA', 'ECC'],
                'quantum_safe_required': True
            }
        )
        
        # European Union - Dual-use restrictions
        self.policies['EU'] = QuantumSovereigntyPolicy(
            country_code='EU',
            sovereignty_level=SovereigntyLevel.RESTRICTED,
            allowed_technologies={
                TechnologyClassification.FUNDAMENTAL_RESEARCH,
                TechnologyClassification.APPLIED_RESEARCH,
                TechnologyClassification.COMMERCIAL,
                TechnologyClassification.DUAL_USE
            },
            export_control_regimes=[
                ExportControlRegime.EU_DUAL_USE,
                ExportControlRegime.WASSENAAR
            ],
            quantum_key_length_limit=256,
            data_localization_required=True
        )
        
        # China - Strategic technology controls
        self.policies['CN'] = QuantumSovereigntyPolicy(
            country_code='CN',
            sovereignty_level=SovereigntyLevel.CONTROLLED,
            allowed_technologies={
                TechnologyClassification.FUNDAMENTAL_RESEARCH,
                TechnologyClassification.APPLIED_RESEARCH
            },
            export_control_regimes=[
                ExportControlRegime.QUANTUM_EXPORT_CONTROL
            ],
            quantum_key_length_limit=512,
            data_localization_required=True,
            audit_level='comprehensive'
        )
        
        # Open research countries
        for country in ['CA', 'AU', 'NZ', 'JP', 'KR']:
            self.policies[country] = QuantumSovereigntyPolicy(
                country_code=country,
                sovereignty_level=SovereigntyLevel.RESTRICTED,
                allowed_technologies={
                    TechnologyClassification.FUNDAMENTAL_RESEARCH,
                    TechnologyClassification.APPLIED_RESEARCH,
                    TechnologyClassification.COMMERCIAL
                },
                export_control_regimes=[ExportControlRegime.WASSENAAR],
                quantum_key_length_limit=256
            )
    
    def _load_policies_from_config(self, config_path: str):
        """Load sovereignty policies from configuration file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            for country_code, policy_data in config_data.get('sovereignty_policies', {}).items():
                self.policies[country_code] = QuantumSovereigntyPolicy(
                    country_code=country_code,
                    sovereignty_level=SovereigntyLevel(policy_data['sovereignty_level']),
                    allowed_technologies={
                        TechnologyClassification(tech) 
                        for tech in policy_data['allowed_technologies']
                    },
                    export_control_regimes=[
                        ExportControlRegime(regime) 
                        for regime in policy_data['export_control_regimes']
                    ],
                    **{k: v for k, v in policy_data.items() 
                       if k not in ['sovereignty_level', 'allowed_technologies', 'export_control_regimes']}
                )
        except Exception as e:
            logger.error(f"Failed to load sovereignty policies: {e}")
    
    async def assess_quantum_technology(self, 
                                      algorithm_description: Dict[str, Any],
                                      source_location: Optional[str] = None,
                                      destination_countries: Optional[List[str]] = None) -> QuantumTechnicalAssessment:
        """Comprehensive technical assessment of quantum technology."""
        algorithm_name = algorithm_description.get('name', 'unknown')
        
        # Classify technology
        classification = self._classify_quantum_technology(algorithm_description)
        
        # Assess quantum advantage
        quantum_advantage = self._calculate_quantum_advantage_factor(algorithm_description)
        
        # Evaluate cryptographic impact
        crypto_impact = self._evaluate_cryptographic_impact(algorithm_description)
        
        # Assess dual-use risk
        dual_use_risk = self._assess_dual_use_risk(algorithm_description)
        
        # Determine export control classification
        export_classification = self._determine_export_control_classification(
            classification, quantum_advantage, crypto_impact
        )
        
        # Check restricted countries
        restricted_countries = self._identify_restricted_countries(
            export_classification, destination_countries
        )
        
        assessment = QuantumTechnicalAssessment(
            algorithm_classification=classification,
            quantum_advantage_factor=quantum_advantage,
            cryptographic_impact_score=crypto_impact,
            dual_use_risk_score=dual_use_risk,
            export_control_classification=export_classification,
            restricted_countries=restricted_countries
        )
        
        # Cache assessment
        self.technical_assessments[algorithm_name] = assessment
        
        # Audit log
        await self._audit_technology_assessment(assessment, source_location)
        
        return assessment
    
    def _classify_quantum_technology(self, algorithm_description: Dict[str, Any]) -> TechnologyClassification:
        """Classify quantum technology based on description."""
        algorithm_type = algorithm_description.get('type', '').lower()
        keywords = algorithm_description.get('keywords', [])
        
        # Classification logic based on algorithm characteristics
        if any(keyword in ['shor', 'cryptanalysis', 'factoring'] for keyword in keywords):
            return TechnologyClassification.DUAL_USE
        elif any(keyword in ['optimization', 'machine_learning', 'simulation'] for keyword in keywords):
            return TechnologyClassification.COMMERCIAL
        elif any(keyword in ['research', 'theoretical', 'experimental'] for keyword in keywords):
            return TechnologyClassification.FUNDAMENTAL_RESEARCH
        elif 'military' in keywords or 'defense' in keywords:
            return TechnologyClassification.DEFENSE_CRITICAL
        else:
            return TechnologyClassification.APPLIED_RESEARCH
    
    def _calculate_quantum_advantage_factor(self, algorithm_description: Dict[str, Any]) -> float:
        """Calculate quantum advantage factor."""
        # Simplified quantum advantage calculation
        complexity_classical = algorithm_description.get('classical_complexity', 'polynomial')
        complexity_quantum = algorithm_description.get('quantum_complexity', 'polynomial')
        
        advantage_map = {
            ('exponential', 'polynomial'): 100.0,  # Exponential speedup (e.g., Shor's)
            ('exponential', 'exponential'): 1.0,   # No advantage
            ('polynomial', 'polynomial'): 2.0,     # Quadratic speedup (e.g., Grover's)
            ('polynomial', 'exponential'): 0.1     # Quantum disadvantage
        }
        
        return advantage_map.get((complexity_classical, complexity_quantum), 1.0)
    
    def _evaluate_cryptographic_impact(self, algorithm_description: Dict[str, Any]) -> float:
        """Evaluate cryptographic impact score."""
        cryptographic_keywords = [
            'encryption', 'decryption', 'cryptanalysis', 'key', 'cipher',
            'hash', 'signature', 'authentication', 'shor', 'factoring'
        ]
        
        keywords = [kw.lower() for kw in algorithm_description.get('keywords', [])]
        description = algorithm_description.get('description', '').lower()
        
        impact_score = 0.0
        
        # Direct cryptographic references
        for keyword in cryptographic_keywords:
            if keyword in keywords or keyword in description:
                if keyword in ['shor', 'factoring', 'cryptanalysis']:
                    impact_score += 10.0  # High impact
                elif keyword in ['encryption', 'decryption', 'key']:
                    impact_score += 5.0   # Medium impact
                else:
                    impact_score += 2.0   # Low impact
        
        # Normalize to 0-100 scale
        return min(100.0, impact_score)
    
    def _assess_dual_use_risk(self, algorithm_description: Dict[str, Any]) -> float:
        """Assess dual-use risk score."""
        high_risk_indicators = [
            'weapons', 'military', 'defense', 'surveillance', 'cryptanalysis',
            'code_breaking', 'intelligence', 'warfare', 'targeting'
        ]
        
        medium_risk_indicators = [
            'optimization', 'simulation', 'modeling', 'prediction',
            'analysis', 'pattern_recognition', 'ai'
        ]
        
        keywords = [kw.lower() for kw in algorithm_description.get('keywords', [])]
        description = algorithm_description.get('description', '').lower()
        
        risk_score = 0.0
        
        for indicator in high_risk_indicators:
            if indicator in keywords or indicator in description:
                risk_score += 20.0
        
        for indicator in medium_risk_indicators:
            if indicator in keywords or indicator in description:
                risk_score += 5.0
        
        # Consider quantum advantage factor
        quantum_advantage = self._calculate_quantum_advantage_factor(algorithm_description)
        if quantum_advantage > 10.0:
            risk_score += 25.0  # High quantum advantage increases dual-use risk
        
        return min(100.0, risk_score)
    
    def _determine_export_control_classification(self, 
                                               classification: TechnologyClassification,
                                               quantum_advantage: float,
                                               crypto_impact: float) -> str:
        """Determine export control classification."""
        if classification == TechnologyClassification.DEFENSE_CRITICAL:
            return "ITAR_CONTROLLED"
        elif classification == TechnologyClassification.DUAL_USE:
            if crypto_impact > 50.0 or quantum_advantage > 50.0:
                return "EAR_CCL_CATEGORY_5_PART_2"
            else:
                return "EAR_CCL_CATEGORY_3_PART_A"
        elif classification == TechnologyClassification.STRATEGIC:
            return "WASSENAAR_CATEGORY_5A002"
        elif crypto_impact > 30.0:
            return "DUAL_USE_CRYPTOGRAPHIC"
        else:
            return "UNCONTROLLED"
    
    def _identify_restricted_countries(self, 
                                     export_classification: str,
                                     destination_countries: Optional[List[str]] = None) -> List[str]:
        """Identify countries with export restrictions."""
        if not destination_countries:
            return []
        
        restricted = []
        
        # High-risk export classifications
        high_restriction_classifications = [
            "ITAR_CONTROLLED", 
            "EAR_CCL_CATEGORY_5_PART_2",
            "WASSENAAR_CATEGORY_5A002"
        ]
        
        # Countries with general export restrictions
        generally_restricted = ['IR', 'KP', 'SY', 'CU']
        
        # Quantum-specific restrictions
        quantum_restricted = ['CN', 'RU', 'IR', 'KP'] if export_classification in high_restriction_classifications else []
        
        for country in destination_countries:
            if (export_classification in high_restriction_classifications and 
                country in quantum_restricted) or country in generally_restricted:
                restricted.append(country)
        
        return restricted
    
    async def _audit_technology_assessment(self, 
                                         assessment: QuantumTechnicalAssessment,
                                         source_location: Optional[str] = None):
        """Audit technology assessment for compliance tracking."""
        audit_entry = {
            'timestamp': assessment.assessment_timestamp.isoformat(),
            'classification': assessment.algorithm_classification.value,
            'export_control': assessment.export_control_classification,
            'quantum_advantage': assessment.quantum_advantage_factor,
            'crypto_impact': assessment.cryptographic_impact_score,
            'dual_use_risk': assessment.dual_use_risk_score,
            'source_location': source_location,
            'restricted_countries': assessment.restricted_countries
        }
        
        self.audit_logger.info(f"Quantum technology assessment: {json.dumps(audit_entry)}")
    
    @requires_auth
    async def validate_quantum_deployment(self, 
                                        deployment_config: Dict[str, Any],
                                        user_context: SecurityContext) -> Dict[str, Any]:
        """Validate quantum deployment against sovereignty policies."""
        source_country = deployment_config.get('source_country', 'unknown')
        target_countries = deployment_config.get('target_countries', [])
        quantum_algorithms = deployment_config.get('algorithms', [])
        
        validation_results = {
            'approved': True,
            'violations': [],
            'warnings': [],
            'required_approvals': [],
            'compliance_score': 100.0
        }
        
        # Assess each quantum algorithm
        for algorithm in quantum_algorithms:
            assessment = await self.assess_quantum_technology(
                algorithm, source_country, target_countries
            )
            
            # Check for sovereignty violations
            violations = self._check_sovereignty_violations(
                assessment, source_country, target_countries
            )
            
            if violations:
                validation_results['violations'].extend(violations)
                validation_results['approved'] = False
                validation_results['compliance_score'] -= 20.0
            
            # Check for required approvals
            if assessment.export_control_classification in [
                "ITAR_CONTROLLED", "EAR_CCL_CATEGORY_5_PART_2"
            ]:
                validation_results['required_approvals'].append({
                    'algorithm': algorithm.get('name'),
                    'approval_type': 'export_license',
                    'classification': assessment.export_control_classification
                })
        
        # Audit validation
        await self._audit_deployment_validation(
            deployment_config, validation_results, user_context
        )
        
        return validation_results
    
    def _check_sovereignty_violations(self, 
                                    assessment: QuantumTechnicalAssessment,
                                    source_country: str,
                                    target_countries: List[str]) -> List[Dict[str, str]]:
        """Check for sovereignty policy violations."""
        violations = []
        
        # Check source country policy
        if source_country in self.policies:
            source_policy = self.policies[source_country]
            
            if assessment.algorithm_classification not in source_policy.allowed_technologies:
                violations.append({
                    'type': 'technology_restriction',
                    'country': source_country,
                    'message': f'Technology classification {assessment.algorithm_classification.value} not allowed in {source_country}'
                })
        
        # Check target country restrictions
        for country in target_countries:
            if country in assessment.restricted_countries:
                violations.append({
                    'type': 'export_restriction',
                    'country': country,
                    'message': f'Export to {country} restricted for {assessment.export_control_classification}'
                })
        
        return violations
    
    async def _audit_deployment_validation(self, 
                                         deployment_config: Dict[str, Any],
                                         validation_results: Dict[str, Any],
                                         user_context: SecurityContext):
        """Audit deployment validation for compliance tracking."""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_context.user_id,
            'deployment_id': deployment_config.get('id'),
            'source_country': deployment_config.get('source_country'),
            'target_countries': deployment_config.get('target_countries', []),
            'approved': validation_results['approved'],
            'compliance_score': validation_results['compliance_score'],
            'violations_count': len(validation_results['violations']),
            'required_approvals_count': len(validation_results['required_approvals'])
        }
        
        self.audit_logger.info(f"Deployment validation: {json.dumps(audit_entry)}")


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