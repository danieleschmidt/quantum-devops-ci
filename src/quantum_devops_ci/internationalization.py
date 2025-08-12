"""
Internationalization (i18n) support for quantum DevOps CI/CD.

This module provides multi-language support for error messages, UI text,
and documentation across different locales and regions.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Supported locales with their language codes
SUPPORTED_LOCALES = {
    'en': 'English',
    'es': 'Español',
    'fr': 'Français', 
    'de': 'Deutsch',
    'ja': '日本語',
    'zh': '中文'
}

# Default locale fallback chain
FALLBACK_CHAIN = ['en', 'es', 'fr']


@dataclass
class LocaleConfig:
    """Configuration for locale-specific settings."""
    code: str
    name: str
    currency: str = 'USD'
    date_format: str = '%Y-%m-%d'
    time_format: str = '%H:%M:%S'
    decimal_separator: str = '.'
    thousands_separator: str = ','
    timezone: str = 'UTC'
    
    def format_currency(self, amount: float) -> str:
        """Format currency amount according to locale."""
        if self.currency == 'USD':
            return f"${amount:,.2f}"
        elif self.currency == 'EUR':
            return f"€{amount:,.2f}"
        elif self.currency == 'JPY':
            return f"¥{amount:,.0f}"
        elif self.currency == 'CNY':
            return f"¥{amount:,.2f}"
        else:
            return f"{amount:,.2f} {self.currency}"
    
    def format_number(self, number: float) -> str:
        """Format number according to locale conventions."""
        if self.decimal_separator == '.':
            return f"{number:,.2f}".replace(',', self.thousands_separator)
        else:
            formatted = f"{number:,.2f}".replace('.', '|').replace(',', self.decimal_separator)
            return formatted.replace('|', self.thousands_separator)


class TranslationManager:
    """Manages translations for quantum DevOps CI/CD system."""
    
    def __init__(self, locale_dir: Optional[str] = None, default_locale: str = 'en'):
        """
        Initialize translation manager.
        
        Args:
            locale_dir: Directory containing translation files
            default_locale: Default locale code
        """
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.locale_dir = Path(locale_dir) if locale_dir else Path(__file__).parent / 'locales'
        
        # Translation cache
        self._translations: Dict[str, Dict[str, str]] = {}
        
        # Locale configurations
        self.locale_configs = self._initialize_locale_configs()
        
        self.logger = logging.getLogger(__name__)
        
        # Load translations
        self._load_all_translations()
    
    def _initialize_locale_configs(self) -> Dict[str, LocaleConfig]:
        """Initialize locale-specific configurations."""
        configs = {
            'en': LocaleConfig(
                code='en',
                name='English',
                currency='USD',
                timezone='UTC'
            ),
            'es': LocaleConfig(
                code='es', 
                name='Español',
                currency='USD',
                timezone='America/Mexico_City'
            ),
            'fr': LocaleConfig(
                code='fr',
                name='Français',
                currency='EUR',
                decimal_separator=',',
                thousands_separator=' ',
                timezone='Europe/Paris'
            ),
            'de': LocaleConfig(
                code='de',
                name='Deutsch',
                currency='EUR',
                decimal_separator=',',
                thousands_separator='.',
                timezone='Europe/Berlin'
            ),
            'ja': LocaleConfig(
                code='ja',
                name='日本語',
                currency='JPY',
                date_format='%Y年%m月%d日',
                time_format='%H時%M分%S秒',
                timezone='Asia/Tokyo'
            ),
            'zh': LocaleConfig(
                code='zh',
                name='中文',
                currency='CNY',
                date_format='%Y年%m月%d日',
                timezone='Asia/Shanghai'
            )
        }
        return configs
    
    def _load_all_translations(self):
        """Load all available translations."""
        for locale_code in SUPPORTED_LOCALES.keys():
            self._load_translations(locale_code)
    
    def _load_translations(self, locale_code: str):
        """Load translations for specific locale."""
        translation_file = self.locale_dir / f'{locale_code}.json'
        
        if translation_file.exists():
            try:
                with open(translation_file, 'r', encoding='utf-8') as f:
                    self._translations[locale_code] = json.load(f)
                self.logger.debug(f"Loaded translations for locale: {locale_code}")
            except Exception as e:
                self.logger.warning(f"Failed to load translations for {locale_code}: {e}")
                self._translations[locale_code] = {}
        else:
            # Create default translation structure if file doesn't exist
            self._translations[locale_code] = self._create_default_translations(locale_code)
            self._save_translations(locale_code)
    
    def _create_default_translations(self, locale_code: str) -> Dict[str, str]:
        """Create default translations structure."""
        default_translations = {
            # Core system messages
            'system.startup': 'Quantum DevOps CI/CD System Starting',
            'system.shutdown': 'System Shutdown Complete',
            'system.ready': 'System Ready',
            
            # Error messages
            'error.circuit_validation': 'Circuit validation failed',
            'error.backend_connection': 'Failed to connect to quantum backend',
            'error.cost_exceeded': 'Cost limit exceeded',
            'error.quota_exceeded': 'Usage quota exceeded',
            'error.authentication': 'Authentication failed',
            'error.authorization': 'Authorization denied',
            
            # Success messages
            'success.circuit_executed': 'Circuit executed successfully',
            'success.tests_passed': 'All tests passed',
            'success.deployment_complete': 'Deployment completed successfully',
            
            # Status messages
            'status.executing': 'Executing quantum circuit',
            'status.optimizing': 'Optimizing circuit for target backend',
            'status.monitoring': 'Monitoring job progress',
            'status.validating': 'Validating circuit structure',
            
            # Units and labels
            'unit.shots': 'shots',
            'unit.seconds': 'seconds',
            'unit.minutes': 'minutes',
            'unit.hours': 'hours',
            'unit.qubits': 'qubits',
            'unit.gates': 'gates',
            'unit.depth': 'depth',
            
            # UI Labels
            'label.circuit_count': 'Circuit Count',
            'label.execution_time': 'Execution Time',
            'label.cost_estimate': 'Cost Estimate',
            'label.fidelity': 'Fidelity',
            'label.backend': 'Backend',
            'label.provider': 'Provider',
            'label.status': 'Status',
            
            # Commands
            'command.run_tests': 'Run Tests',
            'command.deploy': 'Deploy',
            'command.validate': 'Validate',
            'command.optimize': 'Optimize',
            'command.monitor': 'Monitor',
            
            # Validation messages
            'validation.required_field': 'This field is required',
            'validation.invalid_circuit': 'Invalid quantum circuit',
            'validation.invalid_backend': 'Invalid backend selection',
            'validation.invalid_shots': 'Invalid number of shots',
        }
        
        # Add locale-specific variations
        if locale_code == 'es':
            return self._get_spanish_translations()
        elif locale_code == 'fr':
            return self._get_french_translations()
        elif locale_code == 'de':
            return self._get_german_translations()
        elif locale_code == 'ja':
            return self._get_japanese_translations()
        elif locale_code == 'zh':
            return self._get_chinese_translations()
        else:
            return default_translations
    
    def _get_spanish_translations(self) -> Dict[str, str]:
        """Get Spanish translations."""
        return {
            'system.startup': 'Sistema DevOps Cuántico Iniciando',
            'system.shutdown': 'Apagado del Sistema Completado',
            'system.ready': 'Sistema Listo',
            
            'error.circuit_validation': 'Validación de circuito falló',
            'error.backend_connection': 'Falló la conexión al backend cuántico',
            'error.cost_exceeded': 'Límite de costo excedido',
            'error.quota_exceeded': 'Cuota de uso excedida',
            'error.authentication': 'Autenticación fallida',
            'error.authorization': 'Autorización denegada',
            
            'success.circuit_executed': 'Circuito ejecutado exitosamente',
            'success.tests_passed': 'Todas las pruebas pasaron',
            'success.deployment_complete': 'Despliegue completado exitosamente',
            
            'status.executing': 'Ejecutando circuito cuántico',
            'status.optimizing': 'Optimizando circuito para backend objetivo',
            'status.monitoring': 'Monitoreando progreso del trabajo',
            'status.validating': 'Validando estructura del circuito',
            
            'unit.shots': 'disparos',
            'unit.seconds': 'segundos',
            'unit.minutes': 'minutos',
            'unit.hours': 'horas',
            'unit.qubits': 'qubits',
            'unit.gates': 'compuertas',
            'unit.depth': 'profundidad',
            
            'label.circuit_count': 'Cantidad de Circuitos',
            'label.execution_time': 'Tiempo de Ejecución',
            'label.cost_estimate': 'Estimación de Costo',
            'label.fidelity': 'Fidelidad',
            'label.backend': 'Backend',
            'label.provider': 'Proveedor',
            'label.status': 'Estado',
            
            'command.run_tests': 'Ejecutar Pruebas',
            'command.deploy': 'Desplegar',
            'command.validate': 'Validar',
            'command.optimize': 'Optimizar',
            'command.monitor': 'Monitorear',
        }
    
    def _get_french_translations(self) -> Dict[str, str]:
        """Get French translations."""
        return {
            'system.startup': 'Système DevOps Quantique Démarrage',
            'system.shutdown': 'Arrêt du Système Terminé',
            'system.ready': 'Système Prêt',
            
            'error.circuit_validation': 'Validation du circuit échouée',
            'error.backend_connection': 'Échec de connexion au backend quantique',
            'error.cost_exceeded': 'Limite de coût dépassée',
            'error.quota_exceeded': 'Quota d\'usage dépassé',
            'error.authentication': 'Authentification échouée',
            'error.authorization': 'Autorisation refusée',
            
            'success.circuit_executed': 'Circuit exécuté avec succès',
            'success.tests_passed': 'Tous les tests sont passés',
            'success.deployment_complete': 'Déploiement terminé avec succès',
            
            'status.executing': 'Exécution du circuit quantique',
            'status.optimizing': 'Optimisation du circuit pour le backend cible',
            'status.monitoring': 'Surveillance du progrès du travail',
            'status.validating': 'Validation de la structure du circuit',
            
            'unit.shots': 'tirs',
            'unit.seconds': 'secondes',
            'unit.minutes': 'minutes',
            'unit.hours': 'heures',
            'unit.qubits': 'qubits',
            'unit.gates': 'portes',
            'unit.depth': 'profondeur',
        }
    
    def _get_german_translations(self) -> Dict[str, str]:
        """Get German translations."""
        return {
            'system.startup': 'Quantum DevOps System startet',
            'system.shutdown': 'System-Herunterfahren abgeschlossen',
            'system.ready': 'System bereit',
            
            'error.circuit_validation': 'Schaltkreis-Validierung fehlgeschlagen',
            'error.backend_connection': 'Verbindung zu Quantum-Backend fehlgeschlagen',
            'error.cost_exceeded': 'Kostenlimit überschritten',
            'error.quota_exceeded': 'Nutzungskontingent überschritten',
            
            'success.circuit_executed': 'Schaltkreis erfolgreich ausgeführt',
            'success.tests_passed': 'Alle Tests bestanden',
            'success.deployment_complete': 'Bereitstellung erfolgreich abgeschlossen',
            
            'unit.shots': 'Schüsse',
            'unit.seconds': 'Sekunden',
            'unit.minutes': 'Minuten',
            'unit.hours': 'Stunden',
            'unit.qubits': 'Qubits',
            'unit.gates': 'Gatter',
            'unit.depth': 'Tiefe',
        }
    
    def _get_japanese_translations(self) -> Dict[str, str]:
        """Get Japanese translations."""
        return {
            'system.startup': '量子DevOpsシステム起動中',
            'system.shutdown': 'システムシャットダウン完了',
            'system.ready': 'システム準備完了',
            
            'error.circuit_validation': '回路検証に失敗しました',
            'error.backend_connection': '量子バックエンドへの接続に失敗しました',
            'error.cost_exceeded': 'コスト制限を超過しました',
            'error.quota_exceeded': '使用量クォータを超過しました',
            
            'success.circuit_executed': '回路が正常に実行されました',
            'success.tests_passed': 'すべてのテストが合格しました',
            'success.deployment_complete': 'デプロイメントが正常に完了しました',
            
            'unit.shots': 'ショット',
            'unit.seconds': '秒',
            'unit.minutes': '分',
            'unit.hours': '時間',
            'unit.qubits': 'キュービット',
            'unit.gates': 'ゲート',
            'unit.depth': '深度',
        }
    
    def _get_chinese_translations(self) -> Dict[str, str]:
        """Get Chinese translations."""
        return {
            'system.startup': '量子DevOps系统启动中',
            'system.shutdown': '系统关闭完成',
            'system.ready': '系统就绪',
            
            'error.circuit_validation': '电路验证失败',
            'error.backend_connection': '连接量子后端失败',
            'error.cost_exceeded': '超出成本限制',
            'error.quota_exceeded': '超出使用配额',
            
            'success.circuit_executed': '电路执行成功',
            'success.tests_passed': '所有测试通过',
            'success.deployment_complete': '部署成功完成',
            
            'unit.shots': '测量次数',
            'unit.seconds': '秒',
            'unit.minutes': '分钟',
            'unit.hours': '小时',
            'unit.qubits': '量子比特',
            'unit.gates': '量子门',
            'unit.depth': '深度',
        }
    
    def _save_translations(self, locale_code: str):
        """Save translations to file."""
        try:
            self.locale_dir.mkdir(parents=True, exist_ok=True)
            translation_file = self.locale_dir / f'{locale_code}.json'
            
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self._translations[locale_code],
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True
                )
            self.logger.debug(f"Saved translations for locale: {locale_code}")
        except Exception as e:
            self.logger.warning(f"Failed to save translations for {locale_code}: {e}")
    
    def set_locale(self, locale_code: str) -> bool:
        """
        Set current locale.
        
        Args:
            locale_code: Locale code (e.g., 'en', 'es', 'fr')
            
        Returns:
            True if locale was set successfully, False otherwise
        """
        if locale_code not in SUPPORTED_LOCALES:
            self.logger.warning(f"Unsupported locale: {locale_code}")
            return False
        
        self.current_locale = locale_code
        self.logger.info(f"Locale set to: {locale_code}")
        return True
    
    def get_text(self, key: str, locale_code: Optional[str] = None, **kwargs) -> str:
        """
        Get translated text for given key.
        
        Args:
            key: Translation key
            locale_code: Locale code (uses current locale if not specified)
            **kwargs: Format parameters for string interpolation
            
        Returns:
            Translated and formatted text
        """
        locale = locale_code or self.current_locale
        
        # Try to find translation in specified locale
        text = self._get_translation(key, locale)
        
        # Fallback chain if translation not found
        if text is None:
            for fallback_locale in FALLBACK_CHAIN:
                if fallback_locale != locale:
                    text = self._get_translation(key, fallback_locale)
                    if text is not None:
                        break
        
        # Final fallback to key itself
        if text is None:
            text = key
        
        # Apply string formatting if parameters provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError as e:
                self.logger.warning(f"Missing format parameter for key '{key}': {e}")
        
        return text
    
    def _get_translation(self, key: str, locale_code: str) -> Optional[str]:
        """Get translation for specific locale."""
        return self._translations.get(locale_code, {}).get(key)
    
    def get_locale_config(self, locale_code: Optional[str] = None) -> LocaleConfig:
        """Get locale configuration."""
        locale = locale_code or self.current_locale
        return self.locale_configs.get(locale, self.locale_configs['en'])
    
    def format_currency(self, amount: float, locale_code: Optional[str] = None) -> str:
        """Format currency amount according to locale."""
        config = self.get_locale_config(locale_code)
        return config.format_currency(amount)
    
    def format_number(self, number: float, locale_code: Optional[str] = None) -> str:
        """Format number according to locale."""
        config = self.get_locale_config(locale_code)
        return config.format_number(number)
    
    def format_datetime(self, dt: datetime, locale_code: Optional[str] = None) -> str:
        """Format datetime according to locale."""
        config = self.get_locale_config(locale_code)
        date_str = dt.strftime(config.date_format)
        time_str = dt.strftime(config.time_format)
        return f"{date_str} {time_str}"
    
    def get_supported_locales(self) -> Dict[str, str]:
        """Get dictionary of supported locales."""
        return SUPPORTED_LOCALES.copy()
    
    def add_translation(self, key: str, text: str, locale_code: Optional[str] = None):
        """Add or update translation."""
        locale = locale_code or self.current_locale
        
        if locale not in self._translations:
            self._translations[locale] = {}
        
        self._translations[locale][key] = text
        self._save_translations(locale)
    
    def get_translation_completeness(self, locale_code: str) -> float:
        """Get translation completeness percentage for locale."""
        if locale_code not in self._translations:
            return 0.0
        
        en_keys = set(self._translations.get('en', {}).keys())
        locale_keys = set(self._translations.get(locale_code, {}).keys())
        
        if not en_keys:
            return 100.0
        
        return len(locale_keys.intersection(en_keys)) / len(en_keys) * 100
    
    def get_missing_translations(self, locale_code: str) -> List[str]:
        """Get list of missing translation keys for locale."""
        en_keys = set(self._translations.get('en', {}).keys())
        locale_keys = set(self._translations.get(locale_code, {}).keys())
        
        return list(en_keys - locale_keys)


# Global translation manager instance
_translation_manager: Optional[TranslationManager] = None


def get_translation_manager() -> TranslationManager:
    """Get global translation manager instance."""
    global _translation_manager
    if _translation_manager is None:
        _translation_manager = TranslationManager()
    return _translation_manager


def t(key: str, locale_code: Optional[str] = None, **kwargs) -> str:
    """
    Shorthand function for getting translated text.
    
    Args:
        key: Translation key
        locale_code: Locale code (optional)
        **kwargs: Format parameters
        
    Returns:
        Translated text
    """
    return get_translation_manager().get_text(key, locale_code, **kwargs)


def set_locale(locale_code: str) -> bool:
    """Set global locale."""
    return get_translation_manager().set_locale(locale_code)


def format_currency(amount: float, locale_code: Optional[str] = None) -> str:
    """Format currency with locale-specific formatting."""
    return get_translation_manager().format_currency(amount, locale_code)


def format_number(number: float, locale_code: Optional[str] = None) -> str:
    """Format number with locale-specific formatting."""
    return get_translation_manager().format_number(number, locale_code)


def format_datetime(dt: datetime, locale_code: Optional[str] = None) -> str:
    """Format datetime with locale-specific formatting."""
    return get_translation_manager().format_datetime(dt, locale_code)


# Decorator for translating function outputs
def translatable(message_key: str):
    """
    Decorator to make function outputs translatable.
    
    Args:
        message_key: Base translation key
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, str):
                # Try to translate the result
                translated = t(f"{message_key}.{result.lower().replace(' ', '_')}")
                return translated if translated != f"{message_key}.{result.lower().replace(' ', '_')}" else result
            return result
        return wrapper
    return decorator


def main():
    """CLI for internationalization management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum DevOps CI/CD Internationalization')
    parser.add_argument('--locale', help='Set locale')
    parser.add_argument('--check-completeness', help='Check translation completeness for locale')
    parser.add_argument('--list-missing', help='List missing translations for locale')
    parser.add_argument('--export-template', help='Export translation template')
    
    args = parser.parse_args()
    
    tm = get_translation_manager()
    
    if args.locale:
        if tm.set_locale(args.locale):
            print(f"Locale set to: {args.locale}")
        else:
            print(f"Failed to set locale: {args.locale}")
    
    if args.check_completeness:
        completeness = tm.get_translation_completeness(args.check_completeness)
        print(f"Translation completeness for {args.check_completeness}: {completeness:.1f}%")
    
    if args.list_missing:
        missing = tm.get_missing_translations(args.list_missing)
        print(f"Missing translations for {args.list_missing}:")
        for key in missing:
            print(f"  - {key}")
    
    if args.export_template:
        # Export English translations as template
        en_translations = tm._translations.get('en', {})
        template_file = f"translation_template_{args.export_template}.json"
        
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(en_translations, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        print(f"Translation template exported to: {template_file}")


if __name__ == '__main__':
    main()