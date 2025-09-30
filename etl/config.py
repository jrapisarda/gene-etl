"""
ETL Configuration Schema using Pydantic

This module defines the configuration schema for the gene pair correlation ETL pipeline,
providing type-safe configuration management with validation and defaults.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
import uuid


class DatabaseConfig(BaseModel):
    """Database connection configuration"""
    host: str = Field(..., description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    schema: str = Field(default="public", description="Default schema")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('pool_size', 'max_overflow')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Must be positive')
        return v


class GeneFilterConfig(BaseModel):
    """Gene filtering configuration"""
    method: str = Field(default="variance", description="Filtering method: variance, iqr, mad")
    top_n_genes: int = Field(default=3000, description="Number of top genes to keep")
    min_variance_threshold: float = Field(default=0.01, description="Minimum variance threshold")
    min_expression_value: float = Field(default=0.0, description="Minimum expression value")
    
    @validator('method')
    def validate_method(cls, v):
        valid_methods = ['variance', 'iqr', 'mad']
        if v not in valid_methods:
            raise ValueError(f'Method must be one of {valid_methods}')
        return v
    
    @validator('top_n_genes', 'min_variance_threshold')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Must be positive')
        return v


class StatisticalConfig(BaseModel):
    """Statistical analysis configuration"""
    correlation_method: str = Field(default="spearman", description="Correlation method")
    min_samples: int = Field(default=20, description="Minimum samples per illness cohort")
    alpha_fdr: float = Field(default=0.05, description="FDR correction alpha level")
    
    @validator('correlation_method')
    def validate_method(cls, v):
        valid_methods = ['spearman', 'pearson', 'kendall']
        if v not in valid_methods:
            raise ValueError(f'Method must be one of {valid_methods}')
        return v
    
    @validator('min_samples')
    def validate_min_samples(cls, v):
        if v < 10:
            raise ValueError('Minimum samples should be at least 10')
        return v
    
    @validator('alpha_fdr')
    def validate_alpha(cls, v):
        if not 0 < v < 1:
            raise ValueError('Alpha must be between 0 and 1')
        return v


class ProcessingConfig(BaseModel):
    """Processing and performance configuration"""
    batch_size: int = Field(default=1000, description="Batch size for processing")
    block_size: int = Field(default=100, description="Block size for correlation computation")
    max_workers: int = Field(default=4, description="Maximum parallel workers")
    memory_limit_gb: float = Field(default=8.0, description="Memory limit in GB")
    checkpoint_interval: int = Field(default=10000, description="Checkpoint interval")
    
    @validator('batch_size', 'block_size', 'max_workers', 'checkpoint_interval')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Must be positive')
        return v
    
    @validator('memory_limit_gb')
    def validate_memory(cls, v):
        if v <= 0:
            raise ValueError('Memory limit must be positive')
        return v


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="Log format: json, plain")
    file_path: Optional[str] = Field(None, description="Log file path")
    max_file_size_mb: int = Field(default=100, description="Max log file size in MB")
    backup_count: int = Field(default=5, description="Number of backup files")
    
    @validator('level')
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v not in valid_levels:
            raise ValueError(f'Level must be one of {valid_levels}')
        return v
    
    @validator('format')
    def validate_format(cls, v):
        valid_formats = ['json', 'plain']
        if v not in valid_formats:
            raise ValueError(f'Format must be one of {valid_formats}')
        return v


class AlertingConfig(BaseModel):
    """Alerting and monitoring configuration"""
    enabled: bool = Field(default=True, description="Enable alerting")
    email_recipients: List[str] = Field(default_factory=list, description="Email recipients")
    smtp_server: Optional[str] = Field(None, description="SMTP server")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_username: Optional[str] = Field(None, description="SMTP username")
    smtp_password: Optional[str] = Field(None, description="SMTP password")
    
    @validator('email_recipients')
    def validate_emails(cls, v):
        for email in v:
            if '@' not in email:
                raise ValueError(f'Invalid email format: {email}')
        return v


class ETLConfig(BaseModel):
    """Main ETL configuration schema"""
    
    # Metadata
    config_name: str = Field(default="gene_correlation_etl", description="Configuration name")
    config_version: str = Field(default="1.0.0", description="Configuration version")
    environment: str = Field(default="dev", description="Environment: dev, test, prod")
    
    # Core configurations
    database: DatabaseConfig
    gene_filter: GeneFilterConfig = Field(default_factory=GeneFilterConfig)
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    
    # Runtime configuration
    batch_id: Optional[str] = Field(None, description="Batch ID for this run")
    source_snapshot_id: Optional[str] = Field(None, description="Source snapshot ID")
    
    # Optional overrides
    illness_whitelist: Optional[List[str]] = Field(None, description="Illness codes to process")
    illness_blacklist: Optional[List[str]] = Field(None, description="Illness codes to skip")
    gene_whitelist: Optional[List[str]] = Field(None, description="Gene symbols to process")
    
    class Config:
        extra = "forbid"  # No extra fields allowed
        validate_assignment = True
    
    @root_validator
    def validate_batch_id(cls, values):
        """Generate batch ID if not provided"""
        if not values.get('batch_id'):
            values['batch_id'] = str(uuid.uuid4())
        return values
    
    @root_validator
    def validate_mutual_exclusivity(cls, values):
        """Ensure whitelist and blacklist are not both set"""
        illness_whitelist = values.get('illness_whitelist')
        illness_blacklist = values.get('illness_blacklist')
        
        if illness_whitelist and illness_blacklist:
            raise ValueError('Cannot specify both illness_whitelist and illness_blacklist')
        
        return values
    
    def get_config_hash(self) -> str:
        """Generate a hash of the configuration for lineage tracking"""
        import hashlib
        import json
        
        # Create a deterministic string representation
        config_dict = self.dict(exclude={'batch_id'})
        config_str = json.dumps(config_dict, sort_keys=True)
        
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return self.dict()
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ETLConfig':
        """Load configuration from YAML file"""
        import ruamel.yaml
        
        with open(config_path, 'r') as f:
            config_data = ruamel.yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        import ruamel.yaml
        
        with open(config_path, 'w') as f:
            ruamel.yaml.dump(self.dict(), f, default_flow_style=False)


# Example configuration templates
EXAMPLE_CONFIG = {
    "config_name": "gene_correlation_etl_dev",
    "config_version": "1.0.0",
    "environment": "dev",
    "database": {
        "host": "localhost",
        "port": 5432,
        "database": "bioinformatics_dev",
        "username": "etl_user",
        "password": "secure_password",
        "pool_size": 5,
        "max_overflow": 10
    },
    "gene_filter": {
        "method": "variance",
        "top_n_genes": 3000,
        "min_variance_threshold": 0.01,
        "min_expression_value": 0.0
    },
    "statistical": {
        "correlation_method": "spearman",
        "min_samples": 20,
        "alpha_fdr": 0.05
    },
    "processing": {
        "batch_size": 1000,
        "block_size": 100,
        "max_workers": 4,
        "memory_limit_gb": 8.0,
        "checkpoint_interval": 10000
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "file_path": "/var/log/gene_etl/etl.log",
        "max_file_size_mb": 100,
        "backup_count": 5
    },
    "alerting": {
        "enabled": True,
        "email_recipients": ["bioinfo-team@company.com"],
        "smtp_server": "smtp.company.com",
        "smtp_port": 587
    }
}


def create_example_config():
    """Create and return an example configuration"""
    return ETLConfig(**EXAMPLE_CONFIG)