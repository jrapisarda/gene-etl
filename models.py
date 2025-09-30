"""
SQLAlchemy ORM Models for Gene Pair Correlation ETL System

This module defines the database schema for the gene correlation analysis system,
following the BioinformaticsWarehouse conventions with dim, fact, and meta schemas.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, ForeignKey, 
    Text, UniqueConstraint, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class DimGene(Base):
    """Dimension table for gene metadata"""
    __tablename__ = 'dim_gene'
    __table_args__ = {'schema': 'dim'}
    
    gene_key = Column(Integer, primary_key=True, autoincrement=True)
    ensembl_id = Column(String(50), nullable=False, unique=True)
    symbol = Column(String(50), nullable=False)
    gene_name = Column(String(200))
    chromosome = Column(String(10))
    start_position = Column(Integer)
    end_position = Column(Integer)
    strand = Column(String(1))
    gene_type = Column(String(50))
    
    # Relationships
    expression_data = relationship("FactGeneExpression", back_populates="gene")
    correlation_pairs_a = relationship("FactGenePairCorrelation", 
                                     foreign_keys="FactGenePairCorrelation.gene_a_key",
                                     back_populates="gene_a")
    correlation_pairs_b = relationship("FactGenePairCorrelation", 
                                     foreign_keys="FactGenePairCorrelation.gene_b_key",
                                     back_populates="gene_b")


class DimIllness(Base):
    """Dimension table for illness/disease classifications"""
    __tablename__ = 'dim_illness'
    __table_args__ = {'schema': 'dim'}
    
    illness_key = Column(Integer, primary_key=True, autoincrement=True)
    illness_code = Column(String(50), nullable=False, unique=True)
    description = Column(String(200), nullable=False)
    category = Column(String(100))
    
    # Relationships
    samples = relationship("DimSample", back_populates="illness")
    correlations = relationship("FactGenePairCorrelation", back_populates="illness")


class DimSample(Base):
    """Dimension table for biological samples"""
    __tablename__ = 'dim_sample'
    __table_args__ = {'schema': 'dim'}
    
    sample_key = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String(100), nullable=False, unique=True)
    illness_key = Column(Integer, ForeignKey('dim.dim_illness.illness_key'), nullable=False)
    study_key = Column(Integer, ForeignKey('dim.dim_study.study_key'))
    patient_id = Column(String(100))
    age = Column(Integer)
    sex = Column(String(10))
    tissue_type = Column(String(50))
    
    # Relationships
    illness = relationship("DimIllness", back_populates="samples")
    study = relationship("DimStudy", back_populates="samples")
    expression_data = relationship("FactGeneExpression", back_populates="sample")


class DimStudy(Base):
    """Dimension table for research studies"""
    __tablename__ = 'dim_study'
    __table_args__ = {'schema': 'dim'}
    
    study_key = Column(Integer, primary_key=True, autoincrement=True)
    study_accession = Column(String(50), nullable=False, unique=True)
    platform = Column(String(100), nullable=False)
    source_snapshot_id = Column(UUID(as_uuid=True), ForeignKey('meta.source_snapshot.snapshot_id'))
    description = Column(Text)
    
    # Relationships
    samples = relationship("DimSample", back_populates="study")
    source_snapshot = relationship("SourceSnapshot", back_populates="studies")


class FactGeneExpression(Base):
    """Fact table for gene expression measurements"""
    __tablename__ = 'fact_gene_expression'
    __table_args__ = {'schema': 'fact'}
    
    expression_key = Column(Integer, primary_key=True, autoincrement=True)
    gene_key = Column(Integer, ForeignKey('dim.dim_gene.gene_key'), nullable=False)
    sample_key = Column(Integer, ForeignKey('dim.dim_sample.sample_key'), nullable=False)
    expression_value = Column(Float, nullable=False)
    normalized_expression = Column(Float)
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    
    # Relationships
    gene = relationship("DimGene", back_populates="expression_data")
    sample = relationship("DimSample", back_populates="expression_data")
    
    # Indexes for performance
    __table_args__ = (
        UniqueConstraint('gene_key', 'sample_key', name='uq_gene_sample'),
        Index('idx_gene_expression_gene', 'gene_key'),
        Index('idx_gene_expression_sample', 'sample_key'),
        CheckConstraint('expression_value IS NOT NULL', name='ck_expression_value_not_null'),
    )


class FactGenePairCorrelation(Base):
    """Fact table for gene pair correlation results"""
    __tablename__ = 'fact_gene_pair_correlation'
    __table_args__ = {'schema': 'fact'}
    
    correlation_key = Column(Integer, primary_key=True, autoincrement=True)
    gene_a_key = Column(Integer, ForeignKey('dim.dim_gene.gene_key'), nullable=False)
    gene_b_key = Column(Integer, ForeignKey('dim.dim_gene.gene_key'), nullable=False)
    illness_key = Column(Integer, ForeignKey('dim.dim_illness.illness_key'), nullable=False)
    
    # Statistical results
    rho_spearman = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    q_value = Column(Float, nullable=False)
    n_samples = Column(Integer, nullable=False)
    
    # Metadata and lineage
    computed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    source_snapshot_id = Column(UUID(as_uuid=True), ForeignKey('meta.source_snapshot.snapshot_id'))
    config_hash = Column(String(64))
    
    # Relationships
    gene_a = relationship("DimGene", foreign_keys=[gene_a_key], back_populates="correlation_pairs_a")
    gene_b = relationship("DimGene", foreign_keys=[gene_b_key], back_populates="correlation_pairs_b")
    illness = relationship("DimIllness", back_populates="correlations")
    source_snapshot = relationship("SourceSnapshot")
    
    # Indexes for performance
    __table_args__ = (
        UniqueConstraint('gene_a_key', 'gene_b_key', 'illness_key', 'batch_id', name='uq_gene_pair_illness_batch'),
        Index('idx_correlation_illness', 'illness_key'),
        Index('idx_correlation_q_value', 'q_value'),
        Index('idx_correlation_batch', 'batch_id'),
        CheckConstraint('gene_a_key < gene_b_key', name='ck_gene_order'),
        CheckConstraint('rho_spearman >= -1 AND rho_spearman <= 1', name='ck_rho_range'),
        CheckConstraint('p_value >= 0 AND p_value <= 1', name='ck_p_value_range'),
        CheckConstraint('q_value >= 0 AND q_value <= 1', name='ck_q_value_range'),
        CheckConstraint('n_samples > 0', name='ck_n_samples_positive'),
    )


# Meta Schema Tables

class SourceSnapshot(Base):
    """Meta table for tracking data source snapshots"""
    __tablename__ = 'source_snapshot'
    __table_args__ = {'schema': 'meta'}
    
    snapshot_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_name = Column(String(200), nullable=False)
    source_type = Column(String(50), nullable=False)  # 'database', 'file', 'api'
    source_location = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    record_count = Column(Integer)
    checksum = Column(String(64))
    
    # Relationships
    studies = relationship("DimStudy", back_populates="source_snapshot")


class ETLProcessLog(Base):
    """Meta table for ETL process logging"""
    __tablename__ = 'etl_process_log'
    __table_args__ = {'schema': 'meta'}
    
    process_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    process_name = Column(String(100), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime)
    status = Column(String(20), nullable=False)  # 'running', 'success', 'failed', 'degraded'
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    config_hash = Column(String(64))
    illness_count = Column(Integer)
    gene_pairs_processed = Column(Integer)
    error_message = Column(Text)
    performance_metrics = Column(Text)  # JSON string with performance data


class DataValidationLog(Base):
    """Meta table for data validation results"""
    __tablename__ = 'data_validation_log'
    __table_args__ = {'schema': 'meta'}
    
    validation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    validation_type = Column(String(50), nullable=False)  # 'pre_check', 'post_check'
    validation_name = Column(String(100), nullable=False)
    status = Column(String(20), nullable=False)  # 'pass', 'fail', 'warning'
    details = Column(Text)
    validated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Index for querying validation results by batch
    __table_args__ = (
        Index('idx_validation_batch', 'batch_id'),
    )


class Config(Base):
    """Meta table for configuration management"""
    __tablename__ = 'config'
    __table_args__ = {'schema': 'meta'}
    
    config_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    config_name = Column(String(100), nullable=False)
    config_version = Column(String(20), nullable=False)
    config_hash = Column(String(64), nullable=False, unique=True)
    config_data = Column(Text, nullable=False)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    etl_processes = relationship("ETLProcessLog", backref="config")


# Analytics Views (for reference - these would be created as SQL views)
class AnalyticsViews:
    """Container for analytics view definitions"""
    
    VW_GENE_PAIRS_CANDIDATE = """
    CREATE OR REPLACE VIEW analytics.vw_gene_pairs_candidate AS
    SELECT 
        ROW_NUMBER() OVER (ORDER BY rho_spearman DESC) as candidate_rank,
        g1.symbol as gene_a_symbol,
        g2.symbol as gene_b_symbol,
        i.illness_code,
        i.description as illness_description,
        c.rho_spearman,
        c.p_value,
        c.q_value,
        c.n_samples,
        CASE 
            WHEN c.q_value <= 0.01 THEN 'Highly Significant'
            WHEN c.q_value <= 0.05 THEN 'Significant'
            WHEN c.q_value <= 0.1 THEN 'Marginally Significant'
            ELSE 'Not Significant'
        END as significance_category,
        c.computed_at
    FROM fact.fact_gene_pair_correlation c
    JOIN dim.dim_gene g1 ON c.gene_a_key = g1.gene_key
    JOIN dim.dim_gene g2 ON c.gene_b_key = g2.gene_key
    JOIN dim.dim_illness i ON c.illness_key = i.illness_key
    WHERE c.q_value <= 0.1
    ORDER BY rho_spearman DESC;
    """
    
    VW_CORRELATION_SUMMARY = """
    CREATE OR REPLACE VIEW analytics.vw_correlation_summary AS
    SELECT 
        i.illness_code,
        i.description as illness_description,
        COUNT(*) as total_pairs,
        COUNT(CASE WHEN q_value <= 0.05 THEN 1 END) as significant_pairs,
        COUNT(CASE WHEN q_value <= 0.01 THEN 1 END) as highly_significant_pairs,
        AVG(ABS(rho_spearman)) as avg_abs_correlation,
        MIN(rho_spearman) as min_correlation,
        MAX(rho_spearman) as max_correlation,
        AVG(n_samples) as avg_sample_size,
        MAX(computed_at) as last_computed
    FROM fact.fact_gene_pair_correlation c
    JOIN dim.dim_illness i ON c.illness_key = i.illness_key
    GROUP BY i.illness_key, i.illness_code, i.description;
    """