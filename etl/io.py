"""
Data I/O Module for Gene Pair Correlation ETL

This module provides functions for extracting gene expression data from databases,
loading results, and managing data snapshots using Polars for performance.
"""

import polars as pl
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import uuid
from pathlib import Path

from models import (
    DimGene, DimIllness, DimSample, DimStudy, 
    FactGeneExpression, FactGenePairCorrelation,
    SourceSnapshot, ETLProcessLog, DataValidationLog
)
from etl.config import ETLConfig

logger = logging.getLogger(__name__)


class DataIO:
    """Data I/O operations for the ETL pipeline"""
    
    def __init__(self, config: ETLConfig):
        self.config = config
        self.engine = self._create_engine()
        self.Session = sessionmaker(bind=self.engine)
    
    def _create_engine(self):
        """Create SQLAlchemy engine with connection pooling"""
        db_config = self.config.database
        connection_string = (
            f"postgresql://{db_config.username}:{db_config.password}"
            f"@{db_config.host}:{db_config.port}/{db_config.database}"
        )
        
        return create_engine(
            connection_string,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_timeout=db_config.pool_timeout,
            pool_pre_ping=True  # Verify connections before use
        )
    
    def extract_gene_metadata(self) -> pl.DataFrame:
        """Extract gene metadata from dim_gene table"""
        logger.info("Extracting gene metadata")
        
        with self.Session() as session:
            query = session.query(DimGene)
            
            # Apply gene whitelist if configured
            if self.config.gene_whitelist:
                query = query.filter(DimGene.symbol.in_(self.config.gene_whitelist))
            
            genes_df = pd.read_sql(query.statement, self.engine)
            
        return pl.from_pandas(genes_df)
    
    def extract_illness_metadata(self) -> pl.DataFrame:
        """Extract illness metadata from dim_illness table"""
        logger.info("Extracting illness metadata")
        
        with self.Session() as session:
            query = session.query(DimIllness)
            
            # Apply illness filters
            if self.config.illness_whitelist:
                query = query.filter(DimIllness.illness_code.in_(self.config.illness_whitelist))
            elif self.config.illness_blacklist:
                query = query.filter(~DimIllness.illness_code.in_(self.config.illness_blacklist))
            
            illness_df = pd.read_sql(query.statement, self.engine)
            
        return pl.from_pandas(illness_df)
    
    def extract_sample_metadata(self) -> pl.DataFrame:
        """Extract sample metadata with illness associations"""
        logger.info("Extracting sample metadata")
        
        with self.Session() as session:
            query = session.query(DimSample)
            
            # Join with illness to get illness codes
            query = query.join(DimIllness, DimSample.illness_key == DimIllness.illness_key)
            
            # Apply illness filters
            if self.config.illness_whitelist:
                query = query.filter(DimIllness.illness_code.in_(self.config.illness_whitelist))
            elif self.config.illness_blacklist:
                query = query.filter(~DimIllness.illness_code.in_(self.config.illness_blacklist))
            
            samples_df = pd.read_sql(query.statement, self.engine)
            
        return pl.from_pandas(samples_df)
    
    def extract_expression_matrix(
        self, 
        illness_key: int, 
        gene_keys: Optional[List[int]] = None
    ) -> pl.DataFrame:
        """
        Extract expression matrix for a specific illness cohort
        
        Returns DataFrame with columns: gene_key, sample_key, expression_value
        """
        logger.info(f"Extracting expression matrix for illness_key: {illness_key}")
        
        with self.Session() as session:
            # Get samples for this illness
            sample_keys = session.query(DimSample.sample_key).filter(
                DimSample.illness_key == illness_key
            ).all()
            sample_keys = [s[0] for s in sample_keys]
            
            if not sample_keys:
                logger.warning(f"No samples found for illness_key: {illness_key}")
                return pl.DataFrame(schema={'gene_key': pl.Int64, 'sample_key': pl.Int64, 'expression_value': pl.Float64})
            
            # Build expression query
            query = session.query(FactGeneExpression).filter(
                FactGeneExpression.sample_key.in_(sample_keys)
            )
            
            if gene_keys:
                query = query.filter(FactGeneExpression.gene_key.in_(gene_keys))
            
            expr_df = pd.read_sql(query.statement, self.engine)
            
        return pl.from_pandas(expr_df)
    
    def extract_expression_data_by_illness(self) -> Dict[int, pl.DataFrame]:
        """
        Extract expression data for all illnesses, returning a dictionary
        mapping illness_key to expression DataFrame
        """
        logger.info("Extracting expression data by illness")
        
        # Get illness metadata
        illness_df = self.extract_illness_metadata()
        illness_keys = illness_df['illness_key'].to_list()
        
        # Get gene metadata
        gene_df = self.extract_gene_metadata()
        gene_keys = gene_df['gene_key'].to_list()
        
        expression_data = {}
        
        for illness_key in illness_keys:
            expr_df = self.extract_expression_matrix(illness_key, gene_keys)
            if len(expr_df) > 0:
                expression_data[illness_key] = expr_df
                logger.info(f"Extracted {len(expr_df)} expression records for illness {illness_key}")
            else:
                logger.warning(f"No expression data found for illness {illness_key}")
        
        return expression_data
    
    def load_correlation_results(
        self, 
        correlation_results: List[Dict[str, Any]], 
        batch_id: str
    ) -> int:
        """Load correlation results into the database"""
        logger.info(f"Loading {len(correlation_results)} correlation results")
        
        if not correlation_results:
            logger.warning("No correlation results to load")
            return 0
        
        # Convert to DataFrame for efficient loading
        results_df = pd.DataFrame(correlation_results)
        results_df['batch_id'] = batch_id
        results_df['computed_at'] = datetime.utcnow()
        
        # Ensure proper column order
        required_columns = [
            'gene_a_key', 'gene_b_key', 'illness_key', 'rho_spearman',
            'p_value', 'q_value', 'n_samples', 'batch_id', 'computed_at',
            'source_snapshot_id', 'config_hash'
        ]
        
        for col in required_columns:
            if col not in results_df.columns:
                if col == 'source_snapshot_id':
                    results_df[col] = self.config.source_snapshot_id
                elif col == 'config_hash':
                    results_df[col] = self.config.get_config_hash()
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        results_df = results_df[required_columns]
        
        # Load in batches for better performance
        batch_size = self.config.processing.batch_size
        total_loaded = 0
        
        with self.Session() as session:
            for i in range(0, len(results_df), batch_size):
                batch_df = results_df.iloc[i:i+batch_size]
                
                # Convert to records and bulk insert
                records = batch_df.to_dict('records')
                session.bulk_insert_mappings(FactGenePairCorrelation, records)
                session.commit()
                
                total_loaded += len(records)
                logger.info(f"Loaded batch of {len(records)} records, total: {total_loaded}")
        
        return total_loaded
    
    def create_source_snapshot(self, snapshot_info: Dict[str, Any]) -> str:
        """Create a source snapshot record"""
        snapshot_id = str(uuid.uuid4())
        
        with self.Session() as session:
            snapshot = SourceSnapshot(
                snapshot_id=snapshot_id,
                snapshot_name=snapshot_info.get('name', 'gene_expression_snapshot'),
                source_type=snapshot_info.get('type', 'database'),
                source_location=snapshot_info.get('location', 'unknown'),
                record_count=snapshot_info.get('record_count'),
                checksum=snapshot_info.get('checksum', '')
            )
            session.add(snapshot)
            session.commit()
        
        return snapshot_id
    
    def log_etl_process(self, process_info: Dict[str, Any]) -> str:
        """Log ETL process information"""
        process_id = str(uuid.uuid4())
        
        with self.Session() as session:
            process = ETLProcessLog(
                process_id=process_id,
                process_name=process_info.get('name', 'gene_correlation_etl'),
                status=process_info.get('status', 'running'),
                batch_id=process_info.get('batch_id', self.config.batch_id),
                config_hash=process_info.get('config_hash', self.config.get_config_hash()),
                illness_count=process_info.get('illness_count'),
                gene_pairs_processed=process_info.get('gene_pairs_processed'),
                error_message=process_info.get('error_message'),
                performance_metrics=process_info.get('performance_metrics')
            )
            
            if 'end_time' in process_info:
                process.end_time = process_info['end_time']
            
            session.add(process)
            session.commit()
        
        return process_id
    
    def log_validation_result(self, validation_info: Dict[str, Any]) -> str:
        """Log data validation result"""
        validation_id = str(uuid.uuid4())
        
        with self.Session() as session:
            validation = DataValidationLog(
                validation_id=validation_id,
                batch_id=validation_info.get('batch_id', self.config.batch_id),
                validation_type=validation_info.get('validation_type', 'post_check'),
                validation_name=validation_info.get('validation_name', 'unknown'),
                status=validation_info.get('status', 'unknown'),
                details=validation_info.get('details', '')
            )
            session.add(validation)
            session.commit()
        
        return validation_id
    
    def get_existing_batch_ids(self, illness_key: int) -> List[str]:
        """Get existing batch IDs for an illness to avoid recomputation"""
        with self.Session() as session:
            batches = session.query(FactGenePairCorrelation.batch_id).filter(
                FactGenePairCorrelation.illness_key == illness_key
            ).distinct().all()
            
        return [b[0] for b in batches]
    
    def export_results_to_parquet(
        self, 
        output_dir: str, 
        batch_id: str = None
    ) -> List[str]:
        """Export correlation results to Parquet format for archiving"""
        logger.info(f"Exporting results to Parquet in {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        with self.Session() as session:
            # Query correlation results
            query = session.query(FactGenePairCorrelation)
            
            if batch_id:
                query = query.filter(FactGenePairCorrelation.batch_id == batch_id)
            
            # Export in batches
            offset = 0
            batch_size = 100000
            
            while True:
                batch_query = query.offset(offset).limit(batch_size)
                results_df = pd.read_sql(batch_query.statement, self.engine)
                
                if len(results_df) == 0:
                    break
                
                # Convert to Polars for efficient Parquet writing
                pl_df = pl.from_pandas(results_df)
                
                # Generate filename
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                filename = f"correlation_results_{timestamp}_batch_{offset//batch_size}.parquet"
                filepath = output_path / filename
                
                # Write Parquet file
                pl_df.write_parquet(filepath)
                exported_files.append(str(filepath))
                
                logger.info(f"Exported {len(results_df)} records to {filename}")
                offset += batch_size
        
        return exported_files
    
    def close(self):
        """Clean up database connections"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
        logger.info("Database connections closed")