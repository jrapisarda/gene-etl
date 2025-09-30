"""
Data Persistence Module for ETL Pipeline

This module handles the persistence of correlation results to the database
with proper error handling, batching, and lineage tracking.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import polars as pl

from models import (
    FactGenePairCorrelation, ETLProcessLog, DataValidationLog, Config
)
from etl.config import ETLConfig
from etl.io import DataIO

logger = logging.getLogger(__name__)


class ResultPersister:
    """Handles persistence of ETL results with batching and error handling"""
    
    def __init__(self, config: ETLConfig, data_io: DataIO):
        self.config = config
        self.data_io = data_io
        self.batch_size = config.processing.batch_size
        self.config_hash = config.get_config_hash()
    
    def persist_correlation_results(
        self, 
        correlation_results: List[Dict[str, Any]],
        batch_id: str
    ) -> Tuple[int, int]:
        """
        Persist correlation results with batching and error handling
        
        Args:
            correlation_results: List of correlation result dictionaries
            batch_id: Batch ID for this run
            
        Returns:
            Tuple of (successful_records, failed_records)
        """
        logger.info(f"Persisting {len(correlation_results)} correlation results")
        
        if not correlation_results:
            logger.warning("No correlation results to persist")
            return 0, 0
        
        # Prepare results for persistence
        prepared_results = self._prepare_results_for_persistence(correlation_results, batch_id)
        
        # Batch insert with error handling
        successful_count = 0
        failed_count = 0
        failed_records = []
        
        for i in range(0, len(prepared_results), self.batch_size):
            batch = prepared_results[i:i + self.batch_size]
            
            try:
                batch_success, batch_failed = self._persist_batch(batch)
                successful_count += batch_success
                failed_count += batch_failed
                
                if batch_failed > 0:
                    failed_records.extend(batch[len(batch) - batch_failed:])
                
                logger.info(f"Persisted batch: {batch_success} successful, {batch_failed} failed")
                
            except Exception as e:
                logger.error(f"Error persisting batch {i//self.batch_size}: {e}")
                failed_count += len(batch)
                failed_records.extend(batch)
        
        # Log failed records if any
        if failed_records:
            self._log_failed_records(failed_records, batch_id)
        
        logger.info(f"Persistence complete: {successful_count} successful, {failed_count} failed")
        return successful_count, failed_count
    
    def _prepare_results_for_persistence(
        self, 
        correlation_results: List[Dict[str, Any]], 
        batch_id: str
    ) -> List[Dict[str, Any]]:
        """Prepare correlation results for database persistence"""
        
        prepared_results = []
        
        for result in correlation_results:
            # Ensure all required fields are present
            prepared_result = {
                "gene_a_key": result["gene_a_key"],
                "gene_b_key": result["gene_b_key"],
                "illness_key": result["illness_key"],
                "rho_spearman": result["rho_spearman"],
                "p_value": result["p_value"],
                "q_value": result["q_value"],
                "n_samples": result["n_samples"],
                "batch_id": batch_id,
                "computed_at": result.get("computed_at", datetime.utcnow()),
                "source_snapshot_id": result.get("source_snapshot_id", self.config.source_snapshot_id),
                "config_hash": result.get("config_hash", self.config_hash)
            }
            
            # Ensure gene_a_key < gene_b_key for consistency
            if prepared_result["gene_a_key"] > prepared_result["gene_b_key"]:
                prepared_result["gene_a_key"], prepared_result["gene_b_key"] = \
                    prepared_result["gene_b_key"], prepared_result["gene_a_key"]
            
            prepared_results.append(prepared_result)
        
        return prepared_results
    
    def _persist_batch(self, batch: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Persist a batch of correlation results"""
        
        session = self.data_io.Session()
        
        try:
            # Use bulk insert for better performance
            session.bulk_insert_mappings(FactGenePairCorrelation, batch)
            session.commit()
            return len(batch), 0
            
        except IntegrityError as e:
            session.rollback()
            logger.warning(f"Integrity error in batch, falling back to individual inserts: {e}")
            return self._persist_individual_records(session, batch)
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error persisting batch: {e}")
            return 0, len(batch)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error persisting batch: {e}")
            return 0, len(batch)
            
        finally:
            session.close()
    
    def _persist_individual_records(self, session, records: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Fallback method to persist records individually"""
        successful = 0
        failed = 0
        
        for record in records:
            try:
                correlation = FactGenePairCorrelation(**record)
                session.add(correlation)
                session.commit()
                successful += 1
                
            except IntegrityError:
                session.rollback()
                # Check if record already exists
                existing = session.query(FactGenePairCorrelation).filter_by(
                    gene_a_key=record["gene_a_key"],
                    gene_b_key=record["gene_b_key"],
                    illness_key=record["illness_key"],
                    batch_id=record["batch_id"]
                ).first()
                
                if existing:
                    logger.debug(f"Record already exists, skipping: {record}")
                    successful += 1  # Count as successful since it's already there
                else:
                    failed += 1
                    
            except Exception as e:
                session.rollback()
                logger.error(f"Error persisting individual record: {e}")
                failed += 1
        
        return successful, failed
    
    def _log_failed_records(self, failed_records: List[Dict[str, Any]], batch_id: str) -> None:
        """Log failed records for debugging"""
        logger.error(f"Failed to persist {len(failed_records)} records in batch {batch_id}")
        
        for i, record in enumerate(failed_records[:10]):  # Log first 10 failures
            logger.error(f"Failed record {i}: {record}")
    
    def update_process_log(
        self, 
        process_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update ETL process log with new information"""
        
        try:
            session = self.data_io.Session()
            
            process_log = session.query(ETLProcessLog).filter_by(
                process_id=process_id
            ).first()
            
            if not process_log:
                logger.error(f"Process log not found: {process_id}")
                return False
            
            # Update fields
            for key, value in updates.items():
                if hasattr(process_log, key):
                    setattr(process_log, key, value)
                else:
                    logger.warning(f"Invalid field in process log update: {key}")
            
            session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating process log: {e}")
            session.rollback()
            return False
            
        finally:
            session.close()
    
    def log_validation_result(
        self, 
        validation_info: Dict[str, Any]
    ) -> str:
        """Log a validation result"""
        
        try:
            validation_id = self.data_io.log_validation_result(validation_info)
            logger.info(f"Logged validation result: {validation_info.get('validation_name', 'unknown')}")
            return validation_id
            
        except Exception as e:
            logger.error(f"Error logging validation result: {e}")
            return None
    
    def save_configuration(self, config: ETLConfig) -> str:
        """Save configuration to database for lineage tracking"""
        
        try:
            session = self.data_io.Session()
            
            config_record = Config(
                config_name=config.config_name,
                config_version=config.config_version,
                config_hash=config.get_config_hash(),
                config_data=str(config.to_dict()),
                is_active=True
            )
            
            session.add(config_record)
            session.commit()
            
            logger.info(f"Saved configuration: {config_record.config_id}")
            return str(config_record.config_id)
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            session.rollback()
            return None
            
        finally:
            session.close()
    
    def check_existing_results(
        self, 
        illness_key: int, 
        batch_id: str
    ) -> bool:
        """Check if results already exist for this illness and batch"""
        
        try:
            session = self.data_io.Session()
            
            existing_count = session.query(FactGenePairCorrelation).filter_by(
                illness_key=illness_key,
                batch_id=batch_id
            ).count()
            
            return existing_count > 0
            
        except Exception as e:
            logger.error(f"Error checking existing results: {e}")
            return False
            
        finally:
            session.close()
    
    def get_processing_statistics(self, batch_id: str) -> Dict[str, Any]:
        """Get statistics for a batch of processed data"""
        
        try:
            session = self.data_io.Session()
            
            # Count records by illness
            illness_counts = session.query(
                FactGenePairCorrelation.illness_key,
                func.count(FactGenePairCorrelation.correlation_key).label('count')
            ).filter(
                FactGenePairCorrelation.batch_id == batch_id
            ).group_by(
                FactGenePairCorrelation.illness_key
            ).all()
            
            # Get total records and statistics
            total_records = session.query(FactGenePairCorrelation).filter(
                FactGenePairCorrelation.batch_id == batch_id
            ).count()
            
            significant_records = session.query(FactGenePairCorrelation).filter(
                FactGenePairCorrelation.batch_id == batch_id,
                FactGenePairCorrelation.q_value <= 0.05
            ).count()
            
            return {
                "batch_id": batch_id,
                "total_records": total_records,
                "significant_records": significant_records,
                "illness_breakdown": dict(illness_counts),
                "significance_rate": significant_records / total_records if total_records > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting processing statistics: {e}")
            return {}
            
        finally:
            session.close()


def create_checkpoint(
    results: List[Dict[str, Any]], 
    checkpoint_path: str, 
    batch_id: str
) -> bool:
    """Create a checkpoint of results to disk"""
    
    try:
        import json
        import gzip
        
        checkpoint_data = {
            "batch_id": batch_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results_count": len(results),
            "results": results[:1000]  # Save first 1000 results for debugging
        }
        
        with gzip.open(checkpoint_path, 'wt', encoding='utf-8') as f:
            json.dump(checkpoint_data, f)
        
        logger.info(f"Created checkpoint: {checkpoint_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating checkpoint: {e}")
        return False


def load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Load a checkpoint from disk"""
    
    try:
        import json
        import gzip
        
        with gzip.open(checkpoint_path, 'rt', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        logger.info(f"Loaded checkpoint: {checkpoint_data['batch_id']}")
        return checkpoint_data
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return None