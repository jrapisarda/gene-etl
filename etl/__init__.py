"""
Gene Pair Correlation ETL Pipeline

This module orchestrates the complete ETL pipeline for gene pair correlation analysis,
including data extraction, filtering, correlation computation, and result persistence.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

from etl.config import ETLConfig
from etl.io import DataIO
from etl.filters import GeneFilter
from etl.correlate import GeneCorrelationAnalyzer
from etl.persist import ResultPersister
from etl.validate import DataValidator

logger = logging.getLogger(__name__)


class GeneCorrelationETL:
    """Main ETL pipeline orchestrator"""
    
    def __init__(self, config: ETLConfig):
        self.config = config
        self.data_io = DataIO(config)
        self.gene_filter = GeneFilter(config.gene_filter)
        self.correlation_analyzer = GeneCorrelationAnalyzer(config.statistical)
        self.persister = ResultPersister(config, self.data_io)
        self.validator = DataValidator(config, self.data_io)
        
        # Initialize process tracking
        self.process_id = None
        self.start_time = None
        self.performance_metrics = {}
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete ETL pipeline
        
        Returns:
            Dictionary with execution results and statistics
        """
        logger.info(f"Starting Gene Correlation ETL Pipeline - Batch ID: {self.config.batch_id}")
        
        self.start_time = time.time()
        
        try:
            # Initialize process logging
            self._initialize_process_log()
            
            # Phase 1: Data Extraction
            logger.info("Phase 1: Data Extraction")
            extraction_results = self._extract_data()
            
            # Phase 2: Data Validation (Pre-processing)
            logger.info("Phase 2: Pre-processing Validation")
            pre_validation = self.validator.run_pre_processing_validation(
                extraction_results["expression_data"],
                extraction_results["gene_metadata"],
                extraction_results["illness_metadata"]
            )
            
            # Phase 3: Gene Filtering
            logger.info("Phase 3: Gene Filtering")
            filtering_results = self._filter_genes(extraction_results["expression_data"])
            
            # Phase 4: Correlation Analysis
            logger.info("Phase 4: Correlation Analysis")
            correlation_results = self._compute_correlations(
                filtering_results["filtered_expression_data"],
                filtering_results["selected_genes"]
            )
            
            # Phase 5: Data Validation (Post-processing)
            logger.info("Phase 5: Post-processing Validation")
            post_validation = self.validator.run_post_processing_validation(
                correlation_results,
                self.config.batch_id
            )
            
            # Phase 6: Result Persistence
            logger.info("Phase 6: Result Persistence")
            persistence_results = self._persist_results(correlation_results)
            
            # Phase 7: Cleanup and Finalization
            logger.info("Phase 7: Finalization")
            final_results = self._finalize_process(
                extraction_results,
                filtering_results,
                correlation_results,
                persistence_results,
                pre_validation,
                post_validation
            )
            
            logger.info("ETL Pipeline completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            self._handle_failure(e)
            raise
            
        finally:
            self._cleanup()
    
    def _initialize_process_log(self) -> None:
        """Initialize process logging"""
        process_info = {
            "name": "gene_correlation_etl",
            "status": "running",
            "batch_id": self.config.batch_id,
            "config_hash": self.config.get_config_hash()
        }
        
        self.process_id = self.data_io.log_etl_process(process_info)
        logger.info(f"Initialized process log: {self.process_id}")
    
    def _extract_data(self) -> Dict[str, Any]:
        """Extract all required data from database"""
        logger.info("Extracting data from database")
        
        start_time = time.time()
        
        # Extract metadata
        gene_metadata = self.data_io.extract_gene_metadata()
        illness_metadata = self.data_io.extract_illness_metadata()
        
        # Extract expression data
        expression_data = self.data_io.extract_expression_data_by_illness()
        
        extraction_time = time.time() - start_time
        
        results = {
            "gene_metadata": gene_metadata,
            "illness_metadata": illness_metadata,
            "expression_data": expression_data,
            "extraction_time": extraction_time
        }
        
        logger.info(f"Data extraction completed in {extraction_time:.2f} seconds")
        logger.info(f"Extracted {len(gene_metadata)} genes, {len(illness_metadata)} illnesses, {sum(len(df) for df in expression_data.values())} expression records")
        
        return results
    
    def _filter_genes(self, expression_data: Dict[int, pl.DataFrame]) -> Dict[str, Any]:
        """Apply gene filtering to select informative genes"""
        logger.info("Filtering genes for analysis")
        
        start_time = time.time()
        
        selected_genes = {}
        filtered_expression_data = {}
        filtering_statistics = {}
        
        for illness_key, expression_df in expression_data.items():
            logger.info(f"Filtering genes for illness {illness_key}")
            
            # Apply comprehensive filtering
            genes, filtered_df = self.gene_filter.apply_comprehensive_filtering(expression_df)
            
            if len(genes) >= 2:  # Need at least 2 genes for correlation
                selected_genes[illness_key] = genes
                filtered_expression_data[illness_key] = filtered_df
                
                # Get filtering statistics
                stats = self.gene_filter.get_filtering_statistics(
                    expression_df, filtered_df, genes
                )
                filtering_statistics[illness_key] = stats
                
                logger.info(f"Illness {illness_key}: selected {len(genes)} genes")
            else:
                logger.warning(f"Illness {illness_key}: insufficient genes after filtering ({len(genes)})")
        
        filtering_time = time.time() - start_time
        
        results = {
            "selected_genes": selected_genes,
            "filtered_expression_data": filtered_expression_data,
            "filtering_statistics": filtering_statistics,
            "filtering_time": filtering_time
        }
        
        logger.info(f"Gene filtering completed in {filtering_time:.2f} seconds")
        logger.info(f"Selected genes for {len(selected_genes)} illnesses")
        
        return results
    
    def _compute_correlations(
        self, 
        expression_data: Dict[int, pl.DataFrame],
        selected_genes: Dict[int, List[int]]
    ) -> List[Dict[str, Any]]:
        """Compute gene pair correlations"""
        logger.info("Computing gene pair correlations")
        
        start_time = time.time()
        
        correlation_results = self.correlation_analyzer.analyze_multiple_cohorts(
            expression_data,
            selected_genes,
            max_workers=self.config.processing.max_workers
        )
        
        correlation_time = time.time() - start_time
        
        # Get statistics
        statistics = self.correlation_analyzer.get_correlation_statistics(correlation_results)
        
        logger.info(f"Correlation computation completed in {correlation_time:.2f} seconds")
        logger.info(f"Computed {len(correlation_results)} correlations")
        logger.info(f"Statistics: {statistics}")
        
        return correlation_results
    
    def _persist_results(self, correlation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Persist correlation results to database"""
        logger.info("Persisting correlation results")
        
        start_time = time.time()
        
        successful, failed = self.persister.persist_correlation_results(
            correlation_results, self.config.batch_id
        )
        
        persistence_time = time.time() - start_time
        
        results = {
            "successful_records": successful,
            "failed_records": failed,
            "total_records": successful + failed,
            "success_rate": successful / (successful + failed) if (successful + failed) > 0 else 0,
            "persistence_time": persistence_time
        }
        
        logger.info(f"Persistence completed in {persistence_time:.2f} seconds")
        logger.info(f"Results: {results}")
        
        return results
    
    def _finalize_process(
        self,
        extraction_results: Dict[str, Any],
        filtering_results: Dict[str, Any],
        correlation_results: List[Dict[str, Any]],
        persistence_results: Dict[str, Any],
        pre_validation: Dict[str, Any],
        post_validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalize the ETL process and generate results"""
        
        total_time = time.time() - self.start_time
        
        # Update process log with completion
        completion_updates = {
            "status": "success",
            "end_time": datetime.utcnow(),
            "illness_count": len(filtering_results["selected_genes"]),
            "gene_pairs_processed": persistence_results["successful_records"],
            "performance_metrics": {
                "total_time": total_time,
                "extraction_time": extraction_results["extraction_time"],
                "filtering_time": filtering_results["filtering_time"],
                "persistence_time": persistence_results["persistence_time"]
            }
        }
        
        self.persister.update_process_log(self.process_id, completion_updates)
        
        # Generate final results
        final_results = {
            "batch_id": self.config.batch_id,
            "process_id": self.process_id,
            "execution_time": total_time,
            "status": "success",
            "summary": {
                "genes_processed": sum(len(genes) for genes in filtering_results["selected_genes"].values()),
                "illnesses_processed": len(filtering_results["selected_genes"]),
                "correlations_computed": len(correlation_results),
                "correlations_persisted": persistence_results["successful_records"],
                "success_rate": persistence_results["success_rate"]
            },
            "validation": {
                "pre_processing": {
                    "passed": sum(1 for r in pre_validation["results"] if r["status"] == "pass"),
                    "failed": sum(1 for r in pre_validation["results"] if r["status"] == "fail"),
                    "warnings": sum(1 for r in pre_validation["results"] if r["status"] == "warning")
                },
                "post_processing": {
                    "passed": sum(1 for r in post_validation["results"] if r["status"] == "pass"),
                    "failed": sum(1 for r in post_validation["results"] if r["status"] == "fail"),
                    "warnings": sum(1 for r in post_validation["results"] if r["status"] == "warning")
                }
            },
            "performance_metrics": completion_updates["performance_metrics"]
        }
        
        return final_results
    
    def _handle_failure(self, error: Exception) -> None:
        """Handle ETL pipeline failure"""
        if self.process_id:
            failure_updates = {
                "status": "failed",
                "end_time": datetime.utcnow(),
                "error_message": str(error)
            }
            self.persister.update_process_log(self.process_id, failure_updates)
    
    def _cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self.data_io, 'close'):
            self.data_io.close()
        logger.info("ETL cleanup completed")


def run_etl_pipeline(config_path: str = None, config: ETLConfig = None) -> Dict[str, Any]:
    """
    Convenience function to run the ETL pipeline
    
    Args:
        config_path: Path to configuration file
        config: ETLConfig object (alternative to config_path)
        
    Returns:
        ETL execution results
    """
    
    if config_path:
        config = ETLConfig.from_file(config_path)
    elif not config:
        raise ValueError("Either config_path or config must be provided")
    
    etl_pipeline = GeneCorrelationETL(config)
    return etl_pipeline.run()