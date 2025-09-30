"""
Data Validation Module for ETL Pipeline

This module implements comprehensive data validation checks to ensure data quality
and integrity throughout the ETL process.
"""

import polars as pl
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from sqlalchemy.orm import sessionmaker

from etl.config import ETLConfig
from etl.io import DataIO
from models import (
    DimGene, DimIllness, DimSample, FactGeneExpression,
    FactGenePairCorrelation, DataValidationLog
)

logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation for ETL pipeline"""
    
    def __init__(self, config: ETLConfig, data_io: DataIO):
        self.config = config
        self.data_io = data_io
        self.validation_results = []
    
    def run_pre_processing_validation(
        self,
        expression_data: Dict[int, pl.DataFrame],
        gene_metadata: pl.DataFrame,
        illness_metadata: pl.DataFrame
    ) -> Dict[str, Any]:
        """
        Run validation checks before processing
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Running pre-processing validation")
        
        validation_results = {
            "validation_type": "pre_check",
            "timestamp": datetime.utcnow().isoformat(),
            "results": []
        }
        
        # Check 1: Minimum sample count per illness
        self._validate_sample_counts(expression_data, validation_results)
        
        # Check 2: Gene metadata completeness
        self._validate_gene_metadata(gene_metadata, validation_results)
        
        # Check 3: Illness metadata completeness
        self._validate_illness_metadata(illness_metadata, validation_results)
        
        # Check 4: Expression data quality
        self._validate_expression_data(expression_data, validation_results)
        
        # Check 5: Data consistency
        self._validate_data_consistency(expression_data, gene_metadata, illness_metadata, validation_results)
        
        # Log validation results
        self._log_validation_results(validation_results)
        
        # Check if validation passed
        failed_validations = [r for r in validation_results["results"] if r["status"] == "fail"]
        
        if failed_validations:
            logger.error(f"Pre-processing validation failed: {len(failed_validations)} checks failed")
            for validation in failed_validations:
                logger.error(f"Failed check: {validation['name']} - {validation['details']}")
            
            # Fail hard if critical validations fail
            critical_failed = [v for v in failed_validations if v.get("critical", False)]
            if critical_failed:
                raise ValueError(f"Critical validation failures: {[v['name'] for v in critical_failed]}")
        
        logger.info("Pre-processing validation completed")
        return validation_results
    
    def run_post_processing_validation(
        self,
        correlation_results: List[Dict[str, Any]],
        batch_id: str
    ) -> Dict[str, Any]:
        """
        Run validation checks after processing
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Running post-processing validation")
        
        validation_results = {
            "validation_type": "post_check",
            "timestamp": datetime.utcnow().isoformat(),
            "batch_id": batch_id,
            "results": []
        }
        
        # Check 1: Correlation coefficient range
        self._validate_correlation_ranges(correlation_results, validation_results)
        
        # Check 2: P-value and Q-value ranges
        self._validate_p_value_ranges(correlation_results, validation_results)
        
        # Check 3: Sample count validation
        self._validate_sample_counts_results(correlation_results, validation_results)
        
        # Check 4: Gene pair uniqueness
        self._validate_gene_pair_uniqueness(correlation_results, validation_results)
        
        # Check 5: Statistical significance consistency
        self._validate_statistical_consistency(correlation_results, validation_results)
        
        # Check 6: Data completeness
        self._validate_data_completeness(correlation_results, validation_results)
        
        # Log validation results
        self._log_validation_results(validation_results)
        
        # Check if validation passed
        failed_validations = [r for r in validation_results["results"] if r["status"] == "fail"]
        
        if failed_validations:
            logger.error(f"Post-processing validation failed: {len(failed_validations)} checks failed")
            for validation in failed_validations:
                logger.error(f"Failed check: {validation['name']} - {validation['details']}")
        
        logger.info("Post-processing validation completed")
        return validation_results
    
    def _validate_sample_counts(
        self, 
        expression_data: Dict[int, pl.DataFrame], 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate minimum sample counts per illness"""
        
        for illness_key, df in expression_data.items():
            n_samples = df["sample_key"].n_unique()
            
            if n_samples < self.config.statistical.min_samples:
                validation_results["results"].append({
                    "name": "minimum_sample_count",
                    "status": "fail",
                    "details": f"Illness {illness_key}: {n_samples} samples < {self.config.statistical.min_samples} minimum",
                    "illness_key": illness_key,
                    "critical": True
                })
            else:
                validation_results["results"].append({
                    "name": "minimum_sample_count",
                    "status": "pass",
                    "details": f"Illness {illness_key}: {n_samples} samples >= {self.config.statistical.min_samples} minimum",
                    "illness_key": illness_key
                })
    
    def _validate_gene_metadata(
        self, 
        gene_metadata: pl.DataFrame, 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate gene metadata completeness"""
        
        required_fields = ["gene_key", "ensembl_id", "symbol"]
        
        for field in required_fields:
            if field not in gene_metadata.columns:
                validation_results["results"].append({
                    "name": "gene_metadata_completeness",
                    "status": "fail",
                    "details": f"Missing required field: {field}",
                    "critical": True
                })
                return
        
        # Check for missing values
        missing_values = {}
        for field in required_fields:
            missing_count = gene_metadata[field].is_null().sum()
            if missing_count > 0:
                missing_values[field] = missing_count
        
        if missing_values:
            validation_results["results"].append({
                "name": "gene_metadata_completeness",
                "status": "warning",
                "details": f"Missing values found: {missing_values}",
                "missing_values": missing_values
            })
        else:
            validation_results["results"].append({
                "name": "gene_metadata_completeness",
                "status": "pass",
                "details": "All required gene metadata fields present and complete"
            })
    
    def _validate_illness_metadata(
        self, 
        illness_metadata: pl.DataFrame, 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate illness metadata completeness"""
        
        required_fields = ["illness_key", "illness_code", "description"]
        
        for field in required_fields:
            if field not in illness_metadata.columns:
                validation_results["results"].append({
                    "name": "illness_metadata_completeness",
                    "status": "fail",
                    "details": f"Missing required field: {field}",
                    "critical": True
                })
                return
        
        # Check for missing values
        missing_values = {}
        for field in required_fields:
            missing_count = illness_metadata[field].is_null().sum()
            if missing_count > 0:
                missing_values[field] = missing_count
        
        if missing_values:
            validation_results["results"].append({
                "name": "illness_metadata_completeness",
                "status": "warning",
                "details": f"Missing values found: {missing_values}",
                "missing_values": missing_values
            })
        else:
            validation_results["results"].append({
                "name": "illness_metadata_completeness",
                "status": "pass",
                "details": "All required illness metadata fields present and complete"
            })
    
    def _validate_expression_data(
        self, 
        expression_data: Dict[int, pl.DataFrame], 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate expression data quality"""
        
        total_records = 0
        total_missing = 0
        
        for illness_key, df in expression_data.items():
            # Check for missing values
            missing_count = df["expression_value"].is_null().sum()
            total_records += len(df)
            total_missing += missing_count
            
            # Check for negative values (biologically questionable)
            negative_count = (df["expression_value"] < 0).sum()
            
            if missing_count > 0:
                validation_results["results"].append({
                    "name": "expression_data_quality",
                    "status": "warning",
                    "details": f"Illness {illness_key}: {missing_count} missing expression values",
                    "illness_key": illness_key,
                    "missing_count": missing_count
                })
            
            if negative_count > 0:
                validation_results["results"].append({
                    "name": "expression_data_quality",
                    "status": "warning",
                    "details": f"Illness {illness_key}: {negative_count} negative expression values",
                    "illness_key": illness_key,
                    "negative_count": negative_count
                })
        
        if total_missing == 0:
            validation_results["results"].append({
                "name": "expression_data_quality",
                "status": "pass",
                "details": f"No missing expression values in {total_records} total records"
            })
    
    def _validate_data_consistency(
        self, 
        expression_data: Dict[int, pl.DataFrame],
        gene_metadata: pl.DataFrame,
        illness_metadata: pl.DataFrame,
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate data consistency across datasets"""
        
        # Check gene consistency
        expression_genes = set()
        for df in expression_data.values():
            expression_genes.update(df["gene_key"].unique())
        
        metadata_genes = set(gene_metadata["gene_key"].unique())
        
        genes_in_expression_not_in_metadata = expression_genes - metadata_genes
        genes_in_metadata_not_in_expression = metadata_genes - expression_genes
        
        if genes_in_expression_not_in_metadata:
            validation_results["results"].append({
                "name": "data_consistency",
                "status": "fail",
                "details": f"Genes in expression data but not in metadata: {len(genes_in_expression_not_in_metadata)}",
                "gene_count": len(genes_in_expression_not_in_metadata),
                "critical": True
            })
        
        if genes_in_metadata_not_in_expression:
            validation_results["results"].append({
                "name": "data_consistency",
                "status": "warning",
                "details": f"Genes in metadata but not in expression data: {len(genes_in_metadata_not_in_expression)}",
                "gene_count": len(genes_in_metadata_not_in_expression)
            })
        
        # Check illness consistency
        expression_illnesses = set(expression_data.keys())
        metadata_illnesses = set(illness_metadata["illness_key"].unique())
        
        illnesses_in_expression_not_in_metadata = expression_illnesses - metadata_illnesses
        illnesses_in_metadata_not_in_expression = metadata_illnesses - expression_illnesses
        
        if illnesses_in_expression_not_in_metadata:
            validation_results["results"].append({
                "name": "data_consistency",
                "status": "fail",
                "details": f"Illnesses in expression data but not in metadata: {len(illnesses_in_expression_not_in_metadata)}",
                "illness_count": len(illnesses_in_expression_not_in_metadata),
                "critical": True
            })
        
        if len(genes_in_expression_not_in_metadata) == 0 and len(illnesses_in_expression_not_in_metadata) == 0:
            validation_results["results"].append({
                "name": "data_consistency",
                "status": "pass",
                "details": "Data consistency validated across all datasets"
            })
    
    def _validate_correlation_ranges(
        self, 
        correlation_results: List[Dict[str, Any]], 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate correlation coefficient ranges"""
        
        invalid_correlations = []
        for i, result in enumerate(correlation_results):
            rho = result["rho_spearman"]
            if not (-1 <= rho <= 1):
                invalid_correlations.append((i, rho))
        
        if invalid_correlations:
            validation_results["results"].append({
                "name": "correlation_ranges",
                "status": "fail",
                "details": f"Invalid correlation coefficients found: {len(invalid_correlations)}",
                "invalid_count": len(invalid_correlations),
                "examples": invalid_correlations[:5]  # Show first 5 examples
            })
        else:
            validation_results["results"].append({
                "name": "correlation_ranges",
                "status": "pass",
                "details": f"All {len(correlation_results)} correlation coefficients in valid range [-1, 1]"
            })
    
    def _validate_p_value_ranges(
        self, 
        correlation_results: List[Dict[str, Any]], 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate p-value and q-value ranges"""
        
        invalid_p_values = []
        invalid_q_values = []
        
        for i, result in enumerate(correlation_results):
            if not (0 <= result["p_value"] <= 1):
                invalid_p_values.append((i, result["p_value"]))
            
            if not (0 <= result["q_value"] <= 1):
                invalid_q_values.append((i, result["q_value"]))
        
        if invalid_p_values or invalid_q_values:
            if invalid_p_values:
                validation_results["results"].append({
                    "name": "p_value_ranges",
                    "status": "fail",
                    "details": f"Invalid p-values found: {len(invalid_p_values)}",
                    "invalid_count": len(invalid_p_values),
                    "examples": invalid_p_values[:5]
                })
            
            if invalid_q_values:
                validation_results["results"].append({
                    "name": "q_value_ranges",
                    "status": "fail",
                    "details": f"Invalid q-values found: {len(invalid_q_values)}",
                    "invalid_count": len(invalid_q_values),
                    "examples": invalid_q_values[:5]
                })
        else:
            validation_results["results"].append({
                "name": "value_ranges",
                "status": "pass",
                "details": f"All p-values and q-values in valid range [0, 1] for {len(correlation_results)} results"
            })
    
    def _validate_sample_counts_results(
        self, 
        correlation_results: List[Dict[str, Any]], 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate sample counts in results"""
        
        invalid_sample_counts = []
        
        for i, result in enumerate(correlation_results):
            n_samples = result["n_samples"]
            if n_samples <= 0 or n_samples > 10000:  # Reasonable upper bound
                invalid_sample_counts.append((i, n_samples))
        
        if invalid_sample_counts:
            validation_results["results"].append({
                "name": "sample_counts",
                "status": "fail",
                "details": f"Invalid sample counts found: {len(invalid_sample_counts)}",
                "invalid_count": len(invalid_sample_counts),
                "examples": invalid_sample_counts[:5]
            })
        else:
            validation_results["results"].append({
                "name": "sample_counts",
                "status": "pass",
                "details": f"All sample counts valid for {len(correlation_results)} results"
            })
    
    def _validate_gene_pair_uniqueness(
        self, 
        correlation_results: List[Dict[str, Any]], 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate gene pair uniqueness within illness"""
        
        seen_pairs = set()
        duplicates = []
        
        for i, result in enumerate(correlation_results):
            pair_key = (result["illness_key"], result["gene_a_key"], result["gene_b_key"])
            if pair_key in seen_pairs:
                duplicates.append((i, pair_key))
            seen_pairs.add(pair_key)
        
        if duplicates:
            validation_results["results"].append({
                "name": "gene_pair_uniqueness",
                "status": "fail",
                "details": f"Duplicate gene pairs found: {len(duplicates)}",
                "duplicate_count": len(duplicates),
                "examples": duplicates[:5]
            })
        else:
            validation_results["results"].append({
                "name": "gene_pair_uniqueness",
                "status": "pass",
                "details": f"All gene pairs unique within illnesses for {len(correlation_results)} results"
            })
    
    def _validate_statistical_consistency(
        self, 
        correlation_results: List[Dict[str, Any]], 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate statistical consistency"""
        
        inconsistent_results = []
        
        for i, result in enumerate(correlation_results):
            rho = result["rho_spearman"]
            p_value = result["p_value"]
            q_value = result["q_value"]
            
            # Check if correlation magnitude matches significance
            # Very high correlations should have low p-values
            if abs(rho) > 0.8 and p_value > 0.1:
                inconsistent_results.append((i, "high_correlation_high_pvalue", rho, p_value))
            
            # Q-value should be >= p-value (FDR correction)
            if q_value < p_value - 1e-10:  # Allow small numerical errors
                inconsistent_results.append((i, "qvalue_less_than_pvalue", p_value, q_value))
        
        if inconsistent_results:
            validation_results["results"].append({
                "name": "statistical_consistency",
                "status": "warning",
                "details": f"Statistical inconsistencies found: {len(inconsistent_results)}",
                "inconsistent_count": len(inconsistent_results),
                "examples": inconsistent_results[:5]
            })
        else:
            validation_results["results"].append({
                "name": "statistical_consistency",
                "status": "pass",
                "details": f"Statistical consistency validated for {len(correlation_results)} results"
            })
    
    def _validate_data_completeness(
        self, 
        correlation_results: List[Dict[str, Any]], 
        validation_results: Dict[str, Any]
    ) -> None:
        """Validate data completeness"""
        
        required_fields = ["gene_a_key", "gene_b_key", "illness_key", "rho_spearman", "p_value", "q_value", "n_samples"]
        
        incomplete_records = []
        
        for i, result in enumerate(correlation_results):
            missing_fields = [field for field in required_fields if field not in result or result[field] is None]
            if missing_fields:
                incomplete_records.append((i, missing_fields))
        
        if incomplete_records:
            validation_results["results"].append({
                "name": "data_completeness",
                "status": "fail",
                "details": f"Incomplete records found: {len(incomplete_records)}",
                "incomplete_count": len(incomplete_records),
                "examples": incomplete_records[:5]
            })
        else:
            validation_results["results"].append({
                "name": "data_completeness",
                "status": "pass",
                "details": f"All {len(correlation_results)} records have complete required fields"
            })
    
    def _log_validation_results(self, validation_results: Dict[str, Any]) -> None:
        """Log validation results to database"""
        
        for result in validation_results["results"]:
            validation_info = {
                "batch_id": validation_results.get("batch_id", self.config.batch_id),
                "validation_type": validation_results["validation_type"],
                "validation_name": result["name"],
                "status": result["status"],
                "details": result.get("details", "")
            }
            
            self.data_io.log_validation_result(validation_info)
    
    def generate_validation_report(
        self, 
        pre_validation: Dict[str, Any], 
        post_validation: Dict[str, Any]
    ) -> str:
        """Generate a comprehensive validation report"""
        
        report = []
        report.append("=" * 80)
        report.append("ETL VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append(f"Batch ID: {self.config.batch_id}")
        report.append(f"Config Hash: {self.config.get_config_hash()}")
        report.append("")
        
        # Pre-processing validation summary
        report.append("PRE-PROCESSING VALIDATION")
        report.append("-" * 40)
        pre_passed = sum(1 for r in pre_validation["results"] if r["status"] == "pass")
        pre_failed = sum(1 for r in pre_validation["results"] if r["status"] == "fail")
        pre_warnings = sum(1 for r in pre_validation["results"] if r["status"] == "warning")
        
        report.append(f"Passed: {pre_passed}")
        report.append(f"Failed: {pre_failed}")
        report.append(f"Warnings: {pre_warnings}")
        report.append("")
        
        if pre_failed > 0:
            report.append("FAILED CHECKS:")
            for result in pre_validation["results"]:
                if result["status"] == "fail":
                    report.append(f"  - {result['name']}: {result['details']}")
            report.append("")
        
        # Post-processing validation summary
        report.append("POST-PROCESSING VALIDATION")
        report.append("-" * 40)
        post_passed = sum(1 for r in post_validation["results"] if r["status"] == "pass")
        post_failed = sum(1 for r in post_validation["results"] if r["status"] == "fail")
        post_warnings = sum(1 for r in post_validation["results"] if r["status"] == "warning")
        
        report.append(f"Passed: {post_passed}")
        report.append(f"Failed: {post_failed}")
        report.append(f"Warnings: {post_warnings}")
        report.append("")
        
        if post_failed > 0 or post_warnings > 0:
            report.append("ISSUES FOUND:")
            for result in post_validation["results"]:
                if result["status"] in ["fail", "warning"]:
                    report.append(f"  - {result['name']}: {result['details']}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)