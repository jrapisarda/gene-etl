"""
Gene Pair Correlation Analysis Module

This module implements Spearman correlation analysis with Benjamini-Hochberg FDR correction
for gene pair correlation analysis across different illness cohorts.
"""

import polars as pl
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
import itertools
from joblib import Parallel, delayed
import warnings

from etl.config import StatisticalConfig

logger = logging.getLogger(__name__)


class GeneCorrelationAnalyzer:
    """Gene pair correlation analysis with statistical significance testing"""
    
    def __init__(self, config: StatisticalConfig):
        self.config = config
        self.method = config.correlation_method
        self.min_samples = config.min_samples
        self.alpha_fdr = config.alpha_fdr
    
    def compute_correlation_matrix(
        self, 
        expression_matrix: pl.DataFrame,
        block_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Compute correlation matrix for all gene pairs
        
        Args:
            expression_matrix: DataFrame with genes as rows, samples as columns
            block_size: Size of blocks for processing
            
        Returns:
            List of correlation results
        """
        logger.info(f"Computing correlation matrix for {expression_matrix.shape[0]} genes")
        
        # Extract gene keys and expression values
        gene_keys = expression_matrix["gene_key"].to_list()
        expression_values = expression_matrix.select(pl.exclude("gene_key")).to_numpy()
        
        # Remove any rows with all zeros (constant genes)
        non_constant_mask = ~np.all(expression_values == 0, axis=1)
        expression_values = expression_values[non_constant_mask]
        gene_keys = [gene_keys[i] for i, mask in enumerate(non_constant_mask) if mask]
        
        n_genes = len(gene_keys)
        logger.info(f"Processing {n_genes} non-constant genes")
        
        results = []
        
        # Process in blocks to manage memory
        for i in range(0, n_genes, block_size):
            for j in range(i, n_genes, block_size):
                block_results = self._compute_block_correlations(
                    expression_values, gene_keys,
                    i, min(i + block_size, n_genes),
                    j, min(j + block_size, n_genes)
                )
                results.extend(block_results)
                
                if len(results) % 10000 == 0:
                    logger.info(f"Computed {len(results)} correlations")
        
        logger.info(f"Completed {len(results)} total correlations")
        return results
    
    def _compute_block_correlations(
        self,
        expression_values: np.ndarray,
        gene_keys: List[int],
        i_start: int, i_end: int,
        j_start: int, j_end: int
    ) -> List[Dict[str, Any]]:
        """Compute correlations for a block of genes"""
        block_results = []
        
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                if i < j:  # Only compute upper triangle, avoid duplicates
                    gene_a_key = gene_keys[i]
                    gene_b_key = gene_keys[j]
                    
                    # Get expression vectors
                    expr_a = expression_values[i]
                    expr_b = expression_values[j]
                    
                    # Remove samples where either gene has missing data (zero expression)
                    valid_mask = (expr_a != 0) & (expr_b != 0)
                    valid_a = expr_a[valid_mask]
                    valid_b = expr_b[valid_mask]
                    
                    if len(valid_a) >= self.min_samples:
                        correlation_result = self._compute_single_correlation(
                            valid_a, valid_b, gene_a_key, gene_b_key
                        )
                        if correlation_result:
                            block_results.append(correlation_result)
        
        return block_results
    
    def _compute_single_correlation(
        self,
        expr_a: np.ndarray,
        expr_b: np.ndarray,
        gene_a_key: int,
        gene_b_key: int
    ) -> Optional[Dict[str, Any]]:
        """Compute correlation for a single gene pair"""
        try:
            if self.method == "spearman":
                rho, p_value = spearmanr(expr_a, expr_b)
            elif self.method == "pearson":
                rho, p_value = stats.pearsonr(expr_a, expr_b)
            elif self.method == "kendall":
                rho, p_value = stats.kendalltau(expr_a, expr_b)
            else:
                raise ValueError(f"Unknown correlation method: {self.method}")
            
            # Handle NaN results
            if np.isnan(rho) or np.isnan(p_value):
                return None
            
            return {
                "gene_a_key": gene_a_key,
                "gene_b_key": gene_b_key,
                "rho_spearman": float(rho),
                "p_value": float(p_value),
                "n_samples": len(expr_a)
            }
            
        except Exception as e:
            logger.warning(f"Error computing correlation for genes {gene_a_key}-{gene_b_key}: {e}")
            return None
    
    def apply_fdr_correction(
        self, 
        correlation_results: List[Dict[str, Any]],
        illness_key: int
    ) -> List[Dict[str, Any]]:
        """
        Apply Benjamini-Hochberg FDR correction to p-values
        
        Args:
            correlation_results: List of correlation results
            illness_key: Illness key for this batch
            
        Returns:
            List of results with q-values added
        """
        logger.info(f"Applying FDR correction for illness {illness_key}")
        
        if not correlation_results:
            return []
        
        # Extract p-values
        p_values = [result["p_value"] for result in correlation_results]
        
        # Apply BH FDR correction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, q_values, _, _ = multipletests(
                p_values, 
                alpha=self.alpha_fdr, 
                method='fdr_bh',
                is_sorted=False,
                returnsorted=False
            )
        
        # Add q-values to results
        for i, result in enumerate(correlation_results):
            result["q_value"] = float(q_values[i])
            result["illness_key"] = illness_key
        
        logger.info(f"Applied FDR correction to {len(correlation_results)} results")
        return correlation_results
    
    def analyze_illness_cohort(
        self, 
        illness_key: int,
        expression_df: pl.DataFrame,
        gene_keys: List[int],
        block_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Complete correlation analysis for a single illness cohort
        
        Args:
            illness_key: Key of the illness cohort
            expression_df: Expression data for this illness
            gene_keys: List of gene keys to analyze
            block_size: Block size for processing
            
        Returns:
            List of correlation results with FDR correction
        """
        logger.info(f"Analyzing illness cohort {illness_key}")
        
        # Filter expression data to selected genes
        filtered_df = expression_df.filter(pl.col("gene_key").is_in(gene_keys))
        
        # Check sample count
        n_samples = filtered_df["sample_key"].n_unique()
        if n_samples < self.min_samples:
            logger.warning(f"Insufficient samples for illness {illness_key}: {n_samples} < {self.min_samples}")
            return []
        
        # Create expression matrix
        expression_matrix = self._create_expression_matrix(filtered_df)
        
        # Compute correlations
        correlation_results = self.compute_correlation_matrix(expression_matrix, block_size)
        
        if not correlation_results:
            logger.warning(f"No valid correlations computed for illness {illness_key}")
            return []
        
        # Apply FDR correction
        results_with_fdr = self.apply_fdr_correction(correlation_results, illness_key)
        
        logger.info(f"Completed analysis for illness {illness_key}: {len(results_with_fdr)} correlations")
        
        return results_with_fdr
    
    def _create_expression_matrix(self, expression_df: pl.DataFrame) -> pl.DataFrame:
        """Create expression matrix from long format data"""
        # Pivot to get genes as rows, samples as columns
        matrix = expression_df.pivot(
            values="expression_value",
            index="gene_key",
            columns="sample_key"
        ).fill_null(0.0)
        
        return matrix
    
    def analyze_multiple_cohorts(
        self,
        expression_data: Dict[int, pl.DataFrame],
        filtered_genes: Dict[int, List[int]],
        max_workers: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple illness cohorts in parallel
        
        Args:
            expression_data: Dict mapping illness_key to expression DataFrame
            filtered_genes: Dict mapping illness_key to list of selected gene keys
            max_workers: Maximum number of parallel workers
            
        Returns:
            Combined list of correlation results
        """
        logger.info(f"Analyzing {len(expression_data)} illness cohorts")
        
        all_results = []
        
        if max_workers > 1:
            # Parallel processing
            results = Parallel(n_jobs=max_workers, backend='threading')(
                delayed(self.analyze_illness_cohort)(
                    illness_key, 
                    expression_data[illness_key],
                    filtered_genes[illness_key]
                )
                for illness_key in expression_data.keys()
            )
            
            for result_list in results:
                all_results.extend(result_list)
        else:
            # Sequential processing
            for illness_key in expression_data.keys():
                if illness_key in filtered_genes:
                    results = self.analyze_illness_cohort(
                        illness_key,
                        expression_data[illness_key],
                        filtered_genes[illness_key]
                    )
                    all_results.extend(results)
        
        logger.info(f"Completed analysis: {len(all_results)} total correlations")
        return all_results
    
    def get_correlation_statistics(
        self, 
        correlation_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get summary statistics of correlation results"""
        
        if not correlation_results:
            return {
                "total_correlations": 0,
                "significant_correlations": 0,
                "highly_significant_correlations": 0,
                "avg_correlation": None,
                "correlation_range": None
            }
        
        # Extract values
        correlations = [r["rho_spearman"] for r in correlation_results]
        p_values = [r["p_value"] for r in correlation_results]
        q_values = [r["q_value"] for r in correlation_results]
        
        # Calculate statistics
        total = len(correlation_results)
        significant = sum(1 for q in q_values if q <= 0.05)
        highly_significant = sum(1 for q in q_values if q <= 0.01)
        
        return {
            "total_correlations": total,
            "significant_correlations": significant,
            "highly_significant_correlations": highly_significant,
            "avg_correlation": float(np.mean(correlations)),
            "correlation_range": [float(min(correlations)), float(max(correlations))],
            "avg_p_value": float(np.mean(p_values)),
            "avg_q_value": float(np.mean(q_values))
        }


def validate_correlation_results(results: List[Dict[str, Any]]) -> List[str]:
    """Validate correlation results for data quality"""
    errors = []
    
    for i, result in enumerate(results):
        # Check correlation coefficient range
        if not (-1 <= result["rho_spearman"] <= 1):
            errors.append(f"Result {i}: Correlation coefficient out of range: {result['rho_spearman']}")
        
        # Check p-value range
        if not (0 <= result["p_value"] <= 1):
            errors.append(f"Result {i}: P-value out of range: {result['p_value']}")
        
        # Check q-value range
        if not (0 <= result["q_value"] <= 1):
            errors.append(f"Result {i}: Q-value out of range: {result['q_value']}")
        
        # Check sample count
        if result["n_samples"] <= 0:
            errors.append(f"Result {i}: Invalid sample count: {result['n_samples']}")
    
    return errors