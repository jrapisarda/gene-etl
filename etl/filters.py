"""
Gene Filtering Module for ETL Pipeline

This module implements various gene filtering strategies to select the most
informative genes for correlation analysis, improving performance and signal quality.
"""

import polars as pl
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats
import logging

from etl.config import GeneFilterConfig

logger = logging.getLogger(__name__)


class GeneFilter:
    """Gene filtering strategies for selecting informative genes"""
    
    def __init__(self, config: GeneFilterConfig):
        self.config = config
        self.method = config.method
        self.top_n_genes = config.top_n_genes
        self.min_variance_threshold = config.min_variance_threshold
        self.min_expression_value = config.min_expression_value
    
    def filter_genes(
        self, 
        expression_df: pl.DataFrame, 
        gene_metadata: Optional[pl.DataFrame] = None
    ) -> List[int]:
        """
        Filter genes based on the configured method
        
        Args:
            expression_df: DataFrame with columns [gene_key, sample_key, expression_value]
            gene_metadata: Optional gene metadata DataFrame
            
        Returns:
            List of selected gene_keys
        """
        logger.info(f"Filtering genes using method: {self.method}")
        
        if self.method == "variance":
            return self._filter_by_variance(expression_df)
        elif self.method == "iqr":
            return self._filter_by_iqr(expression_df)
        elif self.method == "mad":
            return self._filter_by_mad(expression_df)
        else:
            raise ValueError(f"Unknown filtering method: {self.method}")
    
    def _filter_by_variance(self, expression_df: pl.DataFrame) -> List[int]:
        """Filter genes by expression variance across samples"""
        logger.info("Filtering genes by variance")
        
        # Pivot to get genes as rows, samples as columns
        expression_matrix = expression_df.pivot(
            values="expression_value",
            index="gene_key",
            columns="sample_key"
        ).fill_null(0.0)
        
        # Calculate variance for each gene
        variance_scores = []
        for gene_key in expression_matrix["gene_key"]:
            gene_row = expression_matrix.filter(pl.col("gene_key") == gene_key)
            expression_values = gene_row.select(pl.exclude("gene_key")).to_numpy().flatten()
            
            # Remove zero values (missing data)
            expression_values = expression_values[expression_values != 0]
            
            if len(expression_values) > 1:
                variance = np.var(expression_values)
                # Apply minimum expression threshold
                if np.mean(expression_values) >= self.min_expression_value:
                    variance_scores.append((gene_key, variance))
        
        # Sort by variance and select top N
        variance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply minimum variance threshold
        filtered_genes = [
            gene_key for gene_key, variance in variance_scores 
            if variance >= self.min_variance_threshold
        ][:self.top_n_genes]
        
        logger.info(f"Selected {len(filtered_genes)} genes by variance")
        return filtered_genes
    
    def _filter_by_iqr(self, expression_df: pl.DataFrame) -> List[int]:
        """Filter genes by Interquartile Range (IQR)"""
        logger.info("Filtering genes by IQR")
        
        # Pivot to matrix form
        expression_matrix = expression_df.pivot(
            values="expression_value",
            index="gene_key",
            columns="sample_key"
        ).fill_null(0.0)
        
        iqr_scores = []
        for gene_key in expression_matrix["gene_key"]:
            gene_row = expression_matrix.filter(pl.col("gene_key") == gene_key)
            expression_values = gene_row.select(pl.exclude("gene_key")).to_numpy().flatten()
            
            # Remove zero values
            expression_values = expression_values[expression_values != 0]
            
            if len(expression_values) > 4:  # Need at least 4 samples for IQR
                # Calculate IQR
                q1 = np.percentile(expression_values, 25)
                q3 = np.percentile(expression_values, 75)
                iqr = q3 - q1
                
                # Apply minimum expression threshold
                if np.mean(expression_values) >= self.min_expression_value:
                    iqr_scores.append((gene_key, iqr))
        
        # Sort by IQR and select top N
        iqr_scores.sort(key=lambda x: x[1], reverse=True)
        filtered_genes = [gene_key for gene_key, iqr in iqr_scores][:self.top_n_genes]
        
        logger.info(f"Selected {len(filtered_genes)} genes by IQR")
        return filtered_genes
    
    def _filter_by_mad(self, expression_df: pl.DataFrame) -> List[int]:
        """Filter genes by Median Absolute Deviation (MAD)"""
        logger.info("Filtering genes by MAD")
        
        # Pivot to matrix form
        expression_matrix = expression_df.pivot(
            values="expression_value",
            index="gene_key",
            columns="sample_key"
        ).fill_null(0.0)
        
        mad_scores = []
        for gene_key in expression_matrix["gene_key"]:
            gene_row = expression_matrix.filter(pl.col("gene_key") == gene_key)
            expression_values = gene_row.select(pl.exclude("gene_key")).to_numpy().flatten()
            
            # Remove zero values
            expression_values = expression_values[expression_values != 0]
            
            if len(expression_values) > 1:
                # Calculate MAD
                median = np.median(expression_values)
                mad = np.median(np.abs(expression_values - median))
                
                # Apply minimum expression threshold
                if np.mean(expression_values) >= self.min_expression_value:
                    mad_scores.append((gene_key, mad))
        
        # Sort by MAD and select top N
        mad_scores.sort(key=lambda x: x[1], reverse=True)
        filtered_genes = [gene_key for gene_key, mad in mad_scores][:self.top_n_genes]
        
        logger.info(f"Selected {len(filtered_genes)} genes by MAD")
        return filtered_genes
    
    def filter_low_expressed_genes(
        self, 
        expression_df: pl.DataFrame, 
        min_samples: int = 5,
        min_expression_proportion: float = 0.1
    ) -> List[int]:
        """
        Filter out genes with low expression across samples
        
        Args:
            expression_df: Expression data
            min_samples: Minimum number of samples with expression
            min_expression_proportion: Minimum proportion of samples with expression
        
        Returns:
            List of gene_keys that pass the filter
        """
        logger.info("Filtering low expressed genes")
        
        # Count non-zero expression per gene
        gene_expression_counts = expression_df.filter(
            pl.col("expression_value") > self.min_expression_value
        ).group_by("gene_key").agg(
            pl.count().alias("expressed_samples")
        )
        
        # Get total samples per gene
        total_samples_per_gene = expression_df.group_by("gene_key").agg(
            pl.count().alias("total_samples")
        )
        
        # Join and calculate proportions
        gene_stats = gene_expression_counts.join(
            total_samples_per_gene, on="gene_key", how="left"
        ).fill_null(0)
        
        gene_stats = gene_stats.with_columns(
            (pl.col("expressed_samples") / pl.col("total_samples")).alias("expression_proportion")
        )
        
        # Filter genes
        filtered_genes = gene_stats.filter(
            (pl.col("expressed_samples") >= min_samples) &
            (pl.col("expression_proportion") >= min_expression_proportion)
        )["gene_key"].to_list()
        
        logger.info(f"Kept {len(filtered_genes)} genes after low expression filter")
        return filtered_genes
    
    def apply_comprehensive_filtering(
        self, 
        expression_df: pl.DataFrame,
        gene_metadata: Optional[pl.DataFrame] = None
    ) -> Tuple[List[int], pl.DataFrame]:
        """
        Apply comprehensive filtering pipeline
        
        Returns:
            Tuple of (selected_gene_keys, filtered_expression_df)
        """
        logger.info("Applying comprehensive gene filtering")
        
        # Step 1: Filter low expressed genes
        expressed_genes = self.filter_low_expressed_genes(expression_df)
        filtered_df = expression_df.filter(pl.col("gene_key").is_in(expressed_genes))
        
        # Step 2: Apply main filtering method
        selected_genes = self.filter_genes(filtered_df, gene_metadata)
        
        # Step 3: Create final filtered expression dataframe
        final_df = filtered_df.filter(pl.col("gene_key").is_in(selected_genes))
        
        logger.info(f"Final selection: {len(selected_genes)} genes, {len(final_df)} expression records")
        
        return selected_genes, final_df
    
    def get_filtering_statistics(
        self, 
        original_df: pl.DataFrame, 
        filtered_df: pl.DataFrame,
        selected_genes: List[int]
    ) -> Dict[str, Any]:
        """Get statistics about the filtering process"""
        
        original_genes = original_df["gene_key"].n_unique()
        original_records = len(original_df)
        
        filtered_genes = len(selected_genes)
        filtered_records = len(filtered_df)
        
        return {
            "method": self.method,
            "original_genes": original_genes,
            "original_records": original_records,
            "selected_genes": filtered_genes,
            "selected_records": filtered_records,
            "gene_reduction_pct": (1 - filtered_genes / original_genes) * 100,
            "record_reduction_pct": (1 - filtered_records / original_records) * 100,
            "top_n_genes": self.top_n_genes,
            "min_variance_threshold": self.min_variance_threshold,
            "min_expression_value": self.min_expression_value
        }


def create_expression_matrix(
    expression_df: pl.DataFrame, 
    gene_keys: List[int]
) -> pl.DataFrame:
    """
    Create a gene x sample expression matrix
    
    Args:
        expression_df: Long format expression data
        gene_keys: List of gene keys to include
    
    Returns:
        DataFrame with genes as rows, samples as columns
    """
    logger.info("Creating expression matrix")
    
    # Filter to selected genes
    filtered_df = expression_df.filter(pl.col("gene_key").is_in(gene_keys))
    
    # Pivot to matrix format
    matrix = filtered_df.pivot(
        values="expression_value",
        index="gene_key",
        columns="sample_key"
    )
    
    # Fill missing values with 0
    matrix = matrix.fill_null(0.0)
    
    logger.info(f"Created matrix: {matrix.shape[0]} genes x {matrix.shape[1]-1} samples")
    
    return matrix