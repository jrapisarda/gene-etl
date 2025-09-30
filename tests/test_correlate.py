"""
Unit tests for gene correlation analysis module
"""

import pytest
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from unittest.mock import Mock, patch

from etl.correlate import GeneCorrelationAnalyzer, validate_correlation_results
from etl.config import StatisticalConfig


class TestGeneCorrelationAnalyzer:
    """Test cases for GeneCorrelationAnalyzer"""
    
    def test_initialization(self, test_config):
        """Test analyzer initialization"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        assert analyzer.method == config.correlation_method
        assert analyzer.min_samples == config.min_samples
        assert analyzer.alpha_fdr == config.alpha_fdr
    
    def test_compute_single_correlation_spearman(self, test_config):
        """Test Spearman correlation computation"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test data with known correlation
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = x + np.random.normal(0, 0.1, 50)  # High correlation
        
        result = analyzer._compute_single_correlation(x, y, 1, 2)
        
        assert result is not None
        assert 'gene_a_key' in result
        assert 'gene_b_key' in result
        assert 'rho_spearman' in result
        assert 'p_value' in result
        assert result['gene_a_key'] == 1
        assert result['gene_b_key'] == 2
        assert -1 <= result['rho_spearman'] <= 1
        assert 0 <= result['p_value'] <= 1
    
    def test_compute_single_correlation_pearson(self, test_config):
        """Test Pearson correlation computation"""
        config = StatisticalConfig(
            correlation_method="pearson",
            min_samples=10,
            alpha_fdr=0.05
        )
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test data
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = x + np.random.normal(0, 0.1, 50)
        
        result = analyzer._compute_single_correlation(x, y, 1, 2)
        
        assert result is not None
        assert -1 <= result['rho_spearman'] <= 1
        assert 0 <= result['p_value'] <= 1
    
    def test_compute_single_correlation_kendall(self, test_config):
        """Test Kendall correlation computation"""
        config = StatisticalConfig(
            correlation_method="kendall",
            min_samples=10,
            alpha_fdr=0.05
        )
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test data
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = x + np.random.normal(0, 0.1, 50)
        
        result = analyzer._compute_single_correlation(x, y, 1, 2)
        
        assert result is not None
        assert -1 <= result['rho_spearman'] <= 1
        assert 0 <= result['p_value'] <= 1
    
    def test_compute_single_correlation_insufficient_samples(self, test_config):
        """Test correlation with insufficient samples"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test data with too few samples
        x = np.random.normal(0, 1, 5)  # Less than min_samples
        y = np.random.normal(0, 1, 5)
        
        result = analyzer._compute_single_correlation(x, y, 1, 2)
        
        assert result is None
    
    def test_compute_single_correlation_constant_data(self, test_config):
        """Test correlation with constant data"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create constant data
        x = np.ones(50)
        y = np.ones(50)
        
        result = analyzer._compute_single_correlation(x, y, 1, 2)
        
        # Should handle NaN results gracefully
        assert result is None
    
    def test_apply_fdr_correction(self, test_config):
        """Test FDR correction application"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test correlation results
        correlation_results = [
            {'gene_a_key': 1, 'gene_b_key': 2, 'rho_spearman': 0.8, 'p_value': 0.001, 'n_samples': 20},
            {'gene_a_key': 1, 'gene_b_key': 3, 'rho_spearman': 0.5, 'p_value': 0.05, 'n_samples': 20},
            {'gene_a_key': 2, 'gene_b_key': 3, 'rho_spearman': 0.1, 'p_value': 0.5, 'n_samples': 20},
        ]
        
        illness_key = 1
        results_with_fdr = analyzer.apply_fdr_correction(correlation_results, illness_key)
        
        assert len(results_with_fdr) == 3
        
        for result in results_with_fdr:
            assert 'q_value' in result
            assert 'illness_key' in result
            assert result['illness_key'] == illness_key
            assert 0 <= result['q_value'] <= 1
            # Q-value should be >= p-value (approximately)
            assert result['q_value'] >= result['p_value'] - 1e-10
    
    def test_compute_correlation_matrix(self, test_config):
        """Test correlation matrix computation"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test expression matrix
        np.random.seed(42)
        n_genes = 5
        n_samples = 20
        
        # Create correlated expression data
        expression_data = np.random.normal(0, 1, (n_genes, n_samples))
        # Make some genes correlated
        expression_data[1] = expression_data[0] + np.random.normal(0, 0.1, n_samples)
        expression_data[2] = expression_data[0] + np.random.normal(0, 0.2, n_samples)
        
        # Create DataFrame
        expression_matrix = pl.DataFrame({
            'gene_key': list(range(1, n_genes + 1)),
            **{f'sample_{i}': expression_data[:, i] for i in range(n_samples)}
        })
        
        results = analyzer.compute_correlation_matrix(expression_matrix, block_size=2)
        
        # Should have n_genes choose 2 correlations
        expected_correlations = n_genes * (n_genes - 1) // 2
        assert len(results) == expected_correlations
        
        # Check that all results are valid
        for result in results:
            assert -1 <= result['rho_spearman'] <= 1
            assert 0 <= result['p_value'] <= 1
            assert result['n_samples'] <= n_samples
    
    def test_analyze_illness_cohort(self, test_config):
        """Test complete illness cohort analysis"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test expression data
        np.random.seed(42)
        n_genes = 5
        n_samples = 20
        
        expression_records = []
        for gene_key in range(1, n_genes + 1):
            for sample_key in range(1, n_samples + 1):
                expression_value = np.random.lognormal(2, 1)
                expression_records.append({
                    'gene_key': gene_key,
                    'sample_key': sample_key,
                    'expression_value': expression_value
                })
        
        expression_df = pl.DataFrame(expression_records)
        gene_keys = list(range(1, n_genes + 1))
        illness_key = 1
        
        results = analyzer.analyze_illness_cohort(
            illness_key, expression_df, gene_keys, block_size=2
        )
        
        # Should have computed correlations with FDR correction
        assert len(results) > 0
        
        for result in results:
            assert result['illness_key'] == illness_key
            assert 'q_value' in result
            assert 0 <= result['q_value'] <= 1
    
    def test_analyze_illness_cohort_insufficient_samples(self, test_config):
        """Test analysis with insufficient samples"""
        config = StatisticalConfig(
            correlation_method="spearman",
            min_samples=30,  # Higher than available
            alpha_fdr=0.05
        )
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test expression data with fewer samples than minimum
        expression_records = []
        for gene_key in range(1, 4):
            for sample_key in range(1, 11):  # Only 10 samples
                expression_records.append({
                    'gene_key': gene_key,
                    'sample_key': sample_key,
                    'expression_value': np.random.lognormal(2, 1)
                })
        
        expression_df = pl.DataFrame(expression_records)
        gene_keys = [1, 2, 3]
        illness_key = 1
        
        results = analyzer.analyze_illness_cohort(
            illness_key, expression_df, gene_keys
        )
        
        # Should return empty results due to insufficient samples
        assert len(results) == 0
    
    def test_analyze_multiple_cohorts(self, test_config):
        """Test analysis of multiple illness cohorts"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test data for multiple cohorts
        expression_data = {}
        selected_genes = {}
        
        for illness_key in [1, 2]:
            # Create expression data
            expression_records = []
            for gene_key in range(1, 5):
                for sample_key in range(1, 21):
                    expression_value = np.random.lognormal(2, 1)
                    expression_records.append({
                        'gene_key': gene_key,
                        'sample_key': sample_key,
                        'expression_value': expression_value
                    })
            
            expression_data[illness_key] = pl.DataFrame(expression_records)
            selected_genes[illness_key] = list(range(1, 5))
        
        results = analyzer.analyze_multiple_cohorts(
            expression_data, selected_genes, max_workers=1
        )
        
        # Should have results from both cohorts
        assert len(results) > 0
        
        # Check that results are tagged with correct illness
        illness_keys = set(result['illness_key'] for result in results)
        assert illness_keys == {1, 2}
    
    def test_get_correlation_statistics(self, test_config):
        """Test correlation statistics computation"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        # Create test correlation results
        correlation_results = [
            {'gene_a_key': 1, 'gene_b_key': 2, 'rho_spearman': 0.8, 'p_value': 0.001, 'q_value': 0.002, 'n_samples': 20},
            {'gene_a_key': 1, 'gene_b_key': 3, 'rho_spearman': 0.5, 'p_value': 0.05, 'q_value': 0.06, 'n_samples': 20},
            {'gene_a_key': 2, 'gene_b_key': 3, 'rho_spearman': 0.1, 'p_value': 0.5, 'q_value': 0.6, 'n_samples': 20},
            {'gene_a_key': 2, 'gene_b_key': 4, 'rho_spearman': -0.7, 'p_value': 0.01, 'q_value': 0.02, 'n_samples': 20},
            {'gene_a_key': 3, 'gene_b_key': 4, 'rho_spearman': 0.0, 'p_value': 0.9, 'q_value': 1.0, 'n_samples': 20},
        ]
        
        statistics = analyzer.get_correlation_statistics(correlation_results)
        
        assert statistics['total_correlations'] == 5
        assert statistics['significant_correlations'] == 2  # q_value <= 0.05
        assert statistics['highly_significant_correlations'] == 0  # q_value <= 0.01
        assert abs(statistics['avg_correlation']) < 1.0
        assert statistics['correlation_range'][0] == -0.7
        assert statistics['correlation_range'][1] == 0.8
        assert 0 <= statistics['avg_p_value'] <= 1
        assert 0 <= statistics['avg_q_value'] <= 1
    
    def test_get_correlation_statistics_empty(self, test_config):
        """Test statistics with empty results"""
        config = test_config.statistical
        analyzer = GeneCorrelationAnalyzer(config)
        
        statistics = analyzer.get_correlation_statistics([])
        
        assert statistics['total_correlations'] == 0
        assert statistics['significant_correlations'] == 0
        assert statistics['highly_significant_correlations'] == 0
        assert statistics['avg_correlation'] is None
        assert statistics['correlation_range'] is None


class TestValidateCorrelationResults:
    """Test cases for correlation result validation"""
    
    def test_validate_valid_results(self):
        """Test validation of valid correlation results"""
        results = [
            {'gene_a_key': 1, 'gene_b_key': 2, 'rho_spearman': 0.5, 'p_value': 0.01, 'q_value': 0.02, 'n_samples': 20},
            {'gene_a_key': 1, 'gene_b_key': 3, 'rho_spearman': -0.3, 'p_value': 0.05, 'q_value': 0.06, 'n_samples': 15},
            {'gene_a_key': 2, 'gene_b_key': 3, 'rho_spearman': 0.0, 'p_value': 0.5, 'q_value': 0.6, 'n_samples': 25},
        ]
        
        errors = validate_correlation_results(results)
        
        assert len(errors) == 0
    
    def test_validate_invalid_correlation_coefficients(self):
        """Test validation with invalid correlation coefficients"""
        results = [
            {'gene_a_key': 1, 'gene_b_key': 2, 'rho_spearman': 1.5, 'p_value': 0.01, 'q_value': 0.02, 'n_samples': 20},
            {'gene_a_key': 1, 'gene_b_key': 3, 'rho_spearman': -2.0, 'p_value': 0.05, 'q_value': 0.06, 'n_samples': 15},
        ]
        
        errors = validate_correlation_results(results)
        
        assert len(errors) == 2
        assert "Correlation coefficient out of range" in errors[0]
        assert "Correlation coefficient out of range" in errors[1]
    
    def test_validate_invalid_p_values(self):
        """Test validation with invalid p-values"""
        results = [
            {'gene_a_key': 1, 'gene_b_key': 2, 'rho_spearman': 0.5, 'p_value': -0.1, 'q_value': 0.02, 'n_samples': 20},
            {'gene_a_key': 1, 'gene_b_key': 3, 'rho_spearman': -0.3, 'p_value': 1.5, 'q_value': 0.06, 'n_samples': 15},
        ]
        
        errors = validate_correlation_results(results)
        
        assert len(errors) == 2
        assert "P-value out of range" in errors[0]
        assert "P-value out of range" in errors[1]
    
    def test_validate_invalid_q_values(self):
        """Test validation with invalid q-values"""
        results = [
            {'gene_a_key': 1, 'gene_b_key': 2, 'rho_spearman': 0.5, 'p_value': 0.01, 'q_value': -0.1, 'n_samples': 20},
            {'gene_a_key': 1, 'gene_b_key': 3, 'rho_spearman': -0.3, 'p_value': 0.05, 'q_value': 1.5, 'n_samples': 15},
        ]
        
        errors = validate_correlation_results(results)
        
        assert len(errors) == 2
        assert "Q-value out of range" in errors[0]
        assert "Q-value out of range" in errors[1]
    
    def test_validate_invalid_sample_counts(self):
        """Test validation with invalid sample counts"""
        results = [
            {'gene_a_key': 1, 'gene_b_key': 2, 'rho_spearman': 0.5, 'p_value': 0.01, 'q_value': 0.02, 'n_samples': 0},
            {'gene_a_key': 1, 'gene_b_key': 3, 'rho_spearman': -0.3, 'p_value': 0.05, 'q_value': 0.06, 'n_samples': -5},
        ]
        
        errors = validate_correlation_results(results)
        
        assert len(errors) == 2
        assert "Invalid sample count" in errors[0]
        assert "Invalid sample count" in errors[1]
    
    def test_validate_multiple_errors(self):
        """Test validation with multiple types of errors"""
        results = [
            {'gene_a_key': 1, 'gene_b_key': 2, 'rho_spearman': 1.5, 'p_value': -0.1, 'q_value': 0.02, 'n_samples': 0},
            {'gene_a_key': 1, 'gene_b_key': 3, 'rho_spearman': -0.3, 'p_value': 0.05, 'q_value': 1.5, 'n_samples': -5},
        ]
        
        errors = validate_correlation_results(results)
        
        # Should have multiple errors for each invalid result
        assert len(errors) > 2