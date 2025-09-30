"""
Pytest configuration and fixtures for testing the gene correlation ETL system
"""

import pytest
import tempfile
import os
from datetime import datetime
import uuid
import polars as pl
import numpy as np

from etl.config import ETLConfig, GeneFilterConfig, StatisticalConfig, ProcessingConfig
from models import (
    DimGene, DimIllness, DimSample, FactGeneExpression,
    FactGenePairCorrelation, ETLProcessLog, DataValidationLog
)


@pytest.fixture
def test_config():
    """Create a test configuration"""
    return ETLConfig(
        config_name="test_gene_etl",
        config_version="1.0.0",
        environment="test",
        database={
            "host": "localhost",
            "port": 5432,
            "database": "test_bioinformatics",
            "username": "test_user",
            "password": "test_password",
            "pool_size": 2,
            "max_overflow": 5
        },
        gene_filter=GeneFilterConfig(
            method="variance",
            top_n_genes=100,
            min_variance_threshold=0.01,
            min_expression_value=0.0
        ),
        statistical=StatisticalConfig(
            correlation_method="spearman",
            min_samples=10,
            alpha_fdr=0.05
        ),
        processing=ProcessingConfig(
            batch_size=100,
            block_size=10,
            max_workers=1,
            memory_limit_gb=2.0,
            checkpoint_interval=1000
        )
    )


@pytest.fixture
def sample_gene_metadata():
    """Create sample gene metadata"""
    return pl.DataFrame({
        'gene_key': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'ensembl_id': [
            'ENSG00000000003', 'ENSG00000000005', 'ENSG00000000419',
            'ENSG00000000457', 'ENSG00000000460', 'ENSG00000000938',
            'ENSG00000000971', 'ENSG00000001036', 'ENSG00000001084',
            'ENSG00000001167'
        ],
        'symbol': ['TSPAN6', 'TNMD', 'DPM1', 'SCYL3', 'C1orf112', 'FGR', 'CFH', 'FUCA2', 'GCLC', 'NFYA'],
        'gene_name': [
            'tetraspanin 6', 'tenomodulin', 'dolichyl-phosphate mannosyltransferase',
            'SCY1 like pseudokinase 3', 'chromosome 1 open reading frame 112',
            'FGR proto-oncogene', 'complement factor H', 'fucosidase alpha-2',
            'glutamate-cysteine ligase catalytic subunit', 'nuclear transcription factor Y alpha'
        ],
        'chromosome': ['X', 'X', '20', '1', '1', '1', '1', '6', '6', '6'],
        'gene_type': ['protein_coding', 'protein_coding', 'protein_coding', 'protein_coding',
                     'protein_coding', 'protein_coding', 'protein_coding', 'protein_coding',
                     'protein_coding', 'protein_coding']
    })


@pytest.fixture
def sample_illness_metadata():
    """Create sample illness metadata"""
    return pl.DataFrame({
        'illness_key': [1, 2, 3],
        'illness_code': ['BRCA', 'LUAD', 'COAD'],
        'description': ['Breast Cancer', 'Lung Adenocarcinoma', 'Colon Adenocarcinoma'],
        'category': ['cancer', 'cancer', 'cancer']
    })


@pytest.fixture
def sample_sample_metadata():
    """Create sample sample metadata"""
    samples = []
    for illness_key in [1, 2, 3]:
        for i in range(1, 21):  # 20 samples per illness
            samples.append({
                'sample_key': (illness_key - 1) * 20 + i,
                'sample_id': f'SAMPLE_{illness_key}_{i:03d}',
                'illness_key': illness_key,
                'patient_id': f'PAT_{illness_key}_{i:03d}',
                'age': np.random.randint(30, 80),
                'sex': np.random.choice(['M', 'F']),
                'tissue_type': 'tumor'
            })
    
    return pl.DataFrame(samples)


@pytest.fixture
def sample_expression_data(sample_gene_metadata, sample_sample_metadata):
    """Create sample expression data"""
    expression_data = {}
    
    for illness_key in [1, 2, 3]:
        # Get samples for this illness
        illness_samples = sample_sample_metadata.filter(
            pl.col('illness_key') == illness_key
        )
        
        expression_records = []
        
        for gene_key in sample_gene_metadata['gene_key']:
            for sample_key in illness_samples['sample_key']:
                # Generate realistic expression values
                base_expression = np.random.lognormal(2, 1.5)
                
                # Add some noise
                noise = np.random.normal(0, 0.1 * base_expression)
                expression_value = max(0, base_expression + noise)
                
                expression_records.append({
                    'gene_key': gene_key,
                    'sample_key': sample_key,
                    'expression_value': expression_value
                })
        
        expression_data[illness_key] = pl.DataFrame(expression_records)
    
    return expression_data


@pytest.fixture
def sample_correlation_results():
    """Create sample correlation results"""
    results = []
    
    for illness_key in [1, 2, 3]:
        for i in range(1, 6):  # 5 genes
            for j in range(i + 1, 6):  # Upper triangle
                # Generate realistic correlation data
                rho = np.random.uniform(-0.8, 0.8)
                p_value = np.random.uniform(0.001, 0.1)
                
                results.append({
                    'gene_a_key': i,
                    'gene_b_key': j,
                    'illness_key': illness_key,
                    'rho_spearman': rho,
                    'p_value': p_value,
                    'q_value': p_value * 1.2,  # FDR correction
                    'n_samples': 20
                })
    
    return results


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file"""
    import tempfile
    import json
    
    config_data = {
        "config_name": "test_temp_etl",
        "config_version": "1.0.0",
        "environment": "test",
        "database": {
            "host": "localhost",
            "port": 5432,
            "database": "temp_test_db",
            "username": "temp_user",
            "password": "temp_password"
        },
        "gene_filter": {
            "method": "variance",
            "top_n_genes": 50,
            "min_variance_threshold": 0.01
        },
        "statistical": {
            "correlation_method": "spearman",
            "min_samples": 5,
            "alpha_fdr": 0.05
        },
        "processing": {
            "batch_size": 50,
            "block_size": 5,
            "max_workers": 1
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def mock_database_session(mocker):
    """Create a mock database session"""
    return mocker.Mock()


@pytest.fixture
def mock_data_io(mocker):
    """Create a mock DataIO instance"""
    mock_io = mocker.Mock()
    mock_io.Session.return_value = mock_database_session(mocker)
    return mock_io