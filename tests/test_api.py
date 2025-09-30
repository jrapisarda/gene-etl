"""
API tests for the Flask REST endpoints
"""

import pytest
import json
from unittest.mock import patch, Mock
from app import create_app


@pytest.fixture
def client():
    """Create a test client for the Flask app"""
    app = create_app()
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client


class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check endpoint returns 200"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['service'] == 'gene-etl-api'


class TestConfigEndpoints:
    """Test configuration-related endpoints"""
    
    def test_get_config(self, client):
        """Test getting example configuration"""
        response = client.get('/api/v1/config')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'config' in data
        assert 'message' in data
        assert data['message'] == 'Example configuration retrieved successfully'
        
        # Check config structure
        config = data['config']
        assert 'config_name' in config
        assert 'database' in config
        assert 'gene_filter' in config
        assert 'statistical' in config
        assert 'processing' in config
    
    def test_validate_valid_config(self, client):
        """Test validating a valid configuration"""
        valid_config = {
            "config_name": "test_config",
            "config_version": "1.0.0",
            "environment": "test",
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "test_user",
                "password": "test_password"
            },
            "gene_filter": {
                "method": "variance",
                "top_n_genes": 100,
                "min_variance_threshold": 0.01
            },
            "statistical": {
                "correlation_method": "spearman",
                "min_samples": 10,
                "alpha_fdr": 0.05
            },
            "processing": {
                "batch_size": 100,
                "block_size": 10,
                "max_workers": 1
            }
        }
        
        response = client.post('/api/v1/config/validate', 
                             data=json.dumps(valid_config),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['valid'] is True
        assert 'config_hash' in data
    
    def test_validate_invalid_config(self, client):
        """Test validating an invalid configuration"""
        invalid_config = {
            "config_name": "test_config",
            # Missing required database configuration
            "gene_filter": {
                "method": "invalid_method",  # Invalid method
                "top_n_genes": 100
            }
        }
        
        response = client.post('/api/v1/config/validate',
                             data=json.dumps(invalid_config),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['valid'] is False
        assert 'error' in data
    
    def test_validate_config_no_data(self, client):
        """Test validating with no configuration data"""
        response = client.post('/api/v1/config/validate',
                             data=json.dumps({}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error'] == 'No configuration data provided'


class TestETLEndpoints:
    """Test ETL pipeline endpoints"""
    
    @patch('app.GeneCorrelationETL')
    @patch('app.create_example_config')
    def test_run_etl_with_default_config(self, mock_create_config, mock_etl_class, client):
        """Test running ETL with default configuration"""
        # Mock the ETL pipeline
        mock_etl = Mock()
        mock_etl.run.return_value = {
            'batch_id': 'test-batch-id',
            'status': 'success',
            'summary': {'correlations_computed': 100}
        }
        mock_etl_class.return_value = mock_etl
        
        response = client.post('/api/v1/etl/run')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'completed'
        assert 'job_id' in data
        assert data['results']['status'] == 'success'
    
    @patch('app.GeneCorrelationETL')
    def test_run_etl_with_custom_config(self, mock_etl_class, client):
        """Test running ETL with custom configuration"""
        custom_config = {
            "config_name": "custom_test",
            "config_version": "1.0.0",
            "environment": "test",
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "custom_db",
                "username": "custom_user",
                "password": "custom_password"
            },
            "gene_filter": {
                "method": "variance",
                "top_n_genes": 50
            },
            "statistical": {
                "correlation_method": "spearman",
                "min_samples": 10
            },
            "processing": {
                "batch_size": 50,
                "max_workers": 2
            }
        }
        
        # Mock the ETL pipeline
        mock_etl = Mock()
        mock_etl.run.return_value = {
            'batch_id': 'custom-batch-id',
            'status': 'success',
            'summary': {'correlations_computed': 50}
        }
        mock_etl_class.return_value = mock_etl
        
        response = client.post('/api/v1/etl/run',
                             data=json.dumps(custom_config),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'completed'
        assert 'job_id' in data
    
    @patch('app.GeneCorrelationETL')
    def test_run_etl_failure(self, mock_etl_class, client):
        """Test ETL pipeline failure handling"""
        # Mock ETL pipeline to raise an exception
        mock_etl = Mock()
        mock_etl.run.side_effect = Exception("Test ETL failure")
        mock_etl_class.return_value = mock_etl
        
        response = client.post('/api/v1/etl/run')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'failed'
        assert 'error' in data
        assert 'Test ETL failure' in data['error']


class TestJobEndpoints:
    """Test job management endpoints"""
    
    def test_list_jobs_empty(self, client):
        """Test listing jobs when none exist"""
        # Clear any existing jobs
        from app import job_tracker
        job_tracker.clear()
        
        response = client.get('/api/v1/etl/jobs')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['jobs'] == {}
        assert data['count'] == 0
    
    @patch('app.GeneCorrelationETL')
    def test_list_jobs_with_data(self, mock_etl_class, client):
        """Test listing jobs with existing data"""
        # Run an ETL job first
        mock_etl = Mock()
        mock_etl.run.return_value = {'status': 'success'}
        mock_etl_class.return_value = mock_etl
        
        client.post('/api/v1/etl/run')
        
        response = client.get('/api/v1/etl/jobs')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['count'] == 1
        assert len(data['jobs']) == 1
    
    def test_get_job_status_not_found(self, client):
        """Test getting status of non-existent job"""
        response = client.get('/api/v1/etl/jobs/nonexistent-job-id')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['error'] == 'Job not found'
    
    @patch('app.GeneCorrelationETL')
    def test_get_job_status_found(self, mock_etl_class, client):
        """Test getting status of existing job"""
        # Run an ETL job first
        mock_etl = Mock()
        mock_etl.run.return_value = {'status': 'success'}
        mock_etl_class.return_value = mock_etl
        
        run_response = client.post('/api/v1/etl/run')
        job_id = json.loads(run_response.data)['job_id']
        
        response = client.get(f'/api/v1/etl/jobs/{job_id}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'completed'
        assert 'config' in data
    
    def test_delete_job_not_found(self, client):
        """Test deleting non-existent job"""
        response = client.delete('/api/v1/etl/jobs/nonexistent-job-id')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['error'] == 'Job not found'
    
    @patch('app.GeneCorrelationETL')
    def test_delete_job_found(self, mock_etl_class, client):
        """Test deleting existing job"""
        # Run an ETL job first
        mock_etl = Mock()
        mock_etl.run.return_value = {'status': 'success'}
        mock_etl_class.return_value = mock_etl
        
        run_response = client.post('/api/v1/etl/run')
        job_id = json.loads(run_response.data)['job_id']
        
        response = client.delete(f'/api/v1/etl/jobs/{job_id}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Job deleted successfully'
        
        # Verify job is deleted
        get_response = client.get(f'/api/v1/etl/jobs/{job_id}')
        assert get_response.status_code == 404


class TestResultsEndpoints:
    """Test results retrieval endpoints"""
    
    @patch('app.DataIO')
    def test_get_results_no_filters(self, mock_data_io_class, client):
        """Test getting results without filters"""
        # Mock the database query
        mock_session = Mock()
        mock_query = Mock()
        mock_results = []
        
        # Add some mock results
        for i in range(5):
            mock_result = Mock()
            mock_result.gene_a_key = i + 1
            mock_result.gene_b_key = i + 2
            mock_result.rho_spearman = 0.5 + i * 0.1
            mock_result.p_value = 0.01
            mock_result.q_value = 0.02
            mock_result.n_samples = 20
            mock_result.computed_at = datetime.now()
            mock_result.batch_id = 'test-batch'
            mock_result.symbol = f'GENE_{i+1}'
            mock_result.illness_code = 'BRCA'
            mock_result.illness_description = 'Breast Cancer'
            mock_results.append(mock_result)
        
        mock_query.all.return_value = mock_results
        mock_session.query.return_value = mock_query
        mock_data_io = Mock()
        mock_data_io.Session.return_value = mock_session
        mock_data_io_class.return_value = mock_data_io
        
        response = client.get('/api/v1/results')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'results' in data
        assert data['count'] == 5
        assert data['limit'] == 100
        assert data['offset'] == 0
    
    def test_get_results_with_batch_filter(self, client):
        """Test getting results filtered by batch_id"""
        with patch('app.DataIO') as mock_data_io_class:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.all.return_value = []
            mock_session.query.return_value = mock_query
            mock_data_io = Mock()
            mock_data_io.Session.return_value = mock_session
            mock_data_io_class.return_value = mock_data_io
            
            response = client.get('/api/v1/results?batch_id=test-batch-123')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['results'] == []
    
    def test_get_results_with_pagination(self, client):
        """Test getting results with pagination parameters"""
        with patch('app.DataIO') as mock_data_io_class:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.all.return_value = []
            mock_session.query.return_value = mock_query
            mock_data_io = Mock()
            mock_data_io.Session.return_value = mock_session
            mock_data_io_class.return_value = mock_data_io
            
            response = client.get('/api/v1/results?limit=50&offset=100')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['limit'] == 50
            assert data['offset'] == 100
    
    def test_get_statistics(self, client):
        """Test getting pipeline statistics"""
        with patch('app.DataIO') as mock_data_io_class:
            mock_session = Mock()
            
            # Mock the count queries
            mock_session.query.return_value.count.side_effect = [1000, 200, 50]
            
            # Mock the illness breakdown query
            mock_illness_stats = [
                Mock(illness_code='BRCA', description='Breast Cancer', 
                     total_pairs=400, significant_pairs=100, avg_abs_correlation=0.3),
                Mock(illness_code='LUAD', description='Lung Adenocarcinoma', 
                     total_pairs=300, significant_pairs=50, avg_abs_correlation=0.25),
                Mock(illness_code='COAD', description='Colon Adenocarcinoma', 
                     total_pairs=300, significant_pairs=50, avg_abs_correlation=0.2)
            ]
            
            mock_session.query.return_value.join.return_value.group_by.return_value.all.return_value = mock_illness_stats
            mock_data_io = Mock()
            mock_data_io.Session.return_value = mock_session
            mock_data_io_class.return_value = mock_data_io
            
            response = client.get('/api/v1/statistics')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['total_correlations'] == 1000
            assert data['significant_correlations'] == 200
            assert data['highly_significant_correlations'] == 50
            assert len(data['illness_breakdown']) == 3


class TestValidationEndpoints:
    """Test validation results endpoints"""
    
    def test_get_validation_results_no_batch_filter(self, client):
        """Test getting validation results without batch filter"""
        with patch('app.DataIO') as mock_data_io_class:
            mock_session = Mock()
            mock_query = Mock()
            mock_results = []
            
            for i in range(3):
                mock_result = Mock()
                mock_result.validation_id = f'val-{i}'
                mock_result.batch_id = f'batch-{i}'
                mock_result.validation_type = 'post_check'
                mock_result.validation_name = f'test-validation-{i}'
                mock_result.status = 'pass'
                mock_result.details = f'Test details {i}'
                mock_result.validated_at = datetime.now()
                mock_results.append(mock_result)
            
            mock_query.order_by.return_value.limit.return_value.all.return_value = mock_results
            mock_session.query.return_value = mock_query
            mock_data_io = Mock()
            mock_data_io.Session.return_value = mock_session
            mock_data_io_class.return_value = mock_data_io
            
            response = client.get('/api/v1/validation')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['validations']) == 3
            assert data['count'] == 3
    
    def test_get_validation_results_with_batch_filter(self, client):
        """Test getting validation results filtered by batch_id"""
        with patch('app.DataIO') as mock_data_io_class:
            mock_session = Mock()
            mock_query = Mock()
            mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
            mock_session.query.return_value = mock_query
            mock_data_io = Mock()
            mock_data_io.Session.return_value = mock_session
            mock_data_io_class.return_value = mock_data_io
            
            response = client.get('/api/v1/validation?batch_id=test-batch-123')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['validations'] == []


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get('/api/v1/nonexistent-endpoint')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['error'] == 'Resource not found'
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error"""
        response = client.put('/api/v1/etl/run')
        
        assert response.status_code == 405
    
    @patch('app.DataIO')
    def test_internal_server_error(self, mock_data_io_class, client):
        """Test internal server error handling"""
        # Mock DataIO to raise an exception
        mock_data_io_class.side_effect = Exception("Database connection failed")
        
        response = client.get('/api/v1/statistics')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['error'] == 'Failed to get statistics'
    
    def test_invalid_json_payload(self, client):
        """Test handling of invalid JSON payload"""
        response = client.post('/api/v1/config/validate',
                             data='invalid json{',
                             content_type='application/json')
        
        assert response.status_code == 400