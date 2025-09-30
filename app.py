"""
Flask REST API for Gene Pair Correlation ETL System

This module provides REST endpoints for triggering, monitoring, and managing
the gene pair correlation ETL pipeline.
"""

from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import logging
import os
import uuid
from typing import Dict, Any, Optional

from models import Base
from etl.config import ETLConfig, create_example_config
from etl import run_etl_pipeline, GeneCorrelationETL
from etl.io import DataIO

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://etl_user:password@localhost:5432/bioinformatics_dev')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# In-memory storage for job tracking (in production, use Redis or similar)
job_tracker = {}


def create_app():
    """Application factory for creating Flask app instances"""
    return app


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'gene-etl-api'
    })


@app.route('/api/v1/config', methods=['GET'])
def get_config():
    """Get example configuration"""
    try:
        config = create_example_config()
        return jsonify({
            'config': config.to_dict(),
            'message': 'Example configuration retrieved successfully'
        })
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({
            'error': 'Failed to retrieve configuration',
            'details': str(e)
        }), 500


@app.route('/api/v1/config/validate', methods=['POST'])
def validate_config():
    """Validate configuration"""
    try:
        config_data = request.get_json()
        if not config_data:
            return jsonify({
                'error': 'No configuration data provided'
            }), 400
        
        # Validate configuration
        config = ETLConfig(**config_data)
        
        return jsonify({
            'valid': True,
            'message': 'Configuration is valid',
            'config_hash': config.get_config_hash()
        })
        
    except Exception as e:
        return jsonify({
            'valid': False,
            'error': 'Invalid configuration',
            'details': str(e)
        }), 400


@app.route('/api/v1/etl/run', methods=['POST'])
def run_etl():
    """Trigger ETL pipeline execution"""
    try:
        # Get configuration from request
        config_data = request.get_json()
        
        if not config_data:
            # Use default configuration
            config = create_example_config()
        else:
            config = ETLConfig(**config_data)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Track job
        job_tracker[job_id] = {
            'status': 'running',
            'start_time': datetime.utcnow().isoformat(),
            'config': config.to_dict()
        }
        
        # Run ETL pipeline (synchronously for now - in production, use Celery or similar)
        try:
            etl = GeneCorrelationETL(config)
            results = etl.run()
            
            job_tracker[job_id].update({
                'status': 'completed',
                'end_time': datetime.utcnow().isoformat(),
                'results': results
            })
            
            return jsonify({
                'job_id': job_id,
                'status': 'completed',
                'results': results,
                'message': 'ETL pipeline completed successfully'
            })
            
        except Exception as e:
            job_tracker[job_id].update({
                'status': 'failed',
                'end_time': datetime.utcnow().isoformat(),
                'error': str(e)
            })
            
            return jsonify({
                'job_id': job_id,
                'status': 'failed',
                'error': str(e),
                'message': 'ETL pipeline failed'
            }), 500
            
    except Exception as e:
        logger.error(f"Error running ETL: {e}")
        return jsonify({
            'error': 'Failed to run ETL pipeline',
            'details': str(e)
        }), 500


@app.route('/api/v1/etl/jobs', methods=['GET'])
def list_jobs():
    """List all ETL jobs"""
    try:
        return jsonify({
            'jobs': job_tracker,
            'count': len(job_tracker)
        })
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return jsonify({
            'error': 'Failed to list jobs',
            'details': str(e)
        }), 500


@app.route('/api/v1/etl/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of specific ETL job"""
    try:
        if job_id not in job_tracker:
            return jsonify({
                'error': 'Job not found',
                'job_id': job_id
            }), 404
        
        return jsonify(job_tracker[job_id])
        
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return jsonify({
            'error': 'Failed to get job status',
            'details': str(e)
        }), 500


@app.route('/api/v1/etl/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete job from tracker"""
    try:
        if job_id not in job_tracker:
            return jsonify({
                'error': 'Job not found',
                'job_id': job_id
            }), 404
        
        del job_tracker[job_id]
        
        return jsonify({
            'message': 'Job deleted successfully',
            'job_id': job_id
        })
        
    except Exception as e:
        logger.error(f"Error deleting job: {e}")
        return jsonify({
            'error': 'Failed to delete job',
            'details': str(e)
        }), 500


@app.route('/api/v1/results', methods=['GET'])
def get_results():
    """Get correlation results"""
    try:
        # Get query parameters
        batch_id = request.args.get('batch_id')
        illness_code = request.args.get('illness_code')
        gene_symbol = request.args.get('gene_symbol')
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        
        # Create data IO instance
        config = create_example_config()
        data_io = DataIO(config)
        
        # Build query
        session = data_io.Session()
        
        from models import FactGenePairCorrelation, DimGene, DimIllness
        
        query = session.query(
            FactGenePairCorrelation,
            DimGene.symbol.label('gene_a_symbol'),
            DimGene.symbol.label('gene_b_symbol'),
            DimIllness.illness_code,
            DimIllness.description.label('illness_description')
        ).join(
            DimGene, FactGenePairCorrelation.gene_a_key == DimGene.gene_key
        ).join(
            DimGene, FactGenePairCorrelation.gene_b_key == DimGene.gene_key
        ).join(
            DimIllness, FactGenePairCorrelation.illness_key == DimIllness.illness_key
        )
        
        # Apply filters
        if batch_id:
            query = query.filter(FactGenePairCorrelation.batch_id == batch_id)
        
        if illness_code:
            query = query.filter(DimIllness.illness_code == illness_code)
        
        if gene_symbol:
            query = query.filter(
                (DimGene.symbol == gene_symbol) |
                (DimGene.symbol == gene_symbol)
            )
        
        # Apply pagination
        query = query.offset(offset).limit(limit)
        
        results = query.all()
        
        # Format results
        formatted_results = []
        for result in results:
            correlation, gene_a_symbol, gene_b_symbol, illness_code, illness_description = result
            formatted_results.append({
                'gene_a_key': correlation.gene_a_key,
                'gene_b_key': correlation.gene_b_key,
                'gene_a_symbol': gene_a_symbol,
                'gene_b_symbol': gene_b_symbol,
                'illness_key': correlation.illness_key,
                'illness_code': illness_code,
                'illness_description': illness_description,
                'rho_spearman': correlation.rho_spearman,
                'p_value': correlation.p_value,
                'q_value': correlation.q_value,
                'n_samples': correlation.n_samples,
                'computed_at': correlation.computed_at.isoformat(),
                'batch_id': correlation.batch_id
            })
        
        return jsonify({
            'results': formatted_results,
            'count': len(formatted_results),
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        return jsonify({
            'error': 'Failed to get results',
            'details': str(e)
        }), 500


@app.route('/api/v1/statistics', methods=['GET'])
def get_statistics():
    """Get pipeline statistics"""
    try:
        # Create data IO instance
        config = create_example_config()
        data_io = DataIO(config)
        session = data_io.Session()
        
        from models import FactGenePairCorrelation, DimIllness
        from sqlalchemy import func
        
        # Get basic statistics
        total_correlations = session.query(FactGenePairCorrelation).count()
        
        significant_correlations = session.query(FactGenePairCorrelation).filter(
            FactGenePairCorrelation.q_value <= 0.05
        ).count()
        
        highly_significant_correlations = session.query(FactGenePairCorrelation).filter(
            FactGenePairCorrelation.q_value <= 0.01
        ).count()
        
        # Get illness breakdown
        illness_stats = session.query(
            DimIllness.illness_code,
            DimIllness.description,
            func.count(FactGenePairCorrelation.correlation_key).label('total_pairs'),
            func.count(func.case([(FactGenePairCorrelation.q_value <= 0.05, 1)])).label('significant_pairs'),
            func.avg(func.abs(FactGenePairCorrelation.rho_spearman)).label('avg_abs_correlation')
        ).join(
            FactGenePairCorrelation, DimIllness.illness_key == FactGenePairCorrelation.illness_key
        ).group_by(
            DimIllness.illness_key, DimIllness.illness_code, DimIllness.description
        ).all()
        
        # Format illness statistics
        illness_breakdown = []
        for stat in illness_stats:
            illness_breakdown.append({
                'illness_code': stat.illness_code,
                'illness_description': stat.description,
                'total_pairs': stat.total_pairs,
                'significant_pairs': stat.significant_pairs,
                'avg_abs_correlation': float(stat.avg_abs_correlation) if stat.avg_abs_correlation else 0
            })
        
        return jsonify({
            'total_correlations': total_correlations,
            'significant_correlations': significant_correlations,
            'highly_significant_correlations': highly_significant_correlations,
            'significance_rate': significant_correlations / total_correlations if total_correlations > 0 else 0,
            'illness_breakdown': illness_breakdown
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({
            'error': 'Failed to get statistics',
            'details': str(e)
        }), 500


@app.route('/api/v1/validation', methods=['GET'])
def get_validation_results():
    """Get validation results"""
    try:
        batch_id = request.args.get('batch_id')
        
        # Create data IO instance
        config = create_example_config()
        data_io = DataIO(config)
        session = data_io.Session()
        
        from models import DataValidationLog
        
        query = session.query(DataValidationLog)
        
        if batch_id:
            query = query.filter(DataValidationLog.batch_id == batch_id)
        
        results = query.order_by(DataValidationLog.validated_at.desc()).limit(100).all()
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                'validation_id': result.validation_id,
                'batch_id': result.batch_id,
                'validation_type': result.validation_type,
                'validation_name': result.validation_name,
                'status': result.status,
                'details': result.details,
                'validated_at': result.validated_at.isoformat()
            })
        
        return jsonify({
            'validations': formatted_results,
            'count': len(formatted_results)
        })
        
    except Exception as e:
        logger.error(f"Error getting validation results: {e}")
        return jsonify({
            'error': 'Failed to get validation results',
            'details': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Resource not found',
        'message': 'The requested resource was not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An internal server error occurred'
    }), 500


if __name__ == '__main__':
    # Initialize database tables
    with app.app_context():
        Base.metadata.create_all(bind=db.engine)
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('DEBUG', 'False').lower() == 'true'
    )