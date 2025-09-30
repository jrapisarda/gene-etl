# Gene Pair Correlation ETL System - Runbook

## Overview

This runbook provides comprehensive instructions for installing, configuring, running, and maintaining the Gene Pair Correlation ETL system. The system processes gene expression data to compute Spearman correlations between gene pairs across different illness cohorts.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the ETL Pipeline](#running-the-etl-pipeline)
5. [Testing](#testing)
6. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
7. [Maintenance](#maintenance)
8. [API Usage](#api-usage)

## System Requirements

### Hardware Requirements
- **CPU**: 4+ cores recommended for parallel processing
- **RAM**: 8GB minimum, 16GB+ recommended for large datasets
- **Storage**: 50GB+ available space for data and logs
- **Network**: Stable connection to database server

### Software Requirements
- **Python**: 3.11+ (3.11.5 recommended)
- **PostgreSQL**: 12+ or SQL Server 2019+
- **Operating System**: Linux (Ubuntu 20.04+), macOS (11+), or Windows 10+

### Python Dependencies
See `requirements.txt` for complete list. Key packages include:
- Flask 2.3.3+ (Web framework)
- SQLAlchemy 2.0.20+ (Database ORM)
- Polars 0.19.3+ (Data processing)
- SciPy 1.11.2+ (Statistical analysis)
- Statsmodels 0.14.0+ (FDR correction)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd gene_etl_mvp
```

### 2. Create Virtual Environment
```bash
# Using venv
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n gene_etl python=3.11
conda activate gene_etl
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

### 4. Set Up Database

#### PostgreSQL Setup
```bash
# Create database
createdb bioinformatics_dev

# Create user
psql -c "CREATE USER etl_user WITH PASSWORD 'your_secure_password';"

# Grant permissions
psql -d bioinformatics_dev -c "GRANT ALL PRIVILEGES ON DATABASE bioinformatics_dev TO etl_user;"
```

#### SQL Server Setup
```sql
-- Create database
CREATE DATABASE BioinformaticsWarehouse;
GO

-- Create schemas
USE BioinformaticsWarehouse;
CREATE SCHEMA dim;
CREATE SCHEMA fact;
CREATE SCHEMA meta;
CREATE SCHEMA analytics;
GO
```

### 5. Initialize Database Schema
```bash
# Set database URL environment variable
export DATABASE_URL="postgresql://etl_user:your_secure_password@localhost:5432/bioinformatics_dev"

# Create tables
python -c "
from app import create_app, db
from models import Base
app = create_app()
with app.app_context():
    Base.metadata.create_all(bind=db.engine)
"
```

### 6. Load Seed Data
```bash
# Load dimension data
psql -d bioinformatics_dev -f seeds/01_dim.sql

# Load expression data
psql -d bioinformatics_dev -f seeds/02_expression.sql
```

## Configuration

### Configuration File Structure
Create a YAML configuration file (e.g., `config/production.yaml`):

```yaml
config_name: "gene_correlation_production"
config_version: "1.0.0"
environment: "prod"

database:
  host: "your-db-host"
  port: 5432
  database: "bioinformatics_prod"
  username: "etl_user"
  password: "${DB_PASSWORD}"  # Use environment variable
  pool_size: 10
  max_overflow: 20

gene_filter:
  method: "variance"
  top_n_genes: 3000
  min_variance_threshold: 0.01
  min_expression_value: 0.0

statistical:
  correlation_method: "spearman"
  min_samples: 20
  alpha_fdr: 0.05

processing:
  batch_size: 1000
  block_size: 100
  max_workers: 4
  memory_limit_gb: 8.0
  checkpoint_interval: 10000

logging:
  level: "INFO"
  format: "json"
  file_path: "/var/log/gene_etl/etl.log"
  max_file_size_mb: 100
  backup_count: 5

alerting:
  enabled: true
  email_recipients: ["bioinfo-team@company.com"]
  smtp_server: "smtp.company.com"
  smtp_port: 587
```

### Environment Variables
```bash
# Database connection
export DATABASE_URL="postgresql://user:password@host:5432/database"

# Optional: Override config file path
export ETL_CONFIG_PATH="/path/to/config.yaml"

# Optional: Set log level
export LOG_LEVEL="DEBUG"
```

## Running the ETL Pipeline

### 1. Command Line Execution
```bash
# Run with default configuration
python -m etl

# Run with custom configuration
python -m etl --config /path/to/config.yaml

# Run specific illness cohorts
python -m etl --illness-whitelist BRCA,LUAD

# Run with batch ID for tracking
python -m etl --batch-id custom-batch-123
```

### 2. Python API
```python
from etl import run_etl_pipeline
from etl.config import ETLConfig

# Run with config object
config = ETLConfig.from_file('config/production.yaml')
results = run_etl_pipeline(config=config)

print(f"ETL completed: {results['summary']}")
```

### 3. REST API Execution
```bash
# Start the Flask API
python app.py

# Trigger ETL via API
curl -X POST http://localhost:5000/api/v1/etl/run \
  -H "Content-Type: application/json" \
  -d @config/production.json

# Check job status
curl http://localhost:5000/api/v1/etl/jobs/{job_id}
```

## Testing

### Unit Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_correlate.py

# Run with coverage
pytest --cov=etl --cov-report=html

# Run with verbose output
pytest -v
```

### Integration Tests
```bash
# Run integration tests (requires database)
pytest tests/test_integration.py

# Run with specific database
DATABASE_URL="postgresql://test:test@localhost:5432/test_db" pytest tests/test_integration.py
```

### Load Testing
```bash
# Install locust for load testing
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:5000
```

## Monitoring and Troubleshooting

### Log Files
- **Application logs**: `/var/log/gene_etl/etl.log`
- **Error logs**: `/var/log/gene_etl/error.log`
- **Access logs**: `/var/log/gene_etl/access.log`

### Monitoring Queries
```sql
-- Check ETL process status
SELECT * FROM meta.etl_process_log 
ORDER BY start_time DESC 
LIMIT 10;

-- Check validation results
SELECT * FROM meta.data_validation_log 
WHERE batch_id = 'your-batch-id';

-- Check processing statistics
SELECT 
    illness_code,
    COUNT(*) as total_pairs,
    COUNT(CASE WHEN q_value <= 0.05 THEN 1 END) as significant_pairs,
    AVG(ABS(rho_spearman)) as avg_abs_correlation
FROM fact.fact_gene_pair_correlation c
JOIN dim.dim_illness i ON c.illness_key = i.illness_key
WHERE batch_id = 'your-batch-id'
GROUP BY illness_code;
```

### Common Issues and Solutions

#### 1. Database Connection Issues
```bash
# Test database connection
psql $DATABASE_URL -c "SELECT 1;"

# Check connection pool settings
# Increase pool_size in config if needed
```

#### 2. Memory Issues
```bash
# Monitor memory usage
htop

# Reduce batch_size in config
# Increase memory_limit_gb in config
```

#### 3. Performance Issues
```bash
# Check database indexes
SELECT * FROM pg_indexes WHERE schemaname = 'fact';

# Analyze slow queries
EXPLAIN ANALYZE SELECT * FROM fact.fact_gene_pair_correlation WHERE batch_id = '...';
```

#### 4. Validation Failures
```bash
# Check validation logs
SELECT * FROM meta.data_validation_log 
WHERE status = 'fail' 
ORDER BY validated_at DESC;
```

## Maintenance

### Regular Tasks

#### 1. Database Maintenance
```sql
-- Update table statistics
ANALYZE fact.fact_gene_pair_correlation;
ANALYZE fact.fact_gene_expression;

-- Vacuum and analyze
VACUUM ANALYZE fact.fact_gene_pair_correlation;

-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname IN ('dim', 'fact', 'meta')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

#### 2. Log Rotation
```bash
# Set up logrotate (usually automatic)
# Manual log rotation
sudo logrotate -f /etc/logrotate.d/gene-etl
```

#### 3. Backup Procedures
```bash
# Database backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/
```

### Performance Optimization

#### 1. Database Optimization
```sql
-- Create additional indexes if needed
CREATE INDEX CONCURRENTLY idx_correlation_batch_illness 
ON fact.fact_gene_pair_correlation (batch_id, illness_key);

-- Partition large tables (if needed)
-- CREATE TABLE fact.fact_gene_pair_correlation_2023 
-- PARTITION OF fact.fact_gene_pair_correlation 
-- FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
```

#### 2. Application Optimization
- Adjust batch sizes based on available memory
- Tune parallel processing settings
- Optimize gene filtering parameters

## API Usage

### REST API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Configuration
```bash
# Get example config
curl http://localhost:5000/api/v1/config

# Validate config
curl -X POST http://localhost:5000/api/v1/config/validate \
  -H "Content-Type: application/json" \
  -d @config.json
```

#### ETL Operations
```bash
# Run ETL
curl -X POST http://localhost:5000/api/v1/etl/run

# List jobs
curl http://localhost:5000/api/v1/etl/jobs

# Get job status
curl http://localhost:5000/api/v1/etl/jobs/{job_id}
```

#### Data Retrieval
```bash
# Get results
curl "http://localhost:5000/api/v1/results?limit=10"

# Get statistics
curl http://localhost:5000/api/v1/statistics

# Get validation results
curl http://localhost:5000/api/v1/validation
```

### Python API Examples

#### Running ETL
```python
from etl import run_etl_pipeline
from etl.config import ETLConfig

# Configure and run
config = ETLConfig(
    config_name="my_analysis",
    database={"host": "localhost", "database": "mydb"},
    gene_filter={"top_n_genes": 2000}
)

results = run_etl_pipeline(config=config)
print(f"Processed {results['summary']['correlations_computed']} correlations")
```

#### Custom Analysis
```python
from etl.correlate import GeneCorrelationAnalyzer
from etl.filters import GeneFilter

# Create custom analyzer
config = StatisticalConfig(correlation_method="pearson")
analyzer = GeneCorrelationAnalyzer(config)

# Run analysis on custom data
results = analyzer.analyze_illness_cohort(
    illness_key=1,
    expression_df=my_expression_data,
    gene_keys=my_gene_list
)
```

## Support and Resources

### Documentation
- **API Documentation**: http://localhost:5000/docs (when running)
- **Code Documentation**: `docs/` directory
- **Architecture**: See `arch.md` and RFC documents

### Getting Help
1. Check logs for error messages
2. Review this runbook for common solutions
3. Check validation results in the database
4. Contact the bioinformatics team

### Contributing
1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Submit pull requests with clear descriptions

## Version History

- **v1.0.0**: Initial release with core ETL functionality
- **v1.1.0**: Added REST API and improved error handling
- **v1.2.0**: Enhanced monitoring and validation

---

*Last updated: September 29, 2025*
*For questions or support, contact the Bioinformatics Team*