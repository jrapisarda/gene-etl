# Gene Pair Correlation ETL System

A production-ready ETL (Extract, Transform, Load) system for computing Spearman correlations between gene pairs across different illness cohorts. This system replaces legacy Python scripts with an enterprise-grade solution featuring comprehensive logging, monitoring, error handling, and performance optimization.

## Features

- **High-Performance Correlation Analysis**: Spearman correlation with Benjamini-Hochberg FDR correction
- **Scalable Architecture**: Parallel processing of illness cohorts with configurable workers
- **Comprehensive Validation**: Pre and post-processing data quality checks
- **Production-Ready**: REST API, detailed logging, monitoring, and error handling
- **Flexible Configuration**: YAML-based configuration with environment-specific settings
- **Data Lineage**: Complete tracking of data sources, processing, and results

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd gene_etl_mvp

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Create database (PostgreSQL)
createdb bioinformatics_dev

# Set environment variable
export DATABASE_URL="postgresql://user:password@localhost:5432/bioinformatics_dev"

# Initialize database schema
python -c "
from app import create_app, db
from models import Base
app = create_app()
with app.app_context():
    Base.metadata.create_all(bind=db.engine)
"

# Load seed data
psql -d bioinformatics_dev -f seeds/01_dim.sql
psql -d bioinformatics_dev -f seeds/02_expression.sql
```

### 3. Run the ETL Pipeline

```bash
# Command line
python -m etl

# Or via REST API
python app.py
```

### 4. Access Results

```bash
# Check API health
curl http://localhost:5000/health

# Get statistics
curl http://localhost:5000/api/v1/statistics

# Get correlation results
curl http://localhost:5000/api/v1/results?limit=10
```

## Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Source Data   │    │   ETL Pipeline  │    │   Destination   │
│   (PostgreSQL)  │───▶│  (Python/Polars)│───▶│   (PostgreSQL)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   REST API      │
                       │   (Flask)       │
                       └─────────────────┘
```

### Database Schema

- **dim**: Dimension tables (genes, illnesses, samples)
- **fact**: Fact tables (expression data, correlation results)
- **meta**: Metadata tables (process logs, validation results, configuration)
- **analytics**: Views for analysis and reporting

### Key Modules

- **`etl/`**: Core ETL pipeline components
  - `config.py`: Configuration management with Pydantic
  - `io.py`: Data extraction and loading operations
  - `filters.py`: Gene filtering strategies
  - `correlate.py`: Statistical correlation analysis
  - `persist.py`: Result persistence with batching
  - `validate.py`: Data quality validation
- **`models.py`**: SQLAlchemy ORM models
- **`app.py`**: Flask REST API application
- **`tests/`**: Comprehensive test suite

## Configuration

### Example Configuration

```yaml
config_name: "gene_correlation_dev"
config_version: "1.0.0"
environment: "dev"

database:
  host: "localhost"
  port: 5432
  database: "bioinformatics_dev"
  username: "etl_user"
  password: "secure_password"

gene_filter:
  method: "variance"
  top_n_genes: 3000
  min_variance_threshold: 0.01

statistical:
  correlation_method: "spearman"
  min_samples: 20
  alpha_fdr: 0.05

processing:
  batch_size: 1000
  block_size: 100
  max_workers: 4
```

See `runbook.md` for detailed configuration options.

## Usage Examples

### Python API

```python
from etl import run_etl_pipeline
from etl.config import ETLConfig

# Configure pipeline
config = ETLConfig.from_file('config/production.yaml')

# Run ETL
results = run_etl_pipeline(config=config)

print(f"Processed {results['summary']['correlations_computed']} correlations")
print(f"Success rate: {results['summary']['success_rate']:.2%}")
```

### REST API

```bash
# Trigger ETL pipeline
curl -X POST http://localhost:5000/api/v1/etl/run \
  -H "Content-Type: application/json" \
  -d @config/production.json

# Monitor job status
curl http://localhost:5000/api/v1/etl/jobs/{job_id}

# Get results with filtering
curl "http://localhost:5000/api/v1/results?illness_code=BRCA&limit=100"
```

### Custom Analysis

```python
from etl.correlate import GeneCorrelationAnalyzer
from etl.filters import GeneFilter

# Create custom analyzer
analyzer = GeneCorrelationAnalyzer(config)

# Apply gene filtering
selected_genes, filtered_data = gene_filter.apply_comprehensive_filtering(
    expression_data
)

# Compute correlations
results = analyzer.analyze_multiple_cohorts(
    filtered_data, selected_genes
)
```

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=etl --cov-report=html

# Run specific test file
pytest tests/test_correlate.py -v
```

### Test Data

The system includes comprehensive test fixtures:
- Sample gene metadata (20 genes)
- Sample illness data (5 cancer types)
- Sample expression data (15 samples across 3 illnesses)
- Sample correlation results for validation

## Performance

### Benchmarks

Typical performance on standard hardware:
- **10,000 genes × 100 samples**: ~2-4 hours
- **3,000 genes × 50 samples**: ~30-60 minutes
- **1,000 genes × 20 samples**: ~5-10 minutes

### Optimization Tips

1. **Gene Filtering**: Use variance-based filtering to reduce gene count
2. **Parallel Processing**: Configure max_workers based on available cores
3. **Batch Size**: Tune batch_size for your memory constraints
4. **Database Indexes**: Ensure proper indexing for query performance

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:5000/health

# Database connectivity
curl http://localhost:5000/api/v1/statistics
```

### Logging

The system provides structured JSON logging with:
- Process execution tracking
- Data validation results
- Performance metrics
- Error details with stack traces

### Validation

Comprehensive validation includes:
- Pre-processing data quality checks
- Statistical result validation
- Database constraint enforcement
- Post-processing consistency checks

## Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   # Production configuration
   export ENVIRONMENT=production
   export DATABASE_URL="postgresql://user:pass@prod-db:5432/bioinformatics"
   ```

2. **Docker Deployment**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "app.py"]
   ```

3. **Systemd Service**
   ```ini
   [Unit]
   Description=Gene ETL API
   After=network.target

   [Service]
   User=etl_user
   WorkingDirectory=/opt/gene_etl_mvp
   Environment="DATABASE_URL=postgresql://..."
   ExecStart=/opt/gene_etl_mvp/venv/bin/python app.py
   Restart=always
   ```

### Monitoring Integration

The system integrates with:
- **Prometheus**: Metrics endpoint available
- **Grafana**: Pre-built dashboards
- **ELK Stack**: Structured logging format
- **Custom Monitoring**: REST API for health checks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black etl/ tests/

# Run type checking
mypy etl/

# Run linting
flake8 etl/ tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

- **Documentation**: See `runbook.md` for detailed usage instructions
- **Issues**: Report bugs via GitHub issues
- **Questions**: Contact the bioinformatics team

## Acknowledgments

- **TCGA**: For providing the inspiration and data patterns
- **SciPy/NumPy**: For statistical computing capabilities
- **Polars**: For high-performance data processing
- **SQLAlchemy**: For robust database ORM

---

*For detailed installation and usage instructions, see the [Runbook](runbook.md).*

**Version**: 1.0.0  
**Last Updated**: September 29, 2025  
**Maintained by**: Bioinformatics Team