-- Seed data for gene expression matrix
-- This script populates the fact.gene_expression table with sample expression data

-- Insert sample expression data for BRCA (Breast Cancer)
INSERT INTO fact.fact_gene_expression (gene_key, sample_key, expression_value, batch_id) VALUES
-- Gene 1 (TSPAN6) expressions across BRCA samples
(1, 1, 1234.56, 'batch_001'), (1, 2, 987.65, 'batch_001'), (1, 3, 1456.78, 'batch_001'), 
(1, 4, 876.54, 'batch_001'), (1, 5, 1123.45, 'batch_001'),
-- Gene 2 (TNMD) expressions across BRCA samples
(2, 1, 234.56, 'batch_001'), (2, 2, 345.67, 'batch_001'), (2, 3, 198.76, 'batch_001'),
(2, 4, 287.65, 'batch_001'), (2, 5, 256.78, 'batch_001'),
-- Gene 3 (DPM1) expressions across BRCA samples
(3, 1, 567.89, 'batch_001'), (3, 2, 678.90, 'batch_001'), (3, 3, 456.78, 'batch_001'),
(3, 4, 589.01, 'batch_001'), (3, 5, 612.34, 'batch_001'),
-- Gene 4 (SCYL3) expressions across BRCA samples
(4, 1, 89.12, 'batch_001'), (4, 2, 78.34, 'batch_001'), (4, 3, 92.45, 'batch_001'),
(4, 4, 85.67, 'batch_001'), (4, 5, 91.23, 'batch_001'),
-- Gene 5 (C1orf112) expressions across BRCA samples
(5, 1, 456.78, 'batch_001'), (5, 2, 367.89, 'batch_001'), (5, 3, 498.76, 'batch_001'),
(5, 4, 412.34, 'batch_001'), (5, 5, 445.67, 'batch_001');

-- Insert sample expression data for LUAD (Lung Adenocarcinoma)
INSERT INTO fact.fact_gene_expression (gene_key, sample_key, expression_value, batch_id) VALUES
-- Gene 1 (TSPAN6) expressions across LUAD samples
(1, 6, 1111.11, 'batch_002'), (1, 7, 1222.22, 'batch_002'), (1, 8, 1055.55, 'batch_002'),
(1, 9, 1188.88, 'batch_002'), (1, 10, 1144.44, 'batch_002'),
-- Gene 2 (TNMD) expressions across LUAD samples
(2, 6, 188.88, 'batch_002'), (2, 7, 199.99, 'batch_002'), (2, 8, 177.77, 'batch_002'),
(2, 9, 211.11, 'batch_002'), (2, 10, 194.44, 'batch_002'),
-- Gene 6 (FGR) expressions across LUAD samples
(6, 6, 789.01, 'batch_002'), (6, 7, 876.54, 'batch_002'), (6, 8, 765.43, 'batch_002'),
(6, 9, 823.45, 'batch_002'), (6, 10, 798.76, 'batch_002'),
-- Gene 7 (CFH) expressions across LUAD samples
(7, 6, 345.67, 'batch_002'), (7, 7, 356.78, 'batch_002'), (7, 8, 334.56, 'batch_002'),
(7, 9, 367.89, 'batch_002'), (7, 10, 348.90, 'batch_002'),
-- Gene 8 (FUCA2) expressions across LUAD samples
(8, 6, 234.56, 'batch_002'), (8, 7, 245.67, 'batch_002'), (8, 8, 223.45, 'batch_002'),
(8, 9, 256.78, 'batch_002'), (8, 10, 238.90, 'batch_002');

-- Insert sample expression data for COAD (Colon Adenocarcinoma)
INSERT INTO fact.fact_gene_expression (gene_key, sample_key, expression_value, batch_id) VALUES
-- Gene 3 (DPM1) expressions across COAD samples
(3, 11, 678.90, 'batch_003'), (3, 12, 689.01, 'batch_003'), (3, 13, 667.89, 'batch_003'),
(3, 14, 690.12, 'batch_003'), (3, 15, 675.34, 'batch_003'),
-- Gene 4 (SCYL3) expressions across COAD samples
(4, 11, 123.45, 'batch_003'), (4, 12, 134.56, 'batch_003'), (4, 13, 112.34, 'batch_003'),
(4, 14, 145.67, 'batch_003'), (4, 15, 128.90, 'batch_003'),
-- Gene 9 (GCLC) expressions across COAD samples
(9, 11, 901.23, 'batch_003'), (9, 12, 912.34, 'batch_003'), (9, 13, 889.01, 'batch_003'),
(9, 14, 923.45, 'batch_003'), (9, 15, 906.78, 'batch_003'),
-- Gene 10 (NFYA) expressions across COAD samples
(10, 11, 567.89, 'batch_003'), (10, 12, 578.90, 'batch_003'), (10, 13, 556.78, 'batch_003'),
(10, 14, 589.01, 'batch_003'), (10, 15, 572.34, 'batch_003'),
-- Gene 11 (STPG1) expressions across COAD samples
(11, 11, 345.67, 'batch_003'), (11, 12, 356.78, 'batch_003'), (11, 13, 334.56, 'batch_003'),
(11, 14, 367.89, 'batch_003'), (11, 15, 348.90, 'batch_003');

-- Insert additional expression data for more comprehensive testing
-- High variance genes for filtering tests
INSERT INTO fact.fact_gene_expression (gene_key, sample_key, expression_value, batch_id) VALUES
-- High variance gene 12 (NIPAL3) - BRCA
(12, 1, 100.0, 'batch_001'), (12, 2, 200.0, 'batch_001'), (12, 3, 50.0, 'batch_001'),
(12, 4, 300.0, 'batch_001'), (12, 5, 150.0, 'batch_001'),
-- High variance gene 13 (LAS1L) - LUAD
(13, 6, 50.0, 'batch_002'), (13, 7, 400.0, 'batch_002'), (13, 8, 25.0, 'batch_002'),
(13, 9, 500.0, 'batch_002'), (13, 10, 75.0, 'batch_002'),
-- High variance gene 14 (ENPP4) - COAD
(14, 11, 200.0, 'batch_003'), (14, 12, 50.0, 'batch_003'), (14, 13, 300.0, 'batch_003'),
(14, 14, 25.0, 'batch_003'), (14, 15, 250.0, 'batch_003'),
-- Low variance gene 15 (SEMA3F) - BRCA
(15, 1, 100.0, 'batch_001'), (15, 2, 102.0, 'batch_001'), (15, 3, 99.0, 'batch_001'),
(15, 4, 101.0, 'batch_001'), (15, 5, 100.5, 'batch_001'),
-- Low variance gene 16 (CFTR) - LUAD
(16, 6, 200.0, 'batch_002'), (16, 7, 201.0, 'batch_002'), (16, 8, 199.5, 'batch_002'),
(16, 9, 200.5, 'batch_002'), (16, 10, 200.25, 'batch_002'),
-- Low variance gene 17 (CAV1) - COAD
(17, 11, 150.0, 'batch_003'), (17, 12, 151.0, 'batch_003'), (17, 13, 149.5, 'batch_003'),
(17, 14, 150.5, 'batch_003'), (17, 15, 150.25, 'batch_003');

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_gene_expression_gene_key ON fact.fact_gene_expression (gene_key);
CREATE INDEX IF NOT EXISTS idx_gene_expression_sample_key ON fact.fact_gene_expression (sample_key);
CREATE INDEX IF NOT EXISTS idx_gene_expression_batch_id ON fact.fact_gene_expression (batch_id);
CREATE INDEX IF NOT EXISTS idx_gene_expression_composite ON fact.fact_gene_expression (gene_key, sample_key);