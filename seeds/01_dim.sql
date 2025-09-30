-- Seed data for dimension tables
-- This script populates the dimension tables with sample data for testing

-- Insert sample genes
INSERT INTO dim.dim_gene (gene_key, ensembl_id, symbol, gene_name, chromosome, start_position, end_position, strand, gene_type) VALUES
(1, 'ENSG00000000003', 'TSPAN6', 'tetraspanin 6', 'X', 100627109, 100639285, '-', 'protein_coding'),
(2, 'ENSG00000000005', 'TNMD', 'tenomodulin', 'X', 100584802, 100599885, '-', 'protein_coding'),
(3, 'ENSG00000000419', 'DPM1', 'dolichyl-phosphate mannosyltransferase', '20', 49551404, 49575035, '-', 'protein_coding'),
(4, 'ENSG00000000457', 'SCYL3', 'SCY1 like pseudokinase 3', '1', 169862215, 169894750, '-', 'protein_coding'),
(5, 'ENSG00000000460', 'C1orf112', 'chromosome 1 open reading frame 112', '1', 169631245, 169823221, '-', 'protein_coding'),
(6, 'ENSG00000000938', 'FGR', 'FGR proto-oncogene', '1', 27938574, 27961789, '+', 'protein_coding'),
(7, 'ENSG00000000971', 'CFH', 'complement factor H', '1', 196652048, 196747504, '+', 'protein_coding'),
(8, 'ENSG00000001036', 'FUCA2', 'fucosidase alpha-2', '6', 143734241, 143752013, '-', 'protein_coding'),
(9, 'ENSG00000001084', 'GCLC', 'glutamate-cysteine ligase catalytic subunit', '6', 167654245, 167776140, '+', 'protein_coding'),
(10, 'ENSG00000001167', 'NFYA', 'nuclear transcription factor Y alpha', '6', 41040628, 41067710, '+', 'protein_coding'),
(11, 'ENSG00000001460', 'STPG1', 'steroidogenic protein type 1', '1', 247422870, 247430764, '+', 'protein_coding'),
(12, 'ENSG00000001461', 'NIPAL3', 'NIPA like domain containing 3', '1', 247372602, 247421634, '+', 'protein_coding'),
(13, 'ENSG00000001497', 'LAS1L', 'LAS1 like ribosome biogenesis factor', 'X', 153648315, 153688157, '+', 'protein_coding'),
(14, 'ENSG00000001561', 'ENPP4', 'ectonucleotide pyrophosphatase/phosphodiesterase 4', '6', 97935728, 97977089, '+', 'protein_coding'),
(15, 'ENSG00000001617', 'SEMA3F', 'semaphorin 3F', '3', 50111180, 50285827, '-', 'protein_coding'),
(16, 'ENSG00000001626', 'CFTR', 'cystic fibrosis transmembrane conductance regulator', '7', 117120016, 117308718, '-', 'protein_coding'),
(17, 'ENSG00000001629', 'CAV1', 'caveolin 1', '7', 116672098, 116693636, '-', 'protein_coding'),
(18, 'ENSG00000001630', 'CAV2', 'caveolin 2', '7', 116770633, 116793896, '-', 'protein_coding'),
(19, 'ENSG00000002016', 'RAD52', 'RAD52 homolog, DNA repair protein', '12', 21561883, 21606994, '+', 'protein_coding'),
(20, 'ENSG00000002079', 'ARHGAP25', 'Rho GTPase activating protein 25', '2', 27487548, 27554331, '-', 'protein_coding');

-- Insert sample illnesses
INSERT INTO dim.dim_illness (illness_key, illness_code, description, category) VALUES
(1, 'BRCA', 'Breast Cancer', 'cancer'),
(2, 'LUAD', 'Lung Adenocarcinoma', 'cancer'),
(3, 'COAD', 'Colon Adenocarcinoma', 'cancer'),
(4, 'OV', 'Ovarian Cancer', 'cancer'),
(5, 'GBM', 'Glioblastoma Multiforme', 'cancer');

-- Insert sample samples
INSERT INTO dim.dim_sample (sample_key, sample_id, illness_key, patient_id, age, sex, tissue_type) VALUES
-- BRCA samples
(1, 'TCGA-BRCA-01', 1, 'PAT_001', 45, 'F', 'tumor'),
(2, 'TCGA-BRCA-02', 1, 'PAT_002', 52, 'F', 'tumor'),
(3, 'TCGA-BRCA-03', 1, 'PAT_003', 38, 'F', 'tumor'),
(4, 'TCGA-BRCA-04', 1, 'PAT_004', 61, 'F', 'tumor'),
(5, 'TCGA-BRCA-05', 1, 'PAT_005', 47, 'F', 'tumor'),
-- LUAD samples
(6, 'TCGA-LUAD-01', 2, 'PAT_006', 68, 'M', 'tumor'),
(7, 'TCGA-LUAD-02', 2, 'PAT_007', 55, 'F', 'tumor'),
(8, 'TCGA-LUAD-03', 2, 'PAT_008', 72, 'M', 'tumor'),
(9, 'TCGA-LUAD-04', 2, 'PAT_009', 59, 'F', 'tumor'),
(10, 'TCGA-LUAD-05', 2, 'PAT_010', 64, 'M', 'tumor'),
-- COAD samples
(11, 'TCGA-COAD-01', 3, 'PAT_011', 65, 'M', 'tumor'),
(12, 'TCGA-COAD-02', 3, 'PAT_012', 58, 'F', 'tumor'),
(13, 'TCGA-COAD-03', 3, 'PAT_013', 71, 'M', 'tumor'),
(14, 'TCGA-COAD-04', 3, 'PAT_014', 63, 'F', 'tumor'),
(15, 'TCGA-COAD-05', 3, 'PAT_015', 69, 'M', 'tumor');

-- Insert sample studies
INSERT INTO dim.dim_study (study_key, study_accession, platform, description) VALUES
(1, 'TCGA-BRCA-2012', 'Illumina HiSeq', 'TCGA Breast Cancer Study'),
(2, 'TCGA-LUAD-2012', 'Illumina HiSeq', 'TCGA Lung Adenocarcinoma Study'),
(3, 'TCGA-COAD-2012', 'Illumina HiSeq', 'TCGA Colon Adenocarcinoma Study');

-- Update samples with study information
UPDATE dim.dim_sample SET study_key = 1 WHERE illness_key = 1;
UPDATE dim.dim_sample SET study_key = 2 WHERE illness_key = 2;
UPDATE dim.dim_sample SET study_key = 3 WHERE illness_key = 3;