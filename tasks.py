import pandas as pd
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind as ttest


EXPRESSION_URL = 'https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/hnsc_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt'
PHENOTYPE_URL = 'https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/hnsc_tcga_pan_can_atlas_2018/data_clinical_patient.txt'
GENOTYPE_URL = 'https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/hnsc_tcga_pan_can_atlas_2018/data_mutations.txt'
MUTATED_GENES = Path('Mutated_Genes.txt')


def download_rawrnaseq(product):
    rawrnaseq = pd.read_table(EXPRESSION_URL)
    Path(str(product)).parent.mkdir(exist_ok=True, parents=True)
    rawrnaseq.to_parquet(str(product))

def download_clinical(product):
    clinical = pd.read_table(PHENOTYPE_URL)
    clinical.to_parquet(str(product))

def download_mutations(product):
    mutations = pd.read_table(GENOTYPE_URL)
    mutations.to_parquet(str(product))


def generate_traits(upstream, product):
    clinical = pd.read_parquet(str(upstream['download_clinical']))
    clinical = clinical.drop([0, 1, 2, 3])
    clinical = clinical.set_index('#Patient Identifier')
    traits = clinical[['Subtype']].copy()
    traits.index.name = None
    traits.columns = ['hpv']
    traits = traits.dropna()
    traits.hpv = traits.hpv.replace({'HNSC_HPV-': 0, 'HNSC_HPV+': 1})
    traits.to_parquet(str(product))



def filter_rnaseq(upstream, product):

    rnaseq = pd.read_parquet(str(upstream['download_rawrnaseq']))
    traits = pd.read_parquet(str(upstream['generate_traits']))
    rnaseq.index = rnaseq['Hugo_Symbol']
    rnaseq = rnaseq.drop(['Hugo_Symbol', 'Entrez_Gene_Id'], axis=1)
    rnaseq.columns = rnaseq.columns.str[:-3]
    rnaseq = rnaseq[rnaseq.index.notnull()].dropna().drop_duplicates()
    rnaseq = rnaseq[~rnaseq.index.duplicated(keep='first')]
    rnaseq = rnaseq.loc[:,~rnaseq.columns.duplicated()]
    rnaseq.index.name = None

    rnaseq = rnaseq[rnaseq.mean(1) > 100]
    rnaseq = np.log2(rnaseq + 1)
    rnaseq = rnaseq[~(rnaseq.var(1) < 0.1)]

    def _compare(gene, df=rnaseq):
        hpv_p = df.loc[gene][traits[traits.hpv == 1].index]
        hpv_n = df.loc[gene][traits[traits.hpv == 0].index]
        pvalue = ttest(hpv_p, hpv_n).pvalue

        return pvalue

    pvals = [_compare(gene) for gene in rnaseq.index]

    top_diff_expr_genes = 5000
    rnaseq = rnaseq.loc[pd.DataFrame(pvals, index=rnaseq.index).sort_values(by=0)[:top_diff_expr_genes].index]
    rnaseq.to_parquet(str(product))


def create_mutation_matrix(upstream, product):

    mutations = pd.read_parquet(str(upstream['download_mutations']))
    mutations = mutations[(mutations['Variant_Classification'] != 'Silent')]
    mutations = mutations[(mutations['IMPACT'] != 'LOW')]
    mutations['Barcode'] = mutations['Tumor_Sample_Barcode'].str[:-3]

    mutated_genes = pd.read_csv(MUTATED_GENES, delimiter='\t')
    mutated_genes['Freq'] = mutated_genes['Freq'].str[:-1].astype(float)

    traits = pd.read_parquet(str(upstream['generate_traits']))
    mutation_matrix = pd.DataFrame(columns=set(mutations['Hugo_Symbol']), index = traits.index).fillna(0)
    for patient_id in traits.index:
        for m in set(mutations[mutations.Barcode == patient_id]['Hugo_Symbol']):
            mutation_matrix.loc[patient_id, m] = 1

    mutation_matrix = (mutation_matrix[
            set(mutated_genes
                    .query('Freq > 5')
                    .sort_values(by='Freq', ascending=False).Gene) & 
            set(mutations['Hugo_Symbol'])])

    mutation_matrix.to_parquet(product)


def shape_inputs(upstream, product):
    traits = pd.read_parquet(str(upstream['generate_traits']))
    mutation_matrix = pd.read_parquet(str(upstream['create_mutation_matrix']))
    rnaseq = pd.read_parquet(str(upstream['filter_rnaseq']))

    common_samples = set(rnaseq.columns) & set(mutation_matrix.index) & set(traits.index)
    rnaseq = rnaseq[common_samples].astype(float)
    mutation_matrix = mutation_matrix.loc[common_samples].astype(int)
    traits = traits.loc[common_samples].astype(int)
    rnaseq = rnaseq.subtract(rnaseq.mean(1), 0).div(rnaseq.std(1), 0)

    Z = traits.to_numpy()
    Y = rnaseq.T.to_numpy()
    X = mutation_matrix.to_numpy()

    r = Z.shape[1]
    n = X.shape[0]
    q = Y.shape[1]
    p = X.shape[1]
    assert X.shape[0]==Z.shape[0]
    assert Y.shape[0]<=n

    np.savetxt(product['traits'], Z, delimiter='\t', fmt='%s')
    np.savetxt(product['expression'], Y, delimiter='\t', fmt='%s')
    np.savetxt(product['genotype'], X, delimiter='\t', fmt='%s')

    traits.to_csv(product['traits_csv'])
    mutation_matrix.to_csv(product['mutations_csv'])
    rnaseq.to_csv(product['rnaseq_csv'])