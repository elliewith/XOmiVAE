
import pandas as pd
from Refactoring.dimensionKnockOutExperiments import nullingDimensions
from Refactoring.omiShapExplainer import omiShapExplainer
import numpy as np
import torch


if __name__ == "__main__":
    expr_path = 'data/GDC-PANCAN_htseq_fpkm_'
    input_path = 'DataSources/GDC-PANCAN_'

    print('Loading data...')
    expr_df = pd.read_pickle("/Users/ewithnell/PycharmProjects/OmiVAESecondAttempt/data/expr_df.pkl")
    sample_id = np.loadtxt(input_path + 'both_samples.tsv', delimiter='\t', dtype='str')
    label = pd.read_csv(input_path + 'both_samples_tumour_type_digit.tsv', sep='\t', header=0, index_col=0)
    label_array = label['tumour_type'].to_numpy()


    #explain interim layer. Need to pass in tumour name.
    omiShapExplainer(sample_id, label_array, expr_df, tumourName="TCGA-HNSC", NormalvsTumourInterimExplain=True)
    """
    #Example of knocking out dimension. Pass in dimension number and chosen tumour tissue to evaluate.
    nullingDimensions(sample_id=sample_id,expr_df=expr_df,diseaseCode=17, chosenTissue="TCGA-LUAD", dimension=42)

    # example of explaining the most important genes for a tissue
    omiShapExplainer(sample_id, label_array, expr_df, NormalvsTumourExplain=True, tumourName="TCGA-BRCA")
  

    # explain the most important dimension in the supervised part of the model. Pass in dimension number and tumour name.
    omiShapExplainer(sample_id, label_array, expr_df, tumourName="TCGA-HNSC", dimension=42,
                     NormalvsTumourDimensionExplain=True)
    """
