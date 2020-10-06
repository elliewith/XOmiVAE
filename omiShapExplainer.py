import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from exprVAEwithAdditionalFeatures import ExprOmiVAE
import generalHelperFunctions as GeneralHelper
import shapExplainerHelper as ShapHelper
from dimensionKnockOutExperiments import nullingDimensions

import sys
sys.path.insert(0, '~/PycharmProjects/shapLundberg')
import shapLundberg
from shapLundberg import shap


def omiShapExplainer(sample_id,label_array,expr_df,dimension=0,tumourID=0,device='cpu',LatentSpaceExplain=False, statisticsTest=False, NormalvsTumourExplain=False,
               NormalvsTumourInterimExplain=False, TestingNullingDimensions=False,
               saveLatentSpace=False, statisticsTestEvenGender=False,histogram=False,NormalvsTumourDimensionExplain=False, tumourName="TCGA-KIRC",):
    """
    :param tumourName: String version of TCGA tumour e.g. "TCGA-KIRC" we would like to explain
    :param NormalvsTumourExplain: Explain predictions (from supervised OmiVAE). Pass in tumourName to explain.
    :param NormalvsTumourInterimExplain: Explain most important dimensions (from supervised OmiVAE). Pass in tumourName to explain.
    :param NormalvsTumourDimensionExplain: Explain a specified dimension (either after supervised or unsupervised training). Pass in dimension and tumourName.
    :param statisticsTest: Find the most important dimension in the unsupervised part of the model. Adjust within the code for which latent space (z) file to load in and which factrs we are investigating.
    :param LatentSpaceExplain: Explain the mean latent variables. Need to specifiy dimension. Dimension found by running statisticsTest. Adjust within section
    :param tumourID: TCGA tumour code (see tumour type digit file)  that we want to explain prediction. Required for testing nulling dimensions.
    :return:
    """
    #This class has combined the different analysis' of the Deep SHAP values we conducted.
    #SHAP reference: Lundberg et al., 2017: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf


    input_path = 'DataSources/GDC-PANCAN_'
    vae_model = ExprOmiVAE(input_path, expr_df)

    #load vae trained model
    vae_model.load_state_dict(torch.load('DataSources/vae_saved_model(original).pt',map_location=torch.device('cpu')))

    #Get the most statistically significant dimensions: use for explaining the UNSUPERVISED model (just VAE)
    if NormalvsTumourExplain:
        """
            Explain the most important genes for a specific tumour, when using the normal tissue as a background
            Note: adjusting to use a random training sample as the background is simple, use the 
            ShapHelper.randomTrainingSample(expr) function.
            """
        phenotype = GeneralHelper.processPhenotypeDataForSamples(sample_id)
        conditionone=phenotype['sample_type'] == "Primary Tumor"
        conditiontwo=phenotype['project_id'] == tumourName
        conditionthree=phenotype['sample_type'] == "Solid Tissue Normal"

        conditionaltumour=np.logical_and(conditionone,conditiontwo)
        conditionalnormal = np.logical_and(conditiontwo, conditionthree)
        #recommended to have 100 samples, but in some cases it is not possible to sample 100 for a tissue type, especially for normal tissue
        normal_expr = ShapHelper.splitExprandSample(condition=conditionalnormal, sampleSize=2, expr=expr_df)
        tumour_expr = ShapHelper.splitExprandSample(condition=conditionaltumour, sampleSize=2, expr=expr_df)
        # put on device as correct datatype
        background = GeneralHelper.addToTensor(expr_selection=normal_expr, device=device)
        male_expr_tensor = GeneralHelper.addToTensor(expr_selection=tumour_expr, device=device)
        GeneralHelper.checkPredictions(background, vae_model)

        #output number is based from OmiVAE forward function. 4= predicted y value.
        e = shap.DeepExplainer(vae_model, background, outputNumber=4)
        shap_values_female = e.shap_values(male_expr_tensor, ranked_outputs=1)
        most_imp, least_imp = ShapHelper.getTopShapValues(shap_vals=shap_values_female, numberOfTopFeatures=50,
                                                          expr_df=expr_df, ranked_output=True)

    if NormalvsTumourDimensionExplain:
        """
            Explain the most important genes for a specific dimension for a tumour.            
            Note: adjusting to use a random training sample as the background is simple, use the 
            ShapHelper.randomTrainingSample(expr) function.
            Here we are explaining the mean output (outputNumber=2).
            """
        phenotype = GeneralHelper.processPhenotypeDataForSamples(sample_id)
        conditionone=phenotype['sample_type'] == "Primary Tumor"
        conditiontwo=phenotype['project_id'] == tumourName
        conditionthree=phenotype['sample_type'] == "Solid Tissue Normal"

        conditionaltumour=np.logical_and(conditionone,conditiontwo)
        conditionalnormal = np.logical_and(conditiontwo, conditionthree)
        #recommended to have 100 samples, but in some cases it is not possible to sample 100 for a tissue type, especially for normal tissue
        normal_expr = ShapHelper.splitExprandSample(condition=conditionalnormal, sampleSize=2, expr=expr_df)
        tumour_expr = ShapHelper.splitExprandSample(condition=conditionaltumour, sampleSize=2, expr=expr_df)
        # put on device as correct datatype
        background = GeneralHelper.addToTensor(expr_selection=normal_expr, device=device)
        male_expr_tensor = GeneralHelper.addToTensor(expr_selection=tumour_expr, device=device)
        GeneralHelper.checkPredictions(background, vae_model)

        #output number 2 is the mean (latent space)
        #I edited the SHAP library to allow a dimension to be passed in and edited the output value. important that explainLatentSpace=True.
        e = shap.DeepExplainer(vae_model, background, dim=dimension, outputNumber=2, explainLatentSpace=True)
        #e = shap.DeepExplainer(vae_model, background)
        #as we are explaining the genes for a dimension (unlike previously where we had to choose which prediction we were explaining) we do not need to specifiy to rank the outputs
        shap_values_female = e.shap_values(male_expr_tensor)
        most_imp, least_imp = ShapHelper.getTopShapValues(shap_vals=shap_values_female, numberOfTopFeatures=50,
                                                          expr_df=expr_df, ranked_output=False)

    #Note: this can be easily adjusted to explain any two samples; simply change 'normal_expr' and 'tumour_expr' to the chosen samples
    if NormalvsTumourInterimExplain:
        """
        Explain the interim layer; pass in the interim layer and then it explains the top dimensions.
        """
        phenotype = GeneralHelper.processPhenotypeDataForSamples(sample_id)
        conditionone = phenotype['sample_type'] == "Primary Tumor"
        conditiontwo = phenotype['project_id'] == tumourName
        conditionthree = phenotype['sample_type'] == "Solid Tissue Normal"
        conditionaltumour = np.logical_and(conditionone, conditiontwo)
        conditionalnormal = np.logical_and(conditionthree, conditiontwo)
        # Recommended to have 100 samples, but in some cases it is not possible to sample 100 for a tissue type, especially for normal tissue
        normal_expr = ShapHelper.splitExprandSample(condition=conditionalnormal, sampleSize=2, expr=expr_df)
        tumour_expr = ShapHelper.splitExprandSample(condition=conditionaltumour, sampleSize=2, expr=expr_df)
        # Ensure tensor is on device and has the correct data type
        background = GeneralHelper.addToTensor(expr_selection=normal_expr, device=device)
        male_expr_tensor = GeneralHelper.addToTensor(expr_selection=tumour_expr, device=device)

        GeneralHelper.checkPredictions(background, vae_model)
        GeneralHelper.checkPredictions(male_expr_tensor, vae_model)
        #output number is based from OmiVAE forward function. 4= predicted y value
        e = shap.DeepExplainer((vae_model, vae_model.c_fc1), background, outputNumber=4)

        #explains layer before mean (512 dimensions)
        #e = shap.DeepExplainer((vae_model, vae_model.e_fc4_mean), background)
        shap_values_female = e.shap_values(male_expr_tensor, ranked_outputs=1)
        # Here look at the numbers to left (they should range 1 to no. of dimensions) and this is the most important dimension
        most_imp, least_imp = ShapHelper.getTopShapValues(shap_vals=shap_values_female, numberOfTopFeatures=50,
                                                          expr_df=expr_df, ranked_output=True)


    if LatentSpaceExplain:
        """
            This requires a known dimension to explain (found using the statistical significance above),
            This is for gender, subtype and pathway latent space explanations for the unsupervised part of the model.
            """
        #Example of how to obtain the relevant samples for the subtypes we would like to analyse (source: https://www.sciencedirect.com/science/article/pii/S0092867418303593)
        subtypeConditionalOne, subtypeConditionalTwo = GeneralHelper.processSubtypeSamples(sample_id,subtypeOne="Basal",subtypeTwo="LumB")

        #Example of how to obtain the relevant samples for the pathways we would like to analyse (source: https://www.sciencedirect.com/science/article/pii/S0092867418303593)
        logicalOne, logicalTwo = ShapHelper.pathwayComparison(sample_id=sample_id,pathway='RTK RAS')

        #Example of how to obtain the relevant samples for the different genders to explain the latent space
        femaleCondition, maleCondition=ShapHelper.splitForGenders(sample_id=sample_id,)
        ShapHelper.printConditionalSelection(maleCondition)

        #Change condition to one of the relevant conditions above
        female_expr = ShapHelper.splitExprandSample(condition=femaleCondition,sampleSize=50,expr=expr_df)
        male_expr = ShapHelper.splitExprandSample(condition=maleCondition, sampleSize=50, expr=expr_df)

        #put on device as correct datatype
        background= GeneralHelper.addToTensor(expr_selection=female_expr,device=device)
        male_expr_tensor = GeneralHelper.addToTensor(expr_selection=male_expr, device=device)
        GeneralHelper.checkPredictions(background, vae_model)
        e = shap.DeepExplainer(vae_model, background,outputNumber=2,dim=dimension)
        # If explaining the z/mu dimension then don't need ranked outputs like we used before (as only one output from the model)
        shap_values_female = e.shap_values(male_expr_tensor)
        most_imp, least_imp=ShapHelper.getTopShapValues(shap_vals=shap_values_female, numberOfTopFeatures=50, expr_df=expr_df, ranked_output=False)

    if saveLatentSpace:
        expr_tensor=GeneralHelper.addToTensor(expr_df,device)
        vae_model.load_state_dict(torch.load('data/beta15.pt', map_location=torch.device('cpu')))
        GeneralHelper.saveLatentSpace(vae_model,expr_tensor)

    if statisticsTest:
        # z latent space; explain beta-vae and normal vae latent space (can also feed in the mean here)
        z = np.genfromtxt('data/z_before_supervised_loss32_ForBeta10AnnealingKlLoss8.csv')
        #Split for gender here but can also split for subtype if we want to find the most important dimension for a subtype
        femaleCondition, maleCondition = ShapHelper.splitForGenders(sample_id=sample_id, )

        # Example for adapting this code to measure the statistically significant dimensions in a subtype
        # subtypeConditionalOne, subtypeConditionalTwo = GeneralHelper.processSubtypeSamples(sample_id, subtypeOne="Basal",
        #                                                                                        subtypeTwo="LumB")

        female_z = z[femaleCondition]
        male_z = z[maleCondition]
        statistics = stats.ttest_ind(female_z, male_z, axis=0, equal_var=False, nan_policy='propagate')
        stat_lowest_index = np.argmin(statistics.pvalue)
        female_z_column = female_z[:, stat_lowest_index]
        male_z_column = male_z[:, stat_lowest_index]
        np.savetxt('data/brca_z.csv', female_z_column)
        np.savetxt('data/normal_z.csv', male_z_column)

    # Method to sample the same amount to ensure gender split is even
    if statisticsTestEvenGender:
        # Get the z vector we would like to explain
        z = np.genfromtxt('data/z_before_supervised_loss32_ForBeta10AnnealingKlLoss8.csv')
        phenotype = GeneralHelper.processPhenotypeDataForSamples(sample_id)
        # tumour id's were chosen as they were typically non-gender specific cancers
        femaleCondition, maleCondition = ShapHelper.multipleSampling(label_array=label_array, phenotype=phenotype,
                                                                     tumour_id_one=13,
                                                                     tumour_id_two=18, tumour_id_three=10,
                                                                     tumour_id_four=5, tumour_id_five=6,
                                                                     sample_number=50)
        ShapHelper.saveMostStatisticallySignificantIndex(conditionOne=femaleCondition, conditionTwo=maleCondition,
                                                     fileNameOne="female_z", fileNameTwo="male_z",z=z)

    if histogram:
        feature_importance = pd.read_csv('data/luaddimensions.csv', header=0, index_col=0)
        plt.style.use('seaborn-white')
        hist = feature_importance.hist(column='feature_importance_vals',normed=False, alpha=0.5,
         histtype='stepfilled', color='skyblue',
         edgecolor='none')
        plt.title('SHAP value histogram')
        plt.xlabel('SHAP_value')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.savefig("SHAPvalueLoghistogram.png", dpi=1500)
        plt.show()

    if TestingNullingDimensions:
        #Example of testing four different dimensions
        print("dim 83")
        nullingDimensions(sample_id=sample_id,expr_df=expr_df,diseaseCode=tumourID, chosenTissue=tumourName, dimension=83)

        print("dim 125")
        nullingDimensions(sample_id=sample_id,expr_df=expr_df,diseaseCode=tumourID, chosenTissue=tumourName, dimension=125)

        print(" dim 75")
        nullingDimensions(sample_id=sample_id,expr_df=expr_df,diseaseCode=tumourID, chosenTissue=tumourName, dimension=75)
        print("dim 85")
        nullingDimensions(sample_id=sample_id,expr_df=expr_df,diseaseCode=tumourID, chosenTissue=tumourName, dimension=85)
