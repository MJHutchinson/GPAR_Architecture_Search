import pickle

import pandas as pd

df_hyp_search = pickle.load(open('/home/mjhutchinson/Documents/MachineLearning/architecture_search_gpar/output/fits/weight_pruning_hyperprior3/hypersetting_comparison/results.pkl', 'rb'))

df_hyp_search.columns = ['Dataset',
              'RMSE fit',
              'Final output only',
              'Joint trained',
              'Validation set small',
              'Input scales tied',
              'Linear kernal on inputs',
              'Linear kernal on outputs',
              'Non-linear kernal on outputs',
              'Markov length',
              'Validation loglikelihood',
              'Validation RMSE',
              'Train loglikelihood',
              'Train RMSE',
              'Wall time']

index = ['Markov length', 'Input scales tied', 'Joint trained']
features = ['Validation likelihood', 'Validation RMSE', 'Train likelihood', 'Train RMSE', 'Wall time']

# df.set_index(index, inplace=True)
df_full_data_hypsearch = df_hyp_search[df_hyp_search['Validation set small'] == False]

df_full_data_hypsearch = df_full_data_hypsearch.fillna(value='None')

df_full_data_hypsearch = df_full_data_hypsearch.pivot_table(values=['Validation loglikelihood'], index=index, columns=['Dataset'], dropna=False)

tex_full_data_hypsearch = df_full_data_hypsearch.to_latex(float_format='%6.3f',
                             na_rep='None',
                             multirow=True)

print('\n\n********* Full data hypseach *********\n\n')

print(tex_full_data_hypsearch)

print('\n\n********* Full data hypseach *********\n\n')



df_low_data_hypsearch = df_hyp_search[df_hyp_search['Validation set small'] == True]

df_low_data_hypsearch = df_low_data_hypsearch.fillna(value='None')

df_low_data_hypsearch = df_low_data_hypsearch.pivot_table(values=['Validation loglikelihood'], index=index, columns=['Dataset'], dropna=False)

tex_low_data_hypsearch = df_low_data_hypsearch.to_latex(float_format='%6.3f',
                             na_rep='None',
                             multirow=True)

print('\n\n********* Low data hypseach *********\n\n')

print(tex_low_data_hypsearch)

print('\n\n********* Low data hypseach *********\n\n')



df_kernal_comp = pickle.load(open('/home/mjhutchinson/Documents/MachineLearning/architecture_search_gpar/output/fits/weight_pruning_hyperprior3/kernal_comparison/results.pkl', 'rb'))

df_kernal_comp.columns = ['Dataset',
              'RMSE fit',
              'Final output only',
              'Joint trained',
              'Validation set small',
              'Input scales tied',
              'Linear kernal on inputs',
              'Linear kernal on outputs',
              'Non-linear kernal on outputs',
              'Markov length',
              'Validation loglikelihood',
              'Validation RMSE',
              'Train loglikelihood',
              'Train RMSE',
              'Wall time']

index = ['Linear kernal on inputs', 'Linear kernal on outputs', 'Non-linear kernal on outputs']
features = ['Validation likelihood', 'Validation RMSE', 'Train likelihood', 'Train RMSE', 'Wall time']

# df.set_index(index, inplace=True)
df_full_data_kernal_comp = df_kernal_comp[df_kernal_comp['Validation set small'] == False]

df_full_data_kernal_comp = df_full_data_kernal_comp.fillna(value='None')

df_full_data_kernal_comp = df_full_data_kernal_comp.pivot_table(values=['Validation loglikelihood'], index=index, columns=['Dataset'], dropna=False, )

tex_full_data_kernal_comp = df_full_data_kernal_comp.to_latex(float_format='%6.3f',
                             na_rep='None',
                             multirow=True)

print('\n\n********* Full data kernal comp *********\n\n')

print(tex_full_data_kernal_comp)

print('\n\n********* Full data kernal comp *********\n\n')


# df.set_index(index, inplace=True)
df_low_data_kernal_comp = df_kernal_comp[df_kernal_comp['Validation set small'] == True]

df_low_data_kernal_comp = df_low_data_kernal_comp.fillna(value='None')

df_low_data_kernal_comp = df_low_data_kernal_comp.pivot_table(values=['Validation loglikelihood'], index=index, columns=['Dataset'], dropna=False)

tex_low_data_kernal_comp = df_low_data_kernal_comp.to_latex(float_format='%6.3f',
                             na_rep='None',
                             multirow=True)

print('\n\n********* Low data kernal comp *********\n\n')

print(tex_low_data_kernal_comp)

print('\n\n********* Low data kernal comp *********\n\n')