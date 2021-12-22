addpath('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/Fast-sCGGM')
addpath('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/MATLAB')
rng(0);

% X = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/Viral/MESO/genotype.txt');
% Y = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/Viral/MESO/expression.txt');
% Z = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/Viral/MESO/traits.txt');


X = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/TCGA/HNSC/genotype.txt');
Y = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/TCGA/HNSC/expression.txt');
Z = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/TCGA/HNSC/traits.txt');

options.max_outer_iters = 20;
numSuggestions = 20;
[Lxy, Txy] = regularization_suggestions(Y, X, 'numSuggestions', numSuggestions);
[Lyz, Tyz] = regularization_suggestions(Y, Z, 'numSuggestions', numSuggestions);
