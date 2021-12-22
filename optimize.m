addpath('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/Fast-sCGGM')
addpath('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/MATLAB')
rng(0);

X = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/TCGA/HNSC/genotype.txt');
Y = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/TCGA/HNSC/expression.txt');
Z = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/TCGA/HNSC/traits.txt');

% X = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/Viral/MESO/genotype.txt');
% Y = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/Viral/MESO/expression.txt');
% Z = dlmread('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/Viral/MESO/traits.txt');

% % Warm-starting example
% %figure('name', 'Warm-starting demo');

options.max_outer_iters = 20;
numSuggestions = 25;
[lambdaLambdas, lambdaThetas] = regularization_suggestions(Y, X, 'numSuggestions', numSuggestions);
writematrix(lambdaLambdas, '/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/suggested_lambdas_yx.txt','Delimiter','\t');
writematrix(lambdaThetas, '/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/suggested_thetas_yx.txt','Delimiter','\t');
[estLambda, estTheta, stats] = fast_scggm(Y, X, lambdaLambdas(1), lambdaThetas(1));
bic_yx = [];
aic_yx = [];

for reg_ix=1:numSuggestions
    lambdaLambda = lambdaLambdas(reg_ix);
    lambdaTheta = lambdaThetas(reg_ix);
    options.Lambda0 = estLambda;
    options.Theta0 = estTheta;
    [estLambda, estTheta] = fast_scggm(Y, X, lambdaLambda, lambdaTheta, options);
    bic = BIC(Y, X, estLambda, estTheta);
    aic = AIC(Y, X, estLambda, estTheta);
    bic_yx = [bic_yx bic];
    aic_yx = [aic_yx aic];
    
    fprintf('XY_%d | NNZ: %d; BIC: %d; AIC: %d; L: %d; T: %d\n', reg_ix, nnz(estLambda), bic, aic, lambdaLambda, lambdaTheta);
    save(sprintf('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/xy_estLambda_%d_%d_%d_%d.mat', lambdaLambda, lambdaTheta, bic, aic), 'estLambda')
    save(sprintf('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/xy_estTheta_%d_%d_%d_%d.mat', lambdaLambda, lambdaTheta, bic, aic), 'estTheta')
    

    % subplot(numSuggestions, 2, (reg_ix-1)*2+1); spy(estLambda);
    % subplot(numSuggestions, 2, (reg_ix-1)*2+2); spy(estTheta);
end

writematrix(bic_yx, '/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/bic_yx.txt','Delimiter','\t');
writematrix(aic_yx, '/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/aic_yx.txt','Delimiter','\t');


clear options;
disp('**********************************************')

% Warm-starting example
%figure('name', 'Warm-starting demo');
options.max_outer_iters = 20;
numSuggestions = 25;
[lambdaLambdas, lambdaThetas] = regularization_suggestions(Y, Z, 'numSuggestions', numSuggestions);
writematrix(lambdaLambdas, '/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/suggested_lambdas_yz.txt','Delimiter','\t');
writematrix(lambdaThetas, '/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/suggested_thetas_yz.txt','Delimiter','\t');
[estLambda, estTheta, stats] = fast_scggm(Y, Z, lambdaLambdas(1), lambdaThetas(1));
bic_yz = [];
aic_yz = [];

for reg_ix=1:numSuggestions
    lambdaLambda = lambdaLambdas(reg_ix);
    lambdaTheta = lambdaThetas(reg_ix);
    
    options.Lambda0 = estLambda;
    options.Theta0 = estTheta;
    [estLambda, estTheta] = fast_scggm(Y, Z, lambdaLambda, lambdaTheta, options);
    bic = BIC(Y, Z, estLambda, estTheta);
    aic = AIC(Y, Z, estLambda, estTheta);
    
    bic_yz = [bic_yz bic];
    aic_yz = [aic_yz aic];
    
    fprintf('YZ_%d | NNZ: %d; BIC: %d; AIC: %d; L: %d; T: %d\n', reg_ix, nnz(estLambda), bic, aic, lambdaLambda, lambdaTheta);

    save(sprintf('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/yz_estLambda_%d_%d_%d_%d.mat', lambdaLambda, lambdaTheta, bic, aic), 'estLambda')
    save(sprintf('/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/yz_estTheta_%d_%d_%d_%d.mat', lambdaLambda, lambdaTheta, bic, aic), 'estTheta')
    
    %subplot(numSuggestions, 2, (reg_ix-1)*2+1); spy(estLambda);
    %subplot(numSuggestions, 2, (reg_ix-1)*2+2); spy(estTheta);
end


writematrix(bic_yz, '/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/bic_yz.txt','Delimiter','\t')
writematrix(aic_yz, '/ihome/hosmanbeyoglu/kor11/tools/PerturbNet/scripts/BIC/aic_yz.txt','Delimiter','\t')



