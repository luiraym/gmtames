function res = ad_model(Xtrain,Xtest,options,whattodo)

% ad_model is the function for calculating Appicability Domain (AD) of Xtrain and verify if samples of Xtest are inside or outside the AD
%
% res = ad_model(Xtrain,Xtest,options,whattodo)
% 
% INPUT:
% Xtrain                    training set matrix [n samples x p variables]
% Xtest                     test set matrix [nt samples x p variables]
% options                   is a structure containing the options for calculating AD:
% options.pret_type         data pretreatment:
%                           'cent' for centering
%                           'auto' for autoscaling
%                           'rang' for range scaling between 0 and 1
%                           'none' for no data scaling
% options.distance          distance metric ('euclidean', 'manhattan', or 'mahalanobis')
% options.lev_thr           factor to be multiplied with the leverage average in order to define the leverage threshold (suggested: 2.5)
% options.knnfix_k          k value to be used for the k Nearest Neighbours (kNN) approach with fixed k
% options.dist_pct          percetile to be used to define the threshold for distance to centroid approach (suggested: 95);
% options.knnvar_k_max      maximum number of neighbours considered during the k-optimization procedure 
%                           for the k Nearest Neighbours (kNN) approach with variable k(suggested: 25)
% options.knnvar_k_opt      optimal k value for the k Nearest Neighbours (kNN) approach with variable k
%                           if set to 'auto' (string): automatic selection of the best k value on the basis of montecarlo procedure
%                           if set to 'user' (string): user selection of the best k value on the basis of montecarlo procedure
% options.knnvar_iter       number of iterations of the montecarlo procedure for the selection of the best k value,
%                           for the k Nearest Neighbours (kNN) approach with variable k
% options.knnvar_perc_test  percentage of samples to be used in the validation set (20 to have 20% in test set) of the montecarlo procedure 
%                           for the selection of the best k value (kNN approach with variable k)
% options.pf_smoot          smoothing coefficient to be used for Potential Functions. Smoothing value shlould be similar to variable standard deviations.
%                           if scalar, the coefficient is used to smooth the Potential Functions, if 2-dim vector, [a b], the coefficient is optimised 
%                           between a and b in leave-one-out
% options.pf_kernel         kernel of Potential Functions ('gaus' for gaussian or 'tria' for triangular)
% options.pf_pct            percetile to be used to define the threshold for Potential Functions (suggested: 95);
%
% OPTIONAL INPUT
% whattodo is a structure defining which AD approaches must be calculated (if set to 1) or not (if set to 0):
% whattodo.bounding_box = 1;        bounding box
% whattodo.bounding_box_pca = 1;    bounding box on PCs
% whattodo.convex_hull = 1;         convex hull
% whattodo.leverage = 1;            leverage
% whattodo.dist_centroid = 1;       distance to centroid
% whattodo.dist_knn_fix = 1;        k Nearest Neighbours (kNN) approach with fixed k
% whattodo.dist_knn_var = 1;        k Nearest Neighbours (kNN) approach with variable k
% whattodo.pot_fun = 1;             Potential Functions
%
% OUTPUT
% res is a structure containing the following fields:
% res.scaling                       resume of scaling parameters (average, standard deviation, min and max)
% res.bounding_box.inad             numerical vector defining which test samples are inside (1) or outside (0) the AD defined by means of Bounding Box
% res.bounding_box.min              minimum values of variables for the training set
% res.bounding_box.max              maximum values of variables for the training set
% res.bounding_box_pca.inad         numerical vector defining which test samples are inside (1) or outside (0) the AD defined by means of PCA Bounding Box
% res.bounding_box_pca.min          minimum values of PC scores for the training set
% res.bounding_box_pca.max          maximum values of PC scores for the training set
% res.bounding_box_pca.model_pca    PCA model
% res.convex_hull.inad              numerical vector defining which test samples are inside (1) or outside (0) the AD defined by means of Convex hull
% res.convex_hull.ch                convex hull calculated on the training samples
% res.leverage.inad                 numerical vector defining which test samples are inside (1) or outside (0) the AD defined by means of leverages
% res.leverage.thr                  leverage threshold
% res.leverage.Hcore                leverage core matrix
% res.leverage.h_test               leverages of test samples
% res.leverage.h_train              leverages of training samples
% res.dist_centroid.inad            numerical vector defining which test samples are inside (1) or outside (0) the AD defined by means of distance to centroid
% res.dist_centroid.thr             distance threshold
% res.dist_centroid.Dtest           distances of test samples from centroid
% res.dist_knn_fix.inad             numerical vector defining which test samples are inside (1) or outside (0) the AD defined by means of kNN with fixed k
% res.dist_knn_fix.thr              distance threshold
% res.dist_knn_fix.Dtest            distances of test samples from neighbours
% res.dist_knn_var.k_optimisation   results of k optimisation for the kNN approach with variable k
% res.dist_knn_var.train.thresholds threshold associated to each training sample
% res.dist_knn_var.train.ki         density of the training sample neighbourhoods
% res.dist_knn_var.test.kj          density of the test sample neighbourhoods
% res.dist_knn_var.inad             numerical vector defining which test samples are inside (1) or outside (0) the AD defined by means of kNN with variable k
% res.dist_knn_var.train.D          ranked training distance matrix (n samples x n samples - 1);
% res.dist_knn_var.train.Itrain     training samples(neighbours) associated to distances of res.dist_knn_var.train.D
% res.dist_knn_var.test.Dtest       ranked training distance matrix (nt samples x n samples);
% res.dist_knn_var.test.Itest       training samples(neighbours) associated to distances of res.dist_knn_var.test.Dtest
% res.dist_knn_var.k_opt            optimal k value used for kNN with variable k
% res.pot_fun.inad                  numerical vector defining which test samples are inside (1) or outside (0) the AD defined by means of Potential Functions
% res.pot_fun.thr                   Potential Functions threshold
% res.pot_fun.Stest                 potentials of test samples
% res.pot_fun.smoot                 smoothing coefficient used for Potential Functions
% res.resume_table                  table with the resume of the obtained results for each AD method (number of samples inside and outside the AD, list of samples outside the AD)
% res.options                       options used for the AD calculation
%
% The main function for the AD toolbox is ad_gui, which opens a GUI figure for calculating Applicability Domain;
% in order to open the graphical interface, just type on the matlab command line: ad_gui
% 
% Note that a detailed HTML help is provided with the toolbox.
% See the HTML HELP files (help.htm) for futher details and examples
%
% References for the AD methods:
% - Comparison of different approaches to define the Applicability Domain of QSAR models, Molecules (2012), 17, 4791-4810
% - Stepwise approach for defining the applicability domain of SAR and QSAR models, Journal of Chemical Information Modeling (2005) 45, 839–849.
% - Current status of methods for defining the applicability domain of (quantitative) structure-activity relationships. The report and recommendations of ECVAM Workshop 52. ATLA: Alternatives to Laboratory Animals (2005), 33, 155–173.
% - QSAR applicabilty domain estimation by projection of the training set descriptor space: A review. ATLA: Alternatives to Laboratory Animals (2005), 33, 445–459.
%
% The toolbox is freeware and may be used (but not modified) if proper reference is given to the authors. 
% Preferably refer to the following papers:
%
% F. Sahigara, D. Ballabio, R. Todeschini, V. Consonni
% Assessing the validity of QSARs for ready biodegradability of chemicals: an Applicability Domain perspective
% Current Computer-Aided Drug Design (2014), 10, 137-147
% 
% F. Sahigara, K. Mansouri, D. Ballabio, A. Mauri, V. Consonni, R. Todeschini
% Comparison of different approaches to define the Applicability Domain of QSAR models
% Molecules (2012), 17, 4791-4810
% 
% Applicabilit domain toolbox for MATLAB
% version 1.0 - january 2014
% Milano Chemometrics and QSAR Research Group
% http://michem.disat.unimib.it/chm/

if nargin < 4
    whattodo.bounding_box = 1;
    whattodo.bounding_box_pca = 1;
    whattodo.convex_hull = 1;
    whattodo.leverage = 1;
    whattodo.dist_centroid = 1;
    whattodo.dist_knn_fix = 1;
    whattodo.dist_knn_var = 1;
    whattodo.pot_fun = 1;
end

% Data scaling
[Xtrain_scal,param] = data_pretreatment(Xtrain,options.pret_type);
[Xtest_scal] = test_pretreatment(Xtest,param);

% Store info on variable parameters
res.scaling.param = param;
res.scaling.param_table = variable_parameters(Xtrain,param);

% Bounding box ------------------------------------------------------------
res.bounding_box.inad = [];
if whattodo.bounding_box
    disp('Processing results for Bounding Box')
    inad = ones(size(Xtest,1),1);
    M = param.M;
    m = param.m;
    for num = 1:size(Xtest,1)
        for j = 1:size(Xtest,2)
            if(Xtest(num,j) < m(j) | Xtest(num,j) > M(j))
                inad(num) = 0;
            end
        end
    end
    res.bounding_box.inad = inad;
    res.bounding_box.min = m;
    res.bounding_box.max = M;
end

% PCA Bounding box ------------------------------------------------------------
res.bounding_box_pca.inad = [];
if whattodo.bounding_box_pca
    disp('Processing results for PCA Bounding Box')
    inad = ones(size(Xtest,1),1);
    pchere = min([size(Xtrain_scal,1) size(Xtrain_scal,2)]);
    model_pca = pca_model(Xtrain_scal,pchere);
    model_pca = pca_project(Xtest_scal,model_pca);
    num_comp = length(find(model_pca.E > mean(model_pca.E)));
    T = model_pca.T(:,1:num_comp);
    Tpred = model_pca.Tpred(:,1:num_comp);
    M = max(T);
    m = min(T);
    for num = 1:size(Tpred,1)
        for j = 1:size(Tpred,2)
            if(Tpred(num,j) < m(j) | Tpred(num,j) > M(j))
                inad(num) = 0;
            end
        end
    end
    res.bounding_box_pca.inad = inad;
    res.bounding_box_pca.min = m;
    res.bounding_box_pca.max = M;
    res.bounding_box_pca.model_pca = model_pca;
end

% Convex hull -----------------------------------------------------------------
res.convex_hull.inad = [];
if whattodo.convex_hull
    disp('Processing results for Convex Hull')
    inad = ones(size(Xtest,1),1);
    ch = delaunayn(Xtrain);
    t = tsearchn(Xtrain,ch,Xtest);
    inad(isnan(t)) = 0;
    res.convex_hull.inad = inad;
    res.convex_hull.ch = ch;
end

% Leverage --------------------------------------------------------------------
res.leverage.inad = [];
if whattodo.leverage
    disp('Processing leverage-based approach')
    inad = ones(size(Xtest,1),1);
    [Xtrain_cent,param_cent] = data_pretreatment(Xtrain,'cent');
    Xtest_cent = test_pretreatment(Xtest,param_cent);
    Hcore = pinv(Xtrain_cent'*Xtrain_cent);
    Htrain = Xtrain_cent*Hcore*Xtrain_cent';
    Htest = Xtest_cent*Hcore*Xtest_cent';
    Htrain = diag(Htrain);
    Htest = diag(Htest);
    have = mean(Htrain);
    thr = options.lev_thr*(have+1/length(Htrain));
    inad(find(Htest > thr)) = 0;
    res.leverage.inad = inad;
    res.leverage.thr = thr;
    res.leverage.Hcore = Hcore;
    res.leverage.h_test = Htest;
    res.leverage.h_train = Htrain;
end

% Distance centroid -----------------------------------------------------------
res.dist_centroid.inad = [];
if whattodo.dist_centroid
    disp('Processing centroid distance approach')
    inad = ones(size(Xtest,1),1);
    Dtrain = pdist2(Xtrain_scal,mean(Xtrain_scal),options.distance);
    if strcmp(options.distance,'mahalanobis')
        Smat = nancov(Xtrain_scal);
        Dtest = pdist2(Xtest_scal,mean(Xtrain_scal),options.distance,Smat);
    else
        Dtest = pdist2(Xtest_scal,mean(Xtrain_scal),options.distance);
    end
    [UL,Qperc] = calc_thr_prctile(Dtrain,options.dist_pct);
    if options.dist_pct == 0;
        thr = UL;
    else
        thr = Qperc;
    end
    inad(find(Dtest > thr)) = 0;
    res.dist_centroid.inad = inad;
    res.dist_centroid.thr = thr;
    res.dist_centroid.Dtest = Dtest;
end

% Distance knn (k fixed)--------------------------------------------------------
res.dist_knn_fix.inad = [];
if whattodo.dist_knn_fix
    disp('Processing kNN approach - fixed threshold')
    inad = ones(size(Xtest,1),1);  
    Dtrain_sorted = pdist2(Xtrain_scal,Xtrain_scal,options.distance,'Smallest',options.knnfix_k + 1)';
    if strcmp(options.distance,'mahalanobis')
        Smat = nancov(Xtrain_scal);
        Dtest_sorted = pdist2(Xtrain_scal,Xtest_scal,options.distance,Smat,'Smallest',options.knnfix_k)';
    else
        Dtrain_sorted  = Dtrain_sorted(:,2:end);
        Dtest_sorted = pdist2(Xtrain_scal,Xtest_scal,options.distance,'Smallest',options.knnfix_k)';
    end
    Dtrain = mean(Dtrain_sorted,2);
    Dtest = mean(Dtest_sorted,2);
    [UL,Qperc] = calc_thr_prctile(Dtrain,options.dist_pct);
    if options.dist_pct == 0
        thr = UL;
    else
        thr = Qperc;
    end
    inad(find(Dtest > thr)) = 0;
    res.dist_knn_fix.inad = inad;
    res.dist_knn_fix.thr = thr;
    res.dist_knn_fix.Dtest = Dtest;
end

% Distance knn (k variable)--------------------------------------------------------
res.dist_knn_var.inad = [];
if whattodo.dist_knn_var
    disp('Processing kNN - variable threshold')
    inad = ones(size(Xtest,1),1);
    if isstr(options.knnvar_k_opt)
        % k optimisation with montecarlo
        sit = ceil(size(Xtrain,1)*((100-options.knnvar_perc_test)/100));
        for num=1:options.knnvar_iter
            if mod(num,10) == 0
                disp(['    > Processing iterations = ',num2str(num),' / ',num2str(options.knnvar_iter)]);
            end
            r = randperm(size(Xtrain,1));
            r = r(1:sit);
            in = zeros(size(Xtrain,1),1);
            in(r) = 1;
            Tr = Xtrain(in == 1,:);
            Te = Xtrain(in == 0,:);
            [Tr,ph] = data_pretreatment(Tr,options.pret_type);
            Te = test_pretreatment(Te,ph);
            nnt = size(Tr,1);
            [D,Itrain] = pdist2(Tr,Tr,options.distance,'Smallest',size(Tr,1));
            D = D'; Itrain = Itrain';
            D = D(:,2:end); Itrain = Itrain(:,2:end);
            if strcmp(options.distance,'mahalanobis')
                Smat = nancov(Tr);
                [Dtest,Itest] = pdist2(Tr,Te,options.distance,Smat,'Smallest',size(Tr,1));
                Dtest = Dtest'; Itest = Itest';
            else
                [Dtest,Itest] = pdist2(Tr,Te,options.distance,'Smallest',size(Tr,1));
                Dtest = Dtest'; Itest = Itest';
            end
            for k = 1:options.knnvar_k_max
                meanD = mean(D(:,1:k),2);
                [thr_train,kappa_train] = calc_thr(meanD,D);
                out = AD_inorout(Dtest,Itest,thr_train);
                k_optimisation(num,k) = (sum(out)/length(out))*100;
            end
            res.dist_knn_var.k_optimisation = k_optimisation;
        end
        if strcmp(options.knnvar_k_opt,'user')
            figure
            doboxplot(k_optimisation)
            k_opt = visualize_optk(k_optimisation);
        else
            [k_opt,res.dist_knn_var.opt_stat] = find_k_optimal(k_optimisation,options.knnvar_k_max);
        end
    else
        k_opt = options.knnvar_k_opt;
    end
    % ad calculation
    [D,Itrain] = pdist2(Xtrain_scal,Xtrain_scal,options.distance,'Smallest',size(Xtrain_scal,1));
    D = D';Itrain = Itrain'; D = D(:,2:end); Itrain = Itrain(:,2:end);
    if strcmp(options.distance,'mahalanobis')
        Smat = nancov(Xtrain_scal);
        [Dtest,Itest] = pdist2(Xtrain_scal,Xtest_scal,options.distance,Smat,'Smallest',size(Xtrain_scal,1));
        Dtest = Dtest'; Itest = Itest';
    else
        [Dtest,Itest] = pdist2(Xtrain_scal,Xtest_scal,options.distance,'Smallest',size(Xtrain_scal,1));
        Dtest = Dtest'; Itest = Itest';
    end
    meanD = mean(D(:,1:k_opt),2);
    [thr_train,kappa_train] = calc_thr(meanD,D);
    res.dist_knn_var.train.thresholds = thr_train';
    res.dist_knn_var.train.ki = kappa_train';
    [inad,kappa_test,outside_AD_ko] = AD_inorout(Dtest,Itest,thr_train);
    res.dist_knn_var.test.kj = kappa_test;    
    res.dist_knn_var.inad = inad;
    res.dist_knn_var.train.D = D;
    res.dist_knn_var.train.Itrain = Itrain;
    res.dist_knn_var.test.Dtest = Dtest;
    res.dist_knn_var.test.Itest = Itest;
    res.dist_knn_var.k_opt = k_opt;
end

% Potential functions ---------------------------------------------------------
res.pot_fun.inad = [];
if whattodo.pot_fun
    disp('Processing Potential Functions')
    inad = ones(size(Xtest,1),1);
    if length(options.pf_smoot) > 1
        istep = 6;
        disp(['    > Searching for smoothing coefficient: range [' num2str(options.pf_smoot(1)) ' - ' num2str(options.pf_smoot(2)) ']'])
        [smoot,Vmax1] = smootpf(Xtrain_scal,options.pf_kernel,options.pf_smoot,istep);
        smoot_sav = smoot;
        disp(['    > Found first smoothing coefficient: ' num2str(smoot)])
        del = (options.pf_smoot(2) - options.pf_smoot(1))/(2*istep);
        pf_smoot(1) = smoot - del; pf_smoot(2) = smoot + del;
        disp(['    > Searching for smoothing coefficient: range [' num2str(pf_smoot(1)) ' - ' num2str(pf_smoot(2)) ']'])
        [pf_smoot,Vmax] = smootpf(Xtrain_scal,options.pf_kernel,pf_smoot,istep*2);
        if Vmax1 > Vmax
            pf_smoot = smoot_sav; Vfin = Vmax1;
        else
            Vfin = Vmax;
        end
        disp(['    > Found optimal smoothing coefficient: ' num2str(pf_smoot)])
    else
        pf_smoot = options.pf_smoot;
        V = [];
    end
    for i = 1:size(Xtrain_scal,1)
        S(i,1) = calc_pot_fun(Xtrain_scal,Xtrain_scal(i,:),options.pf_kernel,pf_smoot);
    end
    Q = prctile(S,100-options.pf_pct);
    fthr = Q(1);
    for i = 1:size(Xtest_scal,1)
        Stest(i,1) = calc_pot_fun(Xtrain_scal,Xtest_scal(i,:),options.pf_kernel,pf_smoot);
    end
    inad(find(Stest < fthr)) = 0;
    res.pot_fun.inad = inad;
    res.pot_fun.thr = fthr;
    res.pot_fun.Stest = Stest;
    res.pot_fun.smoot = pf_smoot;
end
res.resume_table = outliers_list(res,whattodo,options);
res.options = options;

%--------------------------------------------------------------------------
function A = outliers_list(res,whattodo,options)
A(1,1)={'Approach'};
A(1,2)={'Options'};
A(1,3)={'Test inside AD'};
A(1,4)={'Test outside AD'};
A(1,5)={'List of samples outside AD'};
cnt_row = 1;
if whattodo.bounding_box
    cnt_row = cnt_row + 1;
    A = writehere(A,'Bounding box','---',res.bounding_box.inad,cnt_row);
end
if whattodo.bounding_box_pca
    cnt_row = cnt_row + 1;
    A = writehere(A,'Bounding box PCA',[num2str(size(res.bounding_box_pca.model_pca.T,2)) ' PCs'],res.bounding_box_pca.inad,cnt_row);
end
if whattodo.convex_hull
    cnt_row = cnt_row + 1;
    A = writehere(A,'Convex hull','---',res.convex_hull.inad,cnt_row);
end
if whattodo.leverage
    cnt_row = cnt_row + 1;
    A = writehere(A,'Leverage','---',res.leverage.inad,cnt_row);
end
if whattodo.dist_centroid
    cnt_row = cnt_row + 1;
    A = writehere(A,'Distance from centroid',options.distance,res.dist_centroid.inad,cnt_row);
end
if whattodo.dist_knn_fix
    cnt_row = cnt_row + 1;
    A = writehere(A,'Distance kNN - fixed k',options.distance,res.dist_knn_fix.inad,cnt_row);
end
if whattodo.dist_knn_var
    cnt_row = cnt_row + 1;
    A = writehere(A,'Distance kNN - variable k',options.distance,res.dist_knn_var.inad,cnt_row);
end
if whattodo.pot_fun
    cnt_row = cnt_row + 1;
    A = writehere(A,'Potential functions',options.pf_kernel,res.pot_fun.inad,cnt_row);
end

%--------------------------------------------------------------------------
function A = writehere(A,methodhere,optionshere,inad,cnt)
A{cnt,1} = methodhere;
A{cnt,2} = optionshere;
A{cnt,3} = num2str(length(find(inad == 1)));
A{cnt,4} = num2str(length(find(inad == 0)));
out = find(inad==0);
if length(out) == 0
    str = '[]';
elseif length(out) == 1
    str= ['[' num2str(out) ']'];
else
    str= ['[' num2str(out(1))];
    for k=2:length(out)-1
         str= [str ' ' num2str(out(k))];
    end
    str = [str ' ' num2str(out(end)) ']'];    
end
A{cnt,5} = str;

% -------------------------------------------------------------------------
function [thr_train,kappa_train] = calc_thr(meanD,D)
Q = prctile(meanD,[25,75]);
UL = Q(2) + 1.2*(Q(2) - Q(1));
ref_val = UL;
B(1:size(D,1),1:size(D,2)) = 0;
for i=1:size(D,1)
    A = find(D(i,:) <= ref_val);
    B(i,1:length(A)) = D(i,A);
    if isempty(A) == 0
        kappa_train(i) = length(A);
    else
        kappa_train(i) = 0;
    end
end
cc = sum(B,2);
for i = 1:size(D,1);
    if kappa_train(i) > 0
        thr_train(i) = cc(i)/kappa_train(i);
    else
        thr_train(i) = 0;
    end
end
QQ = thr_train(find(thr_train > 0));
minthr = min(QQ);
for i=1:size(thr_train,2);
    if thr_train(i) == 0
        thr_train(i) = minthr;
    end
end

% -------------------------------------------------------------------------
function [AD_finaloutput,kappa_test,outside_AD] = AD_inorout(Dtest,Itest,thr_train)
count = 0;
outside_AD = 0;
for i = 1:size(Dtest,1)
    ws = thr_train(Itest(i,:));
    A = find(Dtest(i,:) <= ws);
    if isempty(A) == 0
        kappa_test(i,1) = length(A);
        AD_finaloutput(i,1) = 1;
    else
        kappa_test(i,1) = 0;
        AD_finaloutput(i,1) = 0;
        count = count + 1;
        outside_AD(1,count) = i;
    end
end

%--------------------------------------------------------------------------
function [UL,prc] = calc_thr_prctile(vtrain,perc)
Q = prctile(vtrain,[25,75,perc]);
UL = Q(2) + 1.5*(Q(2) - Q(1));
prc = Q(3);

%--------------------------------------------------------------------------
function pot = calc_pot_fun(X,v,type,smoot)
pot = 0;
v1 = ones(size(X,1),1);
A = v1*v;
Xv = X - A;
[ni,nj] = size(X);
if strcmp(type,'gaus')
    s = std(X);
    pi2 = (2*pi)^0.5;
    for i = 1:ni
        t = 1;
        for j = 1:nj
            d = Xv(i,j);
            sig = smoot*s(j);
            n1 = 1/(pi2*sig);
            n2 = -(d^2)/(2*(sig^2));
            t = t*n1*exp(n2);
        end
        pot = pot + t/ni;
    end
elseif strcmp(type,'tria')
    Xv = Xv.^2;
    Xs = sum(Xv,2);
    Xs = Xs.^0.5;
    sm = smoot^nj;
    for i = 1:ni
        t = Xs(i);
        t = t/smoot;
        if t <= 1
            t = 1 - t;
        else
            t = 0;
        end
        pot = pot + t;
    end
    pot = pot/(ni*sm);
end

%--------------------------------------------------------------------------
function [smoot,Vmax] = smootpf(X,type,smoot,istep)
cnt = 0;
range = smoot(2) - smoot(1);
step = range/istep;
C(1:size(X,1)) = 0;
for s = smoot(1):step:smoot(2)
    cnt = cnt + 1;
    for i=1:size(X,1)
        in = ones(size(X,1),1);
        in(i) = 0;
        Xin  = X(in==1,:);
        Xout = X(in==0,:);
        C(i) = calc_pot_fun(Xin,Xout,type,s);
    end
    V(cnt,1) = s;
    V(cnt,2) = geomean(C);
end
[a,here] = max(V(:,2));
smoot = V(here,1);
Vmax = V(here,2);

%--------------------------------------------------------------------------
function [model] = pca_model(X,num_comp);
n = size(X,1);
[T,E,L] = svd(X,0);     
E = diag(E).^2/(n-1);      
exp_var = E/sum(E);
E = E(1:num_comp);
exp_var = exp_var(1:num_comp);
for k=1:num_comp; cum_var(k) = sum(exp_var(1:k)); end;
L = L(:,1:num_comp);       
T = X*L;
T = T(:,1:num_comp);
model.exp_var = exp_var;
model.cum_var = cum_var';
model.E = E;
model.L = L;
model.T = T;
model.num_comp = num_comp;

%--------------------------------------------------------------------------
function model = pca_project(Xnew,model);
T = Xnew*model.L;
model.Tpred = T;

%--------------------------------------------------------------------------
function [X_scal,param] = data_pretreatment(X,pret_type)
a = mean(X);
s = std(X);
m = min(X);
M = max(X);
if strcmp(pret_type,'cent')
    amat = repmat(a,size(X,1),1);
    X_scal = X - amat;
elseif strcmp(pret_type,'scal')
    f = find(s>0);
    smat = repmat(s,size(X,1),1);
    X_scal = zeros(size(X,1),size(X,2));
    X_scal = X(:,f)./smat(:,f);
elseif strcmp(pret_type,'auto')
    f = find(s>0);
    amat = repmat(a,size(X,1),1);
    smat = repmat(s,size(X,1),1);
    X_scal = zeros(size(X,1),size(X,2));
    X_scal(:,f) = (X(:,f) - amat(:,f))./smat(:,f);
elseif strcmp(pret_type,'rang')
    f = find(M - m > 0);
    mmat = repmat(m,size(X,1),1);
    Mmat = repmat(M,size(X,1),1);
    X_scal = zeros(size(X,1),size(X,2));
    X_scal(:,f) = (X(:,f) - mmat(:,f))./(Mmat(:,f) - mmat(:,f));       
else
    X_scal = X;
end
param.a = a;
param.s = s;
param.m = m;
param.M = M;
param.pret_type = pret_type;

%--------------------------------------------------------------------------
function [X_scal] = test_pretreatment(X,param)
a = param.a;
s = param.s;
m = param.m;
M = param.M;
pret_type = param.pret_type;
if strcmp(pret_type,'cent')
    amat = repmat(a,size(X,1),1);
    X_scal = X - amat;
elseif strcmp(pret_type,'scal')
    f = find(s>0);
    smat = repmat(s,size(X,1),1);
    X_scal = zeros(size(X,1),size(X,2));
    X_scal = X(:,f)./smat(:,f);
elseif strcmp(pret_type,'auto')
    f = find(s>0);
    amat = repmat(a,size(X,1),1);
    smat = repmat(s,size(X,1),1);
    X_scal = zeros(size(X,1),size(X,2));
    X_scal(:,f) = (X(:,f) - amat(:,f))./smat(:,f);
elseif strcmp(pret_type,'rang')
    f = find(M - m > 0);
    mmat = repmat(m,size(X,1),1);
    Mmat = repmat(M,size(X,1),1);
    X_scal = zeros(size(X,1),size(X,2));
    X_scal(:,f) = (X(:,f) - mmat(:,f))./(Mmat(:,f) - mmat(:,f));       
else
    X_scal = X;
end

%--------------------------------------------------------------------------
function out = variable_parameters(Xtrain,param);
modelparamarray = write_results(param,'var');
out.param_table = modelparamarray;
out.param = param;

%--------------------------------------------------------------------------
function modelparamarray = write_results(param,label);
modelparamarray(1,1) = {'Var.ID'};
modelparamarray(1,2) = {'Min'};
modelparamarray(1,3) = {'Max'};
modelparamarray(1,4) = {'Average'};
modelparamarray(1,5) = {'Std.Dev.'};
for j = 1:length(param.m)
    modelparamarray(j+1,1) = {[label ' ' num2str(j)]};
    modelparamarray(j+1,2) = {num2str(param.m(j))};
    modelparamarray(j+1,3) = {num2str(param.M(j))};
    modelparamarray(j+1,4) = {num2str(param.a(j))};
    modelparamarray(j+1,5) = {num2str(param.s(j))};
end

%--------------------------------------------------------------------------
function [k_opt,stat] = find_k_optimal(k_optimisation,k_max)
meanM = mean(k_optimisation);
stdM = std(k_optimisation);
[Q] = prctile(k_optimisation,[1,5,25,75,95,99]);
[aaa,nq] = size(Q');
cvM = stdM./meanM;
dMcv(1:k_max) = 0;
for k = 1:k_max-1
    dMcv(k) = abs(cvM(k+1) - cvM(k));
end
epsi = 0.0001;
flg = 1; cnt = 1;
while flg == 1
    if dMcv(cnt) < epsi && dMcv(cnt+ 1) < epsi
        flg = 0;
        k_opt = cnt;
    else
        cnt = cnt + 1;
    end
    if cnt == k_max
        flg = 0;
        k_opt = k_max;
    end
end
stat(1:k_max,1:nq) = Q';
stat(1:k_max,nq+1) = meanM';
stat(1:k_max,nq+2) = stdM';
stat(1:k_max,nq+3) = dMcv';

%--------------------------------------------------------------------------
function doboxplot(k_optimisation)
boxplot(k_optimisation);
set(gcf,'color','white'); box on;
hold on
plot(mean(k_optimisation),'-*');
xlabel(' k values');
ylabel(' Distribution of samples inside the AD (%)');
title(' Boxplot for the samples inside AD in k optimization');
m = min(min(k_optimisation));
M = max(max(k_optimisation));
R = (M - m)/20;
m = m - R;
M = M + R;
if M > 100; M = 100; end
axis([0 size(k_optimisation,2)+0.5 m M])
hold off
