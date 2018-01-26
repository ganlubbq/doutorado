%----------------------------------------------------------
% Real part in first half and Imaginary in the second half.
%----------------------------------------------------------

clear all;close all;clc

rng(1)

numTrainVectors = 60000;
numTestVectors = 1000;
numPredictionVectors = 100;

max_value = 5;

SNR = 10;                   % Signal-to-noise ratio (SNR) in dB.
linear_SNR = 10.^(SNR./10); % Linear SNR value.

M = 70;             % Number of antennas.
K = 10;             % Number of single-antenna users.
L = 7;              % Number of cells.

N = K;              % Pilot length is set according to K and P.

a = 0.05;           % Constant beta value.
beta111 = 1;

q = linear_SNR/(N*(1 + a*(L-1)));  % Uplink pilot power.

% Generate pilot signals.
S = generatePilotMatrixFFT(N,K);

% Simulation loop starts here.
train_data = zeros(numTrainVectors,M*2);
train_label = zeros(numTrainVectors,M*2);
test_data = zeros(numTestVectors,M*2);
test_label = zeros(numTestVectors,M*2);
prediction_data = zeros(numPredictionVectors,M*2);
prediction_label = zeros(numPredictionVectors,M*2);

for q_idx=1:1:length(q)
    
    %% Train vectors.
    % loop starts here.
    Y1 = zeros(M,N,numTrainVectors);
    g_111 = zeros(M,numTrainVectors);
    Z11_ls = zeros(M,numTrainVectors);
    for trainIter = 1:1:numTrainVectors
        
        beta_sum = 0;
        sum_G = zeros(M,N);
        Gil = zeros(M,K,L);
        % Iterate over all cells (L) in the assumed system.
        % Here we consider the target cell, i, is 1, i.e., i = 1.
        for l=1:1:L
            
            % Generate channel matrix G_{il}.
            beta = a;
            if(l == 1)
                beta = 1;
            end
            betaMatrix = sqrt(beta)*eye(K);
            Gil(:,:,l) = (1/sqrt(2))*complex(randn(M,K),randn(M,K))*betaMatrix;
            
            % Summation of all channels.
            sum_G = sum_G + Gil(:,:,l)*(S');
            
            % Summation of betas.
            beta_sum = beta_sum + beta;
            
            % This is the label.
            if(l == 1)
                g_111(:,trainIter) = Gil(:,1,l);
            end        
        end
        
        % Apply squared pilot power.
        sum_G = sqrt(q(q_idx))*sum_G;
        
        % Generate noise.
        W1 = (1/sqrt(2))*complex(randn(M,N),randn(M,N));
        
        % Received pilot-sequence symbols at BS #1, which is the target cell, i.e., i = 1.
        Y1(:,:,trainIter) = sum_G + W1;
        
        % Least Squares estimation.
        Z11_ls(:,trainIter) = (1/(sqrt(q(q_idx))*N))*Y1(:,:,trainIter)*S(:,1); 
    end
    
    error_train_ls = sum(sum(abs(Z11_ls - g_111),1)/M)/numTrainVectors;
   
    for trainIter = 1:1:numTrainVectors
        
        idx = 0;
        for iq_idx=1:1:2
            for zls_idx=1:1:M
                idx = idx + 1;
                if(iq_idx==1)
                    train_data(trainIter,idx) = real(Z11_ls(zls_idx,trainIter));
                else
                    train_data(trainIter,idx) = imag(Z11_ls(zls_idx,trainIter));
                end
            end
        end
       
        train_data(trainIter,:) = train_data(trainIter,:)./max_value;
        
        idx = 0;
        for iq_idx=1:1:2
            for g_line_idx=1:1:size(g_111,1)
                idx = idx + 1;
                if(iq_idx==1)
                    train_label(trainIter,idx) = real(g_111(g_line_idx,trainIter));
                else
                    train_label(trainIter,idx) = imag(g_111(g_line_idx,trainIter));
                end
            end
        end
        
        train_label(trainIter,:) = train_label(trainIter,:)./max_value;      
    end
    
    error_train_vectors_ls = sum(sum(abs(train_data - train_label),2)/2*M)/numTrainVectors;

    %% Generate test vectors.
    % loop starts here.
    Y1 = zeros(M,N,numTestVectors);
    g_111 = zeros(M,numTestVectors);
    Z11_ls = zeros(M,numTestVectors);
    for testIter = 1:1:numTestVectors
        
        beta_sum = 0;
        sum_G = zeros(M,N);
        Gil = zeros(M,K,L);
        % Iterate over all cells (L) in the assumed system.
        % Here we consider the target cell, i, is 1, i.e., i = 1.
        for l=1:1:L
            
            % Generate channel matrix G_{il}.
            beta = a;
            if(l == 1)
                beta = 1;
            end
            betaMatrix = sqrt(beta)*eye(K);
            Gil(:,:,l) = (1/sqrt(2))*complex(randn(M,K),randn(M,K))*betaMatrix;
            
            % Summation of all channels.
            sum_G = sum_G + Gil(:,:,l)*(S');
            
            % Summation of betas.
            beta_sum = beta_sum + beta;
            
            % This is the label.
            if(l == 1)
                g_111(:,testIter) = Gil(:,1,l);
            end
            
        end
        
        % Apply squared pilot power.
        sum_G = sqrt(q(q_idx))*sum_G;
        
        % Generate noise.
        W1 = (1/sqrt(2))*complex(randn(M,N),randn(M,N));
        
        % Received pilot-sequence symbols at BS #1, which is the target cell, i.e., i = 1.
        Y1(:,:,testIter) = sum_G + W1;
        
        % Least Squares estimation.
        Z11_ls(:,testIter) = (1/(sqrt(q(q_idx))*N))*Y1(:,:,testIter)*S(:,1); 
    end
    
    error_test_ls = sum(sum(abs(Z11_ls - g_111),1)/M)/numTestVectors;
    
    for testIter = 1:1:numTestVectors
        
        idx = 0;
        for iq_idx=1:1:2
            for zls_idx=1:1:M
                idx = idx + 1;
                if(iq_idx==1)
                    test_data(testIter,idx) = real(Z11_ls(zls_idx,testIter));
                else
                    test_data(testIter,idx) = imag(Z11_ls(zls_idx,testIter));
                end
            end
        end
        
        test_data(testIter,:) = test_data(testIter,:)./max_value;        
        
        idx = 0;
        for iq_idx=1:1:2
            for g_line_idx=1:1:size(g_111,1)
                idx = idx + 1;
                if(iq_idx==1)
                    test_label(testIter,idx) = real(g_111(g_line_idx,testIter));
                else
                    test_label(testIter,idx) = imag(g_111(g_line_idx,testIter));
                end
            end
        end

        test_label(testIter,:) = test_label(testIter,:)./max_value; 
    end
    
    error_test_vectors_ls = sum(sum(abs(test_data - test_label),2)/2*M)/numTestVectors;
 
    %% Generate prediction vectors.
    % loop starts here.
    Y1 = zeros(M,N,numPredictionVectors);
    g_111 = zeros(M,numPredictionVectors);
    Z11_ls = zeros(M,numPredictionVectors);
    for predictionIter = 1:1:numPredictionVectors
        
        beta_sum = 0;
        sum_G = zeros(M,N);
        Gil = zeros(M,K,L);
        % Iterate over all cells (L) in the assumed system.
        % Here we consider the target cell, i, is 1, i.e., i = 1.
        for l=1:1:L
            
            % Generate channel matrix G_{il}.
            beta = a;
            if(l == 1)
                beta = 1;
            end
            betaMatrix = sqrt(beta)*eye(K);
            Gil(:,:,l) = (1/sqrt(2))*complex(randn(M,K),randn(M,K))*betaMatrix;
            
            % Summation of all channels.
            sum_G = sum_G + Gil(:,:,l)*(S');
            
            % Summation of betas.
            beta_sum = beta_sum + beta;
            
            % This is the label.
            if(l == 1)
                g_111(:,predictionIter) = Gil(:,1,l);
            end
            
        end
        
        % Apply squared pilot power.
        sum_G = sqrt(q(q_idx))*sum_G;
        
        % Generate noise.
        W1 = (1/sqrt(2))*complex(randn(M,N),randn(M,N));
        
        % Received pilot-sequence symbols at BS #1, which is the target cell, i.e., i = 1.
        Y1(:,:,predictionIter) = sum_G + W1;
        
        % Least Squares estimation.
        Z11_ls(:,predictionIter) = (1/(sqrt(q(q_idx))*N))*Y1(:,:,predictionIter)*S(:,1);  
    end
    
    error_prediction_ls = sum(sum(abs(Z11_ls - g_111),1)/M)/numPredictionVectors;
    
    error_prediction = zeros(1,numPredictionVectors);
    for predictionIter = 1:1:numPredictionVectors        
        idx = 0;
        for iq_idx=1:1:2
            for zls_idx=1:1:M
                idx = idx + 1;
                if(iq_idx==1)
                    prediction_data(predictionIter,idx) = real(Z11_ls(zls_idx,predictionIter));
                else
                    prediction_data(predictionIter,idx) = imag(Z11_ls(zls_idx,predictionIter));
                end
            end
        end
        
        prediction_data(predictionIter,:) = prediction_data(predictionIter,:)./max_value; 
        
        idx = 0;
        for iq_idx=1:1:2
            for g_line_idx=1:1:size(g_111,1)
                idx = idx + 1;
                if(iq_idx==1)
                    prediction_label(predictionIter,idx) = real(g_111(g_line_idx,predictionIter));
                else
                    prediction_label(predictionIter,idx) = imag(g_111(g_line_idx,predictionIter));
                end
            end
        end
        
        prediction_label(predictionIter,:) = prediction_label(predictionIter,:)./max_value;  
        
        sub_vectors = prediction_data(predictionIter,:) - prediction_label(predictionIter,:);
        abs_vectors = abs(sub_vectors);
        sum_vectors = sum(abs_vectors);
        error_prediction(predictionIter) = sum_vectors/(2*M);
    end
    
    %% Save data set for specfic scenario.
    fileName = sprintf('data_set_M_%d_K_%d_SNR_%d_static_scenario_1_ls_est_v1.mat',M,K,SNR(q_idx));
    save(fileName,'train_data','train_label','test_data','test_label','prediction_data','prediction_label','error_prediction','-v7')
end
