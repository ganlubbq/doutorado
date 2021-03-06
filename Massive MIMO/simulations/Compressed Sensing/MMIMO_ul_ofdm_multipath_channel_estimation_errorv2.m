clear all;close all;clc

%% ----------------------------------- Parameters ------------------------------
plot_fig = false;                                                                                   % Disable/enable figure plotting.

Np = 100;                                                                                            % Number of pilots per OFDM symbol.

L = 1;                                                                                              % Number of cells.
K = 10;                                                                                             % Number of single-antenna terminals in each cell, i.e., number of transmitt antennas.
M = 100;                                                                                            % Number of antennas at the base station of each cell (In this case, i.e., uplink, it gives the number of receiving antennas).
NFFT = 2048;                                                                                        % Number of points used by the OFDM.
AlphabetSize = 4;                                                                                   % Modulation alphabet size (QPSK)
modOrd = log2(AlphabetSize);                                                                        % Bits/symbol
numSym = K*(NFFT - K*Np);                                                                           % Number of symbols, i.e., number of terminals. The number of pilots must be leaved out.
NCP = 512;                                                                                          % Number of samples used to create a Extended Cyclic Prefix (CP) according to 3GPP' LTE standards. 12 OFDM symbols per subframe, i.e., 1 ms.
delta_f = 15000;                                                                                    % Frequency space between subcarriers.
Ts = 1/(delta_f*NFFT);                                                                              % System Sampleing Rate.
numSymbInSubframe = 12;                                                                             % Number of symbols in a subframe (1ms). 12 OFDM symbols for extended CP.

add_awgn = true;                                                                                    % Enable/disable addtion of additive gaussian noise to signals.
EbNoVec = 0:2:20;                                                                                  % Eb/No in dB.
EsN0dB = EbNoVec + 10*log10(NFFT/(NFFT+NCP)) + 10*log10((NFFT-K*Np+Np)/NFFT) + 10*log10(modOrd);    % converting to symbol to noise ratio
snr = EsN0dB - 10*log10((NFFT/(NFFT+NCP)));                                                         % Calculate SNR from EsNo in dB.

cellRadius = 1000;                                                                                  % Radius given in meters.
cellHole = 100;                                                                                     % Cell hole in meters.

power_ctrl_enabled = true;                                                                          % enable/disable power control at eNodeB.

nTotalOfBits = 1e8;
nErrors = 1000000;
debug = false;

% Large scale fading.
sshadow = 3;                                                        % Shadow-fading standard deviation in dB.
gamma = 2.8;                                                        % Decay exponent: Urban area cellular radio 2.7 - 3.5

% Small scale fading.
delay = [0 0.977]*1e-6;                                             % Delay in microseconds.
gain  = [-3.010299956639812 -3.010299956639812];                    % Gain in dB (indoor).
numPaths = length(delay);                                           % Number of paths per channel.
totalNumPaths = M*K*numPaths;                                       % Total number of paths between the various antennas and Base Stations.

%% ----------------------------------- Setup Channel -----------------------------------
% Vetor de ganhos.
pos = round(delay/Ts)+1;                                            % Effective position of taps within the vector.
g = zeros(1, round(delay(end)/Ts)+1);                               % +1 is used to include the delay 0 into the vector.
for n = 1:length(delay)
    g( pos(n) ) = sqrt(10^( gain(n)/10 ));
end
delaySpreadMax = pos(length(pos));

fc = 1e9;                                                           % Carrier Freq. in MHz.
c = 3e8;                                                            % Light speed in m/s.
v = 30;                                                             % Speed in m/s.
Fs = 30.72e6;                                                       % Sampling Freq. in MHz.
Ts = 1/Fs;                                                          % Sampling period in seconds.

fd = (v*fc)/c;                                                      % Doppler frequency in Hz.

Pd = 0;                                                             % Relative power in dB.

% Parameters used for generating the Mulipath Masive MIMO Channel.
Fs_chann = 500;                                                     % Channel Sampling Rate in Hz. The channel is sampled every 2 ms. (Periodo de amostragem do canal ~1 amostra / Slot OFDM)
Ts_chann = 1/Fs_chann;
N_chann = 256;                                                      % Number of samples used to sample the channel. Duration of the channel in  number of samples.
delta_f_chann = Fs_chann/N_chann;                                   % in Hz.
f = -Fs_chann/2:delta_f_chann:Fs_chann/2;
snr_idx = find(f<fd);
f = f(snr_idx);
snr_idx = find(f>-fd);
f = f(snr_idx);
f = f.';

LS = length(f);
S = 1/pi/fd./sqrt(1 - (f/fd).^2) * 10^(Pd/10);
S = S * LS / sum(S);                                                % Energy nomalization.
S1 = S;
S = [S((LS+1)/2:LS); zeros(N_chann-LS,1); S(1:(LS-1)/2)];           % Interpolation of the Doppler Spectrum by N_chann/LS.

% ************************* Generate Multipath Massive MIMO Channel. **********************
rng(55);
x = [(randn((LS-1)/2+1,totalNumPaths,'double') + 1i*randn((LS-1)/2+1,totalNumPaths,'double')); zeros(N_chann-LS,totalNumPaths); (randn((LS-1)/2,totalNumPaths,'double')) + 1i*randn((LS-1)/2,totalNumPaths,'double')]/sqrt(2);
ch = ifft(x .* repmat(sqrt(S),1,totalNumPaths)) * N_chann / sqrt(LS);

% for iud=1:1:256
% aaaa=mean((x(iud,:)));
% figure
% plot(real(ch(iud,:)))
% end

% Plot doppler spectrum figures.
if(plot_fig)
    figure;
    plot(f, abs(S1));
    
    figure;
    plot(S)
    
    figure;
    Tsd = 1/Fs_chann;
    TNd = Tsd*N_chann; % Channel duration in seconds.
    plot( (0:N_chann-1)*Tsd, 10*log10(abs(ch)) );
    xlabel('Time (s)')
    ylabel('Power (dB)');
    axis([0 TNd -30 10]);
end

% **************************** Cell Layout. *******************************
% Generate a random radius value within the range cellHole to cellRadius for each one of the terminals.
radius = cellHole + (cellRadius-cellHole).*rand(1,K);

% Generate an random angle value within the range 0 to 2*pi for each one of the terminals.
angle = 2*pi*rand(1,K);

% Plot the position of each terminal inside the cell.
figure;
plot(0, 0, 'rs', 'MarkerFaceColor',[1,0,0], 'MarkerEdgeColor',[1 0 0]);
hold on

Circlex = cellRadius*cos(2*pi*(0:99)/100);
Circley = cellRadius*sin(2*pi*(0:99)/100);
plot(Circlex.', Circley.', 'k');

Circlex = cellHole*cos(2*pi*(0:99)/100);
Circley = cellHole*sin(2*pi*(0:99)/100);
plot(Circlex.', Circley.', 'r');

UEx = radius.*cos(angle);
UEy = radius.*sin(angle);
plot(UEx, UEy, 'b*');

grid on;
hold off;

% Calculate path-loss for all users, in meters.
path_loss = radius.^gamma;

% ************ Apply power delay profile and large-scale fading to the multiptah channel matrix, i.e., small-scale fading. ************
for idx_ch = 1 : N_chann
    
    if(power_ctrl_enabled)
        largeScaleFading = 1;
    else
        % Calculate shadowing for each one of different channels.
        shadowing_att = lognrnd(0,sshadow,1,K);
        % Large scale fading calculated according to Marzetta.
        largeScaleFading = repmat(sqrt(shadowing_att./path_loss), M, 1);
    end
    
    % Atualizacao das matrizes de canal para a celula alvo (l = 1): existe uma matriz de canal por percurso do terminal.
    G = reshape(ch(idx_ch,:), M, K, length(pos));
    G(:,:,1) = (g(pos(1)) * G(:,:,1)) .* largeScaleFading;
    for k = 2:length(pos)
        G(:,:,k) = (g(pos(k)) * G(:,:,k)) .* largeScaleFading;
    end
    ch(idx_ch,:) = reshape(G, 1, totalNumPaths);
end

%% @@@@@@@@@@@@@@@@ Create Pilots @@@@@@@@@@@@@@@@@@@@@@@@@@@@
fo = 1000; % in Hz.                             % Pilot frequency.
P = exp(1i*2*pi*fo*(0:1:Np-1)/Np);              % Normalized pilot signal, i.e., unit power.

Pdiag = diag(P);

F = fft(eye(NFFT));
F = F(:,1:NCP);

%------------ Retrieve Pilot Positions ------------
if(Np==16) % Funciona somente com OMP.
    [ppos, flag_data] = getOptimumPpos16();
elseif(Np==17) % Funciona somente com OMP.
    [ppos, flag_data] = getOptimumPpos17();
elseif(Np==31) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos31();        
elseif(Np==32) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos32();
elseif(Np==34) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos34();
elseif(Np==35) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos35();
elseif(Np==37) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos37();
elseif(Np==36) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos36();
elseif(Np==40) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos40();
elseif(Np==50) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos50();
elseif(Np==53) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos53();
elseif(Np==60) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos60();
elseif(Np==73) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos73();
elseif(Np==80) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos80();
elseif(Np==100) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos100();
elseif(Np==101) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos101();
elseif(Np==151) % Valores �timos para LS e OMP.
    [ppos, flag_data] = getOptimumPpos151();
else
    error('Invalid value for Np!!!!!');
end

%% ----------------------------------- Set up the simulation -----------------------------------
% Create a local random stream to be used by random number generators for repeatability.
hStr = RandStream('mt19937ar');

% Get all bit combinations for ML receiver
bits = de2bi(0:2^(modOrd*K)-1, 'left-msb')';
% Split them per Transmit antenna
b = zeros(K, modOrd, length(bits));
for i = 1:length(bits)
    b(:, :, i) = reshape(bits(:,i), modOrd, K)';
end

% Preallocate variables for speed.
dist = zeros(length(bits), 1);
[BER_MMSE_LE, BER_MFB] = deal(zeros(1, length(EbNoVec)));

% Create QPSK mod-demod objects.
hMod = modem.pskmod('M', 2^modOrd, 'SymbolOrder', 'gray', 'InputType', 'bit');
hDemod = modem.pskdemod(hMod);

% Set up a figure for visualizing BER results.
figura = figure; grid on; hold on;
set(gca,'yscale','log','xlim',[EbNoVec(1)-0.01, EbNoVec(end)],'ylim',[1e-7 1]);
xlabel('Eb/No (dB)'); ylabel('MSE'); set(figura,'NumberTitle','off');
set(figura, 'renderer', 'zbuffer'); set(figura,'Name','OFDM modulated with QPSK Massive MU-MIMO System');
strTitle = sprintf('Massive MU-MIMO Channel Estimation on Uplink - Np: %d',Np);
title(strTitle);

%% ----------------------------------- Loop over selected EbNo points. -----------------------------------
est_error_omp  = zeros(1,length(snr));
est_error_ls   = zeros(1,length(snr));
est_error_mmse = zeros(1,length(snr));
avg_error_omp  = zeros(1,length(snr));
avg_error_ls   = zeros(1,length(snr));
avg_error_mmse = zeros(1,length(snr));
N_chann = 10;
for snr_idx = 1:1:length(snr)
    
    linearSnr = 10^(snr(snr_idx)/10);
    
    ofdm_symbol_number = 0;
    x = complex(zeros(K,NFFT+NCP),zeros(K,NFFT+NCP));
    aux = complex(zeros(K,NFFT+NCP),zeros(K,NFFT+NCP));
    for idx_ch=1:1:N_chann
        
        %---------- Transmission (UE) ----------
        % Create array of bits to modulate.
        msg = randi(hStr, [0 1], modOrd, numSym);
        
        % Modulate data.
        source = modulate(hMod, msg);
        
        % Split source among K terminals.
        Tx = complex(zeros(K,NFFT),zeros(K,NFFT));
        Tx(~flag_data) = reshape(source, K, numel(source)/K); clear source;
        for l_idx=1:1:K
            Tx(l_idx,ppos(l_idx,:)) = P;
        end
        
        % Create OFDM symbol.
        sequence = (NFFT/sqrt(NFFT-Np*K+Np))*ifft(Tx,NFFT,2);
        
        % Add CP.
        ofdm = [sequence(:,NFFT-NCP+1:end), sequence];
        
        %---------- Multipath Channel plus Noise ----------
        H = reshape(ch(idx_ch,:), M, K, length(pos));
        
        x = [H(:,:,1)*ofdm complex(zeros(M,(pos(length(pos))-1)),zeros(M,(pos(length(pos))-1)))];
        for k = 2:length(pos)
            aux = [complex(zeros(M,(pos(k)-1)),zeros(M,(pos(k)-1))) H(:,:,k)*ofdm];
            x = x + aux;
        end
        
        if(add_awgn)
            % Add channel noise power to faded data.
            r = awgn(x, snr(snr_idx), 0, hStr);
        else
            r = x;
        end
        
        %------------------- Reception (base station) ---------------------
        % Get a whole subframe.
        rx = r(:,1:end-(pos(length(pos))-1));
        
        % Remove CP.
        rx = rx(:,NCP+1:end);
        
        % Retrieve modulation symbols by applying FFT to received signal.
        ofdm_rx = (sqrt(NFFT-Np*K+Np)/NFFT)*fft(rx,NFFT,2);
        
        % Massive MIMO Channel Estimation.
        h = zeros(pos(length(pos)),1);
        H_hat_omp = zeros(M, K, length(pos));
        H_hat_ls = zeros(M, K, length(pos));
        H_hat_mmse = zeros(M, K, length(pos));
        iter_error_omp = 0;
        iter_error_ls = 0;
        iter_error_mmse = 0;
        for m_idx=1:1:M
            for k_idx=1:1:K
                
                %% ********* Fourier Basis *********
                Fl = F(ppos(k_idx,:).',:);
                
                y = ofdm_rx(m_idx,ppos(k_idx,:)).';
                
                % --------- Iterative Greedy Algorithm ---------
                A = Pdiag*Fl;
                
                % OMP.
                g_hat_omp = OMP_orig(A,y,numPaths);
                
                % --------- Linear Algorithms ---------
                A = Pdiag*Fl(:,1:pos(length(pos)));
                
                % Least Squares (LS).
                invMatLs = (((A'*A)^(-1))*A');
                g_hat_ls = invMatLs*y;
                
                % MMSE.
                invMatMmse = (A'*A + (numPaths/linearSnr)*eye(delaySpreadMax))^-1;
                invMatMmse = invMatMmse*A';
                g_hat_mmse = invMatMmse*y;
                
                % --------- Estimate of H ---------
                H_hat_omp(m_idx,k_idx,:)  = g_hat_omp(pos);
                H_hat_ls(m_idx,k_idx,:)   = g_hat_ls(pos);
                H_hat_mmse(m_idx,k_idx,:) = g_hat_mmse(pos);
                
                % --------- Iteration error ---------
                h(pos,1) = [H(m_idx,k_idx,1);H(m_idx,k_idx,2)];
                iter_error_omp = iter_error_omp + (sum(abs(h - g_hat_omp(1:pos(length(pos)))).^2)/pos(length(pos)));
                iter_error_ls = iter_error_ls + (sum(abs(h - g_hat_ls).^2)/pos(length(pos)));
                iter_error_mmse = iter_error_mmse + (sum(abs(h - g_hat_mmse).^2)/pos(length(pos)));
            end
        end
        
        est_error_omp(snr_idx)  = est_error_omp(snr_idx) + (sum(sum(sum(abs(H_hat_omp - H).^2)))/(M*K*numPaths));
        est_error_ls(snr_idx)   = est_error_ls(snr_idx) + (sum(sum(sum(abs(H_hat_ls - H).^2)))/(M*K*numPaths));
        est_error_mmse(snr_idx) = est_error_mmse(snr_idx) + (sum(sum(sum(abs(H_hat_mmse - H).^2)))/(M*K*numPaths));
        
        avg_error_omp(snr_idx)  = avg_error_omp(snr_idx) + (iter_error_omp/(M*K));
        avg_error_ls(snr_idx)   = avg_error_ls(snr_idx) + (iter_error_ls/(M*K));
        avg_error_mmse(snr_idx) = avg_error_mmse(snr_idx) + (iter_error_mmse/(M*K));
    end
    
    est_error_omp(snr_idx) = est_error_omp(snr_idx)/N_chann;
    est_error_ls(snr_idx) = est_error_ls(snr_idx)/N_chann;
    est_error_mmse(snr_idx) = est_error_mmse(snr_idx)/N_chann;
    
    avg_error_omp(snr_idx) = avg_error_omp(snr_idx)/N_chann;
    avg_error_ls(snr_idx) = avg_error_ls(snr_idx)/N_chann;
    avg_error_mmse(snr_idx) = avg_error_mmse(snr_idx)/N_chann;
    
    fprintf(1,'OMP  - Reduced vector: %d - Entire vector: %d\n',est_error_omp(snr_idx), avg_error_omp(snr_idx));
    fprintf(1,'LS   - Reduced vector: %d - Entire vector: %d\n',est_error_ls(snr_idx), avg_error_ls(snr_idx));
    fprintf(1,'MMSE - Reduced vector: %d - Entire vector: %d\n',est_error_mmse(snr_idx), avg_error_mmse(snr_idx));
    
end

semilogy(EbNoVec, est_error_ls, 'bs');
semilogy(EbNoVec, est_error_mmse, 'k*');
semilogy(EbNoVec, est_error_omp, 'ro');
legend('LS','MMSE','OMP')
hold off

if(add_awgn)
    m_rx_antennas = '';
    for j=1:length(M)
        m_rx_antennas = strcat(m_rx_antennas, sprintf('_%d',M(j)));
    end
    
    m_tx_antennas = '';
    for j=1:length(K)
        m_tx_antennas = strcat(m_tx_antennas, sprintf('_%d',K(j)));
    end
    
    n_pilots = '';
    for j=1:length(Np)
        n_pilots = strcat(n_pilots, sprintf('_%d',Np(j)));
    end
    
    % Get timestamp for saving files.
    timeStamp = datestr(now,30);
    
    % Save figure to FIG-file.
    fileName = sprintf('Massive_MU_MIMO_M%s_K%s_Np%s_estimation_MSE_%s.fig',m_rx_antennas,m_tx_antennas,n_pilots,timeStamp);
    savefig(figura,fileName); clear figura
    
    % Save workspace to MAT-file.
    fileName = sprintf('Massive_MU_MIMO_M%s_K%s_Np%s_estimation_MSE_%s.mat',m_rx_antennas,m_tx_antennas,n_pilots,timeStamp);
    save(fileName);
end