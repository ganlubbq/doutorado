clear all;clc;close all;

% ------- Definitions -------
u = 129;
Nzc = 839;
NIFFT = 2048;
NIDFT = 24576;
NSUBFRAME = 30720;
Ncp = 3168;
NG = 2976;
v = [10];
Ncs = 13;

K = 1; % Number of Tranmitters, i.e., UEs or number of single-antenna terminals.

prach_offset = 10;

signal = complex(0,0)*zeros(1,NIFFT);
xuv = complex(0,0)*zeros(1,Nzc);

preambles = complex(zeros(K,NSUBFRAME),zeros(K,NSUBFRAME));

show_figures = false;

% ------- Generate Root Zadoff-Chu sequence. -------
n = [0:1:(Nzc-1)];
xu_root = exp(-1i*(pi*u.*n.*(n+1))./Nzc);

%% ****************************** PRACH Transmission ******************************
for i=1:1:K
    
    % ------- Apply Cyclic Shift to Root Zadoff-Chu sequence. -------
    Cv = v(i)*Ncs;
    xuv = xu_root(mod((n+Cv),Nzc)+1);
    
    % ------- Apply DFT to the Preamble. -------
    Xuv = (1/sqrt(Nzc))*fft(xuv,Nzc);
    
    % ------- Subcarrier Mapping. -------
    bb_signal = [complex(zeros(1,prach_offset),zeros(1,prach_offset)), Xuv, complex(zeros(1,NIDFT-prach_offset-Nzc),zeros(1,NIDFT-prach_offset-Nzc))];
    
    % ------- Apply IDFT to the Baseband Signal. -------
    prach = (NIDFT/sqrt(Nzc))*ifft(bb_signal,NIDFT);
    
    % ------- Add CP. -------
    prach_cp = [prach(NIDFT-Ncp+1:NIDFT), prach];
    
    % ------- Add Guard-time (GT) interval. -------
    y = [prach_cp, zeros(1,NG)];
    
    if(show_figures)
        figure(4);
        stem(0:1:NIDFT-1,abs(fft(y(Ncp+1:NIDFT+Ncp),NIDFT)));
        title('Transmitted PRACH signal.');
    end
    
    preambles(i,:) = y;
      
end

% % Make sure the preamble has unit variance.
% std_dev_vector = (sqrt(diag((preambles*preambles')/(NSUBFRAME))));
% std_dev_matrix = diag(1./std_dev_vector);
% preambles = std_dev_matrix*preambles;
% 
% var(preambles)

%% *************************** Multipath Rayleigh Channel ***************************
delay_zcorr1 = 1;
delay1 = delay_zcorr1*30;

delay_zcorr2 = 10;
delay2 = delay_zcorr2*30;

h110 = 0.5 + 1i*0.8;
h111 = 0.2 + 1i*0.2;
h112 = 0.7 + 1i*0.3;
% 
% energy = (abs(h110).^2 + abs(h111).^2 + abs(h112).^2) / 3;
% normalization_factor = 1/sqrt(energy);
% 
% h110 = h110*normalization_factor;
% h111 = h111*normalization_factor;
% h112 = h112*normalization_factor;
% 
% energy = (abs(h110).^2 + abs(h111).^2 + abs(h112).^2) / 3;
% 
% if(delay1 > 0 || delay2 > 0)
%     preambles_delayed1 = [complex(zeros(1,delay1),zeros(1,delay1)), preambles(1,1:end-delay1)];
%     preambles_delayed2 = [complex(zeros(1,delay2),zeros(1,delay2)), preambles(1,1:end-delay2)]; 
%     y_channel = (h110*preambles + h111*preambles_delayed1 + h112*preambles_delayed2);
% else
%     y_channel = h110*preambles;
% end

% preambles_delayed = [complex(zeros(1,delay),zeros(1,delay)), preambles(1,1:end-delay)]; 
% y_channel = preambles_delayed;%h110*preambles_delayed;

y_channel = preambles;

%% ****************************** PRACH Reception ******************************

% ------- CP and GT Removal. -------
rec_signal = y_channel(Ncp+1:NIDFT+Ncp);

if(show_figures)
    figure(5);
    stem(0:1:NIDFT-1,abs(fft(rec_signal,NIDFT)));
    title('Received base-band signal');
end

% ------- Apply DFT to received signal. -------
rec_fft = (1/sqrt(NIDFT))*fft(rec_signal,NIDFT);

% ------- Sub-carrier de-mapping. -------
rec_Xuv = rec_fft(prach_offset+1:prach_offset+Nzc);

% ------- Apply DFT to Root Zadoff-Chu sequence. -------
Xu_root = (1/sqrt(Nzc))*fft(xu_root, Nzc);

% ------- Multiply Local Zadoff-Chu root sequence by received sequence. -------
conj_Xu_root = conj(Xu_root);
multiplied_sequences = (rec_Xuv.*conj_Xu_root);

% ------- Squared modulus used for peak detection. -------
NIFFT_CORR = 839;
pdp_freq = ifft(multiplied_sequences,NIFFT_CORR)/Nzc;

var_pdp_freq = var(pdp_freq);

pdp_freq_adjusted = pdp_freq;

%--------------------------------------------------------------------------
PDP_ADJUST_FACTOR_0 = 1; % delay 0
PDP_ADJUST_FACTOR_1 = 0.989338652615384 + 0.152095818347174i; % delay 1
PDP_ADJUST_FACTOR_2 = 0.957498745916891 + 0.301528270811912i; % delay 2
PDP_ADJUST_FACTOR_3 = 0.904911629047376 + 0.445660126688108i; % delay 3
PDP_ADJUST_FACTOR_10 = 0.050047412688511 + 1.101839958815881i; % delay 10
PDP_ADJUST_FACTOR_13 = -0.473961715008805 + 1.083869828182508i; % delay 13

pdp_freq_adjusted(710) = pdp_freq_adjusted(710);
if(delay1 > 0)
    pdp_freq_adjusted(710+delay_zcorr1) = (PDP_ADJUST_FACTOR_1)*pdp_freq_adjusted(710+delay_zcorr1);
end

if(delay2 > 0)
    pdp_freq_adjusted(710+delay_zcorr2) = (PDP_ADJUST_FACTOR_10)*pdp_freq_adjusted(710+delay_zcorr2);
end

pdp = abs(pdp_freq_adjusted).^2;

stem(0:1:NIFFT_CORR-1,pdp)
%stem(0:1:NIFFT_CORR-1,pdp_freq_adjusted)

h110
h111
h112
pdp_freq_adjusted(710)
pdp_freq_adjusted(710+delay_zcorr1)
pdp_freq_adjusted(710+delay_zcorr2)

