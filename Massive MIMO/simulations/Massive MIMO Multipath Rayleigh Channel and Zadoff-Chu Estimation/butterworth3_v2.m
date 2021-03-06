function Hd = butterworth3_v2
%BUTTERWORTH3_V2 Returns a discrete-time filter object.

% MATLAB Code
% Generated by MATLAB(R) 8.3 and the DSP System Toolbox 8.6.
% Generated on: 26-Mar-2015 09:14:56

% Butterworth Lowpass filter designed using FDESIGN.LOWPASS.

% All frequency values are normalized to 1.

N  = 44;    % Order
Fc = 0.09;  % Cutoff Frequency

% Construct an FDESIGN object and call its BUTTER method.
h  = fdesign.lowpass('N,F3dB', N, Fc);
Hd = design(h, 'butter');

% [EOF]
