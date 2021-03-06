clear all;close all;clc

P = 4;

N = 7; %89;%37;

u = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571];

n = (0:1:N-1);

u = 139;

Pilot = exp((-1i.*pi.*u.*n.*(n-1))./N);

S = zeros(N,P);

for Cv=0:1:P-1
    
    pos = mod((n-Cv),N)+1;
    
    S(:,Cv+1) = Pilot(pos(1:N)).';
   
end

res = (S')*S;

res2 = S(2,1:79)*S(5,1:79)';


res3 = S(1:12,:)*S(1:12,:)';