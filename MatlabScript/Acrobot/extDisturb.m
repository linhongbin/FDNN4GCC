function torques = extDisturb(q1, q2)
    torques = zeros(2,1);
    torques(1) = 0.2*q1^4+0.2*q1^3+0.1*q1^2+1*q1+2+2*q1*q2;
    torques(2) = 0.2*q2^4+0.2*q2^3+0.1*q2^2+1*q2+2+2*q1*q2;
end