function torques = Acrobot_dynamics(q1, q2, std_arr, is_extDisturb)
    torques = Acrobot_gravity(q1, q2);
    torques = torques + gaussian_noise(std_arr);
    if is_extDisturb
        torques = torques + extDisturb(q1, q2);
    end
end