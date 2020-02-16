function torques = gaussian_noise(std_arr)
    torques = [];
    for i =1:size(std_arr,2)
        torque = randn(1)*std_arr(i);
        torques = [torques;torque];
    end
end