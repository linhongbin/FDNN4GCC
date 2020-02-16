% N_arr = [5, 8, 15];
% std_arr = [1, 5, 9];
N_arr = [2,3,4,5,6,7,8,9,10,12,15,17,20];
std_arr = [1];

for j=1:size(std_arr,2)
for i=1:size(N_arr,2)
    train_acrobot_model(N_arr(i), std_arr(j))
end
end

function train_acrobot_model(N, std)
    train_file = sprintf('N%d_std%d\\data\\N%d_std%d.mat', N, std, N, std);
    test_file = sprintf('N%d_std%d\\data\\N%d_std%d.mat', 34, std, 34, std);
    load(train_file)
    R_mat = [];
    tau_vec = [];
    for i = 1:size(input_mat,1)
        q1 = input_mat(i,1);
        q2 = input_mat(i,2);
        R =Regressor(q1,q2);
        R_mat = [R_mat;R];
        tau_vec = [tau_vec; output_mat(i,:).'];
    end

     beta_vec = pinv(R_mat)*tau_vec;


    train_input_mat = input_mat;
    train_output_mat = output_mat;

    load(test_file)
    test_output_mat = [];
    for i = 1:size(input_mat,1)
        q1 = input_mat(i,1);
        q2 = input_mat(i,2);
        tau =Regressor(q1,q2)*beta_vec;
        test_output_mat = [test_output_mat;tau.'];
    end
    test_input_mat = input_mat;

    save(sprintf("N%d_std%d\\result\\dynamic_model.mat", N, std), 'train_input_mat', 'train_output_mat', 'test_input_mat', 'test_output_mat');
end

function R = Regressor(q1,q2)
    s1 =sin(q1);
    s12 = sin(q1+q2);
    R = [s1, s12;0, s12];
end