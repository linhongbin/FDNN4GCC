
[Regressor_Matrix_Pos,Regressor_Matrix_Neg, ~, bool_Regressor_Mat] = symbolic_gc_dynamic(false);
symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat);


train_file = "train\data\D5N5_POS.mat";
test_file = "test\data\rand319.mat";
load(train_file)
R_mat = [];
tau_vec = [];
for i = 1:size(input_mat,1)
    q2 = input_mat(i,1);
    q3 = input_mat(i,2);
    q4 = input_mat(i,3);
    q5 = input_mat(i,4);
    q6 = input_mat(i,5);
    torque = output_mat(i,:).';
    tau_vec = [tau_vec;torque];
    R = analytical_regressor_pos_mat(9.81,q2,q3,q4,q5,q6);
    R_mat = [R_mat;R(2:6,:)];
end

 beta_vec = pinv(R_mat)*tau_vec;


train_input_mat = input_mat;
train_output_mat = output_mat;

load(test_file)
test_input_mat = input_mat;
test_output_mat = output_mat;
target_mat = [];
for i = 1:size(input_mat,1)
    q2 = input_mat(i,1);
    q3 = input_mat(i,2);
    q4 = input_mat(i,3);
    q5 = input_mat(i,4);
    q6 = input_mat(i,5);
    tau =analytical_regressor_pos_mat(9.81,q2,q3,q4,q5,q6)*beta_vec;
    target_mat = [target_mat;tau.'];
end
target_mat = target_mat(:,2:6);


rel_rms = sqrt(sum((target_mat-test_output_mat).^2)./sum((test_output_mat).^2));
rel_rms_vec = rel_rms.'