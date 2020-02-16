function [h1,h2,h3] =  symbolic_to_analytical(Regressor_Matrix_Pos, Regressor_Matrix_Neg, bool_Regressor_Mat)
    if ~exist('./gen_code', 'dir')
       mkdir('./gen_code')
    end
   h1 =  matlabFunction(Regressor_Matrix_Pos,'File','./gen_code/analytical_regressor_pos_mat.m');
   h2 =  matlabFunction(Regressor_Matrix_Neg,'File','./gen_code/analytical_regressor_neg_mat.m');
   h3 =  matlabFunction(bool_Regressor_Mat,'File','./gen_code/analytical_bool_regressor_mat.m');
end





