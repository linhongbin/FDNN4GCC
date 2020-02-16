MTM_model = MTM_DH_Model;
PSM_model = PSM_DH_Model;
q_mtm = zeros(7,1);
q_psm = zeros(6,1);
[T,Jacob] = FK_Jacob_Geometry(q_mtm, MTM_model.DH, MTM_model.tip ,MTM_model.method)