function param_vec_to_latex(vec)
syms L2 L3 L3_parallel L4_z0
    output_list = {};
    for i=1:size(vec,1)
        if i ==1
            a = coeffs(vec(i), L2);
            vec(i) = L2*(a(2))+a(1);
        end
        if i ==3
            a = coeffs(vec(i), L3);
            b = coeffs(a(1), L3_parallel);
            vec(i) = L3*(a(2))+ L3_parallel*(b(2)) +b(1);
        end
        if i ==4
            a = coeffs(vec(i), L4_z0);
            vec(i) = L4_z0*(a(2))+a(1);
        end
        parse_str = char(vec(i));
        parse_str = insertAfter(parse_str,"m","_");
        parse_str = insertAfter(parse_str,"L","_") ;
        parse_str = erase(parse_str,"*");
        parse_str = strrep(parse_str,'_parallel',"^'");
        parse_str = strrep(parse_str,'_parallel',"^'");
        parse_str = strrep(parse_str,"^'_x","_x^'");
        parse_str = strrep(parse_str,"^'_y","_y^'");
        parse_str = strrep(parse_str,"^'_z","_z^'");
        parse_str = strrep(parse_str,"'", "{'}");
        for i=1:6
            parse_str = strrep(parse_str,sprintf("drift%d_pos", i), sprintf("^%da^{+}_0", i));
        end
        for i=1:6
            parse_str = strrep(parse_str,sprintf("drift%d_neg", i), sprintf("^%da^{-}_0", i));
        end
        for i=1:6
            for j=1:4
                parse_str = strrep(parse_str,sprintf("a%d_%d_pos", i, j), sprintf("^%da^{+}_%d", i,j));
            end
        end
        for i=1:6
            for j=1:4
                parse_str = strrep(parse_str,sprintf("a%d_%d_neg", i, j), sprintf("^%da^{-}_%d", i,j));
            end
        end
        

        for i=1:6
            parse_str = strrep(parse_str,sprintf("_%d_x", i), sprintf("_{%dx}", i));
            parse_str = strrep(parse_str,sprintf("_%d_y", i), sprintf("_{%dy}", i));
            parse_str = strrep(parse_str,sprintf("_%d_z", i), sprintf("_{%dz}", i));
        end
        parse_str = strrep(parse_str, "L_2", "l_{arm}");
        parse_str = strrep(parse_str, "L_3", "l_{forearm}");
        parse_str = strrep(parse_str, "L_{4z}0", "h");
        %disp(sprintf('%s', parse_str));
        output_list{end+1} = parse_str;
    end
    
    %to table
    disp(sprintf("\\begin{table*}[t]"));
    disp(sprintf("\\centering"));
    disp(sprintf("\\begin{tabular}{|c|c|}"));
    
    for i=1:5
        disp(sprintf("\\hline"));
        disp(sprintf("\\multirow{2}{*}{${}^g\\beta_%d$}& $%s$\\\\\\cline{2-2}", i+1, output_list{(i-1)*2+1}));
        disp(sprintf("& $%s$ \\\\", output_list{i*2}));
    end
 
    
    for i=1:6
        disp(sprintf("\\hline"));
        disp(sprintf("${}^{ext}\\beta^{+}_%d$ & $%s$,$%s$,$%s$,$%s$,$%s$\\\\", i, output_list{6+i*5}, output_list{7+i*5}, output_list{8+i*5}, output_list{9+i*5}, output_list{10+i*5}));
    end
    
    for i=1:6
        disp(sprintf("\\hline"));
        disp(sprintf("${}^{ext}\\beta^{-}_%d$ & $%s$,$%s$,$%s$,$%s$,$%s$\\\\", i, output_list{36+i*5}, output_list{37+i*5}, output_list{38+i*5}, output_list{39+i*5}, output_list{40+i*5}));
    end
    
    % end of table
     fprintf("\\hline \n")
     fprintf("\\end{tabular}\n")
     fprintf("\\caption{Blabla}\n")
     fprintf("\\label{tab:1}\n")
     fprintf("\\end{table*}\n")
end