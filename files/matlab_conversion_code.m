clear; clc;

load('Shoe55OpenSim.mat')

vars = {'ANGLES_TABLE','ANGLES_TABLE_FILT','ANGULAR_VELOCITY_TABLE_FILT',...
        'GRF_TABLE','MARKERS','CONTACT_ANALOG','CONTACT_KINEMATIC'};

for v = 1:length(vars)
    
    varname = vars{v};
    
    if exist(varname,'var')
        
        disp(['Processing ', varname])
        
        S = eval(varname);
        f = fieldnames(S);
        
        for i = 1:length(f)
            
            name = f{i};
            value = S.(name);
            
            if istable(value)
                
                % convert table to numeric
                data = table2array(value);
                
                % store as matrix instead of table
                S.(name) = data;
                
            end
            
        end
        
        eval([varname ' = S;'])
        
    end
    
end

save('Shoe55OpenSim_python_2.mat','-v7')

disp("Conversion finished")