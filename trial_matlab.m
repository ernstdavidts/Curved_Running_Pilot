clear; clc;

load('Shoe25OpenSim.mat')

vars = {'ANGLES_TABLE','ANGLES_TABLE_FILT','ANGULAR_VELOCITY_TABLE_FILT',...
        'GRF_TABLE','MARKERS','CONTACT_ANALOG','CONTACT_KINEMATIC'};

labels = struct();

for v = 1:length(vars)

    varname = vars{v};

    if exist(varname,'var')

        S = eval(varname);
        f = fieldnames(S);

        for i = 1:length(f)

            name = f{i};
            value = S.(name);

            if istable(value)

                labels.(varname).(name) = value.Properties.VariableNames;
                S.(name) = table2array(value);

            end

        end

        eval([varname ' = S;'])

    end

end

save('ShoeOpenSim_python.mat','ANGLES_TABLE','ANGLES_TABLE_FILT',...
'ANGULAR_VELOCITY_TABLE_FILT','CONTACT_ANALOG','CONTACT_KINEMATIC',...
'GRF_TABLE','MARKERS','OPTIONS','labels','-v7')