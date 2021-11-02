function [Dux,Duy] = ForwardD(U)
            % Forward finite difference operator
            Dux = [diff(U,1,2), U(:,1) - U(:,end)];
            Duy = [diff(U,1,1); U(1,:) - U(end,:)];
        end