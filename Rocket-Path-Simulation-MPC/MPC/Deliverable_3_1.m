addpath(fullfile('..', 'src'));

%% Set constants
Ts = 1/20; % Sample time
H = 7; % Horizon length of 7 seconds
Tf = 7; % Simulation end time
TypeLoop = 'closed'; % open, closed, or both --> for testing purposes
Dimension = 'all'; % x, y, z, roll, or all --> for testing purposes


%% Initialize objects
rocket = Rocket(Ts);
[xs, us] = rocket.trim(); % Compute steady−state for which 0 = f(xs,us)
sys = rocket.linearize(xs, us); % Linearize the nonlinear model about trim point
[sys_x, sys_y, sys_z, sys_roll] = rocket.decompose(sys, xs, us);

%% X
if strcmp(Dimension,'x') || strcmp(Dimension,'all')
    mpc_x = MpcControl_x(sys_x, Ts, H); % Design MCP Controller for sub-system x
    %    wy  beta  vx  x     % Initial state
    x0 = [0,  0,   0,  3]';

    if strcmp(TypeLoop,'open') || strcmp(TypeLoop,'both')
        [u, T_opt, X_opt, U_opt] = mpc_x.get_u(x0);
        U_opt(:,end+1) = NaN;
        ph = rocket.plotvis_sub(T_opt, X_opt, U_opt, sys_x, xs, us); % Plot as usual
        ph.fig.Name = 'X Regulator, Open Loop'; % Set a figure title
    end

    if strcmp(TypeLoop,'closed') || strcmp(TypeLoop,'both')
        [T, X_sub, U_sub] = rocket.simulate_f(sys_x, x0, Tf, @mpc_x.get_u, 0);
        ph = rocket.plotvis_sub(T, X_sub, U_sub, sys_x, xs, us);
        ph.fig.Name = 'X Regulator, Closed Loop'; % Set a figure title
    end

end


%% Y
if strcmp(Dimension,'y') || strcmp(Dimension,'all')
    mpc_y = MpcControl_y(sys_y, Ts, H); % Design MCP Controller for sub-system y
    %    wx alpha  vy   y     % Initial state
    y0 = [0,  0,   0,   3]';

    if strcmp(TypeLoop,'open') || strcmp(TypeLoop,'both')
        [u, T_opt, X_opt, U_opt] = mpc_y.get_u(y0);
        U_opt(:,end+1) = NaN;

        %Account for linearization point:
        U_opt = U_opt + us(2);
        xs_y = repmat(xs(5:8), 1, length(X_opt));
        X_opt = X_opt + xs_y;

        ph = rocket.plotvis_sub(T_opt, X_opt, U_opt, sys_y, xs, us); % Plot as usual
        ph.fig.Name = 'Y Regulator, Open Loop'; % Set a figure title
    end

    if strcmp(TypeLoop,'closed') || strcmp(TypeLoop,'both')
        [T, X_sub, U_sub] = rocket.simulate_f(sys_y, y0, Tf, @mpc_y.get_u, 0);
        ph = rocket.plotvis_sub(T, X_sub, U_sub, sys_y, xs, us);
        ph.fig.Name = 'Y Regulator, Closed Loop'; % Set a figure title
    end

end


%% Z
if strcmp(Dimension,'z') || strcmp(Dimension,'all')
    mpc_z = MpcControl_z(sys_z, Ts, H); % Design MCP Controller for sub-system z
    %    vz  z     % Initial state
    z0 = [0, 3]';

    if strcmp(TypeLoop,'open') || strcmp(TypeLoop,'both')
        [u, T_opt, X_opt, U_opt] = mpc_z.get_u(z0);
        U_opt(:,end+1) = NaN;

        %Account for linearization point:
        U_opt = U_opt + us(3);
        xs_z = repmat(xs(9:10), 1, length(X_opt));
        X_opt = X_opt + xs_z;

        ph = rocket.plotvis_sub(T_opt, X_opt, U_opt, sys_z, xs, us); % Plot as usual
        ph.fig.Name = 'Z Regulator, Open Loop'; % Set a figure title
    end

    if strcmp(TypeLoop,'closed') || strcmp(TypeLoop,'both')
        [T, X_sub, U_sub] = rocket.simulate_f(sys_z, z0, Tf, @mpc_z.get_u, 0);
        ph = rocket.plotvis_sub(T, X_sub, U_sub, sys_z, xs, us);
        ph.fig.Name = 'Z Regulator, Closed Loop'; % Set a figure title
    end

end


%% Roll
if strcmp(Dimension,'roll') || strcmp(Dimension,'all')
    mpc_roll = MpcControl_roll(sys_roll, Ts, H); % Design MCP Controller for sub-system roll
    %        wz     gamma           % Initial state
    roll0 = [0,  deg2rad(30)]';

    if strcmp(TypeLoop,'open') || strcmp(TypeLoop,'both')
        [u, T_opt, X_opt, U_opt] = mpc_roll.get_u(roll0);
        U_opt(:,end+1) = NaN;

        %Account for linearization point:
        U_opt = U_opt + us(4);
        xs_roll = repmat(xs(11:12), 1, length(X_opt));
        X_opt = X_opt + xs_roll;

        ph = rocket.plotvis_sub(T_opt, X_opt, U_opt, sys_roll, xs, us); % Plot as usual
        ph.fig.Name = 'Roll Regulator, Open Loop'; % Set a figure title
    end

    if strcmp(TypeLoop,'closed') || strcmp(TypeLoop,'both')
        [T, X_sub, U_sub] = rocket.simulate_f(sys_roll, roll0, Tf, @mpc_roll.get_u, 0);
        ph = rocket.plotvis_sub(T, X_sub, U_sub, sys_roll, xs, us);
        ph.fig.Name = 'Roll Regulator, Closed Loop'; % Set a figure title
    end
    
end

%close all
%clear all
%clc

%% TODO: This file should produce all the plots for the deliverable