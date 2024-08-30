addpath(fullfile('..', 'src'));

%% Set constants
Ts = 1/20; % Sample time
H = 7; % Horizon length of 7 seconds
Tf = 30; % Simulation end time

%% Design MCP Controller
% Initialize objects
rocket = Rocket(Ts);
[xs, us] = rocket.trim(); % Compute steady−state for which 0 = f(xs,us)
sys = rocket.linearize(xs, us); % Linearize the nonlinear model about trim point
[sys_x, sys_y, sys_z, sys_roll] = rocket.decompose(sys, xs, us);

% Design MPC Controller
mpc_x = MpcControl_x(sys_x, Ts, H);
mpc_y = MpcControl_y(sys_y, Ts, H);
mpc_z = MpcControl_z(sys_z, Ts, H);
mpc_roll = MpcControl_roll(sys_roll, Ts, H);

% Merge four sub−system controllers into one full−system controller,
% accounting for linearization point
mpc = rocket.merge_lin_controllers(xs, us, mpc_x, mpc_y, mpc_z, mpc_roll);

%% Plot simulation
% % Open-loop trajectory
x0 = zeros(12,1);
ref4 = [2 2 2 deg2rad(40)]';
[u, T_opt, X_opt, U_opt] = mpc.get_u(x0, ref4);
U_opt(:,end+1) = NaN;

ph = rocket.plotvis(T_opt, X_opt, U_opt, ref4); % Plot as usual

% Closed-loop trajectory 
ref = @(t_,x_)ref_TVC(t_); % Setup reference function
[T, X, U, Ref] = rocket.simulate(x0, Tf, @mpc.get_u, ref); % Simulate
rocket.anim_rate = 10; % Increase this to make the animation faster
ph = rocket.plotvis(T, X, U, Ref);
ph.fig.Name = 'Merged lin. MPC in nonlinear simulation'; % Set a figure title

%close all
%clear all
%clc

%% TODO: This file should produce all the plots for the deliverable