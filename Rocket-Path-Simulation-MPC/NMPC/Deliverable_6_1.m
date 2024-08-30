addpath(fullfile('/Users/user/Documents/EPFL/MA1/MPC/rocket_project/'));

%% TODO: This file should produce all the plots for the deliverable
Ts = 1/20;
rocket = Rocket(Ts);
Tf = 30;

H = 5; % Horizon length in seconds
nmpc = NmpcControl(rocket, H);

% Open-loop 
x0 = zeros(12, 1);
ref6 = [2 2 2 deg2rad(40)]';
[u, T, X, U] = nmpc.get_u(x0, ref6);
U(:,end+1) = nan;
ph6_0 = rocket.plotvis(T, X, U, ref6);

% MPC reference with default maximum roll = 15 deg
ref = @(t_ , x_ ) ref_TVC(t_);
[T, X, U, Ref] = rocket.simulate(x0, Tf, @nmpc.get_u, ref);
rocket.anim_rate = 5;
ph6_1 = rocket.plotvis(T, X, U, Ref);

% MPC reference with specified maximum roll = 50 deg
roll_max = deg2rad(50);
ref = @(t_ , x_ ) ref_TVC(t_ , roll_max);
[T, X, U, Ref] = rocket.simulate(x0, Tf, @nmpc.get_u, ref);
rocket.anim_rate = 5;
ph6_2 = rocket.plotvis(T, X, U, Ref);
