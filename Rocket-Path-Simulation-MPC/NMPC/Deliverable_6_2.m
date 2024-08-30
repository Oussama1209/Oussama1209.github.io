addpath(fullfile('/Users/user/Documents/EPFL/MA1/MPC/rocket_project/'));

%% TODO: This file should produce all the plots for the deliverable
Ts = 1/40; % Higher sampling rate for this part!

... Define NMPC ...
x0 = zeros(12, 1);
ref = [0.5, 0, 1, deg2rad(65)]';
Tf = 2.5;
rocket.mass = 1.75;

rocket = Rocket(Ts);
H = 5 % Horizon length in seconds

% Simulation with no delay 
nmpc = NmpcControl(rocket, H);
[T, X, U, Ref] = rocket.simulate(x0, Tf, @nmpc.get_u, ref);
rocket.anim_rate = 5;
ph6_1 = rocket.plotvis(T, X, U, Ref);

% Simulation with uncompensated delay of 1
rocket.delay = 1;
rocket_delay = 0;
nmpc = NmpcControl(rocket, H, rocket_delay);
[T, X, U, Ref] = rocket.simulate(x0, Tf, @nmpc.get_u, ref);
rocket.anim_rate = 5; 
ph = rocket.plotvis(T, X, U, Ref);

% Simulation with uncompensated delay of 2
rocket.delay = 2;
rocket_delay = 0;
nmpc = NmpcControl(rocket, H, rocket_delay);
[T, X, U, Ref] = rocket.simulate(x0, Tf, @nmpc.get_u, ref);
rocket.anim_rate = 5; 
ph = rocket.plotvis(T, X, U, Ref);

% Simulation with uncompensated delay of 3
rocket.delay = 3;
rocket_delay = 0;
nmpc = NmpcControl(rocket, H, rocket_delay);
[T, X, U, Ref] = rocket.simulate(x0, Tf, @nmpc.get_u, ref);
rocket.anim_rate = 5; 
ph = rocket.plotvis(T, X, U, Ref);

% Simulation with a patially compensated delay
rocket.delay = 3;
rocket_delay = 2;
nmpc = NmpcControl(rocket, H, rocket_delay);
[T, X, U, Ref] = rocket.simulate(x0, Tf, @nmpc.get_u, ref);
rocket.anim_rate = 5; 
ph = rocket.plotvis(T, X, U, Ref);

% Simulation with a fully compensated delay
rocket.delay = 3;
rocket_delay = 3;
nmpc = NmpcControl(rocket, H, rocket_delay);
[T, X, U, Ref] = rocket.simulate(x0, Tf, @nmpc.get_u, ref);
rocket.anim_rate = 5;
ph = rocket.plotvis(T, X, U, Ref);