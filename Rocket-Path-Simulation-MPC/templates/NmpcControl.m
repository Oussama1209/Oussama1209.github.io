classdef NmpcControl < handle
    
    properties
        solver
        nx, nu, N
        nlp_x0
        nlp_lbx, nlp_ubx
        nlp_lbg, nlp_ubg
        nlp_p
        
        T_opt
        sol
        idx 
        
        % Delay compensation
        rocket
        expected_delay
        mem_u
        
        % Warmstart
        nlp_lam_x0
        nlp_lam_g0
    end
    
      methods
        function obj = NmpcControl(rocket, tf, expected_delay)
            
            if nargin < 3, expected_delay = 0; end
           
            import casadi.*
            
            N_segs = ceil(tf/rocket.Ts); % MPC horizon
            nx = 12; % Number of states
            nu = 4;  % Number of inputs
            
            % Establish symbolic variables for decision making
            N = N_segs + 1;  % Calculation of the index for the last point
            % Define the state and control trajectories as symbolic expressions
            X_sym = SX.sym('X_sym', nx, N);  % Trajectory of states across the horizon
            U_sym = SX.sym('U_sym', nu, N-1);  % Control trajectory, one less than state trajectory
            
            % Symbolic representations for parameters
            x0_sym = SX.sym('x0_sym', nx, 1);  % Symbolic variable for the initial state
            ref_sym = SX.sym('ref_sym', 4, 1);  % Symbolic variable for the target position
            
            % Default state and input constraints
            ubx = inf(nx, 1);
            lbx = -inf(nx, 1);
            ubu = inf(nu, 1);
            lbu = -inf(nu, 1);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE
            
            %Obtaining Linear parameters
            [xs, us] = rocket.trim ();
            sys6 = rocket.linearize (xs,us);
            sys6 = c2d(sys6, rocket.Ts);

            % Establishing the RK4 method for system discretization
            f_disc = @(x,u) RK4(x, u, rocket.Ts, @rocket.f);

            % Cost
            cost = 0;
            
            % Equality constraints (Casadi SX), each entry == 0
            eq_constr = [ ; ];
            
            % Inequality constraints (Casadi SX), each entry <= 0
            ineq_constr = [ ; ];

            % Defining a high symbolic value to represent infinity in constraints
            Infi_sys = 10^10;
            
            % Setting bounds for state and control variables
            for i = 1:nx 
                ubx(i,1) =  Infi_sys;
                lbx(i,1) = -Infi_sys;
            end

            % For box constraints on state and input, overwrite entries of
            % lbx, ubx, lbu, ubu defined above
            
            % Assigning specific bounds to certain state and control variables
            lbx (5,1) = -deg2rad(75);
            ubx (5,1) =  deg2rad(75);

            lbu (1,1) = -0.26;
            ubu (1,1) = 0.26;
            lbu (2,1) = -0.26;
            ubu (2,1) = 0.26;
            lbu (3,1) = 50;
            ubu (3,1) = 80;
            lbu (4,1) = -20;
            ubu (4,1) = 20;

            % Initialize ref_vect as a symbolic vector filled with zeros
            ref_vect = SX.zeros(12, 1); 
            
            % Assign elements from ref_sym to the appropriate positions in ref_vect
            ref_vect(6) = ref_sym(4);
            ref_vect(10) = ref_sym(1);
            ref_vect(11) = ref_sym(2);
            ref_vect(12) = ref_sym(3);
            
            % Adding initial state constraint to the equality constraints
            eq_constr = [eq_constr; X_sym(:,1)-...
                f_disc(x0_sym, zeros(nu,1))];
            
            % Defining the weighting matrices for the cost function
            Q = eye(nx,nx);
            Q(1,1) = 5;
            Q(2,2) = 5;
            Q(3,3) = 5;
            Q(4,4) = 10;
            Q(5,5) = 10;
            Q(6,6) = 100;
            Q(7,7) = 1;
            Q(8,8) = 1;
            Q(9,9) = 1;
            Q(10,10) = 50;
            Q(11,11) = 50;
            Q(12,12) = 100;

            R = eye(nu,nu);
            R(1,1) = 10;
            R(2,2) = 10;
            R(3,3) = 0.001;
            R(4,4) = 0.01;
            
            % Calculating the cost associated with the initial state
            cost_x0 = (x0_sym-ref_vect)'*Q*(x0_sym-ref_vect);

            cost = cost_x0;

            for i = 1:N-1
                % Updating equality and inequality constraints
                eq_constr = [eq_constr; X_sym(:,i+1) - ...
                    f_disc(X_sym(:,i), U_sym(:,i))];

                ineq_constr = [ineq_constr; lbu - U_sym(:,i)];
                ineq_constr = [ineq_constr; U_sym(:,i) - ubu];
                ineq_constr = [ineq_constr; lbx - (X_sym(:,i))];
                ineq_constr = [ineq_constr; X_sym(:,i) - ubx];

                % Computing cost for each state and control input
                cost_x=(X_sym(:,i)-ref_vect)'*Q*(X_sym(:,i)-ref_vect) + ...
                    (U_sym(:,i))'*R*(U_sym(:,i));

                cost = cost + cost_x;
            end

            % Applying final inequality constraints for the terminal state
            ineq_constr = [ineq_constr; lbx - (X_sym(:,N))];
            ineq_constr = [ineq_constr; X_sym(:,N) - ubx];
            
   
            % YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % ---- Assemble NLP ------
            nlp_x = [X_sym(:); U_sym(:)];
            nlp_p = [x0_sym; ref_sym];
            nlp_f = cost;
            nlp_g = [eq_constr; ineq_constr];
            
            nlp = struct('x', nlp_x, 'p', nlp_p, 'f', nlp_f, 'g', nlp_g);
            
            % ---- Setup solver ------
            opts = struct('ipopt', struct('print_level', 0), 'print_time', false);
            obj.solver = nlpsol('solver', 'ipopt', nlp, opts);
            
            % ---- Assemble NLP bounds ----
            obj.nlp_x0  = zeros(size(nlp_x));
            
            obj.nlp_ubx = [repmat(ubx, N, 1); repmat(ubu, (N-1), 1)];
            obj.nlp_lbx = [repmat(lbx, N, 1); repmat(lbu, (N-1), 1)];
            
            obj.nlp_ubg = [zeros(size(eq_constr)); zeros(size(ineq_constr))];
            obj.nlp_lbg = [zeros(size(eq_constr)); -inf(size(ineq_constr))];
            
            obj.nlp_p = [zeros(size(x0_sym)); zeros(size(ref_sym))];
            
            obj.nlp_lam_x0 = [];
            obj.nlp_lam_g0 = [];
            
            obj.nx = nx;
            obj.nu = nu;
            obj.N = N;
            obj.T_opt = linspace(0, N * rocket.Ts, N);
            
            obj.idx.X = [1, obj.N * obj.nx];
            obj.idx.U = obj.idx.X(2) + [1, (obj.N-1) * obj.nu];
            obj.idx.u0 = obj.idx.U(1) + [0, obj.nu-1];
            
            % Members for delay compensation
            obj.rocket = rocket;
            obj.expected_delay = expected_delay;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE

            u_init = us; % Set to the desired setpoint
            
            % YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            obj.mem_u = repmat(u_init, 1, expected_delay);
        end
        
        function [u, T_opt, X_opt, U_opt] = get_u(obj, x0, ref)
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE
            delay = obj.expected_delay;
            mem_u = obj.mem_u;
            
            % Delay compensation: Predict x0 delay timesteps later.
            % Simulate x_ for 'delay' timesteps
            x_ = x0;
            
            % Euler integration
            for i=1:delay
                x_ = x_ + obj.rocket.Ts*obj.rocket.f(x_,mem_u(:,i));
            end
       
            x0 = x_;
            % YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % Compute solution from x0
            obj.solve(x0, ref);
            
            % Evaluate u0
            nlp_x = obj.sol.x;
            id = obj.idx.u0;
            u = full( nlp_x(id(1):id(2)) );      
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE
            % Delay compensation: Save current u
            if obj.expected_delay > 0
                shifted_mem_u = mem_u(:, 2:end);
                obj.mem_u = [shifted_mem_u, u];
            end
            % YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE YOUR CODE HERE
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if nargout > 1, T_opt = obj.get_T_opt(); end
            if nargout > 2, X_opt = obj.get_X_opt(); end
            if nargout > 3, U_opt = obj.get_U_opt(); end
            return
            
            % Additional evaluation
            % Complete trajectory
            % % X_opt = full(reshape(nlp_x(idx_X(1):idx_X(2)), obj.nx, obj.N));
            % % U_opt = full(reshape(nlp_x(idx_U(1):idx_U(2)), obj.nu, obj.N - 1));
            % %
            % % cost_opt = full(sol.f);
            % % constr_opt = full(sol.g);
            % %
            % % stats = obj.solver.stats;
        end
        
        function solve(obj, x0, ref)
            
            % ---- Set the initial state and reference ----
            obj.nlp_p = [x0; ref];     % Initial condition
            obj.nlp_x0(1:obj.nx) = x0; % Initial guess consistent
            
            % ---- Solve the optimization problem ----
            args = {'x0', obj.nlp_x0, ...
                'lbg', obj.nlp_lbg, ...
                'ubg', obj.nlp_ubg, ...
                'lbx', obj.nlp_lbx, ...
                'ubx', obj.nlp_ubx, ...
                'p', obj.nlp_p, ...
                %                 'lam_x0', obj.nlp_lam_x0, ...
                %                 'lam_g0', obj.nlp_lam_g0
                };
            
            obj.sol = obj.solver(args{:});
            if obj.solver.stats.success ~= true
                solve_status_str = obj.solver.stats.return_status;
                fprintf([' [' class(obj) ': ' solve_status_str '] ']);
                obj.sol.x(obj.idx.u0) = nan;
            end
            
            % Use the current solution to speed up the next optimization
            obj.nlp_x0 = obj.sol.x;
            obj.nlp_lam_x0 = obj.sol.lam_x;
            obj.nlp_lam_g0 = obj.sol.lam_g;
        end
        function T_opt = get_T_opt(obj)
            T_opt = obj.T_opt;
        end
        function X_opt = get_X_opt(obj)
            nlp_x = obj.sol.x;
            id = obj.idx.X;
            X_opt = full(reshape(nlp_x(id(1):id(2)), obj.nx, obj.N));
        end
        function U_opt = get_U_opt(obj)
            nlp_x = obj.sol.x;
            id = obj.idx.U;
            U_opt = full(reshape(nlp_x(id(1):id(2)), obj.nu, obj.N - 1));
        end
    end
end

