
% --- 3-Mode Hybrid Leader-Follower Setup (Centimeters) ---
dt = 0.05;         
t = 0:dt:20;       

% Normal PID Gains for Mode 2 (You will tune these)
Kp = 2.0; 
Ki = 0.05; 
Kd = 0.5;

E_sum = 0; 
E_prev = 0;
d_desired = 20;    % Target distance in Mode 2 (cm)

% Robot Specs
max_normal_speed = 30; % Max speed in Mode 2 (cm/s)
burst_speed = 60;      % High speed for Mode 3 (cm/s)

% Initial States [Position in cm]
follower_x = 0;    
leader_x = 70;     % Start far away to test Mode 3

follower_history = zeros(1, length(t));
leader_history = zeros(1, length(t));
mode_history = zeros(1, length(t)); % To track which mode is active

for i = 1:length(t)
    % 1. Simulate Leader Movement (moves away, then comes back)
    leader_vel = 15 * sin(0.3 * t(i)); 
    leader_x = leader_x + leader_vel * dt;
    
    % 2. Calculate Actual Distance
    current_distance = leader_x - follower_x;
    
    % --- THE 3-MODE HYBRID LOGIC ---
    if current_distance <= 8
        % MODE 1: Collision Avoidance (Stop)
        follower_vel = 0;
        E_sum = 0; % Prevent integral windup while stopped!
        current_mode = 1;
        
    elseif current_distance >= 60
        % MODE 3: High-Speed Catch-up
        follower_vel = burst_speed;
        E_sum = 0; % Reset integral so it enters Mode 2 cleanly
        current_mode = 3;
        
    else
        % MODE 2: PID Control (Smooth regulation around 20cm)
        error = current_distance - d_desired;
        
        E_sum = E_sum + error * dt;                  
        deriv = (error - E_prev) / dt;               
        
        follower_vel = (Kp * error) + (Ki * E_sum) + (Kd * deriv);
        E_prev = error;
        
        % Clamp to normal motor speed limit
        follower_vel = max(min(follower_vel, max_normal_speed), -max_normal_speed);
        current_mode = 2;
    end
    
    % 3. Update Follower Position (Kinematics)
    % Prevent the robot from driving backwards if the leader gets too close
    follower_vel = max(0, follower_vel); 
    
    follower_x = follower_x + follower_vel * dt;
    
    % Store for plotting
    leader_history(i) = leader_x;
    follower_history(i) = follower_x;
    mode_history(i) = current_mode;
end

% --- Plotting ---
figure;
subplot(2,1,1);
plot(t, leader_history, 'b', 'LineWidth', 1.5); hold on;
plot(t, follower_history, 'r', 'LineWidth', 1.5);
plot(t, leader_history - d_desired, 'k--', 'LineWidth', 1);
title('Hybrid State + PID Follower (cm)');
ylabel('Position (cm)');
legend('Leader', 'Follower', '20cm Target');
grid on;

subplot(2,1,2);
plot(t, mode_history, 'g', 'LineWidth', 1.5);
title('Active Operational Mode');
xlabel('Time (s)'); ylabel('Mode (1, 2, or 3)');
yticks([1 2 3]);
grid on;