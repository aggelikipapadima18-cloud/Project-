addpath('../Images/Ασκηση 4');
I = im2double(imread('../Images/Ασκηση 4/new_york.png'));
delta = zeros(size(I));
delta(1,1) = 1;
H_impulse = psf(delta);
blurred = psf(I);
save('../outputs/askisi4_restoration_cpp/psf_export.mat', 'H_impulse', 'blurred');
disp('Saved psf_export.mat');
