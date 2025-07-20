function overlay_img = imoverlay_transparent(base_img, masks, colors, alphas)
    % Ensure the inputs are double and normalized
    base_img = im2double(base_img); % Normalize base image
    
    % Ensure the base image is RGB
    if size(base_img, 3) == 1
        base_img = repmat(base_img, [1, 1, 3]); % Convert grayscale to RGB
    end
    
    % Initialize overlay image as the base image
    overlay_img = base_img;

    % Loop through all masks
    for i = 1:length(masks)
        mask = double(masks{i}); % Convert current mask to double
        color = colors{i};       % Get the current color
        alpha = alphas{i};       % Get the current transparency

        % Handle shorthand color names
        switch color
            case 'r', color = [1, 0, 0]; % Red
            case 'g', color = [0, 1, 0]; % Green
            case 'b', color = [0, 0, 1]; % Blue
            case 'y', color = [1, 1, 0]; % Yellow
            case 'm', color = [1, 0, 1]; % Magenta
            case 'c', color = [0, 1, 1]; % Cyan
            case 'k', color = [0, 0, 0]; % Black
            case 'w', color = [1, 1, 1]; % White
        end

        % Create a colored version of the mask
        color_mask = cat(3, color(1) * mask, color(2) * mask, color(3) * mask);

        % Blend the base image and the color mask using transparency (alpha)
        overlay_img = (1 - alpha) * overlay_img + alpha * color_mask;
    end
end
